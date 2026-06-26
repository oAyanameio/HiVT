# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytorch_lightning as pl
import pytorch_lightning.trainer.connectors.logger_connector.result as pl_result
import pytorch_lightning.utilities.data as pl_data
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import LaplaceNLLLoss
from losses import RiskCalibrationLoss
from losses import RiskRankLoss
from losses import SoftTargetCrossEntropyLoss
from metrics import ADE
from metrics import AUROC
from metrics import BrierScore
from metrics import ECE
from metrics import FDE
from metrics import MR
from models import GlobalInteractor
from models import LocalEncoder
from models import MLPDecoder
from models import ReliabilityModule
from models import build_reliability_targets
from models import reconstruct_lane_positions
from models import summarize_reliability_targets
from utils import TemporalData
from utils import extract_lightning_batch_size


pl_data.extract_batch_size = extract_lightning_batch_size
pl_result.extract_batch_size = extract_lightning_batch_size


class HiVT(pl.LightningModule):
    """HiVT 主模型。

    论文对应关系：
    1. `LocalEncoder` 对应局部时空编码部分，用于抽取 actor 的局部历史交互特征。
    2. `GlobalInteractor` 对应全局交互模块，用于建模 agent 级别的全局关系。
    3. `MLPDecoder` 对应多模态轨迹解码头，输出 K 条未来轨迹及其概率。
    """

    def __init__(self,
                 historical_steps: int,
                 future_steps: int,
                 num_modes: int,
                 rotate: bool,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float,
                 num_temporal_layers: int,
                 num_global_layers: int,
                 local_radius: float,
                 parallel: bool,
                 use_reliability: bool,
                 reliability_hidden_dim: int,
                 reliability_rerank_alpha: float,
                 reliability_loss_weight: float,
                 scene_loss_weight: float,
                 rank_loss_weight: float,
                 calib_loss_weight: float,
                 risk_fde_threshold: float,
                 risk_conflict_threshold: float,
                 risk_offroad_threshold: float,
                 lr: float,
                 weight_decay: float,
                 T_max: int,
                 **kwargs) -> None:
        """初始化 HiVT。

        Args:
            historical_steps: 历史轨迹长度 T_h，论文与数据处理中默认使用 20 帧历史。
            future_steps: 未来预测长度 T_f，默认预测 30 帧。
            num_modes: 多模态预测条数 K，即每个 actor 预测多少条候选未来轨迹。
            rotate: 是否将坐标旋转到各 actor 的局部朝向坐标系中。
            node_dim: actor 节点输入维度；本仓库中通常是二维位移/位置特征。
            edge_dim: 边特征维度；这里主要是相对位移向量的维度。
            embed_dim: 隐表示维度 D，也是 HiVT-64/128 中的 64/128。
            num_heads: 多头注意力头数。
            dropout: dropout 比例。
            num_temporal_layers: 时间编码器的 Transformer 层数。
            num_global_layers: 全局交互模块的层数。
            local_radius: 局部交互与 lane-actor 建边时使用的距离阈值。
            parallel: 是否并行计算每个历史时刻的 actor-actor 编码。
            lr: AdamW 学习率。
            weight_decay: AdamW 权重衰减。
            T_max: 余弦退火学习率的周期。
            **kwargs: 预留给 Lightning / argparse 的额外参数。
        """
        super(HiVT, self).__init__()
        self.save_hyperparameters()
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel
        self.use_reliability = use_reliability
        self.reliability_loss_weight = reliability_loss_weight
        self.scene_loss_weight = scene_loss_weight
        self.rank_loss_weight = rank_loss_weight
        self.calib_loss_weight = calib_loss_weight
        self.risk_fde_threshold = risk_fde_threshold
        self.risk_conflict_threshold = risk_conflict_threshold
        self.risk_offroad_threshold = risk_offroad_threshold
        self.lr = lr
        self.weight_decay = weight_decay
        self.T_max = T_max

        self.local_encoder = LocalEncoder(historical_steps=historical_steps,
                                          node_dim=node_dim,
                                          edge_dim=edge_dim,
                                          embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          num_temporal_layers=num_temporal_layers,
                                          local_radius=local_radius,
                                          parallel=parallel)
        self.global_interactor = GlobalInteractor(historical_steps=historical_steps,
                                                  embed_dim=embed_dim,
                                                  edge_dim=edge_dim,
                                                  num_modes=num_modes,
                                                  num_heads=num_heads,
                                                  num_layers=num_global_layers,
                                                  dropout=dropout,
                                                  rotate=rotate)
        self.decoder = MLPDecoder(local_channels=embed_dim,
                                  global_channels=embed_dim,
                                  future_steps=future_steps,
                                  num_modes=num_modes,
                                  uncertain=True)
        self.reliability_module = ReliabilityModule(
            embed_dim=embed_dim,
            future_steps=future_steps,
            num_modes=num_modes,
            hidden_dim=reliability_hidden_dim,
            rerank_alpha=reliability_rerank_alpha,
        ) if use_reliability else None
        self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')
        self.risk_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.rank_loss_fn = RiskRankLoss()
        self.calib_loss_fn = RiskCalibrationLoss()

        self.minADE = ADE()
        self.minFDE = FDE()
        self.minMR = MR()
        if use_reliability:
            self.mode_risk_auroc = AUROC(compute_on_step=False)
            self.mode_risk_brier = BrierScore()
            self.mode_risk_ece = ECE()

    def forward(self, data: TemporalData):
        """前向传播。

        Args:
            data: 单个 batch 的时空图数据，包含历史轨迹、lane 特征、图结构等。

        Returns:
            y_hat: 形状为 [F, N, H, 4] 或 [F, N, H, 2] 的多模态预测结果。
                - F: 模态数 `num_modes`
                - N: batch 中所有 actor 节点数
                - H: `future_steps`
                - 最后一维前 2 项是位置均值，后 2 项是 Laplace 分布尺度
            pi: 形状为 [N, F] 的每个 actor 的模态概率 logits。
        """
        # 论文中会将坐标系旋转到目标 actor 的局部朝向坐标系，以减轻朝向变化带来的学习难度。
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(data['rotate_angles'])
            cos_vals = torch.cos(data['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data['rotate_mat'] = rotate_mat
        else:
            data['rotate_mat'] = None

        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        reliability_outputs = None
        if self.reliability_module is not None:
            batch = getattr(data, 'batch', None)
            if batch is None:
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=y_hat.device)
            reliability_outputs = self.reliability_module(
                local_embed=local_embed,
                global_embed=global_embed,
                y_hat=y_hat,
                pi=pi,
                batch=batch,
            )
        return y_hat, pi, reliability_outputs

    def training_step(self, data, batch_idx):
        """单个训练 step。

        Args:
            data: 当前 batch 的图数据。
            batch_idx: Lightning 传入的 batch 索引。

        Returns:
            总训练损失 = 轨迹回归损失 + 模态分类损失。
        """
        y_hat, pi, reliability_outputs = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        valid_steps = reg_mask.sum(dim=-1)
        cls_mask = valid_steps > 0
        # 以每个 actor 的最优模态作为回归监督目标，对应论文中的 best-of-K 训练思路。
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        # 分类分支不使用硬 one-hot，而是用基于轨迹误差构造的 soft target。
        soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
        cls_loss = self.cls_loss(pi[cls_mask], soft_target)
        loss = reg_loss + cls_loss
        if self.reliability_module is not None:
            batch = getattr(data, 'batch', None)
            if batch is None:
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=y_hat.device)
            current_positions = data['positions'][:, self.historical_steps - 1]
            lane_positions = reconstruct_lane_positions(
                lane_actor_index=data['lane_actor_index'],
                lane_actor_vectors=data['lane_actor_vectors'],
                current_positions=current_positions,
                num_lanes=data['lane_vectors'].size(0),
            )
            reliability_targets = build_reliability_targets(
                y_hat=y_hat.detach(),
                y=data.y,
                reg_mask=reg_mask,
                batch=batch,
                lane_positions=lane_positions,
                lane_actor_index=data['lane_actor_index'],
                lane_actor_vectors=data['lane_actor_vectors'],
                fde_threshold=self.risk_fde_threshold,
                conflict_threshold=self.risk_conflict_threshold,
                offroad_threshold=self.risk_offroad_threshold,
            )
            mode_targets = reliability_targets['mode_targets']
            valid_mask = reliability_targets['valid_mask']
            scene_targets = reliability_targets['scene_targets']
            risk_loss = self.risk_loss(
                reliability_outputs['mode_risk_logits'][valid_mask],
                mode_targets[valid_mask],
            )
            scene_loss = self.risk_loss(
                reliability_outputs['scene_risk_logits'],
                scene_targets,
            ) if scene_targets.numel() > 0 else torch.zeros((), device=loss.device)
            loss = loss + self.reliability_loss_weight * risk_loss + self.scene_loss_weight * scene_loss
            if self.rank_loss_weight > 0:
                rank_loss = self.rank_loss_fn(
                    mode_risk=reliability_outputs['mode_risk'],
                    mode_error=reliability_targets['mode_error'],
                    valid_mask=valid_mask,
                )
                loss = loss + self.rank_loss_weight * rank_loss
                self.log('train_rank_loss', rank_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            if self.calib_loss_weight > 0 and valid_mask.any():
                calib_loss = self.calib_loss_fn(
                    risk=reliability_outputs['mode_risk'][valid_mask],
                    target=mode_targets[valid_mask],
                )
                loss = loss + self.calib_loss_weight * calib_loss
                self.log('train_calib_loss', calib_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            risk_stats = summarize_reliability_targets(reliability_targets)
            self.log('train_risk_loss', risk_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_scene_loss', scene_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_mode_risk_target_rate', risk_stats['mode_positive_rate'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_fde_risk_target_rate', risk_stats['fde_positive_rate'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_conflict_risk_target_rate', risk_stats['conflict_positive_rate'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_offroad_risk_target_rate', risk_stats['offroad_positive_rate'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_scene_risk_target_rate', risk_stats['scene_positive_rate'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_mode_risk_pred_mean', reliability_outputs['mode_risk'].mean(), prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            if reliability_outputs['scene_risk'].numel() > 0:
                self.log('train_scene_risk_pred_mean', reliability_outputs['scene_risk'].mean(), prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss', reg_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        """单个验证 step。

        Args:
            data: 当前 batch 的图数据。
            batch_idx: Lightning 传入的 batch 索引。
        """
        y_hat, pi, reliability_outputs = self(data)
        reg_mask = ~data['padding_mask'][:, self.historical_steps:]
        l2_norm = (torch.norm(y_hat[:, :, :, : 2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes)]
        reg_loss = self.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
        self.log('val_reg_loss', reg_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        # 论文和 README 中汇报的是 agent 目标车辆上的 minADE / minFDE / MR 指标。
        y_hat_agent = y_hat[:, data['agent_index'], :, : 2]
        y_agent = data.y[data['agent_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)
        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(data.num_graphs)]
        self.minADE.update(y_hat_best_agent, y_agent)
        self.minFDE.update(y_hat_best_agent, y_agent)
        self.minMR.update(y_hat_best_agent, y_agent)
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        self.log('val_minMR', self.minMR, prog_bar=True, on_step=False, on_epoch=True, batch_size=y_agent.size(0))
        if self.reliability_module is not None and reliability_outputs is not None:
            batch = getattr(data, 'batch', None)
            if batch is None:
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=y_hat.device)
            current_positions = data['positions'][:, self.historical_steps - 1]
            lane_positions = reconstruct_lane_positions(
                lane_actor_index=data['lane_actor_index'],
                lane_actor_vectors=data['lane_actor_vectors'],
                current_positions=current_positions,
                num_lanes=data['lane_vectors'].size(0),
            )
            reliability_targets = build_reliability_targets(
                y_hat=y_hat.detach(),
                y=data.y,
                reg_mask=reg_mask,
                batch=batch,
                lane_positions=lane_positions,
                lane_actor_index=data['lane_actor_index'],
                lane_actor_vectors=data['lane_actor_vectors'],
                fde_threshold=self.risk_fde_threshold,
                conflict_threshold=self.risk_conflict_threshold,
                offroad_threshold=self.risk_offroad_threshold,
            )
            mode_targets = reliability_targets['mode_targets']
            valid_mask = reliability_targets['valid_mask']
            scene_targets = reliability_targets['scene_targets']
            if valid_mask.any():
                val_risk_loss = self.risk_loss(reliability_outputs['mode_risk_logits'][valid_mask], mode_targets[valid_mask])
                self.log('val_risk_loss', val_risk_loss, prog_bar=False, on_step=False, on_epoch=True,
                         batch_size=int(valid_mask.sum().item()))
            if scene_targets.numel() > 0:
                val_scene_loss = self.risk_loss(reliability_outputs['scene_risk_logits'], scene_targets)
                self.log('val_scene_loss', val_scene_loss, prog_bar=False, on_step=False, on_epoch=True,
                         batch_size=scene_targets.size(0))
            risk_stats = summarize_reliability_targets(reliability_targets)
            self.log('val_mode_risk_target_rate', risk_stats['mode_positive_rate'], prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
            self.log('val_fde_risk_target_rate', risk_stats['fde_positive_rate'], prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
            self.log('val_conflict_risk_target_rate', risk_stats['conflict_positive_rate'], prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
            self.log('val_offroad_risk_target_rate', risk_stats['offroad_positive_rate'], prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
            self.log('val_scene_risk_target_rate', risk_stats['scene_positive_rate'], prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
            self.log('val_mode_risk_pred_mean', reliability_outputs['mode_risk'].mean(), prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
            if reliability_outputs['scene_risk'].numel() > 0:
                self.log('val_scene_risk_pred_mean', reliability_outputs['scene_risk'].mean(), prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
            if valid_mask.any():
                self.mode_risk_auroc.update(reliability_outputs['mode_risk'][valid_mask], mode_targets[valid_mask])
                self.mode_risk_brier.update(reliability_outputs['mode_risk'][valid_mask], mode_targets[valid_mask])
                self.mode_risk_ece.update(reliability_outputs['mode_risk'][valid_mask], mode_targets[valid_mask])
                self.log('val_mode_AUROC', self.mode_risk_auroc, prog_bar=False, on_step=False, on_epoch=True, batch_size=int(valid_mask.sum().item()))
                self.log('val_mode_BrierScore', self.mode_risk_brier, prog_bar=False, on_step=False, on_epoch=True, batch_size=int(valid_mask.sum().item()))
                self.log('val_mode_ECE', self.mode_risk_ece, prog_bar=False, on_step=False, on_epoch=True, batch_size=int(valid_mask.sum().item()))

    def configure_optimizers(self):
        """配置优化器与学习率调度器。

        Returns:
            Lightning 期望格式的 `[optimizer], [scheduler]`。
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        # 按参数类型拆分 weight decay，避免对 bias / LayerNorm / Embedding 施加衰减。
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.T_max, eta_min=0.0)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    @staticmethod
    def add_model_specific_args(parent_parser):
        """向命令行解析器注册 HiVT 专属参数。

        Args:
            parent_parser: 外部传入的 `ArgumentParser`。

        Returns:
            注册完参数后的原 parser。
        """
        parser = parent_parser.add_argument_group('HiVT')
        # 这些默认值与论文公开代码保持一致。
        parser.add_argument('--historical_steps', type=int, default=20)
        parser.add_argument('--future_steps', type=int, default=30)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--rotate', type=bool, default=True)
        parser.add_argument('--node_dim', type=int, default=2)
        parser.add_argument('--edge_dim', type=int, default=2)
        parser.add_argument('--embed_dim', type=int, required=True)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--num_temporal_layers', type=int, default=4)
        parser.add_argument('--num_global_layers', type=int, default=3)
        parser.add_argument('--local_radius', type=float, default=50)
        parser.add_argument('--parallel', type=bool, default=False)
        parser.add_argument('--use_reliability', type=bool, default=False)
        parser.add_argument('--reliability_hidden_dim', type=int, default=128)
        parser.add_argument('--reliability_rerank_alpha', type=float, default=0.5)
        parser.add_argument('--reliability_loss_weight', type=float, default=1.0)
        parser.add_argument('--scene_loss_weight', type=float, default=0.5)
        parser.add_argument('--rank_loss_weight', type=float, default=0.0)
        parser.add_argument('--calib_loss_weight', type=float, default=0.0)
        parser.add_argument('--risk_fde_threshold', type=float, default=2.0)
        parser.add_argument('--risk_conflict_threshold', type=float, default=2.0)
        parser.add_argument('--risk_offroad_threshold', type=float, default=2.0)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--T_max', type=int, default=64)
        return parent_parser
