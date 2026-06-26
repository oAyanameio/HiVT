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
"""可靠性评估指标（idea 文档 §12.4）。

- `BrierScore`：预测风险与失败标签的均方误差（校准质量）。
- `ECE`：分桶期望校准误差（校准质量）。
- `AUROC`：失败检测的 ROC-AUC。
- `AUPRC`：失败检测的 PR-AUC。
- `FPR95TPR`：TPR=95% 时的 FPR（失败检测灵敏度）。
"""
from typing import Any, Callable, List, Optional

import torch
from torchmetrics import Metric


class BrierScore(Metric):
    """预测风险 vs 真实失败标签的均方误差（§12.4 校准指标）。"""

    full_state_update = False

    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Optional[Callable] = None) -> None:
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         dist_sync_fn=dist_sync_fn)
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.sum += ((preds.reshape(-1) - targets.reshape(-1).float()) ** 2).sum()
        self.count += preds.numel()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count.clamp(min=1)


class ECE(Metric):
    """分桶期望校准误差（§12.4 校准指标）。"""

    full_state_update = False

    def __init__(self,
                 num_bins: int = 10,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Optional[Callable] = None) -> None:
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         dist_sync_fn=dist_sync_fn)
        self.num_bins = num_bins
        self.add_state('bin_conf', default=torch.zeros(num_bins), dist_reduce_fx='sum')
        self.add_state('bin_acc', default=torch.zeros(num_bins), dist_reduce_fx='sum')
        self.add_state('bin_count', default=torch.zeros(num_bins), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.reshape(-1)
        targets = targets.reshape(-1).float()
        bin_ids = torch.clamp((preds * self.num_bins).long(), 0, self.num_bins - 1)
        self.bin_conf.scatter_add_(0, bin_ids, preds)
        self.bin_acc.scatter_add_(0, bin_ids, targets)
        self.bin_count.scatter_add_(0, bin_ids, torch.ones_like(preds))

    def compute(self) -> torch.Tensor:
        valid = self.bin_count > 0
        if not valid.any():
            return self.bin_conf.new_zeros(())
        avg_conf = self.bin_conf[valid] / self.bin_count[valid]
        avg_acc = self.bin_acc[valid] / self.bin_count[valid]
        weights = self.bin_count[valid] / self.bin_count.sum()
        return (avg_conf - avg_acc).abs().dot(weights)


class _BinaryRankingMetric(Metric):
    """累积全量 preds/targets 以支持需要全局排序的指标（AUROC / AUPRC / FPR@TPR）。"""

    full_state_update = True

    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Optional[Callable] = None) -> None:
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         dist_sync_fn=dist_sync_fn)
        self.add_state('preds', default=[], dist_reduce_fx='cat')
        self.add_state('targets', default=[], dist_reduce_fx='cat')

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.preds.append(preds.reshape(-1).detach().cpu().float())
        self.targets.append(targets.reshape(-1).detach().cpu().float())

    def _sorted_curves(self):
        """返回按预测分从高到低排列的 (tpr, fpr)。"""
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        order = preds.argsort(descending=True)
        targets = targets[order]
        n_pos = targets.sum().clamp(min=1)
        n_neg = (1 - targets).sum().clamp(min=1)
        tpr = targets.cumsum(0) / n_pos
        fpr = (1 - targets).cumsum(0) / n_neg
        return tpr, fpr


class AUROC(_BinaryRankingMetric):
    """ROC-AUC（§12.4 失败检测指标）。"""

    def compute(self) -> torch.Tensor:
        tpr, fpr = self._sorted_curves()
        return torch.trapz(tpr, fpr).abs()


class AUPRC(_BinaryRankingMetric):
    """PR-AUC（§12.4 失败检测指标）。"""

    def compute(self) -> torch.Tensor:
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        order = preds.argsort(descending=True)
        targets = targets[order]
        n_pos = targets.sum().clamp(min=1)
        tp = targets.cumsum(0)
        fp = (1 - targets).cumsum(0)
        precision = tp / (tp + fp).clamp(min=1e-6)
        recall = tp / n_pos
        return torch.trapz(precision, recall).abs()


class FPR95TPR(_BinaryRankingMetric):
    """FPR@95%TPR（§12.4 失败检测指标）。"""

    def compute(self) -> torch.Tensor:
        tpr, fpr = self._sorted_curves()
        mask = tpr >= 0.95
        if not mask.any():
            return fpr[-1]
        return fpr[mask][0]
