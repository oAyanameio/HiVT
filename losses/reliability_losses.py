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
"""可靠性分支专用损失（idea 文档 10.2 / 10.3）。

- `RiskRankLoss`：pairwise 排序损失。若 mode i 的真实未来误差明显大于 mode j，
  则其风险分数应满足 r_i > r_j，使风险输出不止于二分类拟合，而具备排序意义。
- `RiskCalibrationLoss`：soft-ECE 风格的校准损失，约束预测风险与真实失败率一致。
"""
import torch
import torch.nn as nn


class RiskRankLoss(nn.Module):
    """基于真实误差的 pairwise 风险排序损失（文档 10.3）。

    对每个 actor 的所有 mode 两两配对，当真实误差差距超过 `margin` 时，要求高误差
    mode 的预测风险显著高于低误差 mode，使用 margin ranking 形式。
    """

    def __init__(self, margin: float = 0.1, error_margin: float = 0.5, reduction: str = 'mean') -> None:
        """初始化排序损失。

        Args:
            margin: 风险分差的目标 margin。
            error_margin: 只有当两个 mode 的真实误差差距超过该值时才构成监督对。
            reduction: 聚合方式，`mean` / `sum` / `none`。
        """
        super(RiskRankLoss, self).__init__()
        self.margin = margin
        self.error_margin = error_margin
        self.reduction = reduction

    def forward(self,
                mode_risk: torch.Tensor,
                mode_error: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        """计算 pairwise 排序损失。

        Args:
            mode_risk: [N, F] 预测风险分数（0~1）。
            mode_error: [N, F] 每个 mode 的真实未来误差（如 ADE），越大越该高风险。
            valid_mask: [N] 该 actor 是否有有效未来监督。

        Returns:
            排序损失标量（或 `none` 时的逐对张量）。
        """
        if mode_risk.size(0) == 0 or mode_risk.size(1) < 2:
            return mode_risk.new_zeros(())
        # 两两配对：i, j 遍历 mode 维。
        err_i = mode_error.unsqueeze(2)  # [N, F, 1]
        err_j = mode_error.unsqueeze(1)  # [N, 1, F]
        risk_i = mode_risk.unsqueeze(2)
        risk_j = mode_risk.unsqueeze(1)

        err_diff = err_i - err_j  # >0 表示 i 误差更大，应更高风险
        # 只保留误差差距显著、且 i 比 j 更差的有序对。
        pair_mask = (err_diff > self.error_margin)
        pair_mask = pair_mask & valid_mask.view(-1, 1, 1)
        if pair_mask.sum() == 0:
            return mode_risk.new_zeros(())

        # margin ranking：希望 risk_i - risk_j >= margin。
        loss = torch.clamp(self.margin - (risk_i - risk_j), min=0.0)
        loss = loss[pair_mask]
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))


class RiskCalibrationLoss(nn.Module):
    """soft-ECE 风格的风险校准损失（文档 10.2 中可选的 L_calib）。

    将预测风险按值分桶，约束每个桶的平均预测风险与该桶的真实失败率一致。
    采用软分桶（三角核）以保持可微。
    """

    def __init__(self, num_bins: int = 10) -> None:
        """初始化校准损失。

        Args:
            num_bins: 分桶数量。
        """
        super(RiskCalibrationLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self,
                risk: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """计算 soft-ECE。

        Args:
            risk: 预测风险分数，任意 shape，会被展平。
            target: 对应的 0/1 失败标签，shape 与 `risk` 一致。

        Returns:
            校准损失标量。
        """
        risk = risk.reshape(-1)
        target = target.reshape(-1).float()
        if risk.numel() == 0:
            return risk.new_zeros(())
        device = risk.device
        bin_centers = torch.linspace(0.0, 1.0, self.num_bins, device=device)
        width = 1.0 / max(self.num_bins - 1, 1)
        # 三角核软分配权重：[B, N]
        dist = (risk.unsqueeze(0) - bin_centers.unsqueeze(1)).abs()
        weights = torch.clamp(1.0 - dist / width, min=0.0)
        weight_sum = weights.sum(dim=1)  # [B]
        valid = weight_sum > 1e-6
        if valid.sum() == 0:
            return risk.new_zeros(())
        avg_conf = (weights * risk.unsqueeze(0)).sum(dim=1) / weight_sum.clamp(min=1e-6)
        avg_acc = (weights * target.unsqueeze(0)).sum(dim=1) / weight_sum.clamp(min=1e-6)
        gap = (avg_conf - avg_acc).abs()
        return (gap[valid] * (weight_sum[valid] / weight_sum[valid].sum())).sum()
