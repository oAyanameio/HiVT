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
import torch
import torch.nn as nn


class LaplaceNLLLoss(nn.Module):
    """Laplace 负对数似然损失。

    HiVT 的解码器会同时预测未来位置的均值 `loc` 与尺度 `scale`，
    因而这里使用 Laplace 分布的 NLL 作为轨迹回归目标。
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        """初始化 Laplace NLL 损失。

        Args:
            eps: 尺度的数值稳定下界。
            reduction: 聚合方式，可选 `mean` / `sum` / `none`。
        """
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """计算 Laplace NLL。

        Args:
            pred: 预测张量，最后一维需包含 `(loc, scale)` 两部分。
            target: 真实目标位置。

        Returns:
            按 `reduction` 聚合后的损失值。
        """
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
