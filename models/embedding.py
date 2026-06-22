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
from typing import List, Optional

import torch
import torch.nn as nn

from utils import init_weights


class SingleInputEmbedding(nn.Module):
    """单输入连续特征嵌入层。

    适用于只包含一类连续几何特征的场景，例如单个 actor 的位移向量。
    """

    def __init__(self,
                 in_channel: int,
                 out_channel: int) -> None:
        """初始化单输入嵌入层。

        Args:
            in_channel: 输入特征维度。
            out_channel: 输出嵌入维度。
        """
        super(SingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """嵌入连续输入特征。

        Args:
            x: 输入张量，最后一维大小为 `in_channel`。

        Returns:
            对应的高维嵌入表示。
        """
        return self.embed(x)


class MultipleInputEmbedding(nn.Module):
    """多输入特征嵌入层。

    用于将多类连续几何特征与可选的离散语义嵌入聚合到同一个隐空间中。
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int) -> None:
        """初始化多输入嵌入层。

        Args:
            in_channels: 每类连续输入的维度列表。
            out_channel: 统一映射到的嵌入维度。
        """
        super(MultipleInputEmbedding, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel, out_channel))
             for in_channel in in_channels])
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """聚合多种输入特征。

        Args:
            continuous_inputs: 连续输入张量列表，每项最后一维需与 `in_channels` 对应。
            categorical_inputs: 已经嵌入到 `out_channel` 维的离散特征列表，可为 `None`。

        Returns:
            聚合后的嵌入表示。
        """
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        output = torch.stack(continuous_inputs).sum(dim=0)
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        return self.aggr_embed(output)
