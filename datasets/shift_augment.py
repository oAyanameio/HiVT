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
"""分布偏移数据增强（idea 文档第 11 节）。

四类扰动可以单独使用，也可以通过 `ShiftAugment` 组合调用：

- `HistoryDropout`   (§11.1)：随机删除部分 actor 的最近 k 帧历史。
- `NeighborDropout`  (§11.1)：随机完全移除邻车的历史观测。
- `StateNoise`       (§11.3)：对历史位置和朝向角添加高斯噪声。
- `MapJitter`        (§11.2)：对 lane 几何特征添加噪声，并可随机删除 lane 段。
"""
import random
from typing import List

import torch

from utils import TemporalData


# ─── 辅助函数 ──────────────────────────────────────────────────────────────────

def _recompute_x_from_positions(data: TemporalData) -> None:
    """positions 修改后重新计算 x（相对位移特征）。"""
    pm = data['padding_mask']  # [N, 50]
    x = data['x']              # [N, 20, 2]
    pos = data['positions']    # [N, 50, 2]
    x[:, 0] = 0.0
    x[:, 1:20] = torch.where(
        (pm[:, :19] | pm[:, 1:20]).unsqueeze(-1),
        torch.zeros_like(x[:, 1:20]),
        pos[:, 1:20] - pos[:, :19],
    )


def _recompute_bos_mask(data: TemporalData) -> None:
    """padding_mask 修改后重新计算 bos_mask（段起始标记）。"""
    pm = data['padding_mask']
    bos = data['bos_mask']
    bos[:, 0] = ~pm[:, 0]
    bos[:, 1:20] = pm[:, :19] & ~pm[:, 1:20]


# ─── 四类增强 ─────────────────────────────────────────────────────────────────

class HistoryDropout:
    """随机遮盖 actor 的最近 k 帧历史（§11.1）。

    每个非 agent actor 以概率 `p` 被选中，被选中的 actor 的最近
    k 帧（k ~ Uniform(1, max_drop_steps)）会被置为无效。
    """

    def __init__(self, p: float = 0.3, max_drop_steps: int = 10) -> None:
        self.p = p
        self.max_drop_steps = max_drop_steps

    def __call__(self, data: TemporalData) -> TemporalData:
        agent_idx = int(data['agent_index'])
        pm = data['padding_mask']   # [N, 50]
        x = data['x']               # [N, 20, 2]

        for i in range(data.num_nodes):
            if i == agent_idx or random.random() >= self.p:
                continue
            # 找最后一个有效历史帧
            valid_steps = (~pm[i, :20]).nonzero(as_tuple=False).flatten()
            if valid_steps.numel() == 0:
                continue
            last = int(valid_steps[-1].item())
            k = random.randint(1, self.max_drop_steps)
            drop_start = max(0, last - k + 1)
            pm[i, drop_start:20] = True
            x[i, drop_start:20] = 0.0

        _recompute_bos_mask(data)
        return data


class NeighborDropout:
    """随机完全移除邻车的历史观测（§11.1 邻车观测不完整）。

    每个非 agent、非 AV actor 以概率 `p` 被完全从历史中移除。
    """

    def __init__(self, p: float = 0.2) -> None:
        self.p = p

    def __call__(self, data: TemporalData) -> TemporalData:
        agent_idx = int(data['agent_index'])
        av_idx = int(data['av_index'])
        pm = data['padding_mask']
        x = data['x']

        for i in range(data.num_nodes):
            if i in (agent_idx, av_idx) or random.random() >= self.p:
                continue
            pm[i, :20] = True
            x[i, :] = 0.0

        _recompute_bos_mask(data)
        return data


class StateNoise:
    """对历史位置和朝向角添加高斯噪声（§11.3）。

    仅扰动有效历史帧的位置，然后重新计算位移特征 x；
    对 rotate_angles 添加独立的朝向噪声。
    """

    def __init__(self, position_std: float = 0.1, heading_std: float = 0.05) -> None:
        self.position_std = position_std
        self.heading_std = heading_std

    def __call__(self, data: TemporalData) -> TemporalData:
        pos = data['positions']          # [N, 50, 2]
        angles = data['rotate_angles']   # [N]
        pm = data['padding_mask']        # [N, 50]

        if self.position_std > 0:
            valid = (~pm[:, :20]).float().unsqueeze(-1)   # [N, 20, 1]
            noise = torch.randn_like(pos[:, :20]) * self.position_std
            pos[:, :20] = pos[:, :20] + noise * valid
            _recompute_x_from_positions(data)

        if self.heading_std > 0:
            angles.add_(torch.randn_like(angles) * self.heading_std)

        return data


class MapJitter:
    """对 lane 几何特征添加高斯噪声，可选随机删除 lane 段（§11.2）。

    扰动 lane_vectors 和 lane_actor_vectors；以 `lane_dropout_p` 的概率
    随机删除 lane 段（移除对应 lane_actor 边，不修改 lane_vectors 索引以
    避免重排开销）。
    """

    def __init__(self, std: float = 0.05, lane_dropout_p: float = 0.0) -> None:
        self.std = std
        self.lane_dropout_p = lane_dropout_p

    def __call__(self, data: TemporalData) -> TemporalData:
        lv = data['lane_vectors']          # [L, 2]
        lav = data['lane_actor_vectors']   # [E, 2]
        lai = data['lane_actor_index']     # [2, E]

        if self.std > 0:
            lv.add_(torch.randn_like(lv) * self.std)
            lav.add_(torch.randn_like(lav) * self.std)

        if self.lane_dropout_p > 0 and lai.size(1) > 0:
            keep = torch.rand(lv.size(0), device=lai.device) >= self.lane_dropout_p
            edge_keep = keep[lai[0]]
            if edge_keep.any() and not edge_keep.all():
                data['lane_actor_index'] = lai[:, edge_keep]
                data['lane_actor_vectors'] = lav[edge_keep]

        return data


# ─── 组合接口 ─────────────────────────────────────────────────────────────────

class ShiftAugment:
    """组合分布偏移增强，仅当对应参数大于零时激活子模块（§11）。

    作为 `ArgoverseV1DataModule` 的 `train_transform` 传入，在训练时
    动态应用；验证时不传 transform，以保持评估分布的一致性。

    Example::

        transform = ShiftAugment(
            history_dropout_p=0.3,
            neighbor_dropout_p=0.2,
            position_noise_std=0.1,
            map_jitter_std=0.05,
        )
        datamodule = ArgoverseV1DataModule(..., train_transform=transform)
    """

    def __init__(
        self,
        history_dropout_p: float = 0.0,
        history_max_drop_steps: int = 10,
        neighbor_dropout_p: float = 0.0,
        position_noise_std: float = 0.0,
        heading_noise_std: float = 0.0,
        map_jitter_std: float = 0.0,
        lane_dropout_p: float = 0.0,
    ) -> None:
        self._transforms: List = []
        if history_dropout_p > 0:
            self._transforms.append(
                HistoryDropout(p=history_dropout_p, max_drop_steps=history_max_drop_steps))
        if neighbor_dropout_p > 0:
            self._transforms.append(NeighborDropout(p=neighbor_dropout_p))
        if position_noise_std > 0 or heading_noise_std > 0:
            self._transforms.append(
                StateNoise(position_std=position_noise_std, heading_std=heading_noise_std))
        if map_jitter_std > 0 or lane_dropout_p > 0:
            self._transforms.append(MapJitter(std=map_jitter_std, lane_dropout_p=lane_dropout_p))

    def __call__(self, data: object) -> object:
        for t in self._transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._transforms!r})"
