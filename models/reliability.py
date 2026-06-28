
"""即插即用可靠性模块（InterAct-Risk）。

该文件实现 idea 文档（docs/hivt_reliability_module_idea.md）所述的可靠性分支：

- `TrajectoryReliabilityEncoder`：对应文档 7.1，从单条候选轨迹的几何/动力学特征
  编码出 mode-level 表征。
- `InteractionContextEncoder`：对应文档 7.2，在场景内对 actor 做交互感知注意力，
  编码出 scene/interaction 上下文。
- `ReliabilityModule`：对应文档 7.3 / 7.4，输出 mode-level / scene-level 风险，
  以及风险感知重排后的候选分布。
- 自动风险标签（文档第 9 节）：FDE / ADE / miss / conflict / off-road。
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weights


# 文档 7.1 中轨迹运动学特征向量的维度（见 _trajectory_kinematic_features）。
KINEMATIC_FEATURE_DIM = 21


def apply_risk_reranking(pi: torch.Tensor, mode_risk: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """按风险对 mode logits 做重排（文档 7.4）。"""
    logits = pi - alpha * mode_risk
    return F.softmax(logits, dim=-1)


def _last_valid_step(reg_mask: torch.Tensor) -> torch.Tensor:
    valid_steps = reg_mask.long().sum(dim=-1)
    return torch.clamp(valid_steps - 1, min=0)
def _mode_displacement_error(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    reg_mask: torch.Tensor,
) -> torch.Tensor:
    """计算每个 mode 在有效步上的平均位移误差（ADE），[N, F]。

    用于构造 ADE 风险标签，同时作为 rank loss 的连续监督信号。
    """
    mode_xy = y_hat[..., :2]
    error = torch.norm(mode_xy - y.unsqueeze(0), p=2, dim=-1)  # [F, N, H]
    error = error * reg_mask.unsqueeze(0)
    valid_steps = reg_mask.sum(dim=-1).clamp(min=1)  # [N]
    ade = error.sum(dim=-1) / valid_steps.unsqueeze(0)  # [F, N]
    return ade.transpose(0, 1)  # [N, F]


def compute_mode_risk_targets(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    reg_mask: torch.Tensor,
    fde_threshold: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """根据 FDE 构造 mode-level 风险标签（文档 9.1）。"""
    last_step = _last_valid_step(reg_mask)
    mode_xy = y_hat[..., :2]
    target_xy = y[torch.arange(y.size(0), device=y.device), last_step]
    gather_index = last_step.view(1, -1, 1, 1).expand(mode_xy.size(0), -1, 1, mode_xy.size(-1))
    pred_terminal = torch.gather(mode_xy, dim=2, index=gather_index).squeeze(2)
    fde = torch.norm(pred_terminal - target_xy.unsqueeze(0), p=2, dim=-1)
    fde = fde.transpose(0, 1)
    mode_targets = (fde > fde_threshold).float()
    valid_mask = reg_mask.any(dim=-1)
    mode_targets = mode_targets * valid_mask.unsqueeze(-1).float()
    return mode_targets, valid_mask, fde


def compute_ade_risk_targets(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    reg_mask: torch.Tensor,
    ade_threshold: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """根据 ADE 构造 mode-level 风险标签（文档 9.1 中的 y_ade）。

    Returns:
        `(ade_targets, ade)`，两者形状均为 [N, F]。
    """
    ade = _mode_displacement_error(y_hat, y, reg_mask)  # [N, F]
    valid_nodes = reg_mask.any(dim=-1)
    ade_targets = (ade > ade_threshold).float() * valid_nodes.unsqueeze(-1).float()
    ade = ade.clone()
    ade[~valid_nodes] = 0.0
    return ade_targets, ade


def compute_miss_risk_targets(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    reg_mask: torch.Tensor,
    miss_threshold: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """根据终点是否 miss 构造 mode-level 风险标签（文档 9.1 中的 y_miss）。

    miss 的定义与 MR 指标一致：终点误差超过 `miss_threshold`。它与 FDE 风险共享
    终点误差，但阈值更大、语义上对应“彻底错过 GT”。

    Returns:
        `(miss_targets, fde)`，两者形状均为 [N, F]。
    """
    _, valid_mask, fde = compute_mode_risk_targets(
        y_hat=y_hat, y=y, reg_mask=reg_mask, fde_threshold=miss_threshold)
    miss_targets = (fde > miss_threshold).float() * valid_mask.unsqueeze(-1).float()
    return miss_targets, fde


def _max_consecutive_true(flags: torch.Tensor) -> int:
    """返回布尔序列中 True 的最长连续长度。"""
    best = 0
    run = 0
    for flag in flags.tolist():
        if flag:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def predictions_to_scene_abs(
    y_hat: torch.Tensor,
    positions: torch.Tensor,
    historical_steps: int,
    rotate_mat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """把 per-actor 局部坐标系下的预测轨迹还原到共享 scene 绝对坐标系。

    HiVT 在 `rotate=True` 时会把每个 actor 的 `y` / 预测旋转到其自身局部朝向坐标系，
    不同 actor 处在不同坐标系下，直接做跨 actor 的距离比较在几何上是错误的。
    本函数将预测先按 `rotate_mat^T` 旋回 scene 朝向，再加上当前帧绝对位置。

    Args:
        y_hat: [F, N, H, C]，最后维前 2 项为局部坐标系下、相对当前帧的位移。
        positions: [N, T, 2]，scene 绝对坐标系下的完整轨迹（含历史与未来）。
        historical_steps: 历史步数，当前帧索引为 `historical_steps - 1`。
        rotate_mat: [N, 2, 2] 每个 actor 的旋转矩阵；`None` 表示未旋转。

    Returns:
        [F, N, H, 2] scene 绝对坐标系下的预测轨迹。
    """
    mode_xy = y_hat[..., :2]  # [F, N, H, 2]
    if rotate_mat is not None:
        # data.y_local = y_scene_rel @ R，因此 y_scene_rel = y_local @ R^T。
        inv_rot = rotate_mat.transpose(-1, -2)  # [N, 2, 2]
        scene_rel = torch.einsum("fnhc,ncd->fnhd", mode_xy, inv_rot)
    else:
        scene_rel = mode_xy
    current = positions[:, historical_steps - 1]  # [N, 2]
    return scene_rel + current.view(1, current.size(0), 1, 2)


def compute_scene_risk_targets(
    mode_targets: torch.Tensor,
    batch: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    fde: Optional[torch.Tensor] = None,
    agent_index: Optional[torch.Tensor] = None,
    policy: str = "target_best_mode_fail",
    rate_threshold: float = 0.5,
) -> torch.Tensor:
    """把 mode-level 风险聚合成 scene-level 风险（文档 6.4 / 9.2）。

    支持的 `policy`：
    - `target_best_mode_fail`：目标车 min-FDE（best）mode 是否失败。需要 `fde` 与
      `agent_index`，与 HiVT 的 minFDE/MR 评估口径一致，不易被无关邻车污染。
    - `target_mode_rate`：目标车正样本 mode 比例是否超过 `rate_threshold`。需要
      `agent_index`。
    - `scene_rate`：全场景高风险 mode 比例是否超过 `rate_threshold`（兼容旧逻辑）。
    - `scene_max`：全场景任一 node/mode 为正即为正（旧的 hard-max，过激，不推荐）。

    target-centric 策略缺少 `agent_index` 时直接报错，避免静默退化成全 actor 聚合。
    """
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    scene_targets = torch.zeros(num_graphs, device=mode_targets.device)
    if num_graphs == 0:
        return scene_targets

    if policy in ("target_best_mode_fail", "target_mode_rate"):
        if agent_index is None:
            raise ValueError(
                "scene policy '{}' requires agent_index (target actor 全局索引)".format(policy))
        agent_index = agent_index.view(-1)
        for graph_idx in range(num_graphs):
            if graph_idx >= agent_index.numel():
                continue
            target = int(agent_index[graph_idx].item())
            if valid_mask is not None and not bool(valid_mask[target]):
                scene_targets[graph_idx] = 0.0
                continue
            if policy == "target_best_mode_fail":
                if fde is None:
                    raise ValueError("scene policy 'target_best_mode_fail' requires fde")
                best_mode = int(fde[target].argmin().item())
                scene_targets[graph_idx] = mode_targets[target, best_mode]
            else:  # target_mode_rate
                rate = mode_targets[target].mean()
                scene_targets[graph_idx] = (rate > rate_threshold).float()
        return scene_targets

    # 全场景聚合（兼容旧逻辑）。
    node_risk = mode_targets.max(dim=-1).values
    if valid_mask is not None:
        node_risk = node_risk * valid_mask.float()
    for graph_idx in range(num_graphs):
        node_mask = batch == graph_idx
        if not node_mask.any():
            continue
        if policy == "scene_rate":
            graph_modes = mode_targets[node_mask]
            if valid_mask is not None:
                graph_valid = valid_mask[node_mask]
                graph_modes = graph_modes[graph_valid]
            rate = graph_modes.mean() if graph_modes.numel() > 0 else node_risk.new_zeros(())
            scene_targets[graph_idx] = (rate > rate_threshold).float()
        else:  # scene_max
            scene_targets[graph_idx] = node_risk[node_mask].max()
    return scene_targets


def compute_conflict_risk_targets(
    y_hat: torch.Tensor,
    reg_mask: torch.Tensor,
    batch: torch.Tensor,
    positions: torch.Tensor,
    historical_steps: int,
    rotate_mat: Optional[torch.Tensor] = None,
    agent_index: Optional[torch.Tensor] = None,
    conflict_threshold: float = 1.0,
    min_frames: int = 2,
    scope: str = "target_to_neighbors",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """构造 conflict 风险标签（文档 6.2）。

    第一版默认 `scope='target_to_neighbors'`：
    - 取目标车（`agent_index`）的第 k 条预测 future（还原到 scene 绝对坐标系）；
    - 与同 scene 下每个有效邻车的 GT future 做同一时间步距离比较；
    - 仅在双方 future mask 同时有效的步上计算；
    - 若某邻车的连续接近帧数 >= `min_frames`，则该 mode 记为 conflict。

    `scope='all_valid_pairs'` 保留旧的“预测 vs 预测最小距离”定义，仅供诊断，
    极易在密集交通场景把 conflict 标签打满。

    Returns:
        `(conflict_targets, min_pair_dist)`，均为 [N, F]。
    """
    mode_xy = y_hat[..., :2]
    num_modes, num_nodes, future_steps, _ = mode_xy.shape
    conflict_targets = torch.zeros(num_nodes, num_modes, device=y_hat.device)
    min_pair_dist = torch.full((num_nodes, num_modes), float("inf"), device=y_hat.device)
    valid_nodes = reg_mask.any(dim=-1)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return conflict_targets, min_pair_dist

    if scope == "target_to_neighbors":
        if agent_index is None:
            raise ValueError("conflict scope 'target_to_neighbors' requires agent_index")
        agent_index = agent_index.view(-1)
        pred_abs = predictions_to_scene_abs(
            y_hat=y_hat, positions=positions, historical_steps=historical_steps, rotate_mat=rotate_mat)
        gt_future = positions[:, historical_steps:historical_steps + future_steps]  # [N, H, 2]
        for graph_idx in range(num_graphs):
            if graph_idx >= agent_index.numel():
                continue
            target = int(agent_index[graph_idx].item())
            if not bool(valid_nodes[target]):
                continue
            node_mask = batch == graph_idx
            neighbor_idx = torch.nonzero(node_mask, as_tuple=False).flatten()
            neighbor_idx = neighbor_idx[neighbor_idx != target]
            neighbor_idx = neighbor_idx[valid_nodes[neighbor_idx]]
            if neighbor_idx.numel() == 0:
                continue
            target_valid = reg_mask[target]  # [H]
            neigh_gt = gt_future[neighbor_idx]  # [M, H, 2]
            neigh_valid = reg_mask[neighbor_idx]  # [M, H]
            for mode_idx in range(num_modes):
                traj = pred_abs[mode_idx, target]  # [H, 2]
                dist = torch.norm(traj.unsqueeze(0) - neigh_gt, p=2, dim=-1)  # [M, H]
                joint_valid = target_valid.unsqueeze(0) & neigh_valid  # [M, H]
                masked_dist = dist.masked_fill(~joint_valid, float("inf"))
                if torch.isfinite(masked_dist).any():
                    min_pair_dist[target, mode_idx] = masked_dist.min()
                close = (dist < conflict_threshold) & joint_valid  # [M, H]
                hit = False
                for m in range(neighbor_idx.numel()):
                    if _max_consecutive_true(close[m]) >= min_frames:
                        hit = True
                        break
                conflict_targets[target, mode_idx] = float(hit)
        conflict_targets = conflict_targets * valid_nodes.unsqueeze(-1).float()
        return conflict_targets, min_pair_dist

    # 旧的全场景预测两两比较（all_valid_pairs），仅作诊断。
    for graph_idx in range(num_graphs):
        node_indices = torch.nonzero(batch == graph_idx, as_tuple=False).flatten()
        if node_indices.numel() < 2:
            continue
        for mode_idx in range(num_modes):
            traj = mode_xy[mode_idx, node_indices]  # [M, H, 2]
            pair_dist = torch.norm(traj[:, None, :, :] - traj[None, :, :, :], p=2, dim=-1)  # [M, M, H]
            pair_dist = pair_dist.min(dim=-1).values
            pair_dist = pair_dist + torch.eye(node_indices.numel(), device=y_hat.device) * 1e6
            min_dist = pair_dist.min(dim=-1).values
            min_pair_dist[node_indices, mode_idx] = min_dist
            conflict_targets[node_indices, mode_idx] = (min_dist < conflict_threshold).float()

    conflict_targets = conflict_targets * valid_nodes.unsqueeze(-1).float()
    min_pair_dist[~valid_nodes] = float("inf")
    return conflict_targets, min_pair_dist


def compute_offroad_risk_targets(
    y_hat: torch.Tensor,
    reg_mask: torch.Tensor,
    lane_positions: torch.Tensor,
    lane_actor_index: torch.Tensor,
    lane_actor_vectors: torch.Tensor,
    offroad_threshold: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """基于轨迹到 lane 近邻的距离构造 off-road 风险标签（文档 9.1）。"""
    mode_xy = y_hat[..., :2]
    num_modes, num_nodes, future_steps, _ = mode_xy.shape
    offroad_targets = torch.zeros(num_nodes, num_modes, device=y_hat.device)
    lane_distance = torch.full((num_nodes, num_modes), float("inf"), device=y_hat.device)
    valid_nodes = reg_mask.any(dim=-1)

    lane_lookup = {}
    for lane_idx, node_idx in lane_actor_index.t().tolist():
        lane_lookup.setdefault(node_idx, []).append(lane_idx)

    for node_idx in range(num_nodes):
        lane_ids = lane_lookup.get(node_idx, [])
        if not lane_ids:
            continue
        lane_xy = lane_positions[torch.tensor(lane_ids, device=y_hat.device)]
        for mode_idx in range(num_modes):
            traj = mode_xy[mode_idx, node_idx]
            dists = torch.cdist(traj, lane_xy, p=2)
            min_dist = dists.min()
            lane_distance[node_idx, mode_idx] = min_dist
            offroad_targets[node_idx, mode_idx] = (min_dist > offroad_threshold).float()

    offroad_targets = offroad_targets * valid_nodes.unsqueeze(-1).float()
    lane_distance[~valid_nodes] = float("inf")
    return offroad_targets, lane_distance


def reconstruct_lane_positions(
    lane_actor_index: torch.Tensor,
    lane_actor_vectors: torch.Tensor,
    current_positions: torch.Tensor,
    num_lanes: int,
) -> torch.Tensor:
    """从 lane->actor 相对向量恢复 lane 点坐标。"""
    lane_positions = current_positions.new_zeros((num_lanes, current_positions.size(-1)))
    lane_seen = torch.zeros(num_lanes, dtype=torch.bool, device=current_positions.device)
    for edge_idx in range(lane_actor_index.size(1)):
        lane_idx = int(lane_actor_index[0, edge_idx].item())
        node_idx = int(lane_actor_index[1, edge_idx].item())
        if lane_seen[lane_idx]:
            continue
        lane_positions[lane_idx] = current_positions[node_idx] + lane_actor_vectors[edge_idx]
        lane_seen[lane_idx] = True
    return lane_positions
def build_reliability_targets(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    reg_mask: torch.Tensor,
    batch: torch.Tensor,
    lane_positions: torch.Tensor,
    lane_actor_index: torch.Tensor,
    lane_actor_vectors: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    historical_steps: int = 20,
    rotate_mat: Optional[torch.Tensor] = None,
    agent_index: Optional[torch.Tensor] = None,
    fde_threshold: float = 1.0,
    conflict_threshold: float = 1.0,
    offroad_threshold: float = 2.0,
    ade_threshold: float = 1.0,
    miss_threshold: float = 2.0,
    mode_target_policy: str = "fde_only",
    scene_target_policy: str = "target_best_mode_fail",
    conflict_scope: str = "target_to_neighbors",
    conflict_min_frames: int = 2,
    scene_rate_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """统一构造 mode / scene 可靠性监督信号（文档第 6 / 9 节）。

    本阶段不再默认使用五类标签并集，而是由 `mode_target_policy` 决定主 mode 监督：

    - `fde_only`（默认）：`y_mode = 1(FDE_k > fde_threshold)`，语义清晰、口径与 minFDE 一致。
    - `miss_only`：`y_mode = 1(FDE_k > miss_threshold)`，对应彻底 miss。
    - `fde_or_miss`：FDE 失败或 miss 之一即为正（仅在阈值体检后启用）。
    - `all_union`：旧的五类并集（FDE/ADE/miss/conflict/offroad），保留作对照，易塌缩。

    `conflict` / `offroad` / `ade` 默认仅作为日志子标签返回，不进入主 `mode_targets`，
    避免再次把主监督打满。scene 监督由 `scene_target_policy` 决定（见
    `compute_scene_risk_targets`），默认 `target_best_mode_fail`。

    同时返回每个 mode 的连续位移误差 `mode_error`（[N, F]），供 rank loss 使用。
    """
    mode_targets_fde, valid_mask, fde = compute_mode_risk_targets(
        y_hat=y_hat,
        y=y,
        reg_mask=reg_mask,
        fde_threshold=fde_threshold,
    )
    ade_targets, ade = compute_ade_risk_targets(
        y_hat=y_hat,
        y=y,
        reg_mask=reg_mask,
        ade_threshold=ade_threshold,
    )
    miss_targets, _ = compute_miss_risk_targets(
        y_hat=y_hat,
        y=y,
        reg_mask=reg_mask,
        miss_threshold=miss_threshold,
    )
    if positions is None:
        raise ValueError("build_reliability_targets 需要 positions 才能在 scene 绝对坐标系下计算 conflict")
    conflict_targets, min_pair_dist = compute_conflict_risk_targets(
        y_hat=y_hat,
        reg_mask=reg_mask,
        batch=batch,
        positions=positions,
        historical_steps=historical_steps,
        rotate_mat=rotate_mat,
        agent_index=agent_index,
        conflict_threshold=conflict_threshold,
        min_frames=conflict_min_frames,
        scope=conflict_scope,
    )
    offroad_targets, lane_distance = compute_offroad_risk_targets(
        y_hat=y_hat,
        reg_mask=reg_mask,
        lane_positions=lane_positions,
        lane_actor_index=lane_actor_index,
        lane_actor_vectors=lane_actor_vectors,
        offroad_threshold=offroad_threshold,
    )

    # 主 mode 监督：按 policy 选择，默认 fde_only，避免子标签污染。
    if mode_target_policy == "fde_only":
        mode_targets = mode_targets_fde
    elif mode_target_policy == "miss_only":
        mode_targets = miss_targets
    elif mode_target_policy == "fde_or_miss":
        mode_targets = torch.maximum(mode_targets_fde, miss_targets)
    elif mode_target_policy == "all_union":
        mode_targets = mode_targets_fde
        for component in (ade_targets, miss_targets, conflict_targets, offroad_targets):
            mode_targets = torch.maximum(mode_targets, component)
    else:
        raise ValueError("unknown mode_target_policy: {}".format(mode_target_policy))

    scene_targets = compute_scene_risk_targets(
        mode_targets=mode_targets,
        batch=batch,
        valid_mask=valid_mask,
        fde=fde,
        agent_index=agent_index,
        policy=scene_target_policy,
        rate_threshold=scene_rate_threshold,
    )
    return {
        "mode_targets": mode_targets,
        "fde_targets": mode_targets_fde,
        "ade_targets": ade_targets,
        "miss_targets": miss_targets,
        "conflict_targets": conflict_targets,
        "offroad_targets": offroad_targets,
        "scene_targets": scene_targets,
        "valid_mask": valid_mask,
        "fde": fde,
        "ade": ade,
        "mode_error": ade,
        "min_pair_dist": min_pair_dist,
        "lane_distance": lane_distance,
        "agent_index": agent_index if agent_index is None else agent_index.view(-1),
    }


def summarize_reliability_targets(targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """汇总可靠性监督标签的命中率，便于训练日志观察。"""
    valid_mask = targets["valid_mask"]
    stats = {}
    valid_count = int(valid_mask.sum().item())
    for source_key, stat_key in (
        ("mode_targets", "mode_positive_rate"),
        ("fde_targets", "fde_positive_rate"),
        ("ade_targets", "ade_positive_rate"),
        ("miss_targets", "miss_positive_rate"),
        ("conflict_targets", "conflict_positive_rate"),
        ("offroad_targets", "offroad_positive_rate"),
    ):
        if source_key not in targets:
            continue
        values = targets[source_key]
        if valid_count > 0:
            masked_values = values[valid_mask]
            stats[stat_key] = masked_values.float().mean()
        else:
            stats[stat_key] = values.new_zeros(())
    scene_targets = targets["scene_targets"]
    stats["scene_positive_rate"] = scene_targets.float().mean() if scene_targets.numel() > 0 else scene_targets.new_zeros(())

    # target / non-target actor 拆分，区分“主标签真的恢复可分”与“只是被邻车标签污染得不那么极端”。
    mode_targets = targets["mode_targets"]
    agent_index = targets.get("agent_index")
    fde = targets.get("fde")
    device = valid_mask.device
    if agent_index is not None and agent_index.numel() > 0:
        agent_index = agent_index.view(-1)
        target_mask = torch.zeros(valid_mask.size(0), dtype=torch.bool, device=device)
        target_mask[agent_index] = True
        target_valid = target_mask & valid_mask
        non_target_valid = (~target_mask) & valid_mask
        if target_valid.any():
            stats["target_actor_mode_positive_rate"] = mode_targets[target_valid].float().mean()
            # target_best_mode_fail_rate：目标车 min-FDE mode 失败的比例。
            if fde is not None:
                valid_targets = agent_index[valid_mask[agent_index]]
                if valid_targets.numel() > 0:
                    best_modes = fde[valid_targets].argmin(dim=-1)
                    best_fail = mode_targets[valid_targets, best_modes]
                    stats["target_best_mode_fail_rate"] = best_fail.float().mean()
                else:
                    stats["target_best_mode_fail_rate"] = mode_targets.new_zeros(())
        else:
            stats["target_actor_mode_positive_rate"] = mode_targets.new_zeros(())
            stats["target_best_mode_fail_rate"] = mode_targets.new_zeros(())
        if non_target_valid.any():
            stats["non_target_actor_mode_positive_rate"] = mode_targets[non_target_valid].float().mean()
        else:
            stats["non_target_actor_mode_positive_rate"] = mode_targets.new_zeros(())

    # 计数与 min_pair_dist 分位数，便于挑选 tau_col。
    stats["scene_count"] = torch.tensor(float(scene_targets.numel()), device=device)
    stats["actor_count"] = torch.tensor(float(valid_mask.numel()), device=device)
    stats["valid_actor_count"] = torch.tensor(float(valid_count), device=device)
    min_pair_dist = targets.get("min_pair_dist")
    if min_pair_dist is not None:
        finite = min_pair_dist[torch.isfinite(min_pair_dist)]
        if finite.numel() > 0:
            for q in (0.1, 0.5, 0.9):
                stats["min_pair_dist_q{}".format(int(q * 100))] = torch.quantile(finite, q)
    return stats
def _trajectory_kinematic_features(y_hat: torch.Tensor) -> torch.Tensor:
    """从候选轨迹提取运动学/几何风险特征（文档 7.1）。

    关注“这条轨迹自身看起来稳不稳”，而非场景语义。提取的特征包括终点偏移、
    全程位移分布、速度与加速度变化、航向与曲率变化、形状异常（bbox）等。

    Args:
        y_hat: [F, N, H, 4] 或 [F, N, H, 2]，最后一维前 2 项为位置，后 2 项为尺度。

    Returns:
        [N, F, KINEMATIC_FEATURE_DIM] 的逐 mode 特征。
    """
    num_modes, num_nodes, future_steps, _ = y_hat.shape
    xy = y_hat[..., :2]  # [F, N, H, 2]

    start = xy[:, :, 0]                       # [F, N, 2]
    end = xy[:, :, -1]                        # [F, N, 2]
    net_disp = end - start                    # [F, N, 2]
    net_dist = torch.norm(net_disp, p=2, dim=-1, keepdim=True)  # [F, N, 1]

    # 速度 / 加速度（一阶、二阶差分）。
    vel = xy[:, :, 1:] - xy[:, :, :-1] if future_steps > 1 else torch.zeros_like(xy[:, :, :1])
    speed = torch.norm(vel, p=2, dim=-1)      # [F, N, H-1]
    path_len = speed.sum(dim=-1, keepdim=True)
    speed_mean = speed.mean(dim=-1, keepdim=True)
    speed_std = speed.std(dim=-1, unbiased=False, keepdim=True)
    speed_max = speed.amax(dim=-1, keepdim=True)

    if speed.size(-1) > 1:
        accel = speed[:, :, 1:] - speed[:, :, :-1]
        accel_abs_mean = accel.abs().mean(dim=-1, keepdim=True)
        accel_abs_max = accel.abs().amax(dim=-1, keepdim=True)
    else:
        accel_abs_mean = torch.zeros_like(speed_mean)
        accel_abs_max = torch.zeros_like(speed_mean)

    # 航向变化与曲率。
    heading = torch.atan2(vel[..., 1], vel[..., 0])  # [F, N, H-1]
    if heading.size(-1) > 1:
        dheading = heading[:, :, 1:] - heading[:, :, :-1]
        # wrap 到 [-pi, pi]
        dheading = torch.atan2(torch.sin(dheading), torch.cos(dheading))
        heading_change_abs = dheading.abs().sum(dim=-1, keepdim=True)
        heading_change_max = dheading.abs().amax(dim=-1, keepdim=True)
        seg_len = speed[:, :, 1:].clamp(min=1e-4)
        curvature = (dheading.abs() / seg_len).mean(dim=-1, keepdim=True)
    else:
        heading_change_abs = torch.zeros_like(speed_mean)
        heading_change_max = torch.zeros_like(speed_mean)
        curvature = torch.zeros_like(speed_mean)

    # 直线度：净位移 / 路径长度，越小越绕（形状越不自然）。
    straightness = net_dist / path_len.clamp(min=1e-4)

    # 轨迹包络盒（形状异常线索）。
    bbox = xy.amax(dim=2) - xy.amin(dim=2)    # [F, N, 2]

    # 预测尺度（不确定性），若无则为 0。
    if y_hat.size(-1) >= 4:
        scale = y_hat[..., 2:4]
        scale_mean = scale.mean(dim=(2, 3)).unsqueeze(-1)  # [F, N, 1]
        scale_max = scale.amax(dim=(2, 3)).unsqueeze(-1)
    else:
        scale_mean = torch.zeros_like(net_dist)
        scale_max = torch.zeros_like(net_dist)

    feats = torch.cat(
        [
            start, end, net_disp, net_dist,           # 2+2+2+1
            path_len, speed_mean, speed_std, speed_max,  # 1+1+1+1
            accel_abs_mean, accel_abs_max,            # 1+1
            heading_change_abs, heading_change_max, curvature,  # 1+1+1
            straightness, bbox, scale_mean, scale_max,  # 1+2+1+1
        ],
        dim=-1,
    )  # [F, N, KINEMATIC_FEATURE_DIM]
    return feats.transpose(0, 1)  # [N, F, D]


class TrajectoryReliabilityEncoder(nn.Module):
    """轨迹可靠性编码器（文档 7.1）。

    以单条候选轨迹为中心，把其几何/动力学特征编码成 mode-level 表征。
    """

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(KINEMATIC_FEATURE_DIM + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.apply(init_weights)

    def forward(self, y_hat: torch.Tensor, mode_embed: torch.Tensor) -> torch.Tensor:
        """编码每条候选轨迹。

        Args:
            y_hat: [F, N, H, C] 候选轨迹。
            mode_embed: [N, F, embed_dim] 来自 backbone 的逐 mode 全局表征。

        Returns:
            [N, F, hidden_dim] 的 mode-level 轨迹风险表征。
        """
        kinematic = _trajectory_kinematic_features(y_hat)  # [N, F, D]
        feats = torch.cat([kinematic, mode_embed], dim=-1)
        return self.encoder(feats)
def _neighborhood_density(batch: torch.Tensor) -> torch.Tensor:
    """每个 actor 所在场景的邻域密度描述子（log 缩放的同场景 actor 数）。"""
    if batch.numel() == 0:
        return batch.new_zeros((0, 1), dtype=torch.float)
    num_graphs = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=num_graphs).float()
    density = torch.log1p(counts[batch]).unsqueeze(-1)  # [N, 1]
    return density


def _build_scene_attn_mask(batch: torch.Tensor) -> torch.Tensor:
    """构造场景内注意力 mask，True 表示禁止 attend（跨场景）。"""
    same_scene = batch.unsqueeze(0) == batch.unsqueeze(1)  # [N, N]
    return ~same_scene


class InteractionContextEncoder(nn.Module):
    """交互感知上下文编码器（文档 7.2）。

    不只看单条轨迹，而是在场景内对 actor 做多头自注意力，融合 target / 周边 agent、
    场景结构与邻域密度等信息，编码出 interaction-aware 上下文表征。
    """

    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        # 输入：local_embed + mode 池化后的 global_embed + 邻域密度描述子。
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                "norm1": nn.LayerNorm(hidden_dim),
                "norm2": nn.LayerNorm(hidden_dim),
                "ff": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
            })
            for _ in range(num_layers)
        ])
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """编码场景交互上下文。

        Args:
            local_embed: [N, embed_dim] 局部 actor 表征。
            global_embed: [F, N, embed_dim] 多模态全局表征。
            batch: [N] 每个 actor 的场景归属。

        Returns:
            [N, hidden_dim] 的 interaction-aware 上下文表征。
        """
        num_nodes = local_embed.size(0)
        if num_nodes == 0:
            return local_embed.new_zeros((0, self.layers[0]["norm1"].normalized_shape[0]))
        global_pool = global_embed.mean(dim=0)  # [N, embed_dim]
        density = _neighborhood_density(batch)
        x = self.input_proj(torch.cat([local_embed, global_pool, density], dim=-1))  # [N, hidden]

        # batch_first 注意力：把所有 actor 当成长度 N 的序列，用 mask 限制在同场景内。
        attn_mask = _build_scene_attn_mask(batch)  # [N, N]
        x = x.unsqueeze(0)  # [1, N, hidden]
        for layer in self.layers:
            attn_out, _ = layer["attn"](x, x, x, attn_mask=attn_mask, need_weights=False)
            x = layer["norm1"](x + attn_out)
            x = layer["norm2"](x + layer["ff"](x))
        return x.squeeze(0)  # [N, hidden]
class ReliabilityModule(nn.Module):
    """HiVT 后接的即插即用可靠性分支（文档 7.3 / 7.4）。

    组合 `TrajectoryReliabilityEncoder`（轨迹自身风险）与
    `InteractionContextEncoder`（场景交互上下文），输出：

    - `mode_risk` / `mode_risk_logits`：每条候选轨迹的失败风险。
    - `scene_risk` / `scene_risk_logits`：每个场景整体的预测风险。
    - `reranked_pi`：风险感知重排后的候选分布。
    """

    def __init__(
        self,
        embed_dim: int,
        future_steps: int,
        num_modes: int,
        hidden_dim: int = 128,
        rerank_alpha: float = 0.5,
        num_interaction_heads: int = 4,
        num_interaction_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rerank_alpha = rerank_alpha
        self.hidden_dim = hidden_dim

        self.traj_encoder = TrajectoryReliabilityEncoder(embed_dim=embed_dim, hidden_dim=hidden_dim)
        self.interaction_encoder = InteractionContextEncoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_interaction_heads,
            num_layers=num_interaction_layers,
            dropout=dropout,
        )

        # mode head：轨迹风险表征 + 交互上下文（按 mode 广播）+ 原始 mode score。
        self.mode_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # scene head：场景池化的交互上下文 + 场景内 mode 风险聚合。
        self.scene_head = nn.Sequential(
            nn.Linear(hidden_dim + num_modes + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_weights)

    def forward(
        self,
        local_embed: torch.Tensor,
        global_embed: torch.Tensor,
        y_hat: torch.Tensor,
        pi: torch.Tensor,
        batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        num_modes, num_nodes, _, _ = y_hat.shape
        mode_embed = global_embed.transpose(0, 1)  # [N, F, embed_dim]

        traj_embed = self.traj_encoder(y_hat=y_hat, mode_embed=mode_embed)  # [N, F, hidden]
        context = self.interaction_encoder(local_embed=local_embed, global_embed=global_embed, batch=batch)  # [N, hidden]
        context_expand = context.unsqueeze(1).expand(num_nodes, num_modes, -1)
        pi_feature = torch.softmax(pi, dim=-1).unsqueeze(-1)  # [N, F, 1]

        mode_features = torch.cat([traj_embed, context_expand, pi_feature], dim=-1)
        mode_risk_logits = self.mode_head(mode_features).squeeze(-1)  # [N, F]
        mode_risk = torch.sigmoid(mode_risk_logits)

        scene_mode = mode_risk.max(dim=-1).values  # [N]
        if batch.numel() == 0:
            scene_risk_logits = scene_mode.new_zeros(0)
            scene_risk = scene_mode.new_zeros(0)
        else:
            num_graphs = int(batch.max().item()) + 1
            scene_feats = []
            for graph_idx in range(num_graphs):
                node_mask = batch == graph_idx
                if node_mask.any():
                    context_pool = context[node_mask].mean(dim=0)
                    mode_pool = mode_risk[node_mask].max(dim=0).values
                    scene_feats.append(
                        torch.cat([context_pool, mode_pool, scene_mode[node_mask].max().unsqueeze(0)], dim=0))
                else:
                    scene_feats.append(context.new_zeros(self.hidden_dim + self.num_modes + 1))
            scene_feats = torch.stack(scene_feats, dim=0)
            scene_risk_logits = self.scene_head(scene_feats).squeeze(-1)
            scene_risk = torch.sigmoid(scene_risk_logits)

        reranked_pi = apply_risk_reranking(pi=pi, mode_risk=mode_risk, alpha=self.rerank_alpha)
        return {
            "mode_risk_logits": mode_risk_logits,
            "mode_risk": mode_risk,
            "scene_risk_logits": scene_risk_logits,
            "scene_risk": scene_risk,
            "reranked_pi": reranked_pi,
        }





