# Reliability Label Collapse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore meaningful InterAct-Risk supervision by replacing dense union labels with explicit target policies, stricter conflict labels, target-centric scene labels, and a no-backprop target audit script.

**Architecture:** Keep the HiVT backbone and ReliabilityModule architecture unchanged. Concentrate the supervision protocol in `models/reliability.py`, pass policy knobs through `models/hivt.py` and CLI/preset helpers, and add `scripts/audit_reliability_targets.py` for Exp A label audits before training.

**Tech Stack:** Python, PyTorch, PyTorch Lightning, torch_geometric, pytest, existing HiVT ArgoverseV1DataModule.

## Global Constraints

- Implement only the first-stage conservative fix from `docs/实验问题阶段性解决方案文档.md`.
- Default `mode_target_policy` must be `fde_only`; conflict/off-road/ADE stay logged but do not enter main supervision by default.
- Default `scene_target_policy` must be `target_actor`.
- Default `conflict_scope` must be `target_to_neighbors`.
- Do not change the HiVT backbone, decoder, or reliability network heads.
- Keep `rank_loss_weight` and `calib_loss_weight` defaulted to `0.0`.
- Preserve legacy behavior behind explicit policies: `mode_target_policy=all_union`, `scene_target_policy=scene_max`, `conflict_scope=all_valid_pairs`.
- Do not require a full Argoverse dataset for unit tests.

---

## File Structure

- Modify `models/reliability.py`: Owns target construction policies, conflict target generation, scene target aggregation, and summary stats.
- Modify `models/hivt.py`: Owns model hyperparameters, train/val target construction calls, and reliability logs.
- Modify `models/__init__.py`: Export any new helper only if the audit script needs it directly.
- Modify `training_presets.py`: Keep reliability preset defaults aligned with conservative fix.
- Modify `tests/test_reliability.py`: Cover new policy behavior without importing the full model package.
- Modify `tests/test_reliability_presets.py`: Cover new preset keys and defaults.
- Create `scripts/audit_reliability_targets.py`: No-backprop target audit for train/val batches.
- Modify `run_single_gpu.sh`: Pass environment overrides for reliability label policies and thresholds.

---

### Task 1: Reliability Target Policies

**Files:**
- Modify: `models/reliability.py:53-309`
- Modify: `tests/test_reliability.py:35-233`

**Interfaces:**
- Consumes: existing tensors `y_hat: torch.Tensor [F,N,H,C]`, `y: torch.Tensor [N,H,2]`, `reg_mask: torch.Tensor [N,H]`, `batch: torch.Tensor [N]`.
- Produces: `build_reliability_targets(..., mode_target_policy: str, scene_target_policy: str, agent_index: Optional[torch.Tensor], conflict_min_frames: int, conflict_scope: str, scene_rate_threshold: float) -> Dict[str, torch.Tensor]`.
- Produces: `compute_scene_risk_targets(mode_targets, batch, valid_mask=None, agent_index=None, policy='target_actor', rate_threshold=0.5) -> torch.Tensor`.
- Produces: `compute_conflict_risk_targets(y_hat, reg_mask, batch, conflict_threshold=1.0, conflict_min_frames=2, conflict_scope='target_to_neighbors', agent_index=None) -> Tuple[torch.Tensor, torch.Tensor]`.

- [ ] **Step 1: Add failing tests for mode target policies**

Append these tests to `tests/test_reliability.py` near the existing `test_build_reliability_targets_combines_fde_conflict_and_offroad` test. The first test asserts the new conservative default; the second asserts legacy union remains opt-in.

```python
def test_build_reliability_targets_defaults_to_fde_only_mode_policy():
    y_hat = torch.zeros(2, 2, 3, 2)
    y = torch.zeros(2, 3, 2)
    reg_mask = torch.ones(2, 3, dtype=torch.bool)
    batch = torch.tensor([0, 0])
    agent_index = torch.tensor([0])
    lane_positions = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    lane_actor_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    lane_actor_vectors = torch.tensor([[0.0, 0.0], [1.0, 0.0], [-8.0, 0.0], [-7.0, 0.0]])

    # Mode 0 has a conflict/offroad proxy but good FDE; mode 1 has bad FDE.
    y_hat[0, 0] = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    y_hat[0, 1] = torch.tensor([[0.0, 0.3], [0.1, 0.3], [0.2, 0.3]])
    y_hat[1, 0] = torch.tensor([[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    y_hat[1, 1] = torch.tensor([[8.0, 0.0], [9.0, 0.0], [10.0, 0.0]])

    outputs = build_reliability_targets(
        y_hat=y_hat,
        y=y,
        reg_mask=reg_mask,
        batch=batch,
        lane_positions=lane_positions,
        lane_actor_index=lane_actor_index,
        lane_actor_vectors=lane_actor_vectors,
        agent_index=agent_index,
        fde_threshold=1.0,
        conflict_threshold=0.5,
        conflict_min_frames=2,
        offroad_threshold=2.0,
    )

    assert outputs["mode_targets"].tolist() == [[0.0, 1.0], [0.0, 1.0]]
    assert outputs["conflict_targets"].tolist() == [[1.0, 0.0], [1.0, 0.0]]
    assert outputs["offroad_targets"].tolist() == [[0.0, 0.0], [0.0, 1.0]]


def test_build_reliability_targets_legacy_all_union_policy_is_opt_in():
    y_hat = torch.zeros(2, 2, 3, 2)
    y = torch.zeros(2, 3, 2)
    reg_mask = torch.ones(2, 3, dtype=torch.bool)
    batch = torch.tensor([0, 0])
    agent_index = torch.tensor([0])
    lane_positions = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    lane_actor_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    lane_actor_vectors = torch.tensor([[0.0, 0.0], [1.0, 0.0], [-8.0, 0.0], [-7.0, 0.0]])

    y_hat[0, 0] = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    y_hat[0, 1] = torch.tensor([[0.0, 0.3], [0.1, 0.3], [0.2, 0.3]])
    y_hat[1, 0] = torch.tensor([[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    y_hat[1, 1] = torch.tensor([[8.0, 0.0], [9.0, 0.0], [10.0, 0.0]])

    outputs = build_reliability_targets(
        y_hat=y_hat,
        y=y,
        reg_mask=reg_mask,
        batch=batch,
        lane_positions=lane_positions,
        lane_actor_index=lane_actor_index,
        lane_actor_vectors=lane_actor_vectors,
        agent_index=agent_index,
        fde_threshold=1.0,
        conflict_threshold=0.5,
        conflict_min_frames=2,
        offroad_threshold=2.0,
        mode_target_policy="all_union",
        scene_target_policy="scene_max",
        conflict_scope="all_valid_pairs",
    )

    assert outputs["mode_targets"].tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert outputs["scene_targets"].tolist() == [1.0]
```

- [ ] **Step 2: Add failing tests for scene policies and conflict min frames**

Append these tests to `tests/test_reliability.py` near the existing scene/conflict tests.

```python
def test_scene_risk_target_actor_ignores_non_target_neighbor_risk():
    mode_targets = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ])
    batch = torch.tensor([0, 0, 1, 1])
    valid_mask = torch.ones(4, dtype=torch.bool)
    agent_index = torch.tensor([0, 2])

    target_actor = compute_scene_risk_targets(
        mode_targets=mode_targets,
        batch=batch,
        valid_mask=valid_mask,
        agent_index=agent_index,
        policy="target_actor",
    )
    scene_max = compute_scene_risk_targets(
        mode_targets=mode_targets,
        batch=batch,
        valid_mask=valid_mask,
        agent_index=agent_index,
        policy="scene_max",
    )

    assert target_actor.tolist() == [0.0, 1.0]
    assert scene_max.tolist() == [1.0, 1.0]


def test_conflict_targets_require_minimum_same_time_close_frames():
    y_hat = torch.zeros(1, 2, 4, 2)
    reg_mask = torch.ones(2, 4, dtype=torch.bool)
    batch = torch.tensor([0, 0])
    agent_index = torch.tensor([0])

    y_hat[0, 0] = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    y_hat[0, 1] = torch.tensor([[0.0, 0.2], [1.0, 2.0], [2.0, 0.2], [3.0, 2.0]])

    loose_targets, _ = compute_conflict_risk_targets(
        y_hat=y_hat,
        reg_mask=reg_mask,
        batch=batch,
        agent_index=agent_index,
        conflict_threshold=0.5,
        conflict_min_frames=2,
        conflict_scope="target_to_neighbors",
    )
    strict_targets, _ = compute_conflict_risk_targets(
        y_hat=y_hat,
        reg_mask=reg_mask,
        batch=batch,
        agent_index=agent_index,
        conflict_threshold=0.5,
        conflict_min_frames=3,
        conflict_scope="target_to_neighbors",
    )

    assert loose_targets.tolist() == [[1.0], [1.0]]
    assert strict_targets.tolist() == [[0.0], [0.0]]
```

- [ ] **Step 3: Run reliability tests and verify they fail**

Run:

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability.py -q
```

Expected: FAIL because `agent_index`, `mode_target_policy`, `scene_target_policy`, `conflict_min_frames`, and `conflict_scope` are not implemented yet.

- [ ] **Step 4: Implement policy helpers in `models/reliability.py`**

Insert these helpers above `compute_scene_risk_targets` in `models/reliability.py`:

```python
def _validate_policy(name: str, value: str, choices: Tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name}={value!r} is not supported; expected one of {choices}")


def _graph_agent_indices(batch: torch.Tensor, agent_index: Optional[torch.Tensor]) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    if num_graphs == 0:
        return batch.new_zeros(0, dtype=torch.long)
    if agent_index is None:
        graph_agents = []
        for graph_idx in range(num_graphs):
            nodes = torch.nonzero(batch == graph_idx, as_tuple=False).flatten()
            graph_agents.append(nodes[0] if nodes.numel() > 0 else batch.new_tensor(0))
        return torch.stack(graph_agents).long()
    return agent_index.to(device=batch.device, dtype=torch.long)
```

- [ ] **Step 5: Replace `compute_scene_risk_targets` implementation**

Replace the whole existing `compute_scene_risk_targets` function with:

```python
def compute_scene_risk_targets(
    mode_targets: torch.Tensor,
    batch: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    agent_index: Optional[torch.Tensor] = None,
    policy: str = "target_actor",
    rate_threshold: float = 0.5,
) -> torch.Tensor:
    """Aggregate mode-level labels into scene-level labels.

    Default `target_actor` keeps scene supervision aligned with HiVT's target-agent
    metrics and avoids unrelated neighbors making the whole scene positive.
    """
    _validate_policy("scene_target_policy", policy, ("target_actor", "target_mode_rate", "scene_rate", "scene_max"))
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    scene_targets = torch.zeros(num_graphs, device=mode_targets.device)
    if num_graphs == 0:
        return scene_targets

    node_risk = mode_targets.max(dim=-1).values
    if valid_mask is not None:
        node_risk = node_risk * valid_mask.float()
    graph_agents = _graph_agent_indices(batch=batch, agent_index=agent_index)

    for graph_idx in range(num_graphs):
        node_mask = batch == graph_idx
        if not node_mask.any():
            continue
        if policy in ("target_actor", "target_mode_rate"):
            target_idx = int(graph_agents[graph_idx].item())
            if target_idx < 0 or target_idx >= mode_targets.size(0):
                continue
            if valid_mask is not None and not bool(valid_mask[target_idx].item()):
                continue
            if policy == "target_actor":
                scene_targets[graph_idx] = node_risk[target_idx]
            else:
                scene_targets[graph_idx] = (mode_targets[target_idx].float().mean() > rate_threshold).float()
        elif policy == "scene_rate":
            valid_nodes = node_mask if valid_mask is None else (node_mask & valid_mask)
            if valid_nodes.any():
                scene_targets[graph_idx] = (mode_targets[valid_nodes].float().mean() > rate_threshold).float()
        else:
            scene_targets[graph_idx] = node_risk[node_mask].max()
    return scene_targets
```

- [ ] **Step 6: Replace `compute_conflict_risk_targets` implementation**

Replace the whole existing `compute_conflict_risk_targets` function with:

```python
def compute_conflict_risk_targets(
    y_hat: torch.Tensor,
    reg_mask: torch.Tensor,
    batch: torch.Tensor,
    conflict_threshold: float = 1.0,
    conflict_min_frames: int = 2,
    conflict_scope: str = "target_to_neighbors",
    agent_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build same-time collision-proxy labels for predicted modes.

    A mode is positive when it is close to another valid actor at the same
    future timestep for at least `conflict_min_frames` valid frames.
    """
    _validate_policy("conflict_scope", conflict_scope, ("target_to_neighbors", "all_valid_pairs"))
    mode_xy = y_hat[..., :2]
    num_modes, num_nodes, _, _ = mode_xy.shape
    min_frames = max(int(conflict_min_frames), 1)
    conflict_targets = torch.zeros(num_nodes, num_modes, device=y_hat.device)
    min_pair_dist = torch.full((num_nodes, num_modes), float("inf"), device=y_hat.device)
    valid_nodes = reg_mask.any(dim=-1)
    graph_agents = _graph_agent_indices(batch=batch, agent_index=agent_index)
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

    for graph_idx in range(num_graphs):
        node_indices = torch.nonzero((batch == graph_idx) & valid_nodes, as_tuple=False).flatten()
        if node_indices.numel() < 2:
            continue
        if conflict_scope == "target_to_neighbors":
            target_idx = int(graph_agents[graph_idx].item()) if graph_idx < graph_agents.numel() else int(node_indices[0].item())
            if target_idx not in node_indices.tolist():
                continue
            source_indices = node_indices.new_tensor([target_idx])
        else:
            source_indices = node_indices

        for mode_idx in range(num_modes):
            traj = mode_xy[mode_idx]
            for src_idx in source_indices.tolist():
                neighbor_indices = node_indices[node_indices != src_idx]
                if neighbor_indices.numel() == 0:
                    continue
                valid_steps = reg_mask[src_idx].unsqueeze(0) & reg_mask[neighbor_indices]
                dist = torch.norm(traj[src_idx].unsqueeze(0) - traj[neighbor_indices], p=2, dim=-1)
                masked_dist = dist.masked_fill(~valid_steps, float("inf"))
                min_dist = masked_dist.min()
                close_frames = ((masked_dist < conflict_threshold) & valid_steps).sum(dim=-1)
                has_conflict = close_frames >= min_frames
                min_pair_dist[src_idx, mode_idx] = torch.minimum(min_pair_dist[src_idx, mode_idx], min_dist)
                if has_conflict.any():
                    conflict_targets[src_idx, mode_idx] = 1.0
                    hit_neighbors = neighbor_indices[has_conflict]
                    conflict_targets[hit_neighbors, mode_idx] = 1.0
                    neighbor_min = masked_dist[has_conflict].min(dim=-1).values
                    min_pair_dist[hit_neighbors, mode_idx] = torch.minimum(min_pair_dist[hit_neighbors, mode_idx], neighbor_min)

    conflict_targets = conflict_targets * valid_nodes.unsqueeze(-1).float()
    min_pair_dist[~valid_nodes] = float("inf")
    return conflict_targets, min_pair_dist
```

- [ ] **Step 7: Update `build_reliability_targets` signature and mode policy**

Change the function signature in `models/reliability.py` to include the new parameters:

```python
def build_reliability_targets(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    reg_mask: torch.Tensor,
    batch: torch.Tensor,
    lane_positions: torch.Tensor,
    lane_actor_index: torch.Tensor,
    lane_actor_vectors: torch.Tensor,
    fde_threshold: float = 1.0,
    conflict_threshold: float = 1.0,
    offroad_threshold: float = 2.0,
    ade_threshold: float = 1.0,
    miss_threshold: float = 2.0,
    mode_target_policy: str = "fde_only",
    scene_target_policy: str = "target_actor",
    conflict_min_frames: int = 2,
    conflict_scope: str = "target_to_neighbors",
    scene_rate_threshold: float = 0.5,
    agent_index: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
```

Inside the function, update conflict and mode/scene construction to:

```python
    conflict_targets, min_pair_dist = compute_conflict_risk_targets(
        y_hat=y_hat,
        reg_mask=reg_mask,
        batch=batch,
        conflict_threshold=conflict_threshold,
        conflict_min_frames=conflict_min_frames,
        conflict_scope=conflict_scope,
        agent_index=agent_index,
    )
```

Then replace the legacy union block with:

```python
    _validate_policy("mode_target_policy", mode_target_policy, ("fde_only", "miss_only", "fde_or_miss", "all_union"))
    if mode_target_policy == "fde_only":
        mode_targets = mode_targets_fde
    elif mode_target_policy == "miss_only":
        mode_targets = miss_targets
    elif mode_target_policy == "fde_or_miss":
        mode_targets = torch.maximum(mode_targets_fde, miss_targets)
    else:
        mode_targets = mode_targets_fde
        for component in (ade_targets, miss_targets, conflict_targets, offroad_targets):
            mode_targets = torch.maximum(mode_targets, component)

    scene_targets = compute_scene_risk_targets(
        mode_targets=mode_targets,
        batch=batch,
        valid_mask=valid_mask,
        agent_index=agent_index,
        policy=scene_target_policy,
        rate_threshold=scene_rate_threshold,
    )
```

- [ ] **Step 8: Add ADE/miss rates to summary test and implementation**

Update `test_summarize_reliability_targets_reports_component_positive_rates` in `tests/test_reliability.py` so the test input includes `ade_targets` and `miss_targets`:

```python
        "ade_targets": torch.tensor([
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]),
        "miss_targets": torch.tensor([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]),
```

Add these assertions:

```python
    assert stats["ade_positive_rate"].item() == 0.25
    assert stats["miss_positive_rate"].item() == 0.5
```

Implementation is already mostly present in `summarize_reliability_targets`; verify the tuple includes these pairs:

```python
        ("ade_targets", "ade_positive_rate"),
        ("miss_targets", "miss_positive_rate"),
```

- [ ] **Step 9: Run tests for Task 1**

Run:

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability.py -q
```

Expected: PASS. If an old test expecting union defaults fails, update it to assert `mode_target_policy="all_union"` for legacy behavior or rename it to reflect opt-in legacy semantics.

---

### Task 2: HiVT Training Integration And Presets

**Files:**
- Modify: `models/hivt.py:55-455`
- Modify: `training_presets.py:4-31`
- Modify: `tests/test_reliability_presets.py:12-39`

**Interfaces:**
- Consumes from Task 1: `build_reliability_targets(..., mode_target_policy, scene_target_policy, miss_threshold, conflict_min_frames, conflict_scope, scene_rate_threshold, agent_index)`.
- Produces model args: `--mode_target_policy`, `--scene_target_policy`, `--risk_miss_threshold`, `--risk_conflict_min_frames`, `--risk_conflict_scope`, `--risk_scene_rate_threshold`.

- [ ] **Step 1: Add failing preset tests**

Update `tests/test_reliability_presets.py` defaults test with:

```python
    assert args["mode_target_policy"] == "fde_only"
    assert args["scene_target_policy"] == "target_actor"
    assert args["risk_miss_threshold"] == 2.0
    assert args["risk_conflict_min_frames"] == 2
    assert args["risk_conflict_scope"] == "target_to_neighbors"
    assert args["risk_scene_rate_threshold"] == 0.5
```

Update the override test call with:

```python
        mode_target_policy="fde_or_miss",
        scene_target_policy="target_mode_rate",
        risk_miss_threshold=3.0,
        risk_conflict_min_frames=3,
        risk_conflict_scope="all_valid_pairs",
        risk_scene_rate_threshold=0.25,
```

Add these assertions:

```python
    assert args["mode_target_policy"] == "fde_or_miss"
    assert args["scene_target_policy"] == "target_mode_rate"
    assert args["risk_miss_threshold"] == 3.0
    assert args["risk_conflict_min_frames"] == 3
    assert args["risk_conflict_scope"] == "all_valid_pairs"
    assert args["risk_scene_rate_threshold"] == 0.25
```

- [ ] **Step 2: Run preset tests and verify they fail**

Run:

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_presets.py -q
```

Expected: FAIL because new preset keys are missing.

- [ ] **Step 3: Update `training_presets.py` defaults**

Change `build_reliability_train_args` signature to:

```python
def build_reliability_train_args(
    embed_dim: int,
    reliability_hidden_dim: int = 128,
    reliability_rerank_alpha: float = 0.0,
    reliability_loss_weight: float = 1.0,
    scene_loss_weight: float = 0.2,
    rank_loss_weight: float = 0.0,
    calib_loss_weight: float = 0.0,
    risk_fde_threshold: float = 2.0,
    risk_miss_threshold: float = 2.0,
    risk_conflict_threshold: float = 1.0,
    risk_conflict_min_frames: int = 2,
    risk_offroad_threshold: float = 2.0,
    risk_scene_rate_threshold: float = 0.5,
    mode_target_policy: str = "fde_only",
    scene_target_policy: str = "target_actor",
    risk_conflict_scope: str = "target_to_neighbors",
    **overrides: Any,
) -> Dict[str, Any]:
```

Add these keys to `args`:

```python
        "risk_miss_threshold": risk_miss_threshold,
        "risk_conflict_min_frames": risk_conflict_min_frames,
        "risk_scene_rate_threshold": risk_scene_rate_threshold,
        "mode_target_policy": mode_target_policy,
        "scene_target_policy": scene_target_policy,
        "risk_conflict_scope": risk_conflict_scope,
```

Update old test expectations for `reliability_rerank_alpha` and `scene_loss_weight` to conservative defaults:

```python
    assert args["reliability_rerank_alpha"] == 0.0
    assert args["scene_loss_weight"] == 0.2
```

- [ ] **Step 4: Update `HiVT.__init__` arguments and attributes**

In `models/hivt.py`, add these parameters after `risk_fde_threshold`:

```python
                 risk_miss_threshold: float,
                 risk_conflict_min_frames: int,
                 risk_scene_rate_threshold: float,
                 mode_target_policy: str,
                 scene_target_policy: str,
                 risk_conflict_scope: str,
```

Add these assignments after existing risk threshold assignments:

```python
        self.risk_miss_threshold = risk_miss_threshold
        self.risk_conflict_min_frames = risk_conflict_min_frames
        self.risk_scene_rate_threshold = risk_scene_rate_threshold
        self.mode_target_policy = mode_target_policy
        self.scene_target_policy = scene_target_policy
        self.risk_conflict_scope = risk_conflict_scope
```

- [ ] **Step 5: Pass new target args in training and validation**

In both `training_step` and `validation_step`, extend the `build_reliability_targets` call with:

```python
                miss_threshold=self.risk_miss_threshold,
                mode_target_policy=self.mode_target_policy,
                scene_target_policy=self.scene_target_policy,
                conflict_min_frames=self.risk_conflict_min_frames,
                conflict_scope=self.risk_conflict_scope,
                scene_rate_threshold=self.risk_scene_rate_threshold,
                agent_index=data['agent_index'],
```

- [ ] **Step 6: Log ADE and miss rates in train/val**

In `training_step`, after `train_fde_risk_target_rate`, add:

```python
            self.log('train_ade_risk_target_rate', risk_stats['ade_positive_rate'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_miss_risk_target_rate', risk_stats['miss_positive_rate'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
```

In `validation_step`, after `val_fde_risk_target_rate`, add:

```python
            self.log('val_ade_risk_target_rate', risk_stats['ade_positive_rate'], prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
            self.log('val_miss_risk_target_rate', risk_stats['miss_positive_rate'], prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
```

- [ ] **Step 7: Add model-specific CLI args**

In `HiVT.add_model_specific_args`, replace the reliability defaults with conservative values and add the new args:

```python
        parser.add_argument('--reliability_rerank_alpha', type=float, default=0.0)
        parser.add_argument('--reliability_loss_weight', type=float, default=1.0)
        parser.add_argument('--scene_loss_weight', type=float, default=0.2)
        parser.add_argument('--rank_loss_weight', type=float, default=0.0)
        parser.add_argument('--calib_loss_weight', type=float, default=0.0)
        parser.add_argument('--risk_fde_threshold', type=float, default=2.0)
        parser.add_argument('--risk_miss_threshold', type=float, default=2.0)
        parser.add_argument('--risk_conflict_threshold', type=float, default=1.0)
        parser.add_argument('--risk_conflict_min_frames', type=int, default=2)
        parser.add_argument('--risk_offroad_threshold', type=float, default=2.0)
        parser.add_argument('--risk_scene_rate_threshold', type=float, default=0.5)
        parser.add_argument('--mode_target_policy', type=str, default='fde_only', choices=['fde_only', 'miss_only', 'fde_or_miss', 'all_union'])
        parser.add_argument('--scene_target_policy', type=str, default='target_actor', choices=['target_actor', 'target_mode_rate', 'scene_rate', 'scene_max'])
        parser.add_argument('--risk_conflict_scope', type=str, default='target_to_neighbors', choices=['target_to_neighbors', 'all_valid_pairs'])
```

- [ ] **Step 8: Run tests for Task 2**

Run:

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability.py tests/test_reliability_presets.py -q
```

Expected: PASS.

---

### Task 3: Target Audit Script

**Files:**
- Create: `scripts/audit_reliability_targets.py`

**Interfaces:**
- Consumes: `ArgoverseV1DataModule`, `HiVT`, `build_reliability_targets`, `reconstruct_lane_positions`, `summarize_reliability_targets`.
- Produces CLI: `python scripts/audit_reliability_targets.py --root DATA --embed_dim 64 --split val --max_batches 4`.

- [ ] **Step 1: Create audit script**

Create `scripts/audit_reliability_targets.py` with this complete content:

```python
#!/usr/bin/env python
"""Audit InterAct-Risk target rates without backpropagation."""
from argparse import ArgumentParser
from typing import Dict

import pytorch_lightning as pl
import torch

from datamodules import ArgoverseV1DataModule
from models.hivt import HiVT
from models import build_reliability_targets
from models import reconstruct_lane_positions
from models import summarize_reliability_targets


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _accumulate(totals: Dict[str, float], stats: Dict[str, torch.Tensor]) -> None:
    for key, value in stats.items():
        totals[key] = totals.get(key, 0.0) + _scalar(value)


def _print_table(totals: Dict[str, float], count: int) -> None:
    if count == 0:
        print("No batches were audited.")
        return
    print("metric,value")
    for key in sorted(totals):
        print(f"{key},{totals[key] / count:.6f}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--max_batches', type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--persistent_workers', type=bool, default=False)
    parser.add_argument('--gpus', type=int, default=0)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    pl.seed_everything(2022)
    device = torch.device('cuda:0' if args.gpus > 0 and torch.cuda.is_available() else 'cpu')
    datamodule = ArgoverseV1DataModule.from_argparse_args(args)
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.train_dataloader() if args.split == 'train' else datamodule.val_dataloader()

    model = HiVT(**vars(args)).to(device)
    model.eval()
    totals: Dict[str, float] = {}
    pred_totals: Dict[str, float] = {}
    audited = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            data = data.to(device)
            y_hat, pi, reliability_outputs = model(data)
            reg_mask = ~data['padding_mask'][:, model.historical_steps:]
            graph_batch = getattr(data, 'batch', None)
            if graph_batch is None:
                graph_batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            current_positions = data['positions'][:, model.historical_steps - 1]
            lane_positions = reconstruct_lane_positions(
                lane_actor_index=data['lane_actor_index'],
                lane_actor_vectors=data['lane_actor_vectors'],
                current_positions=current_positions,
                num_lanes=data['lane_vectors'].size(0),
            )
            targets = build_reliability_targets(
                y_hat=y_hat.detach(),
                y=data.y,
                reg_mask=reg_mask,
                batch=graph_batch,
                lane_positions=lane_positions,
                lane_actor_index=data['lane_actor_index'],
                lane_actor_vectors=data['lane_actor_vectors'],
                fde_threshold=model.risk_fde_threshold,
                ade_threshold=1.0,
                miss_threshold=model.risk_miss_threshold,
                conflict_threshold=model.risk_conflict_threshold,
                conflict_min_frames=model.risk_conflict_min_frames,
                conflict_scope=model.risk_conflict_scope,
                offroad_threshold=model.risk_offroad_threshold,
                mode_target_policy=model.mode_target_policy,
                scene_target_policy=model.scene_target_policy,
                scene_rate_threshold=model.risk_scene_rate_threshold,
                agent_index=data['agent_index'],
            )
            _accumulate(totals, summarize_reliability_targets(targets))
            if reliability_outputs is not None:
                pred_totals['mode_risk_pred_mean'] = pred_totals.get('mode_risk_pred_mean', 0.0) + _scalar(reliability_outputs['mode_risk'].mean())
                if reliability_outputs['scene_risk'].numel() > 0:
                    pred_totals['scene_risk_pred_mean'] = pred_totals.get('scene_risk_pred_mean', 0.0) + _scalar(reliability_outputs['scene_risk'].mean())
            audited += 1

    totals.update(pred_totals)
    _print_table(totals, audited)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run audit script help**

Run:

```bash
cd /home/lbh/HiVT
python scripts/audit_reliability_targets.py --help >/tmp/audit_help.txt
```

Expected: command exits 0 and `/tmp/audit_help.txt` contains `--mode_target_policy`, `--scene_target_policy`, and `--risk_conflict_min_frames`.

- [ ] **Step 3: Run import/compile check**

Run:

```bash
cd /home/lbh/HiVT
python -m py_compile scripts/audit_reliability_targets.py
```

Expected: PASS with no output.

---

### Task 4: Shell Entrypoint Overrides

**Files:**
- Modify: `run_single_gpu.sh:28-42`
- Modify: `run_single_gpu.sh:197-207`
- Modify: `run_single_gpu.sh:237-253`

**Interfaces:**
- Consumes: CLI args from Task 2.
- Produces environment overrides: `MODE_TARGET_POLICY`, `SCENE_TARGET_POLICY`, `RISK_FDE_THRESHOLD`, `RISK_MISS_THRESHOLD`, `RISK_CONFLICT_THRESHOLD`, `RISK_CONFLICT_MIN_FRAMES`, `RISK_CONFLICT_SCOPE`, `RISK_SCENE_RATE_THRESHOLD`, `RELIABILITY_RERANK_ALPHA`, `SCENE_LOSS_WEIGHT`.

- [ ] **Step 1: Update usage text**

In `run_single_gpu.sh`, add these lines under `Environment overrides:`:

```bash
  MODE_TARGET_POLICY=fde_only
  SCENE_TARGET_POLICY=target_actor
  RISK_FDE_THRESHOLD=2.0
  RISK_MISS_THRESHOLD=2.0
  RISK_CONFLICT_THRESHOLD=1.0
  RISK_CONFLICT_MIN_FRAMES=2
  RISK_CONFLICT_SCOPE=target_to_neighbors
  RISK_SCENE_RATE_THRESHOLD=0.5
  RELIABILITY_RERANK_ALPHA=0.0
  SCENE_LOSS_WEIGHT=0.2
```

- [ ] **Step 2: Add reliability arg block to `run_train_reliability`**

Before the `exec env` line in `run_train_reliability`, add:

```bash
  local mode_target_policy="${MODE_TARGET_POLICY:-fde_only}"
  local scene_target_policy="${SCENE_TARGET_POLICY:-target_actor}"
  local risk_fde_threshold="${RISK_FDE_THRESHOLD:-2.0}"
  local risk_miss_threshold="${RISK_MISS_THRESHOLD:-2.0}"
  local risk_conflict_threshold="${RISK_CONFLICT_THRESHOLD:-1.0}"
  local risk_conflict_min_frames="${RISK_CONFLICT_MIN_FRAMES:-2}"
  local risk_conflict_scope="${RISK_CONFLICT_SCOPE:-target_to_neighbors}"
  local risk_scene_rate_threshold="${RISK_SCENE_RATE_THRESHOLD:-0.5}"
  local reliability_rerank_alpha="${RELIABILITY_RERANK_ALPHA:-0.0}"
  local scene_loss_weight="${SCENE_LOSS_WEIGHT:-0.2}"
  echo "mode_target_policy=$mode_target_policy scene_target_policy=$scene_target_policy conflict_scope=$risk_conflict_scope"
```

Extend the `python train.py` args after `--use_reliability true`:

```bash
    --reliability_rerank_alpha "$reliability_rerank_alpha" \
    --scene_loss_weight "$scene_loss_weight" \
    --risk_fde_threshold "$risk_fde_threshold" \
    --risk_miss_threshold "$risk_miss_threshold" \
    --risk_conflict_threshold "$risk_conflict_threshold" \
    --risk_conflict_min_frames "$risk_conflict_min_frames" \
    --risk_conflict_scope "$risk_conflict_scope" \
    --risk_scene_rate_threshold "$risk_scene_rate_threshold" \
    --mode_target_policy "$mode_target_policy" \
    --scene_target_policy "$scene_target_policy"
```

- [ ] **Step 3: Add the same reliability args to `run_train_reliability_shift`**

Repeat the local variable block from Step 2 before its `exec env` line. Extend the shift training `python train.py` args by inserting the same reliability args immediately after `--use_reliability true` and before shift augmentation args.

- [ ] **Step 4: Shell syntax check**

Run:

```bash
cd /home/lbh/HiVT
bash -n run_single_gpu.sh
```

Expected: PASS with no output.

---

### Task 5: Final Verification

**Files:**
- Read/verify only unless a previous task revealed a defect.

**Interfaces:**
- Consumes all previous tasks.
- Produces a verified working tree ready for user-run data audit/training.

- [ ] **Step 1: Run focused tests**

Run:

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability.py tests/test_reliability_presets.py -q
```

Expected: PASS.

- [ ] **Step 2: Run Python compile checks**

Run:

```bash
cd /home/lbh/HiVT
python -m py_compile models/reliability.py models/hivt.py training_presets.py scripts/audit_reliability_targets.py
```

Expected: PASS with no output.

- [ ] **Step 3: Run CLI help checks**

Run:

```bash
cd /home/lbh/HiVT
python train.py --help >/tmp/hivt_train_help.txt
python scripts/audit_reliability_targets.py --help >/tmp/hivt_audit_help.txt
rg -- '--mode_target_policy|--scene_target_policy|--risk_conflict_min_frames' /tmp/hivt_train_help.txt /tmp/hivt_audit_help.txt
```

Expected: output contains all three flags in both help files.

- [ ] **Step 4: Run shell syntax check**

Run:

```bash
cd /home/lbh/HiVT
bash -n run_single_gpu.sh
```

Expected: PASS with no output.

- [ ] **Step 5: Optional dataset audit command for the user**

If the Conda env and Argoverse data are available, run:

```bash
cd /home/lbh/HiVT
python scripts/audit_reliability_targets.py \
  --root /home/lbh/HiVT/datasets/argoverse \
  --embed_dim 64 \
  --use_reliability true \
  --split val \
  --max_batches 4 \
  --val_batch_size 8 \
  --num_workers 0
```

Expected: CSV-like output with rates such as `mode_positive_rate`, `conflict_positive_rate`, and `scene_positive_rate`; none should be forced to `1.000000` by union policy alone.

---

## Self-Review

- Spec coverage: Task 1 implements mode policies, conflict same-time/min-frames/scope, scene policies, and component summaries. Task 2 passes parameters through HiVT and CLI. Task 3 adds the audit script. Task 4 exposes shell overrides. Task 5 verifies tests, compile, help, and optional Exp A audit.
- Placeholder scan: No TBD/TODO/fill-in instructions remain; code snippets include exact function signatures, tests, commands, and expected results.
- Type consistency: New names match across tasks: `mode_target_policy`, `scene_target_policy`, `risk_miss_threshold`, `risk_conflict_min_frames`, `risk_conflict_scope`, `risk_scene_rate_threshold`, `agent_index`.
