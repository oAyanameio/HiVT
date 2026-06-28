# Reliability Label Collapse Conservative Fix Design

## Context

`docs/实验问题阶段性解决方案文档.md` diagnoses the current InterAct-Risk supervision collapse: mode, conflict, and scene targets are effectively all positive, so the risk heads learn to predict values near 1.0. This design implements the first-stage conservative fix only. The goal is to restore meaningful positive/negative risk labels before expanding reranking, shift evaluation, or multi-backbone experiments.

## Scope

Implement the document's first round: fix supervision signals, expose the needed policy/threshold controls, and add a target-audit script. Do not add new model heads or change the HiVT backbone.

## Target Policy

`build_reliability_targets` will support explicit mode target policies:

- `fde_only` (default): `mode_targets = fde_targets`.
- `miss_only`: `mode_targets = miss_targets`.
- `fde_or_miss`: union of FDE and miss labels.
- `all_union`: legacy union of FDE, ADE, miss, conflict, and off-road.

ADE, miss, conflict, and off-road component labels stay available for logging. Conflict and off-road are not part of the default main supervision so one dense proxy cannot collapse `mode_targets` again.

## Conflict Targets

`compute_conflict_risk_targets` will move from trajectory-level minimum distance to same-time close-frame counting:

- A predicted target mode is in conflict when same-time distance to another valid actor is below `risk_conflict_threshold` for at least `risk_conflict_min_frames` future steps.
- Default scope is `target_to_neighbors`: compare each graph's target actor against same-scene valid neighbors.
- `all_valid_pairs` remains available for broader diagnostics.
- Valid actor and future masks gate comparisons.

## Scene Targets

`compute_scene_risk_targets` will support explicit scene policies:

- `target_actor` (default): each scene target is positive if the graph's target actor has any positive mode target.
- `target_mode_rate`: positive if the target actor's positive mode rate exceeds `risk_scene_rate_threshold`.
- `scene_rate`: positive if all valid actors' positive mode rate exceeds the threshold.
- `scene_max`: legacy hard max over all valid actors and modes.

The default aligns scene supervision with HiVT's target-agent metrics and avoids unrelated neighbors turning the whole scene positive.

## Training Integration

`HiVT` will accept and store these new arguments:

- `mode_target_policy`
- `scene_target_policy`
- `risk_miss_threshold`
- `risk_conflict_min_frames`
- `risk_conflict_scope`
- `risk_scene_rate_threshold`

Training and validation will pass them into `build_reliability_targets`. Logs will continue reporting mode, FDE, conflict, off-road, and scene target rates, and will add miss/ADE target rates where missing.

## Command-Line And Script Support

`HiVT.add_model_specific_args` exposes all new controls through `train.py`. `run_single_gpu.sh` will pass environment overrides for label-policy and threshold tuning when reliability training is enabled.

Add `scripts/audit_reliability_targets.py` to run a no-backprop target audit on train or val batches. It will load the model and dataloader, build reliability targets, and print aggregate target rates for FDE, miss, conflict, off-road, mode, and scene labels. This supports Exp A before any longer training run.

## Validation

Implementation validation should include:

1. Static import/compile checks for changed Python files.
2. A small synthetic unit-style check for target construction policies if practical without full dataset access.
3. A dry CLI help check for the audit script or training arguments when dependencies are available.
4. If the Argoverse data and environment are ready, run the audit script for a few batches before training.

## Non-Goals

This stage does not implement multi-task paper heads, shift evaluation expansion, or full reranking experiments. Those should wait until target rates and risk predictions are no longer collapsed.
