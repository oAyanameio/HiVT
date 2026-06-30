# Mode Risk Threshold Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the freeze-backbone `mode_risk` objective so the head separates samples near the `FDE=2.0m` decision boundary better, then verify whether reranking `MR` improves or at least stops regressing.

**Architecture:** Keep the existing HiVT backbone, reliability heads, and probability-space reranking unchanged. Add two objective-side controls only: a threshold-aware weighted BCE around `risk_fde_threshold`, and an optional near-threshold / top-k filtered pairwise ranking loss built on the existing `RiskRankLoss`. Wire both through `HiVT`, `training_presets.py`, and `run_single_gpu.sh`, then validate against the existing freeze-backbone checkpoint workflow.

**Tech Stack:** Python 3, PyTorch, PyTorch Lightning, pytest, shell runner `run_single_gpu.sh`, existing reranking evaluator `scripts/eval_reranking.py`.

---

### Task 1: Add failing tests for threshold-aware loss behavior

**Files:**
- Modify: `/home/lbh/HiVT/tests/test_reliability.py`
- Create or Modify: `/home/lbh/HiVT/tests/test_reliability_losses.py`
- Test: `/home/lbh/HiVT/tests/test_reliability.py`
- Test: `/home/lbh/HiVT/tests/test_reliability_losses.py`

- [ ] **Step 1: Write the failing tests for near-threshold sample weighting**

```python
def test_compute_threshold_weights_emphasizes_boundary_samples():
    fde = torch.tensor([[1.0, 1.9, 2.0, 2.1, 3.5]])

    weights = compute_threshold_weights(
        fde=fde,
        threshold=2.0,
        radius=0.25,
        base_weight=1.0,
        peak_weight=3.0,
    )

    assert torch.isclose(weights[0, 0], torch.tensor(1.0))
    assert weights[0, 2] == weights.max()
    assert weights[0, 1] > weights[0, 0]
    assert weights[0, 3] > weights[0, 4]
```

```python
def test_compute_threshold_weights_keeps_invalid_modes_at_zero():
    fde = torch.tensor([[1.9, 2.0], [2.1, 2.2]])
    valid_mask = torch.tensor([True, False])

    weights = compute_threshold_weights(
        fde=fde,
        threshold=2.0,
        radius=0.2,
        base_weight=1.0,
        peak_weight=2.0,
        valid_mask=valid_mask,
    )

    assert torch.all(weights[1] == 0.0)
    assert torch.all(weights[0] >= 1.0)
```

- [ ] **Step 2: Write the failing tests for filtered pairwise ranking**

```python
def test_risk_rank_loss_filters_to_top_k_and_near_threshold_pairs():
    loss_fn = RiskRankLoss(margin=0.1, error_margin=0.05)
    mode_risk = torch.tensor([[0.2, 0.6, 0.8, 0.1]])
    mode_error = torch.tensor([[1.95, 2.05, 3.0, 1.0]])
    valid_mask = torch.tensor([True])
    mode_logits = torch.tensor([[5.0, 4.0, 1.0, 0.5]])

    filtered = loss_fn(
        mode_risk=mode_risk,
        mode_error=mode_error,
        valid_mask=valid_mask,
        mode_logits=mode_logits,
        top_k=2,
        focus_threshold=2.0,
        focus_radius=0.15,
    )
    unfiltered = loss_fn(
        mode_risk=mode_risk,
        mode_error=mode_error,
        valid_mask=valid_mask,
    )

    assert filtered > 0
    assert filtered != unfiltered
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run:

```bash
pytest /home/lbh/HiVT/tests/test_reliability.py /home/lbh/HiVT/tests/test_reliability_losses.py -q
```

Expected: FAIL because `compute_threshold_weights`, `top_k`, `focus_threshold`, and `focus_radius` are not implemented yet.


### Task 2: Implement threshold-aware weighting and filtered ranking controls

**Files:**
- Modify: `/home/lbh/HiVT/models/reliability.py`
- Modify: `/home/lbh/HiVT/losses/reliability_losses.py`
- Modify: `/home/lbh/HiVT/losses/__init__.py`
- Modify: `/home/lbh/HiVT/models/hivt.py`
- Modify: `/home/lbh/HiVT/training_presets.py`
- Modify: `/home/lbh/HiVT/run_single_gpu.sh`

- [ ] **Step 1: Add a reusable threshold-weight helper in `models/reliability.py`**

```python
def compute_threshold_weights(
    fde: torch.Tensor,
    threshold: float,
    radius: float,
    base_weight: float = 1.0,
    peak_weight: float = 2.0,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if radius <= 0:
        weights = torch.full_like(fde, float(base_weight))
    else:
        distance = (fde - threshold).abs()
        proximity = (1.0 - distance / max(radius, eps)).clamp(min=0.0, max=1.0)
        weights = base_weight + (peak_weight - base_weight) * proximity
    if valid_mask is not None:
        weights = weights * valid_mask.unsqueeze(-1).float()
    return weights
```

- [ ] **Step 2: Extend `RiskRankLoss` to support optional top-k and threshold-focused filtering**

```python
def forward(
    self,
    mode_risk: torch.Tensor,
    mode_error: torch.Tensor,
    valid_mask: torch.Tensor,
    mode_logits: Optional[torch.Tensor] = None,
    top_k: Optional[int] = None,
    focus_threshold: Optional[float] = None,
    focus_radius: Optional[float] = None,
) -> torch.Tensor:
    ...
```

Implementation requirements:

```python
pair_mask = (err_diff > self.error_margin) & valid_mask.view(-1, 1, 1)

if top_k is not None and mode_logits is not None:
    topk_idx = torch.topk(mode_logits, k=min(top_k, mode_logits.size(-1)), dim=-1).indices
    topk_mask = torch.zeros_like(mode_risk, dtype=torch.bool)
    topk_mask.scatter_(dim=-1, index=topk_idx, value=True)
    pair_mask = pair_mask & topk_mask.unsqueeze(2) & topk_mask.unsqueeze(1)

if focus_threshold is not None and focus_radius is not None and focus_radius > 0:
    near = (mode_error - focus_threshold).abs() <= focus_radius
    pair_mask = pair_mask & (near.unsqueeze(2) | near.unsqueeze(1))
```

- [ ] **Step 3: Wire objective controls into `HiVT`**

Add constructor arguments and saved attributes:

```python
mode_risk_threshold_weight_enabled: bool,
mode_risk_threshold_weight_radius: float,
mode_risk_threshold_weight_peak: float,
mode_risk_threshold_weight_base: float,
mode_risk_rank_top_k: int,
mode_risk_rank_near_threshold_only: bool,
mode_risk_rank_threshold_radius: float,
```

Then replace the plain BCE block with weighted unreduced BCE:

```python
bce = F.binary_cross_entropy_with_logits(
    reliability_outputs['mode_risk_logits'][valid_mask],
    mode_targets[valid_mask],
    reduction='none',
)
if self.mode_risk_threshold_weight_enabled:
    weights = compute_threshold_weights(
        fde=reliability_targets['fde'],
        threshold=self.risk_fde_threshold,
        radius=self.mode_risk_threshold_weight_radius,
        base_weight=self.mode_risk_threshold_weight_base,
        peak_weight=self.mode_risk_threshold_weight_peak,
        valid_mask=valid_mask,
    )[valid_mask]
    risk_loss = (bce * weights).sum() / weights.sum().clamp(min=1e-6)
else:
    risk_loss = bce.mean()
```

Update rank loss call:

```python
rank_loss = self.rank_loss_fn(
    mode_risk=reliability_outputs['mode_risk'],
    mode_error=reliability_targets['fde'],
    valid_mask=valid_mask,
    mode_logits=pi.detach(),
    top_k=self.mode_risk_rank_top_k if self.mode_risk_rank_top_k > 0 else None,
    focus_threshold=self.risk_fde_threshold if self.mode_risk_rank_near_threshold_only else None,
    focus_radius=self.mode_risk_rank_threshold_radius if self.mode_risk_rank_near_threshold_only else None,
)
```

- [ ] **Step 4: Expose the new controls through presets and shell entrypoints**

Add defaults in `/home/lbh/HiVT/training_presets.py`:

```python
mode_risk_threshold_weight_enabled: bool = False,
mode_risk_threshold_weight_radius: float = 0.25,
mode_risk_threshold_weight_peak: float = 2.0,
mode_risk_threshold_weight_base: float = 1.0,
mode_risk_rank_top_k: int = 0,
mode_risk_rank_near_threshold_only: bool = False,
mode_risk_rank_threshold_radius: float = 0.25,
```

Add parser args in `/home/lbh/HiVT/models/hivt.py` and pass them through `/home/lbh/HiVT/run_single_gpu.sh` with env overrides:

```bash
local mode_risk_threshold_weight_enabled="${MODE_RISK_THRESHOLD_WEIGHT_ENABLED:-false}"
local mode_risk_threshold_weight_radius="${MODE_RISK_THRESHOLD_WEIGHT_RADIUS:-0.25}"
local mode_risk_threshold_weight_peak="${MODE_RISK_THRESHOLD_WEIGHT_PEAK:-2.0}"
local mode_risk_threshold_weight_base="${MODE_RISK_THRESHOLD_WEIGHT_BASE:-1.0}"
local mode_risk_rank_top_k="${MODE_RISK_RANK_TOP_K:-0}"
local mode_risk_rank_near_threshold_only="${MODE_RISK_RANK_NEAR_THRESHOLD_ONLY:-false}"
local mode_risk_rank_threshold_radius="${MODE_RISK_RANK_THRESHOLD_RADIUS:-0.25}"
```


### Task 3: Run focused verification and freeze-backbone validation

**Files:**
- Modify: `/home/lbh/HiVT/tests/test_reliability_presets.py`
- Modify: `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md`
- Modify: `/home/lbh/HiVT/docs/可靠性实验结果汇总.md`

- [ ] **Step 1: Update preset coverage for the new controls**

```python
def test_build_reliability_train_args_accepts_threshold_objective_overrides():
    args = training_presets.build_reliability_train_args(
        embed_dim=64,
        mode_risk_threshold_weight_enabled=True,
        mode_risk_threshold_weight_radius=0.2,
        mode_risk_threshold_weight_peak=3.0,
        mode_risk_rank_top_k=3,
        mode_risk_rank_near_threshold_only=True,
        mode_risk_rank_threshold_radius=0.15,
    )

    assert args["mode_risk_threshold_weight_enabled"] is True
    assert args["mode_risk_threshold_weight_radius"] == 0.2
    assert args["mode_risk_threshold_weight_peak"] == 3.0
    assert args["mode_risk_rank_top_k"] == 3
    assert args["mode_risk_rank_near_threshold_only"] is True
    assert args["mode_risk_rank_threshold_radius"] == 0.15
```

- [ ] **Step 2: Run unit tests to verify all targeted behavior passes**

Run:

```bash
pytest /home/lbh/HiVT/tests/test_reliability.py /home/lbh/HiVT/tests/test_reliability_losses.py /home/lbh/HiVT/tests/test_reliability_presets.py /home/lbh/HiVT/tests/test_reliability_analysis.py -q
```

Expected: PASS.

- [ ] **Step 3: Run a short freeze-backbone training validation from the known good checkpoint**

Run:

```bash
RUN_VERSION=freeze_backbone_threshold_objective_smoke \
INIT_CKPT_PATH=/home/lbh/HiVT/runs/hivt_reliability/freeze_backbone_budget128_targetmode067/checkpoints/epoch=00-val_minFDE=0.8031.ckpt \
FREEZE_BACKBONE=true \
LIMIT_TRAIN_BATCHES=0.1 \
LIMIT_VAL_BATCHES=0.1 \
MAX_EPOCHS=1 \
MODE_RISK_THRESHOLD_WEIGHT_ENABLED=true \
MODE_RISK_THRESHOLD_WEIGHT_RADIUS=0.25 \
MODE_RISK_THRESHOLD_WEIGHT_PEAK=3.0 \
MODE_RISK_RANK_TOP_K=3 \
MODE_RISK_RANK_NEAR_THRESHOLD_ONLY=true \
MODE_RISK_RANK_THRESHOLD_RADIUS=0.2 \
RANK_LOSS_WEIGHT=0.05 \
./run_single_gpu.sh train_reliability 64 0
```

Expected: training exits 0 and writes a new run under `/home/lbh/HiVT/runs/hivt_reliability/freeze_backbone_threshold_objective_smoke/`.

- [ ] **Step 4: Re-evaluate reranking MR on the original and new freeze-backbone checkpoints**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python /home/lbh/HiVT/scripts/eval_reranking.py \
  --root /datasets/argoverse \
  --ckpt_path /home/lbh/HiVT/runs/hivt_reliability/freeze_backbone_budget128_targetmode067/checkpoints/epoch=00-val_minFDE=0.8031.ckpt \
  --rerank_method prob_product \
  --rerank_alpha 1.0 \
  --rerank_top_k 3 \
  --max_batches 32 \
  --val_batch_size 8 \
  --num_workers 0
```

and

```bash
CUDA_VISIBLE_DEVICES=0 python /home/lbh/HiVT/scripts/eval_reranking.py \
  --root /datasets/argoverse \
  --ckpt_path /home/lbh/HiVT/runs/hivt_reliability/freeze_backbone_threshold_objective_smoke/checkpoints/last.ckpt \
  --rerank_method prob_product \
  --rerank_alpha 1.0 \
  --rerank_top_k 3 \
  --max_batches 32 \
  --val_batch_size 8 \
  --num_workers 0
```

Expected: both commands exit 0 and produce comparable `original_minMR`, `reranked_minMR`, and case breakdown rows.

- [ ] **Step 5: Record the result in the docs**

Update `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md` and `/home/lbh/HiVT/docs/可靠性实验结果汇总.md` with:

```text
- objective variant name
- threshold weighting hyperparameters
- rank filtering hyperparameters
- whether reranked MR improved / matched / regressed
- recommendation: keep / tune / drop this objective branch
```
