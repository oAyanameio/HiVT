# Reliability Review Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 吸收最新评审意见，优先修复 reranking 概率语义、重审 conflict 标签可训练性、定位 scene 校准失配，并补联合训练 MR 退化来源分析，为后续中等预算实验建立更稳的论据基础。

**Architecture:** 保持当前 HiVT + ReliabilityModule 结构不变，不新增复杂 head。先围绕现有 freeze-backbone checkpoint 做低成本高信息密度分析：把 reranking 切回概率空间；把 conflict 标签重新 audit；把 scene head 的校准问题单独剥离；最后用训练日志和梯度诊断解释联合训练为何伤害宿主指标。只有这些机制问题明确后，再扩大预算。

**Tech Stack:** Python 3, PyTorch, PyTorch Lightning, torchmetrics-style custom metrics, pytest, shell scripts, TensorBoard event logs.

---

## Scope Check

本计划只覆盖以下五项：

- 概率空间 reranking 对照
- conflict 标签阈值 audit
- scene head 校准分析
- 联合训练 MR 退化来源分析
- 中等预算实验的前置口径准备

不包含：

- 新 backbone 泛化
- 新多头结构设计
- 论文正文全面重写
- 全量长训练

---

## File Structure

- Modify: `/home/lbh/HiVT/scripts/eval_reranking.py`
  - 增加概率空间 reranking 公式、top-k 限制和公式选择参数。
- Modify: `/home/lbh/HiVT/tests/test_reliability_analysis.py`
  - 增加概率空间 reranking 与 conflict audit 聚合逻辑测试。
- Create: `/home/lbh/HiVT/scripts/audit_conflict_thresholds.py`
  - 对不同 `risk_conflict_threshold` 输出 conflict 正样本率。
- Create: `/home/lbh/HiVT/scripts/analyze_scene_calibration.py`
  - 输出 scene 分桶统计、ECE/Brier 和 reliability curve 基础数据。
- Create: `/home/lbh/HiVT/scripts/analyze_joint_training_regression.py`
  - 汇总联合训练与 freeze-backbone run 的关键日志，并对比 `L_pred`、`MR`、相关 loss 曲线。
- Modify: `/home/lbh/HiVT/run_single_gpu.sh`
  - 如需参数透传，补充 reranking 公式选择或 audit 用环境变量。
- Modify: `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md`
  - 回填 follow-up 结果与口径。
- Modify: `/home/lbh/HiVT/docs/可靠性实验结果汇总.md`
  - 只保留一处 reranking 主表，并增加“概率空间 reranking”与“conflict audit”结果区。

---

### Task 1: 概率空间 reranking 对照

**Files:**
- Modify: `/home/lbh/HiVT/scripts/eval_reranking.py`
- Modify: `/home/lbh/HiVT/tests/test_reliability_analysis.py`

- [ ] **Step 1: 先写失败测试，固定概率空间 reranking 的最小语义**

在 `/home/lbh/HiVT/tests/test_reliability_analysis.py` 增加：

```python
import torch

from scripts.eval_reranking import rerank_scores


def test_probability_space_reranking_penalizes_high_risk_mode():
    pi = torch.tensor([[3.0, 2.0]])
    risk = torch.tensor([[0.9, 0.1]])
    scores = rerank_scores(pi, risk, method="prob_product", alpha=1.0, top_k=None)
    assert scores.shape == pi.shape
    assert scores[0, 1] > scores[0, 0]


def test_top_k_reranking_only_changes_candidates_inside_top_k():
    pi = torch.tensor([[5.0, 4.0, 0.0]])
    risk = torch.tensor([[0.9, 0.1, 0.0]])
    scores = rerank_scores(pi, risk, method="prob_product", alpha=1.0, top_k=2)
    assert scores[0, 2] < scores[0, 1]
```

- [ ] **Step 2: 运行测试并确认当前失败**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_analysis.py::test_probability_space_reranking_penalizes_high_risk_mode -q
```

预期：失败，因为 `rerank_scores` 还不支持新的概率空间公式。

- [ ] **Step 3: 在 `eval_reranking.py` 中实现统一 reranking 接口**

在 `/home/lbh/HiVT/scripts/eval_reranking.py` 中补：

```python
def rerank_scores(
    pi: torch.Tensor,
    risk: torch.Tensor,
    method: str,
    alpha: float,
    top_k: int | None,
    eps: float = 1e-6,
) -> torch.Tensor:
    log_prob = torch.log_softmax(pi, dim=-1)
    if method == "prob_product":
        reranked = log_prob + torch.log1p(-risk.clamp(max=1.0 - eps) + eps)
    elif method == "logprob_minus_norm_risk":
        norm_risk = (risk - risk.mean(dim=-1, keepdim=True)) / (risk.std(dim=-1, keepdim=True) + eps)
        reranked = log_prob - alpha * norm_risk
    else:
        raise ValueError(f"Unknown reranking method: {method}")
    if top_k is None or top_k >= pi.size(-1):
        return reranked
    masked = torch.full_like(reranked, float("-inf"))
    topk_idx = torch.topk(pi, k=top_k, dim=-1).indices
    masked.scatter_(dim=-1, index=topk_idx, src=reranked.gather(dim=-1, index=topk_idx))
    return masked
```

并为脚本增加参数：

```text
--rerank_method prob_product|logprob_minus_norm_risk
--rerank_top_k
```

- [ ] **Step 4: 跑测试确认通过**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_analysis.py -q
```

预期：新增 reranking 测试通过。

- [ ] **Step 5: 跑概率空间 reranking 对照**

运行：

```bash
cd /home/lbh/HiVT
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate HiVT
CUDA_VISIBLE_DEVICES=0 python scripts/eval_reranking.py \
  --root /home/lbh/HiVT/datasets/argoverse \
  --ckpt_path /home/lbh/HiVT/runs/hivt_reliability/freeze_backbone_budget128_targetmode067/checkpoints/epoch=00-val_minFDE=0.8031.ckpt \
  --gpus 1 \
  --embed_dim 64 \
  --use_reliability true \
  --mode_target_policy fde_only \
  --scene_target_policy target_mode_rate \
  --risk_scene_rate_threshold 0.67 \
  --freeze_backbone true \
  --rerank_method prob_product \
  --rerank_alpha 1.0 \
  --rerank_top_k 3 \
  --max_batches 32 \
  --train_batch_size 8 \
  --val_batch_size 8 \
  --num_workers 8
```

预期：输出新的 `original_* / reranked_* / rerank_top1_change_rate`，用于和旧公式对照。

---

### Task 2: Conflict 标签阈值 audit

**Files:**
- Create: `/home/lbh/HiVT/scripts/audit_conflict_thresholds.py`
- Modify: `/home/lbh/HiVT/tests/test_reliability_analysis.py`

- [ ] **Step 1: 写失败测试，固定 conflict rate 聚合行为**

在 `/home/lbh/HiVT/tests/test_reliability_analysis.py` 增加：

```python
import torch

from scripts.audit_conflict_thresholds import summarize_positive_rate


def test_summarize_positive_rate_returns_fraction():
    values = torch.tensor([0.0, 1.0, 1.0, 0.0])
    rate = summarize_positive_rate(values)
    assert float(rate) == 0.5
```

- [ ] **Step 2: 运行测试并确认当前失败**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_analysis.py::test_summarize_positive_rate_returns_fraction -q
```

预期：失败，因为脚本还不存在。

- [ ] **Step 3: 实现 conflict threshold audit 脚本**

在 `/home/lbh/HiVT/scripts/audit_conflict_thresholds.py` 新增：

```python
def summarize_positive_rate(values: torch.Tensor) -> torch.Tensor:
    values = values.float().reshape(-1)
    if values.numel() == 0:
        return values.new_zeros(())
    return values.mean()
```

脚本主流程应：

1. 加载宿主 checkpoint；
2. 对 `risk_conflict_threshold` 依次取 `1.0 / 1.5 / 2.0 / 3.0`；
3. 调 `build_reliability_targets`；
4. 输出每个阈值的：

```text
conflict_threshold
conflict_risk_target_rate
mode_risk_target_rate
```

- [ ] **Step 4: 跑测试确认通过**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_analysis.py -q
```

- [ ] **Step 5: 运行 conflict audit**

运行：

```bash
cd /home/lbh/HiVT
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate HiVT
CUDA_VISIBLE_DEVICES=0 python scripts/audit_conflict_thresholds.py \
  --root /home/lbh/HiVT/datasets/argoverse \
  --ckpt_path /home/lbh/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt \
  --gpus 1 \
  --embed_dim 64 \
  --mode_target_policy fde_only \
  --scene_target_policy target_mode_rate \
  --risk_scene_rate_threshold 0.67 \
  --max_batches 32 \
  --train_batch_size 8 \
  --val_batch_size 8 \
  --num_workers 8
```

预期：得到不同 `tau_col` 下的 conflict 正样本率，为是否保留 conflict 主论据提供依据。

---

### Task 3: Scene head 校准分析

**Files:**
- Create: `/home/lbh/HiVT/scripts/analyze_scene_calibration.py`
- Modify: `/home/lbh/HiVT/tests/test_reliability_analysis.py`

- [ ] **Step 1: 写失败测试，固定分桶统计输出结构**

在 `/home/lbh/HiVT/tests/test_reliability_analysis.py` 增加：

```python
import torch

from scripts.analyze_scene_calibration import bucketize_binary_calibration


def test_bucketize_binary_calibration_returns_counts_and_rates():
    probs = torch.tensor([0.1, 0.2, 0.8, 0.9])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    rows = bucketize_binary_calibration(probs, targets, num_bins=2)
    assert len(rows) == 2
    assert rows[0]["count"] == 2
    assert rows[1]["count"] == 2
```

- [ ] **Step 2: 运行测试并确认当前失败**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_analysis.py::test_bucketize_binary_calibration_returns_counts_and_rates -q
```

- [ ] **Step 3: 实现 scene calibration 分析脚本**

在 `/home/lbh/HiVT/scripts/analyze_scene_calibration.py` 新增：

```python
def bucketize_binary_calibration(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10,
) -> list[dict[str, float]]:
    rows = []
    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1)
    for idx in range(num_bins):
        left = bin_edges[idx]
        right = bin_edges[idx + 1]
        if idx == num_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_left": float(left), "bin_right": float(right), "count": 0, "avg_prob": 0.0, "avg_target": 0.0})
            continue
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "avg_prob": float(probs[mask].mean()),
                "avg_target": float(targets[mask].float().mean()),
            }
        )
    return rows
```

脚本输出至少包括：

```text
scene_BrierScore
scene_ECE
scene_pred_mean
scene_target_rate
bin_left, bin_right, count, avg_prob, avg_target
```

- [ ] **Step 4: 跑测试确认通过**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_analysis.py -q
```

- [ ] **Step 5: 运行 scene calibration 分析**

运行：

```bash
cd /home/lbh/HiVT
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate HiVT
CUDA_VISIBLE_DEVICES=0 python scripts/analyze_scene_calibration.py \
  --root /home/lbh/HiVT/datasets/argoverse \
  --ckpt_path /home/lbh/HiVT/runs/hivt_reliability/freeze_backbone_budget128_targetmode067/checkpoints/epoch=00-val_minFDE=0.8031.ckpt \
  --gpus 1 \
  --embed_dim 64 \
  --use_reliability true \
  --mode_target_policy fde_only \
  --scene_target_policy target_mode_rate \
  --risk_scene_rate_threshold 0.67 \
  --freeze_backbone true \
  --max_batches 32 \
  --train_batch_size 8 \
  --val_batch_size 8 \
  --num_workers 8
```

预期：得到 scene calibration 分桶统计，支撑“当前 scene 概率不宜直接使用”的结论。

---

### Task 4: 联合训练 MR 退化来源分析

**Files:**
- Create: `/home/lbh/HiVT/scripts/analyze_joint_training_regression.py`
- Modify: `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md`

- [ ] **Step 1: 先写分析脚本的最小接口**

在 `/home/lbh/HiVT/scripts/analyze_joint_training_regression.py` 新增参数：

```text
--baseline_run_dir
--joint_run_dir
--freeze_run_dir
```

脚本先读取：

```text
hparams.yaml
events.out.tfevents.*
```

并打印可用 scalar tags。

- [ ] **Step 2: 运行脚本确认能读到标量标签**

运行：

```bash
cd /home/lbh/HiVT
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate HiVT
python scripts/analyze_joint_training_regression.py \
  --joint_run_dir /home/lbh/HiVT/runs/hivt_reliability/stage1_fix_budget128_fdeonly_targetmode067 \
  --freeze_run_dir /home/lbh/HiVT/runs/hivt_reliability/freeze_backbone_budget128_targetmode067
```

预期：输出可用 tags，例如 `val_minMR`、`val_reg_loss`、`val_risk_loss`、`val_scene_loss`。

- [ ] **Step 3: 增加最小对比摘要**

脚本补充输出：

```text
joint.val_minADE / joint.val_minFDE / joint.val_minMR
freeze.val_minADE / freeze.val_minFDE / freeze.val_minMR
joint.val_reg_loss
freeze.val_reg_loss
```

并给出结论模板：

```text
regression_degradation_source = selection_shift | trajectory_regression_shift | unresolved
```

第一轮即使只能先做基于日志的弱结论也可以，后续再加梯度诊断。

- [ ] **Step 4: 将分析结论回填文档**

在 `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md` 的联合训练退化章节补：

- 当前证据能说明什么
- 还不能说明什么
- 是否需要补梯度范数分析

---

### Task 5: 统一文档口径并准备中等预算入口

**Files:**
- Modify: `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md`
- Modify: `/home/lbh/HiVT/docs/可靠性实验结果汇总.md`

- [ ] **Step 1: 统一小预算结论口径**

将相关表述统一成：

```text
小预算下趋势明确，但绝对数值仍待中等预算验证。
```

重点修正：

- 联合训练退化的结论边界
- reranking 局部验证子集与全量验证的口径区分

- [ ] **Step 2: 消除重复表格**

只在 `/home/lbh/HiVT/docs/可靠性实验结果汇总.md` 保留 reranking 主表，  
在 `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md` 中只保留摘要并引用结果汇总文档。

- [ ] **Step 3: 给出中等预算实验入口**

在文档中补一个明确入口，例如：

```text
在完成 reranking / conflict / scene / MR 来源四项前置分析后，
再运行 20%~50% 训练集覆盖的中等预算实验，验证：
1. plugin > naive baseline 的相对排序是否稳定
2. freeze-backbone + 新 reranking 是否至少不劣化
```

---

## Self-Review

- 已覆盖评审最核心的五个问题：概率空间 reranking、conflict 标签有效性、scene 校准、MR 退化来源、小预算口径边界。
- 没有引入新结构或新 backbone，范围控制在当前实验闭环内。
- 每个任务都能独立执行并产生可直接写回文档的结果。
