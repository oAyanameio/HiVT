# Reliability Next Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前标签修复已完成的基础上，补齐无训练诊断、冻结主干训练、reranking-only 验证和 shift 评估闭环，判断 reliability 分支是否相对宿主原始 `pi` 具有真实增量价值，且不继续拉低宿主预测指标。

**Architecture:** 保持当前 HiVT + ReliabilityModule 主体结构不变，不新增复杂多头。先补齐评估链路，再做 freeze-backbone 训练入口，最后基于同一 checkpoint 做 reranking-only 与 shift-aware 验证。核心实现集中在 `models/hivt.py` 的验证日志、`metrics/reliability_metrics.py` 的复用、以及新增的分析脚本中。

**Tech Stack:** Python 3, PyTorch, PyTorch Lightning, torchmetrics-style custom metrics, TensorBoard logs, shell runner `run_single_gpu.sh`, pytest.

---

## Scope Check

本计划只覆盖当前文档 `docs/实验问题阶段性解决方案文档.md` 中“仍会影响实验推进”的四类事项：

- 无训练诊断
- freeze backbone 训练
- reranking-only 验证
- shift-aware reliability evaluation

不包含：

- 多 backbone 泛化
- 新 conflict/off-road 独立 head
- 大规模论文消融
- 新网络结构重构

---

## File Structure

- Modify: `/home/lbh/HiVT/models/hivt.py`
  - 增加 scene 校准指标、可选 freeze-backbone 参数、reranking 对照日志。
- Modify: `/home/lbh/HiVT/models/__init__.py`
  - 导出新分析函数（如果脚本需要直接 import）。
- Modify: `/home/lbh/HiVT/metrics/reliability_metrics.py`
  - 复用现有 metric 风格，新增 Spearman 或 scene 级统计支持。
- Modify: `/home/lbh/HiVT/run_single_gpu.sh`
  - 增加 freeze-backbone 训练模式和 analysis/eval 模式参数透传。
- Modify: `/home/lbh/HiVT/eval.py`
  - 支持 reliability checkpoint 验证时输出新增 mode/scene/reranking 指标。
- Create: `/home/lbh/HiVT/scripts/analyze_reliability_baselines.py`
  - 计算 naive baseline AUROC/AUPRC、mode_risk-FDE Spearman、scene 校准诊断。
- Create: `/home/lbh/HiVT/scripts/eval_reranking.py`
  - 同一 checkpoint 下对比 `original pi` 与 `reranked_pi`。
- Create: `/home/lbh/HiVT/tests/test_reliability_analysis.py`
  - 覆盖 naive baseline、Spearman、scene metric 聚合逻辑。
- Modify: `/home/lbh/HiVT/tests/test_reliability.py`
  - 增补 reranking / freeze 相关的最小单测。
- Modify: `/home/lbh/HiVT/tests/test_reliability_presets.py`
  - 若增加 CLI/default 参数，同步覆盖。
- Modify: `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md`
  - 在实验完成后回填结果；本计划阶段只在需要时补“执行入口”说明。

---

### Task 1: 补齐无训练诊断脚本

**Files:**
- Create: `/home/lbh/HiVT/scripts/analyze_reliability_baselines.py`
- Modify: `/home/lbh/HiVT/models/__init__.py`
- Test: `/home/lbh/HiVT/tests/test_reliability_analysis.py`

- [ ] **Step 1: 写失败测试，先固定 naive baseline 和 Spearman 的计算口径**

在 `/home/lbh/HiVT/tests/test_reliability_analysis.py` 新增：

```python
import math
import torch

from metrics.reliability_metrics import AUPRC, AUROC


def _naive_risk_from_pi(pi: torch.Tensor) -> torch.Tensor:
    log_prob = torch.log_softmax(pi, dim=-1)
    return -log_prob


def _spearman_rank_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_rank = x.argsort().argsort().float()
    y_rank = y.argsort().argsort().float()
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    denom = x_rank.norm() * y_rank.norm()
    if denom == 0:
        return x_rank.new_zeros(())
    return (x_rank * y_rank).sum() / denom


def test_naive_risk_from_pi_gives_higher_score_to_lower_prob_mode():
    pi = torch.tensor([[3.0, 1.0, -2.0]])
    risk = _naive_risk_from_pi(pi)
    assert risk.shape == pi.shape
    assert risk[0, 0] < risk[0, 1] < risk[0, 2]


def test_spearman_rank_corr_returns_positive_one_for_identical_order():
    x = torch.tensor([0.1, 0.2, 0.3, 0.4])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    corr = _spearman_rank_corr(x, y)
    assert math.isclose(float(corr), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_naive_risk_can_be_scored_with_existing_auroc_auprc():
    preds = torch.tensor([0.1, 0.8, 0.7, 0.2])
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
    auroc = AUROC(compute_on_step=False)
    auprc = AUPRC(compute_on_step=False)
    auroc.update(preds, targets)
    auprc.update(preds, targets)
    assert float(auroc.compute()) > 0.9
    assert float(auprc.compute()) > 0.9
```

- [ ] **Step 2: 运行测试并确认当前失败**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_analysis.py -q
```

预期：失败，因为文件和函数还不存在。

- [ ] **Step 3: 实现最小分析脚本**

在 `/home/lbh/HiVT/scripts/analyze_reliability_baselines.py` 新增：

```python
#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated as an API.*", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytorch_lightning as pl
import torch

from datamodules import ArgoverseV1DataModule
from metrics import AUPRC, AUROC, BrierScore, ECE
from models import build_reliability_targets
from models import reconstruct_lane_positions
from models.hivt import HiVT


def naive_risk_from_pi(pi: torch.Tensor) -> torch.Tensor:
    return -torch.log_softmax(pi, dim=-1)


def spearman_rank_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    x_rank = x.argsort().argsort().float()
    y_rank = y.argsort().argsort().float()
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    denom = x_rank.norm() * y_rank.norm()
    if float(denom) == 0.0:
        return x_rank.new_zeros(())
    return (x_rank * y_rank).sum() / denom


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--max_batches", type=int, default=16)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--persistent_workers", type=bool, default=False)
    parser.add_argument("--gpus", type=int, default=0)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    pl.seed_everything(2022)
    device = torch.device("cuda:0" if args.gpus > 0 and torch.cuda.is_available() else "cpu")
    datamodule = ArgoverseV1DataModule.from_argparse_args(args)
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.train_dataloader() if args.split == "train" else datamodule.val_dataloader()

    model = HiVT.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        map_location=device,
        strict=False,
        **vars(args),
    ).to(device)
    model.eval()

    naive_auroc = AUROC(compute_on_step=False)
    naive_auprc = AUPRC(compute_on_step=False)
    scene_brier = BrierScore()
    scene_ece = ECE()
    risk_list = []
    fde_list = []

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            data = data.to(device)
            y_hat, pi, reliability_outputs = model(data)
            reg_mask = ~data["padding_mask"][:, model.historical_steps:]
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            current_positions = data["positions"][:, model.historical_steps - 1]
            lane_positions = reconstruct_lane_positions(
                lane_actor_index=data["lane_actor_index"],
                lane_actor_vectors=data["lane_actor_vectors"],
                current_positions=current_positions,
                num_lanes=data["lane_vectors"].size(0),
            )
            targets = build_reliability_targets(
                y_hat=y_hat.detach(),
                y=data.y,
                reg_mask=reg_mask,
                batch=batch,
                lane_positions=lane_positions,
                lane_actor_index=data["lane_actor_index"],
                lane_actor_vectors=data["lane_actor_vectors"],
                positions=data["positions"],
                historical_steps=model.historical_steps,
                rotate_mat=data["rotate_mat"],
                agent_index=data["agent_index"],
                fde_threshold=model.risk_fde_threshold,
                conflict_threshold=model.risk_conflict_threshold,
                offroad_threshold=model.risk_offroad_threshold,
                miss_threshold=model.risk_miss_threshold,
                mode_target_policy=model.mode_target_policy,
                scene_target_policy=model.scene_target_policy,
                conflict_scope=model.risk_conflict_scope,
                conflict_min_frames=model.risk_conflict_min_frames,
                scene_rate_threshold=model.risk_scene_rate_threshold,
            )
            valid_mask = targets["valid_mask"]
            mode_targets = targets["mode_targets"]
            naive_risk = naive_risk_from_pi(pi)
            naive_auroc.update(naive_risk[valid_mask], mode_targets[valid_mask])
            naive_auprc.update(naive_risk[valid_mask], mode_targets[valid_mask])
            fde = targets["fde"]
            if reliability_outputs is not None:
                risk_list.append(reliability_outputs["mode_risk"][valid_mask].reshape(-1).detach().cpu())
                fde_list.append(fde[valid_mask].reshape(-1).detach().cpu())
                if targets["scene_targets"].numel() > 0:
                    scene_brier.update(reliability_outputs["scene_risk"], targets["scene_targets"])
                    scene_ece.update(reliability_outputs["scene_risk"], targets["scene_targets"])

    risk_all = torch.cat(risk_list) if risk_list else torch.tensor([])
    fde_all = torch.cat(fde_list) if fde_list else torch.tensor([])
    spearman = spearman_rank_corr(risk_all, fde_all) if risk_all.numel() > 0 else torch.tensor(0.0)

    print("metric,value")
    print(f"naive_mode_AUROC,{float(naive_auroc.compute()):.6f}")
    print(f"naive_mode_AUPRC,{float(naive_auprc.compute()):.6f}")
    print(f"mode_risk_fde_spearman,{float(spearman):.6f}")
    if risk_all.numel() > 0:
        print(f"scene_BrierScore,{float(scene_brier.compute()):.6f}")
        print(f"scene_ECE,{float(scene_ece.compute()):.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 导出需要的 symbols（如果脚本中复用）**

如果测试或脚本需要从 `models` 顶层导入，修改 `/home/lbh/HiVT/models/__init__.py`：

```python
from models.reliability import predictions_to_scene_abs
```

只在确实需要时添加，避免无意义暴露。

- [ ] **Step 5: 运行测试确认通过**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_analysis.py -q
```

预期：PASS。

- [ ] **Step 6: 运行一次短诊断**

运行：

```bash
cd /home/lbh/HiVT
python scripts/analyze_reliability_baselines.py \
  --root /home/lbh/HiVT/datasets/argoverse \
  --ckpt_path /home/lbh/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt \
  --gpus 1 \
  --max_batches 16 \
  --train_batch_size 8 \
  --val_batch_size 8 \
  --use_reliability true \
  --embed_dim 64 \
  --mode_target_policy fde_only \
  --scene_target_policy target_mode_rate \
  --risk_scene_rate_threshold 0.67
```

预期：输出 `metric,value` 表，至少包含：

```text
naive_mode_AUROC
naive_mode_AUPRC
mode_risk_fde_spearman
scene_BrierScore
scene_ECE
```

- [ ] **Step 7: Commit**

```bash
git add scripts/analyze_reliability_baselines.py tests/test_reliability_analysis.py models/__init__.py
git commit -m "Add reliability baseline analysis script"
```

### Task 2: 补齐 scene 校准与 reranking 对照日志

**Files:**
- Modify: `/home/lbh/HiVT/models/hivt.py`
- Modify: `/home/lbh/HiVT/metrics/reliability_metrics.py`
- Test: `/home/lbh/HiVT/tests/test_reliability_analysis.py`

- [ ] **Step 1: 写失败测试，固定 scene metric 和 reranking 对照输出的最小语义**

在 `/home/lbh/HiVT/tests/test_reliability_analysis.py` 追加：

```python
def test_scene_metrics_accept_binary_scene_probs():
    preds = torch.tensor([0.2, 0.9, 0.8, 0.1])
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
    brier = BrierScore()
    ece = ECE()
    brier.update(preds, targets)
    ece.update(preds, targets)
    assert float(brier.compute()) < 0.1
    assert float(ece.compute()) < 0.2
```

- [ ] **Step 2: 运行测试并确认失败或缺少 import**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_analysis.py::test_scene_metrics_accept_binary_scene_probs -q
```

- [ ] **Step 3: 在 `models/hivt.py` 中新增 scene 校准日志与 reranking 对照指标**

在 `__init__` 里增加：

```python
if use_reliability:
    self.scene_risk_brier = BrierScore()
    self.scene_risk_ece = ECE()
```

在 `validation_step` 中，在 scene 分支下追加：

```python
if scene_targets.numel() > 0:
    self.scene_risk_brier.update(reliability_outputs['scene_risk'], scene_targets)
    self.scene_risk_ece.update(reliability_outputs['scene_risk'], scene_targets)
    self.log('val_scene_BrierScore', self.scene_risk_brier, prog_bar=False, on_step=False, on_epoch=True,
             batch_size=scene_targets.size(0))
    self.log('val_scene_ECE', self.scene_risk_ece, prog_bar=False, on_step=False, on_epoch=True,
             batch_size=scene_targets.size(0))
```

同时补充 reranking-only 对照基础量：

```python
reranked_pi = reliability_outputs['reranked_pi']
original_top1 = pi.argmax(dim=-1)
reranked_top1 = reranked_pi.argmax(dim=-1)
top1_changed = (original_top1 != reranked_top1).float().mean()
self.log('val_rerank_top1_change_rate', top1_changed, prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
```

- [ ] **Step 4: 运行定向测试**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_analysis.py -q
```

预期：PASS。

- [ ] **Step 5: 用宿主 checkpoint 跑一次标准验证**

运行：

```bash
cd /home/lbh/HiVT
./run_single_gpu.sh eval /home/lbh/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt 0 8
```

预期：验证日志里新增：

```text
val_scene_BrierScore
val_scene_ECE
val_rerank_top1_change_rate
```

- [ ] **Step 6: Commit**

```bash
git add models/hivt.py tests/test_reliability_analysis.py
git commit -m "Log scene calibration and reranking stats"
```

### Task 3: 增加 freeze-backbone 训练入口

**Files:**
- Modify: `/home/lbh/HiVT/models/hivt.py`
- Modify: `/home/lbh/HiVT/train.py`
- Modify: `/home/lbh/HiVT/run_single_gpu.sh`
- Modify: `/home/lbh/HiVT/tests/test_reliability_presets.py`

- [ ] **Step 1: 写失败测试，先固定 freeze 参数语义**

在 `/home/lbh/HiVT/tests/test_reliability_presets.py` 新增：

```python
def test_build_reliability_train_args_accepts_freeze_backbone_flag():
    from training_presets import build_reliability_train_args

    args = build_reliability_train_args(
        embed_dim=64,
        freeze_backbone=True,
    )

    assert args["freeze_backbone"] is True
```

- [ ] **Step 2: 运行测试确认失败**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_presets.py::test_build_reliability_train_args_accepts_freeze_backbone_flag -q
```

- [ ] **Step 3: 在 `models/hivt.py` 中实现 freeze-backbone 参数与 optimizer 过滤**

在 `HiVT.__init__` 参数中加入：

```python
freeze_backbone: bool,
```

保存到：

```python
self.freeze_backbone = freeze_backbone
```

在 `configure_optimizers` 开头加入：

```python
if self.freeze_backbone:
    for module in (self.local_encoder, self.global_interactor, self.decoder):
        for param in module.parameters():
            param.requires_grad = False
```

并在构建 `param_dict` 时只保留：

```python
param_dict = {
    param_name: param
    for param_name, param in self.named_parameters()
    if param.requires_grad
}
```

同时保护空集合断言：

```python
assert len(param_dict) > 0
```

- [ ] **Step 4: 在 CLI 与 shell 入口透传 freeze 选项**

在 `/home/lbh/HiVT/models/hivt.py` 的 `add_model_specific_args` 里加入：

```python
parser.add_argument('--freeze_backbone', type=bool, default=False)
```

在 `/home/lbh/HiVT/run_single_gpu.sh` 中新增环境变量：

```bash
FREEZE_BACKBONE=false
```

并在 `train_reliability` / `train_reliability_shift` 的命令中透传：

```bash
--freeze_backbone "${FREEZE_BACKBONE:-false}"
```

- [ ] **Step 5: 运行测试确认通过**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_presets.py -q
```

预期：PASS。

- [ ] **Step 6: 用小预算跑一次 freeze-backbone 训练**

运行：

```bash
cd /home/lbh/HiVT
FREEZE_BACKBONE=true \
TRAIN_BATCH_SIZE=8 \
VAL_BATCH_SIZE=8 \
NUM_WORKERS=8 \
RUN_VERSION=freeze_backbone_budget128_targetmode067 \
MODE_TARGET_POLICY=fde_only \
SCENE_TARGET_POLICY=target_mode_rate \
RISK_SCENE_RATE_THRESHOLD=0.67 \
./run_single_gpu.sh train_reliability 64 0
```

若需要强制小预算，直接在 `train.py` 临时增加：

```python
parser.add_argument('--limit_train_batches', type=int, default=1.0)
parser.add_argument('--limit_val_batches', type=int, default=1.0)
```

并透传 `128/32`。

- [ ] **Step 7: Commit**

```bash
git add models/hivt.py run_single_gpu.sh train.py tests/test_reliability_presets.py
git commit -m "Add freeze-backbone reliability training mode"
```

### Task 4: 增加 reranking-only 评估脚本

**Files:**
- Create: `/home/lbh/HiVT/scripts/eval_reranking.py`
- Modify: `/home/lbh/HiVT/models/hivt.py`
- Test: `/home/lbh/HiVT/tests/test_reliability_analysis.py`

- [ ] **Step 1: 写失败测试，固定 reranking 改变量的计算**

在 `/home/lbh/HiVT/tests/test_reliability_analysis.py` 追加：

```python
def test_reranking_changes_top1_when_high_risk_mode_has_best_pi():
    pi = torch.tensor([[3.0, 2.0]])
    risk = torch.tensor([[1.0, 0.0]])
    reranked = torch.softmax(pi - risk * 2.0, dim=-1)
    assert int(pi.argmax(dim=-1)[0]) == 0
    assert int(reranked.argmax(dim=-1)[0]) == 1
```

- [ ] **Step 2: 运行测试确认失败或尚未接入脚本**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_analysis.py::test_reranking_changes_top1_when_high_risk_mode_has_best_pi -q
```

- [ ] **Step 3: 实现 `eval_reranking.py`**

在 `/home/lbh/HiVT/scripts/eval_reranking.py` 新增：

```python
#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytorch_lightning as pl
import torch

from datamodules import ArgoverseV1DataModule
from metrics import ADE, FDE, MR
from models.hivt import HiVT


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--rerank_alpha", type=float, default=0.5)
    parser.add_argument("--max_batches", type=int, default=32)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=False)
    parser.add_argument("--persistent_workers", type=bool, default=False)
    parser.add_argument("--gpus", type=int, default=0)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    pl.seed_everything(2022)
    device = torch.device("cuda:0" if args.gpus > 0 and torch.cuda.is_available() else "cpu")
    datamodule = ArgoverseV1DataModule.from_argparse_args(args)
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.val_dataloader()
    model = HiVT.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        map_location=device,
        strict=False,
        **vars(args),
    ).to(device)
    model.eval()
    model.reliability_module.rerank_alpha = args.rerank_alpha

    orig_ade = ADE()
    orig_fde = FDE()
    orig_mr = MR()
    rerank_ade = ADE()
    rerank_fde = FDE()
    rerank_mr = MR()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            data = data.to(device)
            y_hat, pi, reliability_outputs = model(data)
            y_agent = data.y[data["agent_index"]]

            y_hat_agent = y_hat[:, data["agent_index"], :, :2]
            orig_idx = pi[data["agent_index"]].argmax(dim=-1)
            rerank_idx = reliability_outputs["reranked_pi"][data["agent_index"]].argmax(dim=-1)
            orig_best = y_hat_agent[orig_idx, torch.arange(data.num_graphs, device=device)]
            rerank_best = y_hat_agent[rerank_idx, torch.arange(data.num_graphs, device=device)]

            orig_ade.update(orig_best, y_agent)
            orig_fde.update(orig_best, y_agent)
            orig_mr.update(orig_best, y_agent)
            rerank_ade.update(rerank_best, y_agent)
            rerank_fde.update(rerank_best, y_agent)
            rerank_mr.update(rerank_best, y_agent)

    print("metric,value")
    print(f"original_minADE,{float(orig_ade.compute()):.6f}")
    print(f"original_minFDE,{float(orig_fde.compute()):.6f}")
    print(f"original_minMR,{float(orig_mr.compute()):.6f}")
    print(f"reranked_minADE,{float(rerank_ade.compute()):.6f}")
    print(f"reranked_minFDE,{float(rerank_fde.compute()):.6f}")
    print(f"reranked_minMR,{float(rerank_mr.compute()):.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

运行：

```bash
cd /home/lbh/HiVT
pytest tests/test_reliability_analysis.py -q
```

- [ ] **Step 5: 运行 reranking-only 评估**

运行：

```bash
cd /home/lbh/HiVT
python scripts/eval_reranking.py \
  --root /home/lbh/HiVT/datasets/argoverse \
  --ckpt_path /home/lbh/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt \
  --gpus 1 \
  --embed_dim 64 \
  --use_reliability true \
  --mode_target_policy fde_only \
  --scene_target_policy target_mode_rate \
  --risk_scene_rate_threshold 0.67 \
  --rerank_alpha 0.5
```

预期：输出 `original_*` 与 `reranked_*` 六个指标。

- [ ] **Step 6: 扫描多个 alpha**

运行：

```bash
cd /home/lbh/HiVT
for alpha in 0.2 0.5 1.0; do
  python scripts/eval_reranking.py \
    --root /home/lbh/HiVT/datasets/argoverse \
    --ckpt_path /home/lbh/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt \
    --gpus 1 \
    --embed_dim 64 \
    --use_reliability true \
    --mode_target_policy fde_only \
    --scene_target_policy target_mode_rate \
    --risk_scene_rate_threshold 0.67 \
    --rerank_alpha "$alpha"
done
```

- [ ] **Step 7: Commit**

```bash
git add scripts/eval_reranking.py tests/test_reliability_analysis.py
git commit -m "Add reranking-only evaluation script"
```

### Task 5: 跑 freeze-backbone / reranking / shift 三组实验并整理结果

**Files:**
- Modify: `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md`
- Modify: `/home/lbh/HiVT/docs/可靠性实验结果汇总.md`
- Modify: `/home/lbh/HiVT/docs/可靠性实验结果汇总.csv`

- [ ] **Step 1: 跑 freeze-backbone 小预算对照**

运行：

```bash
cd /home/lbh/HiVT
FREEZE_BACKBONE=true \
TRAIN_BATCH_SIZE=8 \
VAL_BATCH_SIZE=8 \
NUM_WORKERS=8 \
RUN_VERSION=freeze_backbone_budget128_targetmode067 \
MODE_TARGET_POLICY=fde_only \
SCENE_TARGET_POLICY=target_mode_rate \
RISK_SCENE_RATE_THRESHOLD=0.67 \
./run_single_gpu.sh train_reliability 64 0
```

记录：

```text
val_minADE
val_minFDE
val_minMR
val_mode_AUROC
val_mode_AUPRC
val_mode_BrierScore
val_mode_ECE
val_scene_BrierScore
val_scene_ECE
```

- [ ] **Step 2: 跑 reranking-only 三个 alpha**

运行 Task 4 的 alpha 扫描命令。

记录：

```text
alpha
original_minADE / original_minFDE / original_minMR
reranked_minADE / reranked_minFDE / reranked_minMR
```

- [ ] **Step 3: 跑 shift-aware 诊断**

运行：

```bash
cd /home/lbh/HiVT
SHIFT_HISTORY_DROPOUT_P=0.3 \
SHIFT_NEIGHBOR_DROPOUT_P=0.2 \
SHIFT_POSITION_NOISE_STD=0.1 \
SHIFT_HEADING_NOISE_STD=0.05 \
SHIFT_MAP_JITTER_STD=0.05 \
SHIFT_LANE_DROPOUT_P=0.1 \
RUN_VERSION=shift_diag_targetmode067 \
MODE_TARGET_POLICY=fde_only \
SCENE_TARGET_POLICY=target_mode_rate \
RISK_SCENE_RATE_THRESHOLD=0.67 \
./run_single_gpu.sh train_reliability_shift 64 0
```

重点检查：

```text
failure rate 是否上升
mode_risk_pred_mean 是否同步上升
naive baseline 与 mode_risk 的 AUROC 差异
```

- [ ] **Step 4: 回填结果文档**

在 `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md` 中补：

- naive baseline AUROC/AUPRC
- mode_risk-FDE Spearman
- scene_BrierScore / scene_ECE
- freeze-backbone 对照结果
- reranking-only 对照结果
- shift 诊断结果

同时更新：

- `/home/lbh/HiVT/docs/可靠性实验结果汇总.md`
- `/home/lbh/HiVT/docs/可靠性实验结果汇总.csv`

- [ ] **Step 5: 写结论边界**

结论必须按下面三选一归档：

```text
A. reliability 相对 naive pi 有明确增量，且 freeze backbone 不再破坏宿主指标
B. reliability 只有校准/检测价值，但 mode selection 增量有限
C. 当前 risk score 尚不足以支撑 reranking，优先继续做排序能力增强
```

- [ ] **Step 6: Commit**

```bash
git add docs/实验问题阶段性解决方案文档.md docs/可靠性实验结果汇总.md docs/可靠性实验结果汇总.csv
git commit -m "Document next-stage reliability experiment results"
```

## Self-Review

1. **Spec coverage**
- 已覆盖文档中的三项当前目标：增量价值、排序能力、避免宿主退化。
- 已覆盖推荐顺序：无训练诊断 -> freeze backbone -> reranking-only -> shift eval。
- 未扩展到多 backbone 或复杂论文 head，符合当前范围。

2. **Placeholder scan**
- 计划中没有 `TODO/TBD/implement later`。
- 每个任务都给了具体文件、命令和最小代码骨架。

3. **Type consistency**
- 统一使用：
  - `mode_risk`
  - `scene_risk`
  - `reranked_pi`
  - `freeze_backbone`
  - `naive_risk_from_pi`
  - `spearman_rank_corr`

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-28-reliability-next-stage-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
