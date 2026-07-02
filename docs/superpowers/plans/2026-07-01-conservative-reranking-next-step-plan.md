# Conservative Reranking Next-Step Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 吸收最新评审意见后，先把 `conservative reranking rule` 固化成统一复评口径，再在该口径下判断当前 freeze-backbone objective 是否仍有额外增益，最后才决定是否改写成更直接的 decision-oriented supervision。

**Architecture:** 保持当前 HiVT + ReliabilityModule 结构和已实现的概率空间 reranking 不变，不再继续扩展 `scene/conflict/joint training` 主线。执行顺序调整为 `rule first -> cross-checkpoint verification -> objective re-eval -> objective redesign if needed`。所有判断都围绕 `freeze-backbone -> mode_risk -> reranking -> MR` 这条链闭环。

**Evidence Boundary:** 当前已有证据足以支持“先锁 rule、再判 objective”的方向调整，但还不足以支持论文级强结论。后续文档和实验判读必须始终区分：

- 可以当作方向性证据的结果
- 可以当作较强结论的结果
- 仍可能被样本量、checkpoint 依赖性或单次 run 波动解释掉的结果

**Tech Stack:** Python 3, PyTorch, PyTorch Lightning, existing scripts `scripts/eval_reranking.py`, `scripts/scan_reranking_rules.py`, `scripts/analyze_reranking_case_comparison.py`, shell runner `run_single_gpu.sh`, markdown docs.

---

## Scope Check

本计划只覆盖以下四项：

- 固化 conservative reranking 默认口径
- 跨 checkpoint 复验 rule 的稳定性
- 在保守 rule 下复评当前 objective 是否还有额外收益
- 为下一轮中等预算 run 给出明确入口与判读标准

不包含：

- 继续扩展 scene / conflict 大矩阵
- 回到 joint training 主线
- 新 backbone 或复杂多头结构
- 论文级全面重写

---

## Canonical Evaluation Settings

后续所有 reranking 复评都必须同时报告两套口径：

1. 基础复评口径

```text
rerank_method   = prob_product
rerank_alpha    = 1.0
rerank_top_k    = 3
rerank_margin   = 0.0
rerank_guard    = 0.0
max_batches     = 32
```

2. 保守决策口径

```text
rerank_method   = prob_product
rerank_alpha    = 1.0
rerank_top_k    = 3
rerank_margin   = candidate in {0.1, 0.15, 0.2}
rerank_guard    = 0.0
max_batches     = 32
```

判读原则：

- 先看保守决策口径下是否 `reranked_MR <= original_MR`
- 再看 `hit_to_miss` 是否为 `0` 或显著低于基础复评口径
- 最后看 `miss_to_hit` 是否还能保留至少少量强收益翻转

说明：

- 在原始 checkpoint 上，`margin=0.1` 曾是最优点；
- 但在新增独立 freeze-backbone checkpoint（`seed=3407`）上，
  真正满足 `hit_to_miss = 0` 且 `reranked_MR < original_MR` 的是 `margin=0.15 / 0.20`；
- 其中 `margin=0.15` 在两次扫描里的形态更一致；
- `margin=0.20` 更适合作为收益可能归零的参考上界，而不是同等级默认候选；
- 因此当前默认不再把 `0.1` 视为已锁定单点，而把 `0.15` 作为当前主候选继续复验。

其中“稳定”暂时量化为：

```text
1. 至少 2 个相互独立的 freeze-backbone checkpoint 上，
   同一个固定 margin 值满足 hit_to_miss = 0
2. 且 miss_to_hit >= 1
3. 且 reranked_MR 不劣于 original_MR
```

若只满足部分条件，则只能记为“趋势成立”，不能记为“rule 已稳定成立”。

---

## File Structure

- Modify: `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md`
  - 把“接下来实验顺序”改成 rule-first 口径，并补一个明确四步计划。
- Modify: `/home/lbh/HiVT/docs/可靠性实验结果汇总.md`
  - 补一段“后续复评统一口径”与结果记录模板。
- Create: `/home/lbh/HiVT/docs/superpowers/plans/2026-07-01-conservative-reranking-next-step-plan.md`
  - 记录当前执行计划。

---

### Task 1: 固化 conservative reranking 为默认复评逻辑

**Files:**
- Modify: `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md`
- Modify: `/home/lbh/HiVT/docs/可靠性实验结果汇总.md`

- [ ] **Step 1: 在文档中明确默认 rule**

统一写成：

```text
prob_product + alpha=1.0 + top_k=3 + guard=0.0
with margin tracked at 0.1 / 0.15 / 0.2
```

并明确其身份是：

- 当前默认 reranking 决策逻辑
- 不是历史可比主表的唯一口径
- 所有新 checkpoint 都必须同时报告基础复评口径与保守决策口径
- 其中保守决策口径当前不是单点，而是待收敛的 `conservative margin family`

- [ ] **Step 2: 给出统一记录字段**

后续每次复评至少记录：

```text
checkpoint_name
eval_setting_label
original_minADE / original_minFDE / original_minMR
reranked_minADE / reranked_minFDE / reranked_minMR
rerank_top1_change_rate
case_hit_to_miss_count
case_miss_to_hit_count
case_mean_fde_delta
```

- [ ] **Step 3: 明确当前默认结论边界**

统一表述为：

```text
当前已经证明存在一个保守 rule，
可以在不修改风险头参数的前提下避免激进翻转导致的 MR 退化；
但该结论仍需跨 checkpoint 复验其稳定性。
```

---

### Task 2: 跨 checkpoint 复验 conservative rule 是否稳定成立

**Files:**
- Use existing script: `/home/lbh/HiVT/scripts/eval_reranking.py`
- Use existing script: `/home/lbh/HiVT/scripts/scan_reranking_rules.py`
- Modify: `/home/lbh/HiVT/docs/可靠性实验结果汇总.md`

- [ ] **Step 1: 复评 freeze-backbone 参考 checkpoint**

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
  --rerank_method prob_product \
  --rerank_alpha 1.0 \
  --rerank_top_k 3 \
  --rerank_margin 0.0 \
  --rerank_guard 0.0 \
  --max_batches 32 \
  --train_batch_size 8 \
  --val_batch_size 8 \
  --num_workers 8
```

再运行同样命令但把 `--rerank_margin` 改为 `0.1`。

- [ ] **Step 2: 复评 top-k only 小预算 checkpoint**

运行同样两套口径，checkpoint 改为：

```text
/home/lbh/HiVT/runs/hivt_reliability/reliability_20260629_191045_dim64_gpu0/checkpoints/epoch=00-val_minFDE=0.8031.ckpt
```

但文档中必须明确标注：

```text
该 checkpoint 已混入新 objective（rank loss + threshold weighting），
不能被当成“rule 通用性”的独立 freeze-backbone 样本；
它只能用于观察 rule 与 objective 叠加后的表现，不能单独支撑 rule 稳定性结论。
```

- [x] **Step 2.5: 补一个与 objective 无关的独立 freeze-backbone checkpoint**

已完成：新增 `seed=9103` 的独立 freeze-backbone checkpoint（配置与 `seed=3407` 相同，`rank_loss_weight=0.0`、`threshold_weighting=false`），训练后 `val_minADE=0.574 / val_minFDE=0.803 / val_minMR=0.0508`，与参考 checkpoint 一致。

对该 checkpoint 跑了 `margin=0.10/0.15/0.20`（`guard=0.0`）scan，结果见 `docs/reranking_rule_scan_seed9103.csv`：

```text
margin=0.10 -> reranked_MR=0.421875, hit_to_miss=0, miss_to_hit=1   (原始 MR=0.425781)
margin=0.15 -> reranked_MR=0.421875, hit_to_miss=0, miss_to_hit=1
margin=0.20 -> reranked_MR=0.421875, hit_to_miss=0, miss_to_hit=1
```

- [ ] **Step 3: 如有必要，对单个 checkpoint 再跑 margin scan**

当 `margin=0.1` 未显示稳定收益时，再运行：

```bash
cd /home/lbh/HiVT
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate HiVT
CUDA_VISIBLE_DEVICES=0 python scripts/scan_reranking_rules.py \
  --root /home/lbh/HiVT/datasets/argoverse \
  --ckpt_path /home/lbh/HiVT/runs/hivt_reliability/reliability_20260629_191045_dim64_gpu0/checkpoints/epoch=00-val_minFDE=0.8031.ckpt \
  --gpus 1 \
  --embed_dim 64 \
  --use_reliability true \
  --mode_target_policy fde_only \
  --scene_target_policy target_mode_rate \
  --risk_scene_rate_threshold 0.67 \
  --rerank_method prob_product \
  --rerank_alpha 1.0 \
  --rerank_top_k 3 \
  --margin_grid 0.0,0.05,0.1,0.15,0.2 \
  --guard_grid 0.0,0.1,0.2,0.3 \
  --max_batches 32 \
  --train_batch_size 8 \
  --val_batch_size 8 \
  --num_workers 8
```

- [x] **Step 4: 记录稳定性判读**

当前判定 “rule 已稳定” 至少要求：

```text
1. 至少 2 个相互独立的 freeze-backbone checkpoint 上，
   某个固定 margin 值（当前重点跟踪 0.10 / 0.15 / 0.20 之一）均满足：
   - reranked_MR <= original_MR
   - hit_to_miss = 0
   - miss_to_hit >= 1
2. 若只在“参考 checkpoint + 混入新 objective 的 checkpoint”上成立，
   只能记为趋势，不能写成稳定结论
3. 不需要依赖 guard 才能成立
```

**结论：`margin=0.15` 已满足稳定性门槛。**

两个相互独立、无新 objective 的 freeze-backbone checkpoint（`seed=3407`、`seed=9103`）在 `margin=0.15` 下都得到：

```text
reranked_MR = 0.421875 < original_MR = 0.425781
hit_to_miss = 0
miss_to_hit = 1
```

不依赖 guard（两次扫描里 guard=0.0 都能达成上述结果）。因此 `margin=0.15` 正式升级为默认 conservative rule 的 margin 值。

最终状态：

```text
- margin=0.10: 在原 checkpoint 与 seed9103 上成立，但在 seed3407 上不成立（hit_to_miss=1）—— 不稳定
- margin=0.15: 在 seed3407、seed9103 两个独立 checkpoint 上均成立 —— 已达稳定性门槛，升级为默认
- margin=0.20: 在 seed3407、seed9103 两个独立 checkpoint 上也成立，但在 mixed-objective checkpoint 上更接近收益归零，因此只作参考上界，不作为默认
```

---

### Task 3: 在保守 rule 下复评当前 objective 是否仍有额外收益

**Files:**
- Use existing runner: `/home/lbh/HiVT/run_single_gpu.sh`
- Modify: `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md`
- Modify: `/home/lbh/HiVT/docs/可靠性实验结果汇总.md`

- [x] **Step 1: 先把当前 objective 的结论边界写清楚**

进入本任务前必须先满足：

```text
至少再补 1 个纯 freeze-backbone、无新 objective 的独立种子，
并完成 margin=0.10 / 0.15 / 0.20 scan。
```

如果 `margin=0.15` 未能在第二个干净独立种子上复现：

```text
暂停 objective re-eval；
继续收敛 rule，而不是把 rule 层不确定性带进 Task 3。
```

统一表述为：

```text
当前 near-threshold / top-k objective 已实现并进入训练，
但其是否真正改善 MR，必须在 conservative reranking rule 下重新判断；
在 margin=0.0 的激进 rule 下观测到的负信号，不能直接当成 objective 的最终结论。
```

- [x] **Step 2: 运行下一轮中等预算 freeze-backbone objective re-eval**

本计划将“中等预算”固定为：

```text
max_epochs          = 1
limit_train_batches = 256
limit_val_batches   = 64
train_batch_size    = 8
val_batch_size      = 8
```

该预算的结论边界固定写成：

```text
它足以回答“链路是否打通 + 方向是否继续成立”，
但不足以单次支撑论文级 A/B 结论。
```

实际执行入口（保持 mixed-objective 小预算 run 的配置，只放大预算，不再额外切换 `scene_target_policy`）：

```bash
cd /home/lbh/HiVT
RUN_VERSION=freeze_backbone_threshold_objective_budget256 \
INIT_CKPT_PATH=/home/lbh/HiVT/runs/hivt_reliability/freeze_backbone_budget128_targetmode067/checkpoints/epoch=00-val_minFDE=0.8031.ckpt \
FREEZE_BACKBONE=true \
TRAIN_BATCH_SIZE=8 \
VAL_BATCH_SIZE=8 \
NUM_WORKERS=8 \
LIMIT_TRAIN_BATCHES=256 \
LIMIT_VAL_BATCHES=64 \
MAX_EPOCHS=1 \
SEED=9103 \
MODE_TARGET_POLICY=fde_only \
SCENE_TARGET_POLICY=target_best_mode_fail \
RISK_SCENE_RATE_THRESHOLD=0.5 \
RANK_LOSS_WEIGHT=0.1 \
MODE_RISK_THRESHOLD_WEIGHT_ENABLED=true \
MODE_RISK_THRESHOLD_WEIGHT_RADIUS=0.25 \
MODE_RISK_THRESHOLD_WEIGHT_PEAK=3.0 \
MODE_RISK_THRESHOLD_WEIGHT_BASE=1.0 \
MODE_RISK_RANK_TOP_K=3 \
MODE_RISK_RANK_NEAR_THRESHOLD_ONLY=false \
MODE_RISK_RANK_THRESHOLD_RADIUS=0.2 \
SCENE_LOSS_WEIGHT=0.05 \
./run_single_gpu.sh train_reliability 64 0
```

- [x] **Step 3: 对中等预算 checkpoint 立即做双口径复评**

训练结束后，对新 checkpoint 分别运行：

1. 基础复评口径：`margin=0.0`
2. 保守决策口径：`margin=0.15`

若 `last.ckpt` 与 best checkpoint 同时存在，优先复评监控 best checkpoint。

- [x] **Step 4: 用统一判读规则给出去留结论**

判读规则：

```text
A. 若新 checkpoint 在 margin=0.15 下相对原始 pi 仍有稳定额外收益，
   则保留当前 objective 分支，后续继续围绕它做小范围调参。

B. 若 margin=0.15 已经吃掉大部分收益，而新 objective 既不增加 miss_to_hit，
   也不改善 reranked_MR，则当前 objective 不再作为默认主线。
```

补充约束：

```text
若结果刚好处在边界（例如 MR 几乎不变、只出现 0/1 个离散 case 差异），
不能用单次中等预算 run 直接做最终去留判断；
应至少再补一次复验（复跑或换 seed）再下结论。
```

**2026-07-02 实际结果：**

- 训练已完成，best checkpoint 为：

```text
/home/lbh/HiVT/runs/hivt_reliability/freeze_backbone_threshold_objective_budget256_seed9103_v3/checkpoints/epoch=00-val_minFDE=1.0136.ckpt
```

- 评估环境中出现过一次 `protobuf` 的 `GLIBCXX_3.4.29` ABI 问题；
  通过固定 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 后已成功补跑评估。
- 统一使用 `64 val batches` 口径，对比 reference freeze-backbone checkpoint 与本轮 mixed-objective checkpoint：

```text
reference checkpoint
  margin=0.0  -> original_MR=0.449219, reranked_MR=0.470703, hit_to_miss=17, miss_to_hit=6
  margin=0.15 -> original_MR=0.449219, reranked_MR=0.447266, hit_to_miss=0,  miss_to_hit=1

budget256 mixed-objective checkpoint
  margin=0.0  -> original_MR=0.449219, reranked_MR=0.462891, hit_to_miss=18, miss_to_hit=11
  margin=0.15 -> original_MR=0.449219, reranked_MR=0.447266, hit_to_miss=0,  miss_to_hit=1
```

**Task 3 当前结论：**

```text
conservative rule 在中等预算 mixed-objective checkpoint 上继续有效，
但当前 objective 暂未显示超出 reference baseline 的额外 MR 增量。
```

因此 Task 3 已经完成到“单个中等预算 seed 的证据边界”这一层；
下一步不应直接跳到最终 redesign，而应先补一个 mixed-objective 的第二独立 seed / 复跑，再决定是否进入 Task 4。

这里还要补一个执行层面的注意点：

```text
margin=0.15 下 reference 与 mixed-objective 都只保留了极少数真实翻转，
当前“无增量”并不自动等价于“objective 真无效”；
还存在 conservative rule ceiling effect 掩盖细小差异的可能性。
```

因此下一步的复评不应只盯：

```text
margin = 0.15
```

而应扩成两层：

```text
主结论层：
  margin = 0.15

诊断层：
  margin = 0.05 / 0.10
```

用途分别是：

```text
margin=0.15:
  判断当前默认 conservative rule 下是否有稳定额外 MR 增量

margin=0.05 / 0.10:
  判断是否存在被更保守门槛掩盖的 miss_to_hit 增量趋势，
  用于区分“objective 真无效”与“rule ceiling”
```

---

### Task 4: 若当前 objective 仍无增益，切换到更直接的 decision-oriented objective 设计

**Files:**
- Future design target: `/home/lbh/HiVT/models/hivt.py`
- Future design target: `/home/lbh/HiVT/losses/reliability_losses.py`
- Modify: `/home/lbh/HiVT/docs/实验问题阶段性解决方案文档.md`

- [ ] **Step 1: 只在 Task 3 明确失败后才进入本任务**

触发条件：

```text
conservative rule 已稳定成立
+ 当前 objective 在 conservative rule 下仍无额外收益
```

- [ ] **Step 2: 设计方向改成 top1-vs-best_alt / miss-oriented supervision**

设计原则固定为：

```text
1. 当前 top-1 已 hit 时，优先保护而不是鼓励对称翻转
2. 当前 top-1 已 miss 且存在明显更低风险替代 mode 时，再增强翻转监督
3. 监督对象优先缩到 target actor 的 top-k 候选，而不是全 mode 全排序
```

- [ ] **Step 3: 暂缓进一步放大 near-threshold BCE 权重**

在新 objective 明确之前：

- 不继续默认增大 `mode_risk_threshold_weight_peak`
- 不把 `rerank_guard` 作为主调参轴
- 不把 `scene/conflict` 重新抬回主线

关于 `guard` 的说明必须写清楚：

```text
在当前实现中，guard 只有在非 None 时才参与门控；
而当 guard=0.0 时，条件近似等于 orig_risk > 0，
对当前风险分布通常几乎恒为真。
```

因此当前扫描里 `guard` 几乎不改变结果，更像是这套 margin 门控结构下的自然结果，而不只是“当前 checkpoint 恰好不需要 guard”。

---

## Execution Preconditions

在继续任何新 run 前，先确认：

1. `/home` 磁盘空间充足，避免再次因 `save_last` 失败中断结果保存
2. 需要复评的 checkpoint 文件都可读
3. 文档中的 checkpoint 路径与实际 run 目录一致

当前磁盘状态参考：

```text
/home avail ≈ 850G
```

---

## Done Criteria

本计划完成的标志是：

1. 文档中已统一采用“双口径 reranking 复评”写法
2. 至少两个 checkpoint 完成 `margin=0.0` 与 `margin=0.15` 对照复评
3. 已完成 rule 稳定性复验；中等预算 objective re-eval 也已完成到单个中等预算 seed 的结论边界
4. 已能明确回答下面这个问题：

```text
在 conservative reranking rule 已锁稳后，
当前 objective 是否还能为 MR 提供额外增量？
```

当前答案是：

```text
在已完成的单个中等预算 mixed-objective seed 上，未观察到额外 MR 增量；
但仍需至少再补一个独立 seed / 复跑，
并结合 margin=0.05 / 0.10 / 0.15 的趋势一起判断，
才能决定是否正式转向 Task 4。
```
