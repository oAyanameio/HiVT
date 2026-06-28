# 面向自动驾驶轨迹预测的即插即用可靠性模块 Idea 文档

日期：2026-06-25

## 1. 题目定位

这项工作的核心不是再设计一个新的完整轨迹预测 backbone，而是提出一个能够接入现有轨迹预测模型的可靠性模块，使模型在输出未来轨迹的同时，也能显式输出“这次预测是否可靠”。

建议题目可以表述为：

- 中文：面向自动驾驶轨迹预测的即插即用交互感知可靠性评估模块
- 英文：InterAct-Risk: A Plug-and-Play Interaction-Aware Reliability Module for Motion Forecasting

其中：

- `Plug-and-Play` 强调方法不是替换原模型，而是以后接分支形式接入
- `Interaction-Aware` 强调风险判断不是只看单条轨迹，而要看多车交互和场景上下文
- `Reliability` 强调研究目标是可信性、失败预警和风险估计，而不是单纯追求 minADE / minFDE

首个验证宿主模型设为 HiVT，后续再扩展到 QCNet、MTR、Wayformer 等 backbone。

## 2. 研究背景

近年来，多智能体轨迹预测已经成为自动驾驶感知到规划链路中的关键中间模块。像 HiVT、QCNet、MTR、Wayformer 这类模型，已经可以输出较高质量的多模态未来轨迹预测，在 Argoverse、Waymo、nuScenes 等数据集上也取得了较强结果。

但从部署角度看，仅有“预测结果”是不够的。自动驾驶系统真正关心的不只是：

- 目标车辆未来可能怎么走

还包括：

- 当前预测是否可靠
- 哪个候选 mode 更值得信任
- 在交互激烈、地图噪声大、输入不完整时，模型是否能够意识到自己可能会出错

现有主流工作仍主要以 `minADE / minFDE / MR` 作为核心指标，这类指标关注“平均预测误差”，却不能回答下面这些更接近真实部署的问题：

- 某个高分 mode 是否其实是高风险 mode
- 复杂路口场景中模型是否存在“自信地犯错”
- 分布偏移下模型输出的 score 与真实失败概率是否匹配

因此，这个 idea 的本质是在轨迹预测研究中补上“可靠性建模”这一层，使预测模块具备可解释、可监控、可降级的能力。

## 3. 核心问题定义

这个工作不把主要问题定义成：

- 如何让 HiVT 再降低一点点 minFDE

而是定义成下面三个问题：

1. 预测器当前输出的 mode score，是否真的代表这条预测可靠
2. 在多车强交互、路口汇入、历史缺失、地图扰动等条件下，模型何时更容易失败
3. 能否设计一个不改 backbone 主干、但能统一输出风险信息的可插拔模块

换句话说，这个工作研究的是：

“让模型不仅给出未来轨迹，还给出它对自己预测失败风险的判断。”

## 4. 研究目标

目标可以拆成五条：

1. 对每个预测 mode 输出 `mode-level risk`
2. 对每个场景输出 `scene-level risk`
3. 在复杂交互和分布偏移条件下，提高失败检测能力
4. 在基本不损害原始预测精度的前提下，增强可解释性和可部署性
5. 证明该模块不绑定 HiVT，而是具备跨 backbone 的迁移潜力

更偏工程地说，这个模块应当成为轨迹预测系统的“健康度分支”：

- 主分支负责预测未来
- 可靠性分支负责判断这次预测能不能信

## 5. 方法定位

### 5.1 它是什么

它是一个可靠性估计模块，而不是新的全模型。

### 5.2 它不是什么

它不是：

- 单纯的 calibration 后处理
- 单纯的 OOD detector
- 单纯的不确定性估计打分器
- 单纯做 test-time adaptation

### 5.3 它和 HiVT 的关系

HiVT 在这个工作中是首个宿主 backbone。真正的方法贡献是“可靠性模块本身”以及围绕它的一整套风险标签、训练目标、评估协议和 shift-aware 实验设计。

### 5.4 即插即用的含义

这里的即插即用不是说只吃最终轨迹坐标、完全黑盒；更合理的定义是：

在不改 backbone 主体建模逻辑的前提下，以后接独立分支接入，通过统一接口读取预测结果和中间特征，再输出风险结果。

因此模块可以读取：

- 候选轨迹 `Y`
- mode logits / mode score `P`
- actor feature
- scene feature
- interaction feature
- 可选的 map-related feature

但不会要求重写原有 encoder / decoder 主干。

## 6. 方法总览

整体框架可以写成：

`Input -> Backbone Predictor -> Predicted Trajectories + Intermediate Features -> Plug-in Reliability Module -> Risk Outputs`

其中：

1. `Backbone Predictor`
   - 例如 HiVT
   - 输出 K 条候选轨迹和中间表征

2. `Plug-in Reliability Module`
   - 利用候选轨迹、场景上下文、交互特征估计可靠性

3. `Risk Outputs`
   - `mode-level risk`
   - `scene-level risk`
   - 可选 `risk-aware reranked mode score`

整个研究的关键点，在于把“预测输出”和“失败风险”这两个层面解耦：

- 主干网络学“未来会怎么走”
- 可靠性分支学“哪些输出更容易失败”

## 7. 模块架构设计

建议采用三段式结构。

### 7.1 Trajectory Reliability Encoder

这部分以单条候选轨迹为中心，对轨迹本身做风险编码。

它关注的不是场景语义，而是这条轨迹从几何和动力学上“看起来是否不稳”。可提取的特征包括：

- 终点偏移趋势
- 全轨迹位移分布
- 速度变化
- 方向变化与曲率变化
- 是否存在不自然形状
- 与其他候选或其他 agent 未来轨迹的潜在冲突线索

它的作用是把“这条轨迹自身像不像一个高风险输出”编码成 mode-level feature。

### 7.2 Interaction-Aware Context Encoder

这部分不只看单条轨迹，而是看当前场景和交互结构。

输入可包括：

- target actor feature
- surrounding agent features
- global scene token
- interaction summary feature
- map context
- 邻域密度、可观测性等辅助指标

它关注的问题包括：

- 当前是否是多车强交互场景
- 是否存在路口、汇入、冲突结构
- 场景复杂度是否较高
- 输入是否不完整
- 地图是否可能存在局部缺失或噪声

这部分的核心意义在于：很多预测失败不是单条轨迹本身坏，而是场景整体就难。

### 7.3 Risk Heads

这一层负责把前面的表征真正变成可监督的风险输出。

#### Mode-level Risk Head

输入：

- trajectory feature
- mode embedding
- actor feature
- interaction feature

输出：

- `r_k in [0,1]`

含义：

- 第 `k` 条候选轨迹失败的概率或风险分数

#### Scene-level Risk Head

输入：

- global / scene feature
- aggregate interaction feature
- map completeness / density descriptor
- mode risk aggregate

输出：

- `r_scene in [0,1]`

含义：

- 当前场景整体的预测风险

### 7.4 Risk-aware Reranking

为了让模块具备更直接的实际价值，可以进一步把风险分数反馈到 mode 选择上。

可采用简单形式：

`p'_k = softmax(log p_k - alpha * r_k)`

直观含义：

- 原始 score 高但风险也高的 mode 会被降权
- 更稳妥的 mode 会被提前

这样模块不只是“报警器”，还可以成为一个轻量的风险感知重排器。

## 8. 模块输入与输出定义

### 8.1 输入

对每个场景，模块接收：

#### 来自 backbone 的直接输出

- `Y = {y_k}`：K 条候选未来轨迹
- `P = {p_k}`：每个 mode 的原始 score / logits

#### 来自 backbone 的中间特征

- target actor feature
- global / scene feature
- interaction summary feature
- mode embedding

#### 可选附加信息

- map-related feature
- neighborhood density
- observation completeness indicator
- corruption / shift indicator

### 8.2 输出

模块输出三类量：

- `r_k`：每个 mode 的风险分数
- `r_scene`：场景级风险分数
- `p'_k`：经风险修正后的候选分布

语义上：

- `r_k` 越高，表示对应 mode 越可能失败
- `r_scene` 越高，表示该场景整体越难预测
- `p'_k` 用于更稳健的下游 mode 选择

## 9. 风险标签设计

这个工作最合理的监督方式不是人工标风险，而是自动构造标签。

### 9.1 Mode-level 标签

可以从以下几个方面定义失败事件：

- `y_fde = 1(FDE_k > tau_fde)`
- `y_ade = 1(ADE_k > tau_ade)`
- `y_offroad = 1(trajectory leaves drivable area)`
- `y_conflict = 1(target prediction stays close to neighbor GT future for m frames)`
- `y_miss = 1(mode misses GT beyond threshold)`

当前默认主标签定义为：

`y_risk = y_fde`

`ADE / miss / conflict / off-road` 保留为辅助统计或显式对照；五类并集只作为 `mode_target_policy=all_union` 的 legacy 对照，不再作为默认主监督。

### 9.2 Scene-level 标签

场景风险当前采用 target-centric 聚合：

- `y_scene = 1(target actor positive mode rate > rho)`

当前推荐 `rho=0.67`。全场景 hard max 只作为 `scene_target_policy=scene_max` 的 legacy 对照保留，不再作为默认 scene label。

### 9.3 MVP 标签建议

当前 conservative fix 已采用：

- `FDE failure`
- `target_mode_rate` scene label
- `target prediction vs neighbor GT future` conflict 代理

`off-road` 与 `conflict` 当前默认不进入主 `mode_targets`，只作为日志和安全代理指标。

## 10. 训练方式

### 10.1 两种接入模式

#### 模式 A：推理后插件

- 训练时不参与 backbone 学习
- 推理时后接做风险估计

优点：

- 简单
- 改动小

缺点：

- 上限偏低
- 风险分支和 backbone 表征耦合较弱

#### 模式 B：联合训练插件

- 风险模块作为后接分支参与联合训练
- 推理时依然可以保持模块独立输出

优点：

- 可靠性头能充分利用 backbone 表征
- 实验更完整，更适合论文

建议第一版论文采用模式 B。

### 10.2 损失函数

总损失可写成：

`L = L_pred + lambda1 * L_mode + lambda2 * L_scene + lambda3 * L_rank + lambda4 * L_calib`

其中：

- `L_pred`：原始轨迹预测损失
- `L_mode`：mode-level risk 分类损失
- `L_scene`：scene-level risk 分类损失
- `L_rank`：风险排序损失
- `L_calib`：可选的校准损失

第一版最小可行版本可简化为：

`L = L_pred + lambda1 * BCE(r_k, y_risk) + lambda2 * BCE(r_scene, y_scene)`

后续增强版可再加入：

- pairwise rank loss（`losses/reliability_losses.py` 中的 `RiskRankLoss`，已实现并通过 `--rank_loss_weight` 接入训练）
- calibration loss（`losses/reliability_losses.py` 中的 `RiskCalibrationLoss`，已实现并通过 `--calib_loss_weight` 接入训练，默认 0.0 = 关闭）
- consistency regularization

### 10.3 Rank Loss 的意义

若两个 mode 中：

- mode i 的真实未来误差明显大于 mode j

则风险分数应满足：

- `r_i > r_j`

这样输出的 risk 才不仅是二分类标签拟合，而是具有更强排序意义。

## 11. 分布偏移与鲁棒性设计

这是这个 idea 非常关键的部分，因为它决定了工作是否真正有部署意义。

目标不是证明模型在所有 shift 下都变得鲁棒，而是证明：

当输入和场景变难时，可靠性分数能更准确地反映失败概率。

建议构造四类 shift：

### 11.1 历史轨迹缺失

- 随机删除历史点
- 删除最近若干帧
- 邻车观测不完整

> **已实现**：`datasets/shift_augment.py` 中的 `HistoryDropout`（逐 actor 随机遮盖最近 k 帧）和 `NeighborDropout`（整体移除邻车历史）。训练时通过 `--shift_history_dropout_p` / `--shift_neighbor_dropout_p` 启用。

### 11.2 地图扰动

- lane polyline jitter
- 删除局部 lane segment
- lane connectivity 断裂

> **已实现**：`datasets/shift_augment.py` 中的 `MapJitter`，支持对 `lane_vectors` / `lane_actor_vectors` 加高斯噪声（`--shift_map_jitter_std`）以及随机删除 lane-actor 边（`--shift_lane_dropout_p`）。

### 11.3 状态噪声

- heading noise
- velocity noise
- localization jitter

> **已实现**：`datasets/shift_augment.py` 中的 `StateNoise`，对历史位置加噪后自动重算位移特征 x，并对 `rotate_angles` 加朝向噪声。通过 `--shift_position_noise_std` / `--shift_heading_noise_std` 启用。

### 11.4 场景难例子集

- 路口
- 汇入
- 高密度交互
- 急转和罕见行为

通过这些构造，可以回答：

- 高风险分数是否对应更高失败率
- shift 增大时风险分数是否同步上升
- 风险分支是否比原始 mode score 更能揭示不可靠输出

## 12. 实验设计

### 12.1 宿主模型

第一阶段：

- HiVT

第二阶段泛化验证：

- QCNet
- MTR 或 Wayformer

### 12.2 数据集

建议：

- 主数据集：Argoverse 2 或 Argoverse 1（若现有工程基于 AV1）
- 补充：Waymo Open Motion Dataset
- 可选：nuScenes Prediction

在当前仓库上下文里，短期最现实的是先基于 Argoverse 1 的 HiVT 代码完成原型。

### 12.3 对比方法

主对比：

- 原始 HiVT
- HiVT + temperature scaling
- HiVT + 简单 uncertainty baseline
- HiVT + reliability module

泛用性对比：

- QCNet
- QCNet + reliability module
- MTR / Wayformer
- MTR / Wayformer + reliability module

### 12.4 指标体系

#### 预测精度

- minADE
- minFDE
- MR

#### 可靠性 / 校准

- ECE（`metrics/reliability_metrics.py` → `ECE`，已接入 val_step，log 为 `val_mode_ECE`）
- Brier Score（`metrics/reliability_metrics.py` → `BrierScore`，已接入 val_step，log 为 `val_mode_BrierScore`）
- NLL（由原始 `val_reg_loss` 的 Laplace NLL 覆盖）

#### 失败检测

- AUROC（`metrics/reliability_metrics.py` → `AUROC`，已接入 val_step，log 为 `val_mode_AUROC`）
- AUPRC（`metrics/reliability_metrics.py` → `AUPRC`，已实现，可按需接入）
- FPR@95TPR（`metrics/reliability_metrics.py` → `FPR95TPR`，已实现，可按需接入）

#### 安全相关代理指标

- off-road violation rate
- future conflict / collision proxy

### 12.5 消融实验

建议至少做：

- 去掉 interaction feature
- 去掉 scene-level risk
- 去掉 mode-level risk
- 去掉 rank loss
- 去掉 shift augmentation
- 仅后处理 vs 联合训练
- 是否做 risk-aware reranking

## 13. 预期创新点

这项工作可以归纳成三条核心创新。

### 创新点 1：即插即用可靠性分支

提出一个不重写 backbone 主干、而是以后接模块形式接入的可靠性估计框架，使现有预测器具备风险输出能力。

### 创新点 2：交互感知失败估计

不是只基于单轨迹误差做二分类，而是融合轨迹几何、场景结构和多车交互信息，提升复杂场景中的失败识别能力。

### 创新点 3：面向分布偏移的统一评估协议

系统地在历史缺失、地图扰动、状态噪声和难场景子集上验证风险估计能力，使工作更接近真实部署问题，而不是只停留在离线精度指标。

## 14. 实际意义

### 14.1 对安全

系统可以更早知道：

- 哪些轨迹不能太信
- 哪些场景应该更保守

### 14.2 对工程部署

为轨迹预测模块增加健康度监控：

- 可监控
- 可解释
- 可降级

### 14.3 对学术

把问题从“轨迹准不准”扩展到：

- 预测是否可信
- 模型是否知道自己什么时候会错

## 15. 风险与挑战

### 15.1 看起来像简单加头

风险在于审稿人可能觉得只是额外接了一个 MLP。

应对方式：

- 强调 interaction-aware
- 强调 mode-level + scene-level 双层风险
- 强调 shift-aware evaluation
- 强调统一 plug-in 接口

### 15.2 风险标签不够 convincing

应对方式：

- 使用自动构造且与安全相关的标签
- 不只用 FDE，还加入 off-road / conflict

### 15.3 泛化证据不足

应对方式：

- 至少补 1 到 2 个额外 backbone

### 15.4 精度不提升但可靠性提升

这不是失败。该类工作本来就应该把“更可信”作为一等目标，而不是只看 minFDE 是否继续下降。

## 16. 最小可行版本

第一版 MVP 已基本完成，当前仓库状态：

- Backbone：HiVT
- 输出：`mode_risk` + `scene_risk` + `reranked_pi` ✅
- 标签：FDE / ADE / miss / conflict / off-road 五类并集 ✅
- 损失：`L_pred + λ1·BCE_mode + λ2·BCE_scene`，可选 `λ3·L_rank + λ4·L_calib` ✅
- 重排：`risk-aware reranking` ✅
- 扰动：`HistoryDropout` / `NeighborDropout` / `StateNoise` / `MapJitter`，通过 `ShiftAugment` 统一接入 ✅
- 指标：`val_minFDE` / `val_mode_AUROC` / `val_mode_BrierScore` / `val_mode_ECE` ✅

这样已经足够构成一篇完整论文的骨架。

## 17. 后续扩展方向

如果第一版成立，后续可以扩展到：

1. 跨 backbone 迁移
2. 与 planner 联动，形成 risk-aware planning
3. 加入 conformal calibration
4. 做 online reliability monitoring
5. 把风险信息反向反馈到 decoder

## 18. 与当前 HiVT 仓库实现的对应关系

结合当前仓库，各模块与文档章节的对应如下：

| 文件 | 内容 | 文档章节 |
|---|---|---|
| `models/hivt.py` | 主干预测器 + 可靠性分支集成 + 训练/验证 step | §5, §10 |
| `models/reliability.py` | TrajectoryReliabilityEncoder / InteractionContextEncoder / ReliabilityModule / 自动标签构造 | §7, §8, §9 |
| `losses/reliability_losses.py` | `RiskRankLoss` / `RiskCalibrationLoss` | §10.2, §10.3 |
| `datasets/shift_augment.py` | `HistoryDropout` / `NeighborDropout` / `StateNoise` / `MapJitter` / `ShiftAugment` | §11.1–11.3 |
| `metrics/reliability_metrics.py` | `BrierScore` / `ECE` / `AUROC` / `AUPRC` / `FPR95TPR` | §12.4 |
| `training_presets.py` | `build_reliability_train_args`（预设超参） | §10 |
| `run_single_gpu.sh` | `train_reliability` / `train_reliability_shift` / `eval` 入口 | §12 |

**当前已完成：**

- `mode_risk` / `scene_risk` / `reranked_pi`（文档§7.3/7.4）
- 五类自动风险标签：FDE / ADE / miss / conflict / off-road（§9）
- 联合训练模式 B，完整损失 `L_pred + λ1·L_mode + λ2·L_scene + λ3·L_rank + λ4·L_calib`（§10）
- 四类分布偏移增强，统一通过 `ShiftAugment` + `--shift_*` CLI 参数接入（§11）
- 五项可靠性评估指标，`AUROC`/`BrierScore`/`ECE` 已接入 `validation_step`（§12.4）
- 通过 `--use_reliability true` 开启联合训练，`--rank_loss_weight` / `--calib_loss_weight` 默认 0 可选开启
- `train_reliability_shift` 快捷入口，默认偏移强度可通过环境变量覆盖

**仍需推进（论文版本）：**

- 更精确的 off-road 几何标签（目前为 lane 中心线距离代理，非 drivable-area polygon）
- 更强的 conflict 时序/交互标签（当前已从轨迹最小距离改为 target prediction vs neighbor GT future 的持续接近代理；论文版本可继续增强为更精确碰撞语义）
- §11.4 场景难例子集的系统化构造与评估协议
- 多 backbone 泛化验证（QCNet / MTR / Wayformer）

**训练 log 字段（`train_*` / `val_*` 前缀）：**

- `{reg/risk/scene/rank/calib}_loss`：各项损失
- `{mode/fde/ade/conflict/offroad/scene}_risk_target_rate`：自动标签正样本率（监控标签偏斜）
- `{mode/scene}_risk_pred_mean`：预测风险均值（监控头是否塌缩）
- `mode_{AUROC/BrierScore/ECE}`：epoch 级可靠性评估指标

## 19. 适合汇报的一段总结

我们计划研究一个面向自动驾驶轨迹预测的即插即用可靠性模块。与现有方法主要关注预测精度不同，这项工作关注模型在复杂交互和分布偏移条件下，是否能够识别自身预测的失败风险。该模块可以接入 HiVT 等现有轨迹预测 backbone，在不显著影响原始预测性能的前提下，对候选轨迹和场景级风险进行估计，并可进一步通过风险感知重排输出更稳健的候选分布。最终目标是让轨迹预测模型不仅“会预测”，还“知道自己什么时候不可靠”，从而为下游规划提供更可信的决策依据。

## 20. 阶段性实验记录（2026-06-26）

### 20.1 实验目的

先做一组不含消融的小预算对照，只比较：

- 原始宿主模型 `HiVT-64`
- `HiVT-64 + reliability plugin`

目标不是得出论文最终结论，而是先回答：

- 当前插件是否显著破坏宿主模型
- 当前插件是否已经能带来可见收益
- 当前 reliability 分支是否真的学到了有判别性的风险输出

### 20.2 实验设置

- 数据：Argoverse 1（当前仓库默认数据）
- 模型：
  - Baseline：`HiVT-64`
  - Plugin：`HiVT-64 + reliability module`
- 训练预算：
  - `max_epochs = 1`
  - `limit_train_batches = 64`
  - `limit_val_batches = 16`
  - `train_batch_size = 8`
  - `val_batch_size = 8`
- reliability 训练配置：
  - `use_reliability = true`
  - `reliability_hidden_dim = 128`
  - `reliability_rerank_alpha = 0.5`
  - `reliability_loss_weight = 1.0`
  - `scene_loss_weight = 0.5`
  - `rank_loss_weight = 0.0`
  - `calib_loss_weight = 0.0`

说明：

- 这是“快速判断当前方向是否可行”的阶段性实验，不是正式主结果。
- 两组实验训练预算完全一致，因此可以做当前版本下的直接对比。

### 20.3 实验结果

#### Baseline：HiVT-64

- `val_reg_loss = 2.7914`
- `val_minADE = 13.2153`
- `val_minFDE = 25.5616`
- `val_minMR = 1.0000`

#### Plugin：HiVT-64 + reliability module

- `val_reg_loss = 2.8305`
- `val_minADE = 13.3392`
- `val_minFDE = 25.4756`
- `val_minMR = 0.9922`
- `val_risk_loss = 0.000375`
- `val_scene_loss = 0.000589`
- `val_mode_AUROC = 0.0`
- `val_mode_BrierScore = 1.41e-7`
- `val_mode_ECE = 3.34e-4`

### 20.4 当前阶段结论

基于这组小预算对照，可以得到当前版本的直接结论：

1. 当前 reliability 插件 **没有明显破坏** 宿主 HiVT。

2. 当前 reliability 插件 **没有表现出清晰、稳定的预测性能提升**：
   - `minFDE` 有极小幅改善：`25.56 -> 25.48`
   - `MR` 有极小幅改善：`1.0000 -> 0.9922`
   - 但 `minADE` 略差：`13.22 -> 13.34`
   - `val_reg_loss` 也略差：`2.79 -> 2.83`

3. 因此目前更合理的表述不是“插件提升了宿主模型预测精度”，而是：
   - 当前插件对宿主的预测性能影响较小
   - 对 mode 选择可能有轻微正向作用
   - 但尚未形成有说服力的精度收益

### 20.5 更关键的历史发现：risk 标签曾经塌缩

本次实验里，reliability 分支曾暴露出一个比精度更重要的问题：

- `train_mode_risk_target_rate = 1.0`
- `val_mode_risk_target_rate = 1.0`
- `train_conflict_risk_target_rate = 1.0`
- `val_conflict_risk_target_rate = 1.0`
- `train_scene_risk_target_rate = 1.0`
- `val_scene_risk_target_rate = 1.0`
- `train_mode_risk_pred_mean ≈ 0.98 ~ 1.00`
- `val_mode_risk_pred_mean ≈ 0.9996`
- `train_scene_risk_pred_mean ≈ 0.99 ~ 1.00`
- `val_scene_risk_pred_mean ≈ 0.9994`

这说明在当时的标签定义与阈值设置下：

- 风险标签几乎全是正样本
- risk head 学到的是“几乎所有 mode 都高风险”
- reliability 分支出现明显塌缩

因此第 20 节中的 `AUROC / Brier / ECE` 数值不能被直接解读成插件已经具备可靠的风险估计能力，因为当时标签分布本身已经失去判别性。该问题已在第 21 节通过 conservative fix 修复。

### 20.6 当前最合理的判断

截至 2026-06-26，修复前插件状态可以总结为：

- 结构层面：已打通
- 训练层面：已可联合训练
- 精度层面：尚未证明明显收益
- 可靠性层面：当时标签设计过密，risk learning 出现塌缩，未形成可信结论

### 20.7 后续处理情况

第 20 节提出的下一步是先修正 reliability supervision 的可判别性，重点包括：

1. 调整 `conflict` 风险阈值，降低过密正样本
2. 调整 `scene risk` 聚合方式，避免场景标签恒为 1
3. 改进 `off-road` 代理标签，使其不只依赖 lane 中心线距离
4. 在标签分布恢复正常后，再重新比较：
   - 宿主模型 vs 插件模型
   - reranking 前后表现
   - shift 条件下的失败检测能力

这些标签修复工作已经在第 21 节完成。当前新的优先级不再是修复标签塌缩，而是验证 `freeze backbone` 与 `reranking-only` 是否能在不破坏宿主预测指标的前提下发挥可靠性模块价值。

## 21. 标签塌缩修复与阶段性复验（2026-06-27）

### 21.1 修复动机

第 20 节实验暴露的核心问题不是插件结构无法训练，而是自动风险标签协议过密：

- `mode_risk_target_rate = 1.0`
- `conflict_risk_target_rate = 1.0`
- `scene_risk_target_rate = 1.0`

这会让 reliability head 的最优策略退化成“所有 mode / scene 都预测为高风险”，导致 `AUROC / Brier / ECE` 失去解释意义。

因此本轮修复优先解决 supervision collapse，而不是扩大训练规模。

### 21.2 已完成的 conservative fix

本轮实现了第一阶段保守修复：

- 主 `mode_targets` 默认改为 `fde_only`，不再默认并入 `ADE / miss / conflict / offroad`。
- `conflict_targets` 改为 target actor 预测轨迹与邻车 GT future 的同时间步持续接近代理。
- `scene_targets` 支持 `target_best_mode_fail`、`target_mode_rate`、`scene_rate`、`scene_max`。
- `agent_index` 显式传入标签构造函数，保证 target-centric 策略不会退化成全 actor 聚合。
- 新增 `scripts/audit_reliability_targets.py`，支持加载宿主 checkpoint 做无反传标签体检。
- 训练与验证日志补齐 `ADE / miss / AUPRC` 等指标。

### 21.3 Label audit 结果

使用已训练宿主 checkpoint：

```text
/home/lbh/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt
```

在 `val` split 上审计前 4 个 batch，默认 `target_best_mode_fail` 得到：

```text
mode_positive_rate          = 0.4085
fde_positive_rate           = 0.4085
miss_positive_rate          = 0.2156
conflict_positive_rate      = 0.0008
offroad_positive_rate       = 0.1815
scene_positive_rate         = 0.0625
target_actor_mode_rate      = 0.6823
target_best_mode_fail_rate  = 0.0625
```

结论：

- 主 `mode` 标签已经恢复到可训练区间。
- `conflict` 已降至稀疏安全事件区间，且不再污染主监督。
- `target_best_mode_fail` 作为 scene label 过于保守，scene 正样本率偏低。

进一步扫描 `scene_target_policy=target_mode_rate`：

| scene policy | rho | scene_positive_rate | 结论 |
|---|---:|---:|---|
| `target_mode_rate` | 0.50 | 0.7500 | 偏高 |
| `target_mode_rate` | 0.67 | 0.4063 | 合适 |
| `target_mode_rate` | 0.75 | 0.4063 | 合适 |

阶段性推荐配置：

```text
mode_target_policy=fde_only
scene_target_policy=target_mode_rate
risk_scene_rate_threshold=0.67
risk_fde_threshold=2.0
risk_miss_threshold=4.0
risk_conflict_threshold=1.0
risk_conflict_min_frames=2
risk_conflict_scope=target_to_neighbors
reliability_rerank_alpha=0.0
scene_loss_weight=0.2
rank_loss_weight=0.0
calib_loss_weight=0.0
```

### 21.4 小预算训练复验

实验设置：

- 初始化：宿主 `HiVT-64` checkpoint，`reliability_module` 随机初始化。
- 预算：`max_epochs=1`，`limit_train_batches=64`，`limit_val_batches=16`。
- batch size：`train_batch_size=8`，`val_batch_size=8`。
- 配置：`fde_only + target_mode_rate(rho=0.67)`。
- 日志目录：`/home/lbh/HiVT/runs/hivt_reliability/stage1_fix_budget64_fdeonly_targetmode067`。

关键结果：

```text
train_mode_risk_target_rate     = 0.4386
train_scene_risk_target_rate    = 0.4805
train_conflict_risk_target_rate = 0.0007
train_mode_risk_pred_mean       = 0.4569
train_scene_risk_pred_mean      = 0.4815

val_mode_risk_target_rate       = 0.4571
val_scene_risk_target_rate      = 0.3359
val_conflict_risk_target_rate   = 0.0007
val_mode_risk_pred_mean         = 0.5314
val_scene_risk_pred_mean        = 0.4160

val_mode_AUROC                  = 0.8522
val_mode_AUPRC                  = 0.7964
val_mode_BrierScore             = 0.1592
val_mode_ECE                    = 0.0691

val_minADE                      = 0.6223
val_minFDE                      = 0.9509
val_minMR                       = 0.0703
```

阶段性结论：

1. 标签塌缩已经解除。`mode / scene / conflict` 三类标签都处于可解释区间。
2. `mode` 风险监督具备明显判别性，`AUROC=0.8522`、`AUPRC=0.7964`。
3. risk prediction 没有继续塌缩到全 1。
4. 当前阶段可以说 reliability supervision 已恢复有效，但还不能宣称插件显著提升预测精度。

### 21.5 更完整小预算对照

为确认上述结果不是 `64/16` 小样本偶然现象，进一步使用相同推荐配置进行 `128/32` 小预算复验。

实验设置：

- 宿主 checkpoint：`/home/lbh/HiVT/checkpoints/HiVT-64/checkpoints/epoch=63-step=411903.ckpt`
- Baseline：直接验证宿主 `HiVT-64` checkpoint。
- Plugin：宿主 checkpoint 初始化 backbone，`reliability_module` 随机初始化。
- Plugin 训练预算：`max_epochs=1`，`limit_train_batches=128`，`limit_val_batches=32`。
- Baseline 验证预算：`limit_val_batches=32`。
- batch size：`train_batch_size=8`，`val_batch_size=8`。
- Plugin 配置：`fde_only + target_mode_rate(rho=0.67)`。
- Plugin 日志目录：`/home/lbh/HiVT/runs/hivt_reliability/stage1_fix_budget128_fdeonly_targetmode067`。

Baseline 结果：

```text
val_reg_loss = -0.3638
val_minADE   = 0.5738
val_minFDE   = 0.8031
val_minMR    = 0.0508
```

Plugin 结果：

```text
val_reg_loss                  = -0.2850
val_minADE                    = 0.6110
val_minFDE                    = 0.8664
val_minMR                     = 0.0703

val_mode_risk_target_rate     = 0.4591
val_scene_risk_target_rate    = 0.3945
val_conflict_risk_target_rate = 0.0007
val_mode_risk_pred_mean       = 0.4738
val_scene_risk_pred_mean      = 0.6350

val_mode_AUROC                = 0.8735
val_mode_AUPRC                = 0.8358
val_mode_BrierScore           = 0.1445
val_mode_ECE                  = 0.0302
```

对照结论：

1. 标签修复效果在更大的小预算设置下仍稳定，`mode / scene / conflict` 标签均处于可解释区间。
2. 可靠性判别能力进一步增强：
   - `AUROC: 0.8522 -> 0.8735`
   - `AUPRC: 0.7964 -> 0.8358`
   - `ECE: 0.0691 -> 0.0302`
3. 当前 plugin 联合训练会轻微牺牲宿主原始预测指标：
   - `minADE: 0.5738 -> 0.6110`
   - `minFDE: 0.8031 -> 0.8664`
   - `MR: 0.0508 -> 0.0703`

因此当前阶段的结论应调整为：

```text
标签协议修复已经成功，可靠性监督恢复可判别性；但在当前联合训练设置下，插件尚未证明能提升宿主预测精度，下一步应优先做 reranking-only 或冻结 backbone 的对照，避免 reliability loss 干扰已经训练好的预测主干。
```

后续优先级：

1. 冻结 HiVT backbone，只训练 `reliability_module`。
2. 单独评估 `original pi` 与 `reranked_pi`，验证 risk-aware reranking 是否改善 `minFDE / MR`。
3. 在确认不破坏宿主预测指标后，再进入 shift evaluation 和更大预算实验。
