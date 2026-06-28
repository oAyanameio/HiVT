# 可靠性实验结果汇总

数据来源：`/home/lbh/HiVT/runs/hivt_reliability/` 下保留的 TensorBoard event 文件。

## 指标含义与统计口径

### 1. 数据集与时间窗口

- 数据集：`ArgoverseV1Dataset` 验证集
- 历史帧数：`20`
- 未来帧数：`30`
- 采样频率：`10 Hz`
- 历史时长：`2.0 秒`
- 预测时长：`3.0 秒`

当前代码里，每个样本总共使用 `50` 个时间戳：

- 历史部分：`timestamps[:20]`
- 未来部分：`timestamps[20:]`

对应代码：

- [argoverse_v1_dataset.py](/home/lbh/HiVT/datasets/argoverse_v1_dataset.py:167)
- [argoverse_v1_dataset.py](/home/lbh/HiVT/datasets/argoverse_v1_dataset.py:229)

### 2. 这些 `val_*` 指标是不是在整个验证集上算的？

是。

这些 `val_*` 指标都不是单个 batch 的数，而是在整个 `val_dataloader()` 上逐 batch 累积，最后在一个 validation epoch 结束时输出。

对应代码：

- [argoverse_v1_datamodule.py](/home/lbh/HiVT/datamodules/argoverse_v1_datamodule.py:60)
- [hivt.py](/home/lbh/HiVT/models/hivt.py:338)

### 3. 预测精度指标

#### `val_minADE`

`val_minADE` 不是在所有 actor 上算的，而是**只在每个场景的目标车 `AGENT` 上算**。

当前代码流程是：

1. 用 `data['agent_index']` 取出每个场景的目标车
2. 模型输出 `K=6` 条预测轨迹
3. 先按终点误差最小选出 best mode
4. 再用这条 best mode 计算整条未来轨迹的 ADE
5. 最后在整个验证集所有场景上取平均

当前实现对应的含义是：

```text
best mode = argmin_k FDE_k
ADE = mean_t ||y_pred(t) - y_gt(t)||_2
val_minADE = 验证集所有场景目标车的平均值
```

对应代码：

- [hivt.py](/home/lbh/HiVT/models/hivt.py:340)
- [ade.py](/home/lbh/HiVT/metrics/ade.py:32)

#### `val_minFDE`

同样只在目标车上计算。

流程是先选出终点误差最小的 best mode，再计算这条轨迹最后一个未来时刻的终点误差：

```text
best mode = argmin_k FDE_k
FDE = ||y_pred(T) - y_gt(T)||_2
val_minFDE = 验证集所有场景目标车的平均值
```

对应代码：

- [hivt.py](/home/lbh/HiVT/models/hivt.py:343)
- [fde.py](/home/lbh/HiVT/metrics/fde.py:32)

#### `val_minMR`

同样只在目标车上计算。

当前实现里 miss 的定义是：

```text
MR = mean( 1(FDE > 2.0 m) )
```

也就是 best mode 的终点误差如果大于 `2.0m`，这个样本就记为 miss。

对应代码：

- [hivt.py](/home/lbh/HiVT/models/hivt.py:344)
- [mr.py](/home/lbh/HiVT/metrics/mr.py:16)

### 4. 可靠性指标

#### `val_mode_AUROC`

这是 **mode 级失败检测** 的 ROC-AUC。

- 样本单位：一个 `(actor, mode)` 对
- 样本范围：所有有效 actor，经 `valid_mask` 过滤
- 分数：预测的 `mode_risk`
- 标签：`mode_targets`

直观上，它表示：

> 随机抽一个正样本 mode 和一个负样本 mode，模型给正样本更高风险分数的概率有多大。

对应代码：

- [hivt.py](/home/lbh/HiVT/models/hivt.py:407)
- [reliability_metrics.py](/home/lbh/HiVT/metrics/reliability_metrics.py:120)

#### `val_mode_AUPRC`

这是同一个 mode 级失败检测任务上的 PR-AUC。

- 样本单位：一个 `(actor, mode)` 对
- 样本范围：所有有效 actor，经 `valid_mask` 过滤

当正样本比例不均衡时，AUPRC 往往比 AUROC 更敏感。

对应代码：

- [hivt.py](/home/lbh/HiVT/models/hivt.py:408)
- [reliability_metrics.py](/home/lbh/HiVT/metrics/reliability_metrics.py:128)

#### `val_mode_BrierScore`

这是预测风险概率与二值失败标签之间的均方误差：

```text
Brier = mean( (risk - target)^2 )
```

- 样本单位：一个 `(actor, mode)` 对
- 样本范围：所有有效 actor，经 `valid_mask` 过滤

越低越好。

对应代码：

- [hivt.py](/home/lbh/HiVT/models/hivt.py:409)
- [reliability_metrics.py](/home/lbh/HiVT/metrics/reliability_metrics.py:19)

#### `val_mode_ECE`

这是 mode 级风险概率的校准误差。

- 样本单位：一个 `(actor, mode)` 对
- 样本范围：所有有效 actor，经 `valid_mask` 过滤
- 当前实现：`10` 个 bin

当前代码里的定义是：

```text
ECE = Σ_b (n_b / N) * |avg_conf_b - avg_target_b|
```

越低越好。

对应代码：

- [hivt.py](/home/lbh/HiVT/models/hivt.py:410)
- [reliability_metrics.py](/home/lbh/HiVT/metrics/reliability_metrics.py:42)

### 5. 风险标签命中率

#### `val_mode_risk_target_rate`

这是主 `mode_targets` 的正样本率，统计范围是所有有效 `(actor, mode)` 对。

在当前保留的实验设置里：

```text
mode_target_policy = fde_only
```

所以它本质上就是：

```text
1(FDE_k > risk_fde_threshold)
```

这个标签在所有有效 `(actor, mode)` 上的平均值。

对应代码：

- [reliability.py](/home/lbh/HiVT/models/reliability.py:387)
- [reliability.py](/home/lbh/HiVT/models/reliability.py:491)

#### `val_fde_risk_target_rate`

这是原始 FDE 风险标签的正样本率，统计范围也是所有有效 `(actor, mode)` 对。

当前阈值：

- `risk_fde_threshold = 2.0`

对应代码：

- [reliability.py](/home/lbh/HiVT/models/reliability.py:54)
- [reliability.py](/home/lbh/HiVT/models/reliability.py:491)

#### `val_ade_risk_target_rate`

这是原始 ADE 风险标签的正样本率：

```text
1(ADE_k > ade_threshold)
```

统计范围仍然是所有有效 `(actor, mode)` 对。

对应代码：

- [reliability.py](/home/lbh/HiVT/models/reliability.py:72)
- [reliability.py](/home/lbh/HiVT/models/reliability.py:491)

#### `val_miss_risk_target_rate`

这是 miss 风险标签的正样本率：

```text
1(FDE_k > miss_threshold)
```

在保留的 stage-1 实验里：

- `risk_miss_threshold = 4.0`

对应代码：

- [reliability.py](/home/lbh/HiVT/models/reliability.py:88)
- [reliability.py](/home/lbh/HiVT/models/reliability.py:491)

#### `val_conflict_risk_target_rate`

这是 conflict 标签的正样本率，统计范围是所有有效 `(actor, mode)` 对。

在保留实验里，conflict 的定义是：

- `risk_conflict_scope = target_to_neighbors`
- 用目标车预测未来和邻车 GT 未来比较
- 距离低于阈值
- 至少连续 `risk_conflict_min_frames = 2` 帧接近

它是一个 mode 级安全代理指标，不是真实碰撞标注。

对应代码：

- [reliability.py](/home/lbh/HiVT/models/reliability.py:219)
- [reliability.py](/home/lbh/HiVT/models/reliability.py:491)

#### `val_offroad_risk_target_rate`

这是 off-road 代理标签的正样本率，统计范围是所有有效 `(actor, mode)` 对。

当前实现里，如果预测轨迹与局部 lane support 的最小距离大于设定阈值，就记为正样本。

对应代码：

- [reliability.py](/home/lbh/HiVT/models/reliability.py:290)
- [reliability.py](/home/lbh/HiVT/models/reliability.py:491)

#### `val_scene_risk_target_rate`

这是 scene 级标签的正样本率，统计范围是整个验证集的场景。

在当前保留实验里：

```text
scene_target_policy = target_mode_rate
risk_scene_rate_threshold = 0.67
```

也就是说：

> 如果某个场景里目标车的正样本 mode 比例超过 `67%`，这个 scene 就记为正样本。

对应代码：

- [reliability.py](/home/lbh/HiVT/models/reliability.py:157)
- [reliability.py](/home/lbh/HiVT/models/reliability.py:478)

### 6. 风险预测均值

#### `val_mode_risk_pred_mean`

这是 validation 过程中，所有 actor、所有 mode 的 `mode_risk` 预测均值。

对应代码：

- [hivt.py](/home/lbh/HiVT/models/hivt.py:404)

#### `val_scene_risk_pred_mean`

这是 validation 过程中，所有场景的 `scene_risk` 预测均值。

对应代码：

- [hivt.py](/home/lbh/HiVT/models/hivt.py:405)

## 保留的实验

### `stage1_fix_budget64_fdeonly_targetmode067`

配置要点：

- `mode_target_policy = fde_only`
- `scene_target_policy = target_mode_rate`
- `risk_scene_rate_threshold = 0.67`
- `risk_conflict_scope = target_to_neighbors`
- `reliability_rerank_alpha = 0.0`

关键验证指标：

| 指标 | 数值 |
| --- | ---: |
| `val_minADE` | `0.622291` |
| `val_minFDE` | `0.950928` |
| `val_minMR` | `0.0703125` |
| `val_mode_AUROC` | `0.852163` |
| `val_mode_AUPRC` | `0.796411` |
| `val_mode_BrierScore` | `0.159168` |
| `val_mode_ECE` | `0.0691175` |
| `val_risk_loss` | `0.478258` |
| `val_scene_loss` | `0.646788` |
| `val_mode_risk_target_rate` | `0.45711` |
| `val_scene_risk_target_rate` | `0.335938` |
| `val_mode_risk_pred_mean` | `0.531409` |
| `val_scene_risk_pred_mean` | `0.415996` |

子标签命中率：

| 指标 | 数值 |
| --- | ---: |
| `val_fde_risk_target_rate` | `0.45711` |
| `val_ade_risk_target_rate` | `0.456845` |
| `val_miss_risk_target_rate` | `0.261163` |
| `val_conflict_risk_target_rate` | `0.000723963` |
| `val_offroad_risk_target_rate` | `0.230842` |

### `stage1_fix_budget128_fdeonly_targetmode067`

配置要点：

- `mode_target_policy = fde_only`
- `scene_target_policy = target_mode_rate`
- `risk_scene_rate_threshold = 0.67`
- `risk_conflict_scope = target_to_neighbors`
- `reliability_rerank_alpha = 0.0`

关键验证指标：

| 指标 | 数值 |
| --- | ---: |
| `val_minADE` | `0.610951` |
| `val_minFDE` | `0.866362` |
| `val_minMR` | `0.0703125` |
| `val_mode_AUROC` | `0.873523` |
| `val_mode_AUPRC` | `0.835783` |
| `val_mode_BrierScore` | `0.144478` |
| `val_mode_ECE` | `0.0301806` |
| `val_risk_loss` | `0.43723` |
| `val_scene_loss` | `0.788139` |
| `val_mode_risk_target_rate` | `0.459108` |
| `val_scene_risk_target_rate` | `0.394531` |
| `val_mode_risk_pred_mean` | `0.473831` |
| `val_scene_risk_pred_mean` | `0.634967` |

子标签命中率：

| 指标 | 数值 |
| --- | ---: |
| `val_fde_risk_target_rate` | `0.459108` |
| `val_ade_risk_target_rate` | `0.456631` |
| `val_miss_risk_target_rate` | `0.264216` |
| `val_conflict_risk_target_rate` | `0.000677719` |
| `val_offroad_risk_target_rate` | `0.223118` |

## 当前最佳实验

按当前验证结果看，表现最好的保留实验是：

- `stage1_fix_budget128_fdeonly_targetmode067`

主要原因：

- `val_minADE` 更低
- `val_minFDE` 更低
- `val_mode_AUROC` 更高
- `val_mode_AUPRC` 更高
- `val_mode_BrierScore` 更低
- `val_mode_ECE` 更低

## 当前已知问题

- `val_conflict_risk_target_rate` 在两个 run 里都接近 0，说明 conflict 目前还不能作为有效主监督信号
- `stage1_fix_budget128_fdeonly_targetmode067` 里，`val_scene_risk_pred_mean` 明显高于 `val_scene_risk_target_rate`，scene risk 的校准仍然弱于 mode risk
