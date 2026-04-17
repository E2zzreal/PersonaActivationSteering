# 2026-04-12 Gate 与思考泄露归因设计

日期：2026-04-12
最后验证日期：2026-04-12
状态：提议（已通过讨论确认）

## 1. 背景

当前项目已确认：
- 训练数据中显式伪思考比例极低；
- baseline 泄露很少；
- 注入模型，尤其后期阶段模型，泄露明显更高；
- 保守型评估侧止损已接近规则上限。

因此，当前最值得优先验证的归因假设是：

> 是否是 gate / 注入强度行为，使 injected 模型更容易进入 planning / reasoning 子空间并外显为思考泄露？

## 2. 目标

本设计的目标是回答以下问题：

1. 泄露样本的 gate 分布，是否与非泄露样本显著不同；
2. Stage 1 / Stage 2 / Stage 3 的 gate 行为演化，是否与泄露率上升一致；
3. 泄露是否集中对应某些层的高 gate；
4. 后续应优先做：
   - gate 约束优化；
   - 还是转向注入层位置分析。

## 3. 非目标

本轮不做：
- 直接修改 gate 策略；
- 直接修改注入层位置；
- 重新训练模型；
- 引入新的损失函数。

本轮只做：
- 样本级 gate 记录；
- 泄露与 gate 的相关分析；
- 阶段间对照归因。

## 4. 为什么先查 gate

在 PersonaSteer 当前结构中，gate 是最直接控制注入强度和层间分配的组件。既然：
- baseline 基本不泄露；
- injected 模型明显泄露；
- 数据侧污染很低；

那么最自然的怀疑就是：
- gate 是否过高；
- gate 是否过于集中在少数层；
- gate 在 stage 演化中是否逐渐把模型推向“先规划再输出”的状态。

## 5. 最小归因实验集

建议先使用小规模但足够有趋势的信息量实验：
- baseline
- stage1
- stage2_v2
- stage3_v2

在同一批样本、同一生成配置下，记录每条样本的：
- 泄露标记
- gate 分布
- 回复文本

首轮样本量可控制在：
- 50–100 条样本

目的是先判断趋势是否明显，而不是一开始追求统计学完整性。

## 6. 核心指标

### 6.1 样本级指标
每条样本记录：
- `gate_values`（逐层）
- `gate_mean`
- `gate_std`
- `gate_max`
- `gate_min`
- `leak_detected`
- `stage`

### 6.2 分组对照指标
对每个 checkpoint：
- `leak` 组 gate mean/std
- `non-leak` 组 gate mean/std
- 层级平均 gate 分布

### 6.3 阶段演化指标
比较：
- stage1 → stage2 → stage3
- 泄露率变化
- gate 均值变化
- gate 集中层变化

## 7. 实现落点

### 7.1 主评估脚本
优先接入：
- `scripts/v4_eval_qwen3.py`

因为它已经：
- 作为当前主评估入口之一；
- 已具备 dualtrack 泄露输出；
- 容易扩展样本级附加字段。

### 7.2 模型侧数据来源
优先使用：
- `src/models/injection.py` 中的 `current_gate_values`

理由：
- 无需额外大改模型结构；
- gate 已在注入阶段缓存；
- 直接读取即可。

## 8. 数据流设计

```text
model.generate()
  -> raw_response / clean_response / leak_detected
  -> 读取 model.injection.current_gate_values
  -> 计算 gate_mean/std/max/min
  -> 写入样本级结果
  -> 后处理分析脚本做聚合统计
```

## 9. 输出产物

### 9.1 样本级结果
在评估结果中新增：
- `gate_values`
- `gate_mean`
- `gate_std`
- `gate_max`
- `gate_min`

### 9.2 聚合分析
新增分析脚本，例如：
- `scripts/analyze_gate_leak_correlation.py`

输出：
- `results/gate_analysis_<timestamp>/summary.json`
- `results/gate_analysis_<timestamp>/heatmap.png` 或文本摘要
- `results/gate_analysis_<timestamp>/examples.json`

## 10. 验证标准

本轮成功标准不是“已经修好 gate”，而是“归因是否清晰”。最低标准：

1. 能稳定拿到样本级 gate 数据；
2. 能把 `leak_detected` 和 gate 统计绑定起来分析；
3. 至少能明确以下其一：
   - 泄露样本对应更高或更偏置的 gate；
   - gate 与泄露无明显关系，应转向注入层位置分析。

## 11. 与 B（注入层位置）的衔接

A 的作用不是取代 B，而是筛选 B 的优先级：

### 若发现少数层 gate 异常偏高
则 B 重点检查：
- 是否这些层的选择本身有问题；
- 是否应避开这些层或降低其权重。

### 若发现所有层整体偏高
则优先考虑 gate 约束，而不是先改层位置。

### 若发现 gate 与泄露关系很弱
则 B 升级为主线，转向“层位置是否天然更靠近 reasoning 子空间”。

## 12. 后续决策逻辑

### 若 gate 与泄露强相关
下一步：
- 设计 gate 约束 / clipping / regularization 调整方案

### 若 gate 与泄露弱相关
下一步：
- 设计注入层位置归因方案

## 13. 一句话结论

> Gate 是当前最值得先查的注入机制归因入口；只要拿到样本级 gate × 泄露对照，我们就能判断下一步该优先修“注入强度”还是“注入位置”。
