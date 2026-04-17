# Probing 实验优化建议

**日期**: 2026-04-10
**基于**: Qwen3-4B Probing 实验 (100样本)

---

## 1. 当前配置问题诊断

### 1.1 当前配置
```yaml
inject_layers: [8, 9, 10, 11, 12, 13, 14, 15]  # 8层
```

### 1.2 问题分析

| 特质 | 峰值层 | 相关性 | 当前配置覆盖 | 问题 |
|------|--------|--------|-------------|------|
| neuroticism | 10 | **0.395** | ✓ | 已覆盖 |
| introversion | 28 | **0.393** | **✗** | **未覆盖** |
| gender | 3 | **0.377** | **✗** | **未覆盖** |
| openness | 15 | 0.324 | ✓ | 已覆盖 |
| conscientiousness | 0 | 0.326 | **✗** | **未覆盖** |
| agreeableness | 11 | 0.242 | ✓ | 已覆盖 |

**结论**: 当前配置错过了 3 个重要特质的峰值层，包括相关性最高的 introversion 和 gender。

---

## 2. 优化方案

### 方案A: 精简高效 (6层)

```yaml
inject_layers: [0, 3, 10, 11, 15, 28]
num_inject_layers: 6
```

| 特质 | 峰值层 | 是否覆盖 |
|------|--------|---------|
| conscientiousness | 0 | ✓ |
| gender | 3 | ✓ |
| neuroticism | 10 | ✓ |
| agreeableness | 11 | ✓ |
| openness | 15 | ✓ |
| introversion | 28 | ✓ |

**优点**:
- 减少 25% 层数 (8→6)
- 覆盖所有特质峰值层
- 计算效率高

**缺点**:
- 缺少冗余，可能不够稳定
- 单层失效影响大

**配置文件**: `configs/train_stage1_qwen3_probing_minimal.yaml`

---

### 方案B: 稳定可靠 (11层) - 推荐

```yaml
inject_layers: [0, 2, 3, 4, 9, 10, 11, 15, 27, 28, 29]
num_inject_layers: 11
```

**选择策略**:
- 高相关性特质 (>0.35): neuroticism, introversion, gender → 峰值±1
- 低相关性特质 (≤0.35): openness, conscientiousness, agreeableness → 只选峰值

| 特质 | 相关性 | 选择层数 | 选择层 |
|------|--------|---------|--------|
| neuroticism | 0.395 | 3 | [9, 10, 11] |
| introversion | 0.393 | 3 | [27, 28, 29] |
| gender | 0.377 | 3 | [2, 3, 4] |
| openness | 0.324 | 1 | [15] |
| conscientiousness | 0.326 | 1 | [0] |
| agreeableness | 0.242 | 1 | [11] |

**优点**:
- 高相关性特质有冗余，更稳定
- 低相关性特质节省资源
- 平衡效率与稳定性

**缺点**:
- 层数从 8 增加到 11
- 计算量略有增加

**配置文件**: `configs/train_stage1_qwen3_probing_optimized.yaml`

---

### 方案C: 特质分离 (3层) - 车机助手专用

```yaml
# 目标: 降低神经质
inject_layers: [9, 10, 11]
num_inject_layers: 3
```

**适用场景**:
- 车机助手: 专注稳定性 (神经质)
- 聊天助手: 专注共情 (宜人性) → 需要深层 [30, 31, 32]

**优点**:
- 层数最少 (3层)
- 目标明确，干预精准
- 计算效率最高

**缺点**:
- 只能干预单一特质
- 需要针对不同场景训练不同模型

**配置文件**: `configs/train_stage1_qwen3_neuroticism.yaml`

---

## 3. 是否可以减少层数？

### 3.1 答案: 可以，但要选对层

| 方案 | 层数 | vs 当前 | 覆盖率 | 稳定性 | 推荐场景 |
|------|------|--------|--------|--------|---------|
| 当前 | 8 | - | 50% | 中 | 不推荐 |
| 方案A | **6** | **-25%** | **100%** | 低 | 资源受限 |
| 方案B | 11 | +38% | 100% | **高** | **通用推荐** |
| 方案C | **3** | **-63%** | 单特质 | 中 | 特定场景 |

### 3.2 关键结论

1. **当前配置不是层数问题，是选层问题**
   - 8层足够，但选错了层
   - [8-15] 错过了 0, 3, 28 三个关键层

2. **可以减少到 6 层**
   - 方案A: [0, 3, 10, 11, 15, 28]
   - 覆盖所有特质峰值层
   - 减少 25% 计算量

3. **可以减少到 3 层**
   - 如果只干预单一特质 (如神经质)
   - 车机助手场景推荐

---

## 4. 推荐行动

### 4.1 短期 (立即执行)

1. **使用方案B重新训练**
   ```bash
   python scripts/train.py --config configs/train_stage1_qwen3_probing_optimized.yaml
   ```

2. **对比实验**
   ```bash
   # 当前配置
   python scripts/train.py --config configs/train_stage1_qwen3.yaml
   
   # 优化配置
   python scripts/train.py --config configs/train_stage1_qwen3_probing_optimized.yaml
   
   # 精简配置
   python scripts/train.py --config configs/train_stage1_qwen3_probing_minimal.yaml
   ```

### 4.2 中期 (后续优化)

1. **特质分离训练**
   - 车机助手: `train_stage1_qwen3_neuroticism.yaml`
   - 聊天助手: 创建 agreeableness 专用配置

2. **动态层选择**
   - 根据输入内容判断需要干预的特质
   - 动态选择注入层

---

## 5. 配置文件清单

| 文件 | 方案 | 层数 | 用途 |
|------|------|------|------|
| `train_stage1_qwen3.yaml` | 当前 | 8 | 基线对比 |
| `train_stage1_qwen3_probing_optimized.yaml` | B | 11 | **推荐** |
| `train_stage1_qwen3_probing_minimal.yaml` | A | 6 | 资源受限 |
| `train_stage1_qwen3_neuroticism.yaml` | C | 3 | 车机助手 |

---

## 附录: Qwen2.5-3B 配置建议

基于 Qwen2.5-3B Probing 结果，推荐配置:

```yaml
# Qwen2.5-3B 优化配置
inject_layers: [6, 7, 8, 11, 12, 13, 19, 20, 21, 26, 27, 28, 30, 31, 32]
num_inject_layers: 15
```

特点:
- 峰值层分散，需要更多层覆盖
- 神经质需要深层干预 (层27)
- 宜人性需要深层干预 (层31)