# PersonaSteer 快速参考指南

> 📁 **文件整理说明**: 项目已整理，删除187GB过时文件。详见 [file_organization.md](file_organization.md)

## 目录结构

```
PersonaSteer/
├── checkpoints/          # 模型检查点（331GB，已清理）
│   ├── stage1/          # Stage 1 训练结果
│   ├── stage1_qwen3/    # Stage 1 (Qwen3) ⭐推荐
│   ├── exp_gate_*/      # Stage 2 实验
│   └── stage3_*/        # Stage 3 实验 ⭐推荐
├── configs/             # 配置文件
├── data/                # 数据目录
├── docs/                # 文档
│   ├── experiments/     # 实验记录
│   ├── analysis/        # 问题分析
│   └── guides/          # 使用指南
│       ├── quick_reference.md      # 本文件
│       └── file_organization.md    # 文件组织规范
├── logs/                # 训练日志（精简后）
├── results/             # 生成结果（9个成功模型）
├── scripts/             # 工具脚本
└── src/                 # 源代码
```

**存储统计**: 清理前 ~520GB → 清理后 ~333GB（节省187GB）

## 训练流程

### Stage 1: 训练 HyperNetwork

```bash
python scripts/train.py --config configs/train_stage1_qwen3.yaml
```

**输出**: `checkpoints/stage1_qwen3/best.pt`

### Stage 2: 联合训练 (解冻 Gate)

```bash
python scripts/train.py \
    --config configs/exp_gate_init_neg3.yaml \
    --resume checkpoints/stage1_qwen3/best.pt
```

**输出**: `checkpoints/exp_gate_init_neg3/best.pt`

### Stage 3: 加入对比学习

```bash
python scripts/train.py \
    --config configs/train_stage3_auto.yaml \
    --resume checkpoints/exp_gate_init_neg3/best.pt
```

**输出**: `checkpoints/stage3_auto/best.pt`

## 生成对话

### 单模型生成

```bash
python scripts/generate_all_conversations_fixed.py \
    --config configs/train_stage3_auto.yaml \
    --checkpoint checkpoints/stage3_auto/best.pt \
    --output results/conversations_stage3_auto.json \
    --num_samples 50
```

### 批量生成所有模型

```bash
bash scripts/run_generation_pipeline.sh
```

## 评估模型

### 模拟评估（无需API）

```bash
bash scripts/evaluate_all_models_mock.sh
```

### 真实LLM Judge评估（需要API Key）

```bash
export BLSC_API_KEY="your_api_key"
bash scripts/evaluate_all_models.sh
```

## 关键配置参数

### 模型配置

```yaml
model:
  inject_layers: [8, 9, 10, 11, 12, 13, 14, 15]  # 注入层
  v_dim: 1024                                      # 干预向量维度
  hidden_dim: 4096                                 # MLP隐藏层维度
  gate_hidden_dim: 256                             # Gate隐藏层维度
```

### 训练配置

```yaml
training:
  num_epochs: 4
  learning_rate: 0.00005
  batch_size: 2
  max_turns: 6
  
  # Gate配置（重要！）
  gate_init_bias: -2.0      # 当前硬编码无效
  gate_reg_weight: 0.01
```

## 常见问题

### Q: 为什么模型生成空回复？
A: 可能原因：
1. gate_init_bias 设置不当（当前硬编码-2.0）
2. 模型参数加载问题
3. 随机初始化导致的数值不稳定

### Q: 为什么baseline和stage1表现相当？
A: 因为：
1. 系统提示词已经足够引导角色扮演
2. gate值太小（~0.12），干预强度弱
3. 需要增大gate_init_bias到-1.0或更高

### Q: 如何修复gate_init_bias？
A: 修改 `src/models/injection.py:134`：
```python
self.gate = DynamicGate(
    v_dim,
    self.num_inject_layers,
    gate_hidden_dim,
    gate_max=gate_max,           # 从参数传入
    gate_init_bias=gate_init_bias,  # 从参数传入
)
```

## 模型选择建议

### 生产环境推荐

1. **stage1_qwen3** - 最佳表现，稳定
2. **baseline** - 简单可靠
3. **stage3_auto** - 完整训练流程

### 避免使用

- exp_gate_init_0
- exp_gate_init_neg1
- exp_gate_init_neg2
- exp_gate_reg_0.01_lr3e5
- exp_gate_reg_0.05_lr5e5
- stage3_gate_reg_0.05_lr5e5

（以上模型生成空回复）

---

*最后更新: 2026-04-10*
