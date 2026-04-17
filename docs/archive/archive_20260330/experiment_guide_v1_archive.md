# PersonaSteer V2 实验执行文档

**项目路径**: `/home/kemove/Desktop/Projects/3-PersonaSteer_V2`
**文档更新**: 2026-03-26

---

## 当前状态与任务说明

**需要从头重新训练两个模型：Qwen2.5-3B 和 Qwen3-4B。**

`checkpoints/` 目录中已有旧的训练结果，但那批 checkpoint 是在代码存在以下 bug 时训练的，结果不可用：

1. 每个 turn 执行了两次 forward pass（梯度计算错误）
2. `v_norm_target=5.0` 与 LayerNorm 输出不匹配（导致向量方向趋同、生成循环）
3. padding 行参与了 loss 计算（污染梯度）

这些 bug 已于 2026-03-26 全部修复。**实习生的任务是使用修复后的代码并行训练两个模型，并对每个阶段进行评估，将结果与 baseline 对比。**

---

## 目录

1. [环境准备](#1-环境准备)
2. [项目结构说明](#2-项目结构说明)
3. [训练实验](#3-训练实验)
4. [评估实验](#4-评估实验)
5. [结果查看与对比](#5-结果查看与对比)
6. [常见问题排查](#6-常见问题排查)
7. [关键指标说明](#7-关键指标说明)

---

## 1. 环境准备

### 1.1 激活虚拟环境

**所有命令必须在激活虚拟环境后执行**，否则 transformers/torch 版本不匹配会报错。

```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
source venv/bin/activate

# 验证环境
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
# 预期输出: PyTorch: 2.10.0 / Transformers: 5.3.0
```

### 1.2 确认 GPU 状态

```bash
nvidia-smi
```

服务器共有 4 块 RTX 4090 (23GB)，编号 0~3。

| GPU | 训练任务 | 评估任务 |
|-----|---------|---------|
| cuda:0 | Qwen2.5-3B 训练 | Qwen2.5 评估 |
| cuda:1 | 备用 | Qwen2.5 评估 |
| cuda:2 | Qwen3-4B 训练 | Qwen3 评估 |
| cuda:3 | 备用 | Qwen3 评估 |

**执行前先确认 GPU 是否空闲：**

```bash
nvidia-smi | grep -E "GPU|MiB"
```

### 1.3 确认 API Key

LLM Judge 评估需要访问 https://llmapi.blsc.cn，API Key 应已在环境变量中配置。

```bash
echo $BLSC_API_KEY   # 有输出则正常
```

---

## 2. 项目结构说明

```
3-PersonaSteer_V2/
├── configs/                    # 训练/评估配置文件
│   ├── train_stage1.yaml       # Stage 1 配置 (Qwen2.5-3B)
│   ├── train_stage2.yaml       # Stage 2 配置 (Qwen2.5-3B)
│   ├── train_stage3.yaml       # Stage 3 配置 (Qwen2.5-3B)
│   ├── train_stage1_qwen3.yaml # Stage 1 配置 (Qwen3-4B)
│   ├── train_stage2_qwen3.yaml # Stage 2 配置 (Qwen3-4B)
│   └── train_stage3_qwen3.yaml # Stage 3 配置 (Qwen3-4B)
│
├── data/split/
│   ├── train.jsonl             # 训练集 (2777 条)
│   └── val.jsonl               # 验证集 (661 条)
│
├── checkpoints/                # 保存的模型权重
│   ├── stage1_fixed/           # Qwen2.5-3B Stage 1
│   ├── stage2_fixed/           # Qwen2.5-3B Stage 2
│   ├── stage3_fixed/           # Qwen2.5-3B Stage 3
│   ├── stage1_qwen3_fixed/     # Qwen3-4B Stage 1
│   ├── stage2_qwen3_fixed/     # Qwen3-4B Stage 2
│   └── stage3_qwen3_fixed/     # Qwen3-4B Stage 3
│
├── scripts/
│   ├── train.py                # 训练脚本（核心）
│   ├── evaluate_fixed.py       # 评估脚本（核心）
│   └── train_pipeline_venv.sh  # 自动化三阶段训练流水线
│
├── results/                    # 评估结果 JSON 文件
├── logs/                       # 训练日志
└── src/                        # 源代码
```

---

## 3. 训练实验

### 3.1 三阶段训练概述

PersonaSteer 采用三阶段渐进训练：

| 阶段 | 训练组件 | Epochs | 学习率 | 损失函数 |
|------|---------|--------|--------|---------|
| Stage 1 | HyperNetwork（Gate 冻结） | 3 | 1e-4 | SFT + v_norm 约束 |
| Stage 2 | HyperNetwork + Gate | 3 | 5e-5 | SFT + Gate 正则化 |
| Stage 3 | 全部参数 | 5 | 3e-5 | SFT + Gate 正则化 + 对比学习 |

**Stage 1 → Stage 2 → Stage 3 必须顺序执行。**

### 3.2 并行训练两个模型

打开**两个终端窗口**，同时运行：

#### 终端 1：Qwen2.5-3B (cuda:0)

```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
source venv/bin/activate

bash scripts/train_pipeline_venv.sh qwen25 cuda:0
```

**输出位置**：`checkpoints/stage1_fixed/`, `stage2_fixed/`, `stage3_fixed/`
**日志位置**：`logs/train_stage1_qwen25_fixed.log` 等

#### 终端 2：Qwen3-4B (cuda:2)

```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
source venv/bin/activate

bash scripts/train_pipeline_venv.sh qwen3 cuda:2
```

**输出位置**：`checkpoints/stage1_qwen3_fixed/`, `stage2_qwen3_fixed/`, `stage3_qwen3_fixed/`
**日志位置**：`logs/train_stage1_qwen3_fixed.log` 等

### 3.3 实时查看训练日志

```bash
# Qwen2.5-3B
tail -f logs/train_stage1_qwen25_fixed.log

# Qwen3-4B
tail -f logs/train_stage1_qwen3_fixed.log
```

### 3.4 训练过程正常输出

```
2026-03-26 10:00:00 - INFO - Backbone hidden_size: 2048
2026-03-26 10:00:15 - INFO - Training dataset: 2777 samples
2026-03-26 10:00:15 - INFO - Stage 1: Training HyperNetwork only (gate frozen)
Epoch 1/3: 100%|████████| 1388/1388 [45:20<00:00, loss=0.2341, sft=0.2341]
2026-03-26 10:45:35 - INFO - Epoch 1: loss=0.2341, sft=0.2341, scl=0.0000
2026-03-26 10:45:36 - INFO - New best model saved (loss=0.2341)
```

**关注指标**：
- `loss`：总损失，应持续下降
- `sft`：语言模型损失，Stage 1 结束时应 ~0.10~0.20

### 3.5 预计时间

| 模型 | Stage 1 | Stage 2 | Stage 3 | 总计 |
|------|---------|---------|---------|------|
| Qwen2.5-3B | ~2h | ~2h | ~3h | ~7h |
| Qwen3-4B | ~3h | ~3h | ~4h | ~10h |

---

## 4. 评估实验

### 4.1 评估任务概览

训练完成后需评估 8 个检查点：

| 序号 | 模型 | Checkpoint | 输出文件 | 推荐 GPU |
|------|------|-----------|---------|---------|
| 1 | Qwen2.5 | baseline | results/qwen25_baseline_eval.json | cuda:0 |
| 2 | Qwen2.5 | stage1_fixed | results/qwen25_stage1_eval.json | cuda:0 |
| 3 | Qwen2.5 | stage2_fixed | results/qwen25_stage2_eval.json | cuda:1 |
| 4 | Qwen2.5 | stage3_fixed | results/qwen25_stage3_eval.json | cuda:1 |
| 5 | Qwen3 | baseline | results/qwen3_baseline_eval.json | cuda:2 |
| 6 | Qwen3 | stage1_qwen3_fixed | results/qwen3_stage1_eval.json | cuda:2 |
| 7 | Qwen3 | stage2_qwen3_fixed | results/qwen3_stage2_eval.json | cuda:3 |
| 8 | Qwen3 | stage3_qwen3_fixed | results/qwen3_stage3_eval.json | cuda:3 |

### 4.2 并行评估（推荐）

打开 4 个终端，每个 GPU 运行 2 个评估任务：

#### 终端 1 (cuda:0)

```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
source venv/bin/activate

# Qwen2.5 Baseline
python scripts/evaluate_fixed.py \
    --config configs/train_stage1.yaml --baseline \
    --data data/split/val.jsonl --num_samples 30 \
    --output results/qwen25_baseline_eval.json --device cuda:0

# Qwen2.5 Stage 1
python scripts/evaluate_fixed.py \
    --config configs/train_stage1.yaml \
    --checkpoint checkpoints/stage1_fixed/best.pt \
    --data data/split/val.jsonl --num_samples 30 \
    --output results/qwen25_stage1_eval.json --device cuda:0
```

#### 终端 2 (cuda:1)

```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
source venv/bin/activate

# Qwen2.5 Stage 2
python scripts/evaluate_fixed.py \
    --config configs/train_stage2.yaml \
    --checkpoint checkpoints/stage2_fixed/best.pt \
    --data data/split/val.jsonl --num_samples 30 \
    --output results/qwen25_stage2_eval.json --device cuda:1

# Qwen2.5 Stage 3
python scripts/evaluate_fixed.py \
    --config configs/train_stage3.yaml \
    --checkpoint checkpoints/stage3_fixed/best.pt \
    --data data/split/val.jsonl --num_samples 30 \
    --output results/qwen25_stage3_eval.json --device cuda:1
```

#### 终端 3 (cuda:2)

```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
source venv/bin/activate

# Qwen3 Baseline
python scripts/evaluate_fixed.py \
    --config configs/train_stage1_qwen3.yaml --baseline \
    --data data/split/val.jsonl --num_samples 30 \
    --output results/qwen3_baseline_eval.json --device cuda:2

# Qwen3 Stage 1
python scripts/evaluate_fixed.py \
    --config configs/train_stage1_qwen3.yaml \
    --checkpoint checkpoints/stage1_qwen3_fixed/best.pt \
    --data data/split/val.jsonl --num_samples 30 \
    --output results/qwen3_stage1_eval.json --device cuda:2
```

#### 终端 4 (cuda:3)

```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
source venv/bin/activate

# Qwen3 Stage 2
python scripts/evaluate_fixed.py \
    --config configs/train_stage2_qwen3.yaml \
    --checkpoint checkpoints/stage2_qwen3_fixed/best.pt \
    --data data/split/val.jsonl --num_samples 30 \
    --output results/qwen3_stage2_eval.json --device cuda:3

# Qwen3 Stage 3
python scripts/evaluate_fixed.py \
    --config configs/train_stage3_qwen3.yaml \
    --checkpoint checkpoints/stage3_qwen3_fixed/best.pt \
    --data data/split/val.jsonl --num_samples 30 \
    --output results/qwen3_stage3_eval.json --device cuda:3
```

### 4.3 快速验证（跳过 LLM Judge）

如需快速验证模型是否正常，可先跳过 LLM Judge：

```bash
python scripts/evaluate_fixed.py \
    --config configs/train_stage1.yaml \
    --checkpoint checkpoints/stage1_fixed/best.pt \
    --data data/split/val.jsonl \
    --skip_llm_judge \
    --output results/qwen25_stage1_auto.json --device cuda:0
```

---

## 5. 结果查看与对比

### 5.1 查看单个结果

```bash
cat results/qwen25_stage1_eval.json | python3 -m json.tool
```

### 5.2 汇总对比（两个模型）

```bash
python3 << 'EOF'
import json
import glob

models = {
    'Qwen2.5-3B': {
        'baseline': 'results/qwen25_baseline_eval.json',
        'stage1': 'results/qwen25_stage1_eval.json',
        'stage2': 'results/qwen25_stage2_eval.json',
        'stage3': 'results/qwen25_stage3_eval.json',
    },
    'Qwen3-4B': {
        'baseline': 'results/qwen3_baseline_eval.json',
        'stage1': 'results/qwen3_stage1_eval.json',
        'stage2': 'results/qwen3_stage2_eval.json',
        'stage3': 'results/qwen3_stage3_eval.json',
    }
}

for model_name, files in models.items():
    print(f"\n{'='*70}")
    print(f"模型: {model_name}")
    print(f"{'='*70}")
    print(f"{'实验':<12} {'SFT Loss':<12} {'PPL':<10} {'v_variance':<14} {'AL(K)_AVG':<12}")
    print('-' * 60)

    for name, path in files.items():
        try:
            with open(path) as f:
                d = json.load(f)
            am = d.get('auto_metrics', {})
            lj = d.get('llm_judge', {})
            sft = am.get('loss_sft', '-')
            ppl = am.get('ppl', '-')
            vvar = am.get('v_variance', '-')
            alk = lj.get('al_k_avg', '-')
            print(f"{name:<12} {sft:<12.4f} {ppl:<10.4f} {vvar:<14.6f} {alk}")
        except FileNotFoundError:
            print(f"{name:<12} (结果文件不存在)")
EOF
```

### 5.3 预期结果范围

| 实验 | SFT Loss | PPL | v_variance | AL(K)_AVG |
|------|----------|-----|------------|-----------|
| Baseline | ~14.0 | >1000 | ~0.0001 | ~1.1 |
| Stage 1 | ~0.18 | ~1.2 | >0.05 | >2.5 |
| Stage 2 | ~0.15 | ~1.2 | >0.08 | >3.0 |
| Stage 3 | ~0.15 | ~1.2 | >0.10 | >3.5 |

**异常判断**：
- PPL > 100：生成崩溃
- v_variance < 0.001：向量趋同，个性化失败
- AL(K)_AVG ≤ 1.5：模型未学到人格对齐

---

## 6. 常见问题排查

### Q1: `ModuleNotFoundError`

未激活虚拟环境：

```bash
source venv/bin/activate
```

### Q2: `CUDA out of memory`

1. 换用空闲 GPU
2. 检查占用进程：`nvidia-smi`

### Q3: Loss 不下降或 NaN

| 现象 | 原因 | 处理 |
|------|------|------|
| Loss = NaN | 学习率过高 | 降低 `learning_rate` |
| Loss 不变 | checkpoint 加载失败 | 检查 `--resume` 路径 |

### Q4: LLM Judge 返回默认 3.0

API 调用失败，检查：

```bash
echo $BLSC_API_KEY
curl -s https://llmapi.blsc.cn -o /dev/null -w "%{http_code}"
```

### Q5: 生成回复重复

注入向量趋同，检查 `v_variance`，若 < 0.001 需重训。

---

## 7. 关键指标说明

### 自动指标

| 指标 | 含义 | 正常范围 |
|------|------|---------|
| `loss_sft` | 语言模型损失 | 0.10~0.25 |
| `ppl` | 困惑度 | 1.1~2.0 |
| `v_variance` | 样本间向量方差 | >0.05 |
| `gate_distribution.mean` | 门控均值 | 0.3~0.7 |

### LLM Judge 指标

| 指标 | 含义 | 范围 |
|------|------|------|
| `al_k_avg` | 人格对齐分数 | 1~5 |
| `n_ir` | 相对改进率 | 正值为改进 |

---

## 附录：完整流程速查

```bash
# === 训练阶段 ===
# 终端 1
bash scripts/train_pipeline_venv.sh qwen25 cuda:0

# 终端 2
bash scripts/train_pipeline_venv.sh qwen3 cuda:2

# === 等待训练完成 ===

# === 评估阶段（4终端并行）===
# 终端 1 (cuda:0): qwen25 baseline + stage1
# 终端 2 (cuda:1): qwen25 stage2 + stage3
# 终端 3 (cuda:2): qwen3 baseline + stage1
# 终端 4 (cuda:3): qwen3 stage2 + stage3

# === 汇总结果 ===
python3 << 'EOF'
import json
models = {
    'Qwen2.5-3B': ['baseline', 'stage1', 'stage2', 'stage3'],
    'Qwen3-4B': ['baseline', 'stage1', 'stage2', 'stage3'],
}
for model, stages in models.items():
    print(f"\n{model}:")
    prefix = 'qwen25' if '2.5' in model else 'qwen3'
    suffix = '_fixed' if '2.5' in model else '_qwen3_fixed'
    for s in stages:
        path = f'results/{prefix}_{s}_eval.json' if s == 'baseline' else f'results/{prefix}_{s}{suffix}.json'
        try:
            with open(path) as f: d = json.load(f)
            lj = d.get('llm_judge', {})
            print(f"  {s}: AL(K)_AVG = {lj.get('al_k_avg', '-')}")
        except: print(f"  {s}: (未找到)")
EOF
```
