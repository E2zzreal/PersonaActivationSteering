# PersonaSteer V2 实验执行指南

**项目路径**: `/home/kemove/Desktop/Projects/3-PersonaSteer_V2`
**文档版本**: v3.0
**更新日期**: 2026-03-27

---

## 当前状态

🔧 **代码已修复**：chat template 问题已解决
⚠️ **需要重新训练**：旧 checkpoint 使用错误格式训练，需从头重训
📋 **待执行**：重新训练 → 重新评估

---

## 修复内容

### 关键修复
1. **训练数据格式** (`src/data/aloe_dataset.py`)
   - 修复前：直接拼接 `user_ids + asst_ids`
   - 修复后：使用 `tokenizer.apply_chat_template()` 添加正确的特殊 token

2. **生成格式** (`src/evaluation/llm_judge.py`)
   - 修复前：直接 encode user_text
   - 修复后：使用 chat template 构建输入

### 影响
- 旧 checkpoint 使用错误格式训练，模型学习到错误的输入输出模式
- 必须重新训练所有 6 个 checkpoint

---

## 目录

1. [环境准备](#1-环境准备)
2. [重新训练](#2-重新训练)
3. [评估实验](#3-评估实验)
4. [结果分析](#4-结果分析)

---

## 1. 环境准备

### 1.1 激活虚拟环境

```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
source venv/bin/activate
```

### 1.2 确认 GPU 状态

```bash
nvidia-smi
```

### 1.3 归档旧 checkpoint

```bash
mkdir -p checkpoints/archive_20260327
mv checkpoints/stage* checkpoints/archive_20260327/
```

---

## 2. 重新训练

### 2.1 Qwen2.5-3B 训练

**终端 1 (cuda:0) - Stage 1**
```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
source venv/bin/activate

python scripts/train.py \
    --config configs/train_stage1.yaml \
    --device cuda:0
```

**终端 2 (cuda:1) - Stage 2**
```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
source venv/bin/activate

# 等待 Stage 1 完成后执行
python scripts/train.py \
    --config configs/train_stage2.yaml \
    --device cuda:1
```

**终端 3 (cuda:2) - Stage 3**
```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2
source venv/bin/activate

# 等待 Stage 2 完成后执行
python scripts/train.py \
    --config configs/train_stage3.yaml \
    --device cuda:2
```

### 2.2 Qwen3-4B 训练

**终端 1 (cuda:0) - Stage 1**
```bash
python scripts/train.py \
    --config configs/train_stage1_qwen3.yaml \
    --device cuda:0
```

**终端 2 (cuda:1) - Stage 2**
```bash
# 等待 Stage 1 完成后执行
python scripts/train.py \
    --config configs/train_stage2_qwen3.yaml \
    --device cuda:1
```

**终端 3 (cuda:2) - Stage 3**
```bash
# 等待 Stage 2 完成后执行
python scripts/train.py \
    --config configs/train_stage3_qwen3.yaml \
    --device cuda:2
```

### 2.3 训练监控

```bash
# 查看训练日志
tail -f logs/train_stage1.log

# 查看 checkpoint
ls -lh checkpoints/stage1/
```

---

## 3. 评估实验

训练完成后，运行完整评估（8 个检查点）。

### 3.1 快速执行

```bash
bash EVALUATION_COMMANDS.sh
```

按提示在 4 个终端中分别执行对应命令。

### 3.2 单个评估示例

```bash
# Qwen2.5 Baseline
python scripts/evaluate_fixed.py \
    --config configs/train_stage1.yaml \
    --baseline \
    --data data/split/val.jsonl \
    --num_samples 30 \
    --output results/qwen25_baseline_eval.json \
    --device cuda:0
```

---

## 4. 结果分析

### 4.1 查看结果汇总

```bash
python3 << 'EOF'
import json

files = [
    ('qwen25_baseline', 'results/qwen25_baseline_eval.json'),
    ('qwen25_stage1', 'results/qwen25_stage1_eval.json'),
    ('qwen25_stage2', 'results/qwen25_stage2_eval.json'),
    ('qwen25_stage3', 'results/qwen25_stage3_eval.json'),
    ('qwen3_baseline', 'results/qwen3_baseline_eval.json'),
    ('qwen3_stage1', 'results/qwen3_stage1_eval.json'),
    ('qwen3_stage2', 'results/qwen3_stage2_eval.json'),
    ('qwen3_stage3', 'results/qwen3_stage3_eval.json'),
]

print(f"\n{'='*80}")
print("评估结果汇总")
print(f"{'='*80}\n")
print(f"{'实验':<20} {'Loss':<10} {'PPL':<10} {'v_var':<12} {'AL(K)':<10}")
print('-' * 80)

for name, path in files:
    try:
        with open(path) as f:
            d = json.load(f)
        am = d.get('auto_metrics', {})
        lj = d.get('llm_judge', {})
        loss = am.get('loss_sft', 'N/A')
        ppl = am.get('ppl', 'N/A')
        vvar = am.get('v_variance', 'N/A')
        alk = lj.get('al_k_avg', 'N/A')

        if isinstance(loss, float):
            print(f"{name:<20} {loss:<10.4f} {ppl:<10.4f} {vvar:<12.6f} {alk}")
        else:
            print(f"{name:<20} {loss:<10} {ppl:<10} {vvar:<12} {alk}")
    except FileNotFoundError:
        print(f"{name:<20} (未找到)")
    except Exception as e:
        print(f"{name:<20} (错误)")

print(f"\n{'='*80}\n")
EOF
```

### 4.2 预期结果

修复后应该看到：
- ✅ 生成内容不再复制用户消息
- ✅ AL(K) 分数显著提升（baseline ~1.0 → stage3 >3.0）
- ✅ v_variance 正常（>0.01）
- ✅ 所有指标完整

---

## 5. 关键文件

- **训练脚本**: `scripts/train.py`
- **评估脚本**: `scripts/evaluate_fixed.py`
- **数据加载**: `src/data/aloe_dataset.py` (已修复)
- **生成逻辑**: `src/evaluation/llm_judge.py` (已修复)
- **配置文件**: `configs/train_stage*.yaml`

---

**更新时间**: 2026-03-27
**修复版本**: v3.0 (chat template 修复)
