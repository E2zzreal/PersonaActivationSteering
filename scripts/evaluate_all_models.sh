#!/bin/bash
# 评估所有有效模型（跳过有问题的模型）

set -e

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/eval_all_models_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "LLM Judge评估 - ${TIMESTAMP}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 有效模型列表（跳过exp_gate_init_0和exp_gate_init_neg1）
MODELS=(
    "baseline"
    "stage1"
    "stage1_qwen3"
    "exp_gate_init_neg2"
    "exp_gate_init_neg3"
    "exp_gate_reg_0.001_lr5e5"
    "exp_gate_reg_0.01_lr1e4"
    "exp_gate_reg_0.01_lr3e5"
    "exp_gate_reg_0.05_lr5e5"
    "stage3_auto"
    "stage3_gate_init_0"
    "stage3_gate_reg_0.01_lr1e4"
    "stage3_gate_reg_0.05_lr5e5"
)

total=${#MODELS[@]}
echo "总计模型数: $total" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 评估单个模型
evaluate_model() {
    local name=$1
    local conv_file="results/conversations_${name}_20260410_103948.json"
    local eval_file="results/judge_eval_${name}_${TIMESTAMP}.json"

    if [ ! -f "$conv_file" ]; then
        echo "[$name] ✗ 对话文件不存在: $conv_file" | tee -a "$LOG_FILE"
        return 1
    fi

    echo "[$name] 开始评估..." | tee -a "$LOG_FILE"

    python scripts/llm_judge_eval.py \
        --conversations "$conv_file" \
        --output "$eval_file" \
        --judge_model "GPT-4o-mini" \
        2>&1 | tee -a "$LOG_FILE"

    if [ -f "$eval_file" ]; then
        echo "[$name] ✓ 完成: $eval_file" | tee -a "$LOG_FILE"
    else
        echo "[$name] ✗ 失败" | tee -a "$LOG_FILE"
    fi
}

# 执行评估
current=0
for name in "${MODELS[@]}"; do
    current=$((current + 1))
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "[$current/$total] 评估: $name" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    evaluate_model "$name"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "评估完成!" | tee -a "$LOG_FILE"
echo "日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 生成汇总报告
python3 << 'PYEOF'
import json
import glob
from pathlib import Path

print("\n" + "="*70)
print("评估结果汇总")
print("="*70)

eval_files = sorted(glob.glob("results/judge_eval_*_*.json"))

if not eval_files:
    print("未找到评估结果文件")
else:
    print(f"\n找到 {len(eval_files)} 个评估结果:")

    all_results = {}
    for eval_file in eval_files:
        name = Path(eval_file).stem.replace("judge_eval_", "")
        try:
            with open(eval_file) as f:
                data = json.load(f)

            avg = data.get("average_score", 0)
            all_results[name] = avg

            print(f"\n{name}:")
            print(f"  平均分: {avg:.2f}")
        except Exception as e:
            print(f"\n{name}: 读取失败 - {e}")

    # 排序
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "="*70)
    print("排名:")
    print("="*70)
    for i, (name, score) in enumerate(sorted_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i}. {name}: {score:.2f}")

PYEOF
