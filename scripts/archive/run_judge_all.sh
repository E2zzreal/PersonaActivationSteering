#!/bin/bash
# 对所有已生成的对话进行Judge评估（V1和V3并行）
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

CONV_DIR="results/conversations_20260331_080511"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/judge_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "开始Judge评估（全部8个模型） - $(date)"
echo "对话目录: $CONV_DIR"
echo "=========================================="

# V1评估
echo -e "\n[1/2] 运行 Judge V1 (严格评分)..."
for model in baseline_qwen25 baseline_qwen3 stage1_qwen25 stage1_qwen3 stage2_qwen25 stage2_qwen3 stage3_qwen25 stage3_qwen3; do
    echo "  评估: $model"
    python scripts/judge_v1_from_conversations.py --conversations "$CONV_DIR/${model}.json" --output "results/judge_v1_${model}_${TIMESTAMP}.json" > "$LOG_DIR/v1_${model}.log" 2>&1 &
done
wait
echo "  V1评估完成"

# V3评估
echo -e "\n[2/2] 运行 Judge V3 (多维度)..."
for model in baseline_qwen25 baseline_qwen3 stage1_qwen25 stage1_qwen3 stage2_qwen25 stage2_qwen3 stage3_qwen25 stage3_qwen3; do
    echo "  评估: $model"
    python scripts/judge_v3_from_conversations.py --conversations "$CONV_DIR/${model}.json" --output "results/judge_v3_${model}_${TIMESTAMP}.json" > "$LOG_DIR/v3_${model}.log" 2>&1 &
done
wait
echo "  V3评估完成"

echo -e "\n=========================================="
echo "Judge评估完成 - $(date)"
echo "结果保存在: results/judge_*_${TIMESTAMP}.json"
echo "=========================================="
