#!/bin/bash
# V3多维度评估
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

CONV_DIR="results/conversations_20260331_080511"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/judge_v3_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "开始V3多维度评估 - $(date)"
echo "=========================================="

for model in baseline_qwen25 baseline_qwen3 stage1_qwen25 stage1_qwen3 stage2_qwen25 stage2_qwen3 stage3_qwen25 stage3_qwen3; do
    echo "  评估: $model"
    python scripts/judge_v3_from_conversations.py --conversations "$CONV_DIR/${model}.json" --output "results/judge_v3_${model}_${TIMESTAMP}.json" > "$LOG_DIR/v3_${model}.log" 2>&1 &
done

wait

echo -e "\n=========================================="
echo "V3评估完成 - $(date)"
echo "=========================================="
