#!/bin/bash
# 对已生成的对话进行Judge评估（V1和V3并行）
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

CONV_DIR="results/conversations_20260331_080511"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/judge_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "开始Judge评估 - $(date)"
echo "对话目录: $CONV_DIR"
echo "=========================================="

# V1评估（已完成的4个模型）
echo -e "\n[1/2] 运行 Judge V1 (严格评分)..."
python scripts/judge_v1_from_conversations.py --conversations "$CONV_DIR/baseline_qwen25.json" --output "results/judge_v1_baseline_qwen25_${TIMESTAMP}.json" > "$LOG_DIR/v1_baseline_qwen25.log" 2>&1 &
python scripts/judge_v1_from_conversations.py --conversations "$CONV_DIR/baseline_qwen3.json" --output "results/judge_v1_baseline_qwen3_${TIMESTAMP}.json" > "$LOG_DIR/v1_baseline_qwen3.log" 2>&1 &
python scripts/judge_v1_from_conversations.py --conversations "$CONV_DIR/stage1_qwen25.json" --output "results/judge_v1_stage1_qwen25_${TIMESTAMP}.json" > "$LOG_DIR/v1_stage1_qwen25.log" 2>&1 &
python scripts/judge_v1_from_conversations.py --conversations "$CONV_DIR/stage1_qwen3.json" --output "results/judge_v1_stage1_qwen3_${TIMESTAMP}.json" > "$LOG_DIR/v1_stage1_qwen3.log" 2>&1 &

# V3评估（已完成的4个模型）
echo -e "\n[2/2] 运行 Judge V3 (多维度)..."
python scripts/judge_v3_from_conversations.py --conversations "$CONV_DIR/baseline_qwen25.json" --output "results/judge_v3_baseline_qwen25_${TIMESTAMP}.json" > "$LOG_DIR/v3_baseline_qwen25.log" 2>&1 &
python scripts/judge_v3_from_conversations.py --conversations "$CONV_DIR/baseline_qwen3.json" --output "results/judge_v3_baseline_qwen3_${TIMESTAMP}.json" > "$LOG_DIR/v3_baseline_qwen3.log" 2>&1 &
python scripts/judge_v3_from_conversations.py --conversations "$CONV_DIR/stage1_qwen25.json" --output "results/judge_v3_stage1_qwen25_${TIMESTAMP}.json" > "$LOG_DIR/v3_stage1_qwen25.log" 2>&1 &
python scripts/judge_v3_from_conversations.py --conversations "$CONV_DIR/stage1_qwen3.json" --output "results/judge_v3_stage1_qwen3_${TIMESTAMP}.json" > "$LOG_DIR/v3_stage1_qwen3.log" 2>&1 &

wait

echo -e "\n=========================================="
echo "Judge评估完成 - $(date)"
echo "结果保存在: results/judge_*_${TIMESTAMP}.json"
echo "=========================================="
