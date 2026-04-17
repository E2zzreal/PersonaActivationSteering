#!/bin/bash
# V2 A/B对比评估
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

CONV_DIR="results/conversations_20260331_080511"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/judge_v2_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "开始V2 A/B对比评估 - $(date)"
echo "=========================================="

# Baseline vs Stage1
echo -e "\n[1/6] Baseline vs Stage1 (Qwen2.5)..."
python scripts/judge_v2_from_conversations.py \
    --conversations_a "$CONV_DIR/baseline_qwen25.json" \
    --conversations_b "$CONV_DIR/stage1_qwen25.json" \
    --model_a_name "baseline_qwen25" \
    --model_b_name "stage1_qwen25" \
    --output "results/judge_v2_baseline_vs_stage1_qwen25_${TIMESTAMP}.json" \
    > "$LOG_DIR/v2_baseline_vs_stage1_qwen25.log" 2>&1 &

echo "[2/6] Baseline vs Stage1 (Qwen3)..."
python scripts/judge_v2_from_conversations.py \
    --conversations_a "$CONV_DIR/baseline_qwen3.json" \
    --conversations_b "$CONV_DIR/stage1_qwen3.json" \
    --model_a_name "baseline_qwen3" \
    --model_b_name "stage1_qwen3" \
    --output "results/judge_v2_baseline_vs_stage1_qwen3_${TIMESTAMP}.json" \
    > "$LOG_DIR/v2_baseline_vs_stage1_qwen3.log" 2>&1 &

# Stage1 vs Stage2
echo "[3/6] Stage1 vs Stage2 (Qwen2.5)..."
python scripts/judge_v2_from_conversations.py \
    --conversations_a "$CONV_DIR/stage1_qwen25.json" \
    --conversations_b "$CONV_DIR/stage2_qwen25.json" \
    --model_a_name "stage1_qwen25" \
    --model_b_name "stage2_qwen25" \
    --output "results/judge_v2_stage1_vs_stage2_qwen25_${TIMESTAMP}.json" \
    > "$LOG_DIR/v2_stage1_vs_stage2_qwen25.log" 2>&1 &

echo "[4/6] Stage1 vs Stage2 (Qwen3)..."
python scripts/judge_v2_from_conversations.py \
    --conversations_a "$CONV_DIR/stage1_qwen3.json" \
    --conversations_b "$CONV_DIR/stage2_qwen3.json" \
    --model_a_name "stage1_qwen3" \
    --model_b_name "stage2_qwen3" \
    --output "results/judge_v2_stage1_vs_stage2_qwen3_${TIMESTAMP}.json" \
    > "$LOG_DIR/v2_stage1_vs_stage2_qwen3.log" 2>&1 &

# Stage2 vs Stage3
echo "[5/6] Stage2 vs Stage3 (Qwen2.5)..."
python scripts/judge_v2_from_conversations.py \
    --conversations_a "$CONV_DIR/stage2_qwen25.json" \
    --conversations_b "$CONV_DIR/stage3_qwen25.json" \
    --model_a_name "stage2_qwen25" \
    --model_b_name "stage3_qwen25" \
    --output "results/judge_v2_stage2_vs_stage3_qwen25_${TIMESTAMP}.json" \
    > "$LOG_DIR/v2_stage2_vs_stage3_qwen25.log" 2>&1 &

echo "[6/6] Stage2 vs Stage3 (Qwen3)..."
python scripts/judge_v2_from_conversations.py \
    --conversations_a "$CONV_DIR/stage2_qwen3.json" \
    --conversations_b "$CONV_DIR/stage3_qwen3.json" \
    --model_a_name "stage2_qwen3" \
    --model_b_name "stage3_qwen3" \
    --output "results/judge_v2_stage2_vs_stage3_qwen3_${TIMESTAMP}.json" \
    > "$LOG_DIR/v2_stage2_vs_stage3_qwen3.log" 2>&1 &

wait

echo -e "\n=========================================="
echo "V2评估完成 - $(date)"
echo "=========================================="
