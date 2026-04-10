#!/bin/bash
# 完整评估脚本 - 对所有模型运行V1/V2/V3三种Judge评估
# 生成时间: 2026-03-31

set -e

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/full_eval_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "开始完整评估 - $(date)"
echo "日志目录: $LOG_DIR"
echo "=========================================="

# V1评估 - 严格评分 (顺序执行避免OOM)
echo -e "\n[1/3] 运行 Judge V1 (严格评分)..."
python scripts/judge_v1_strict_rubric.py --config configs/train_stage1.yaml --checkpoint baseline --output "results/judge_v1_baseline_qwen25_${TIMESTAMP}.json" > "$LOG_DIR/v1_baseline_qwen25.log" 2>&1
python scripts/judge_v1_strict_rubric.py --config configs/train_stage1_qwen3.yaml --checkpoint baseline --output "results/judge_v1_baseline_qwen3_${TIMESTAMP}.json" > "$LOG_DIR/v1_baseline_qwen3.log" 2>&1
python scripts/judge_v1_strict_rubric.py --config configs/train_stage1.yaml --checkpoint checkpoints/stage1/best.pt --output "results/judge_v1_stage1_qwen25_${TIMESTAMP}.json" > "$LOG_DIR/v1_stage1_qwen25.log" 2>&1
python scripts/judge_v1_strict_rubric.py --config configs/train_stage1_qwen3.yaml --checkpoint checkpoints/stage1_qwen3/best.pt --output "results/judge_v1_stage1_qwen3_${TIMESTAMP}.json" > "$LOG_DIR/v1_stage1_qwen3.log" 2>&1
python scripts/judge_v1_strict_rubric.py --config configs/train_stage2.yaml --checkpoint checkpoints/stage2/best.pt --output "results/judge_v1_stage2_qwen25_${TIMESTAMP}.json" > "$LOG_DIR/v1_stage2_qwen25.log" 2>&1
python scripts/judge_v1_strict_rubric.py --config configs/train_stage2_qwen3.yaml --checkpoint checkpoints/stage2_qwen3/best.pt --output "results/judge_v1_stage2_qwen3_${TIMESTAMP}.json" > "$LOG_DIR/v1_stage2_qwen3.log" 2>&1
python scripts/judge_v1_strict_rubric.py --config configs/train_stage3.yaml --checkpoint checkpoints/stage3/best.pt --output "results/judge_v1_stage3_qwen25_${TIMESTAMP}.json" > "$LOG_DIR/v1_stage3_qwen25.log" 2>&1
python scripts/judge_v1_strict_rubric.py --config configs/train_stage3_qwen3.yaml --checkpoint checkpoints/stage3_qwen3/best.pt --output "results/judge_v1_stage3_qwen3_${TIMESTAMP}.json" > "$LOG_DIR/v1_stage3_qwen3.log" 2>&1
echo "  V1评估完成"

# V3评估 - 多维度 (顺序执行避免OOM)
echo -e "\n[2/3] 运行 Judge V3 (多维度)..."
python scripts/judge_v3_multidim.py --config configs/train_stage1.yaml --checkpoint baseline --output "results/judge_v3_baseline_qwen25_${TIMESTAMP}.json" > "$LOG_DIR/v3_baseline_qwen25.log" 2>&1
python scripts/judge_v3_multidim.py --config configs/train_stage1_qwen3.yaml --checkpoint baseline --output "results/judge_v3_baseline_qwen3_${TIMESTAMP}.json" > "$LOG_DIR/v3_baseline_qwen3.log" 2>&1
python scripts/judge_v3_multidim.py --config configs/train_stage1.yaml --checkpoint checkpoints/stage1/best.pt --output "results/judge_v3_stage1_qwen25_${TIMESTAMP}.json" > "$LOG_DIR/v3_stage1_qwen25.log" 2>&1
python scripts/judge_v3_multidim.py --config configs/train_stage1_qwen3.yaml --checkpoint checkpoints/stage1_qwen3/best.pt --output "results/judge_v3_stage1_qwen3_${TIMESTAMP}.json" > "$LOG_DIR/v3_stage1_qwen3.log" 2>&1
python scripts/judge_v3_multidim.py --config configs/train_stage2.yaml --checkpoint checkpoints/stage2/best.pt --output "results/judge_v3_stage2_qwen25_${TIMESTAMP}.json" > "$LOG_DIR/v3_stage2_qwen25.log" 2>&1
python scripts/judge_v3_multidim.py --config configs/train_stage2_qwen3.yaml --checkpoint checkpoints/stage2_qwen3/best.pt --output "results/judge_v3_stage2_qwen3_${TIMESTAMP}.json" > "$LOG_DIR/v3_stage2_qwen3.log" 2>&1
python scripts/judge_v3_multidim.py --config configs/train_stage3.yaml --checkpoint checkpoints/stage3/best.pt --output "results/judge_v3_stage3_qwen25_${TIMESTAMP}.json" > "$LOG_DIR/v3_stage3_qwen25.log" 2>&1
python scripts/judge_v3_multidim.py --config configs/train_stage3_qwen3.yaml --checkpoint checkpoints/stage3_qwen3/best.pt --output "results/judge_v3_stage3_qwen3_${TIMESTAMP}.json" > "$LOG_DIR/v3_stage3_qwen3.log" 2>&1
echo "  V3评估完成"

# V2评估 - A/B对比 (两两对比)
echo -e "\n[3/3] 运行 Judge V2 (A/B对比)..."
# Baseline vs Stage1
python scripts/judge_v2_ab_compare.py \
    --config_a configs/train_stage1.yaml --checkpoint_a baseline \
    --config_b configs/train_stage1.yaml --checkpoint_b checkpoints/stage1/best.pt \
    --output "results/judge_v2_baseline_vs_stage1_qwen25_${TIMESTAMP}.json" \
    > "$LOG_DIR/v2_baseline_vs_stage1_qwen25.log" 2>&1 &

python scripts/judge_v2_ab_compare.py \
    --config_a configs/train_stage1_qwen3.yaml --checkpoint_a baseline \
    --config_b configs/train_stage1_qwen3.yaml --checkpoint_b checkpoints/stage1_qwen3/best.pt \
    --output "results/judge_v2_baseline_vs_stage1_qwen3_${TIMESTAMP}.json" \
    > "$LOG_DIR/v2_baseline_vs_stage1_qwen3.log" 2>&1 &

# Stage1 vs Stage2
python scripts/judge_v2_ab_compare.py \
    --config_a configs/train_stage1.yaml --checkpoint_a checkpoints/stage1/best.pt \
    --config_b configs/train_stage2.yaml --checkpoint_b checkpoints/stage2/best.pt \
    --output "results/judge_v2_stage1_vs_stage2_qwen25_${TIMESTAMP}.json" \
    > "$LOG_DIR/v2_stage1_vs_stage2_qwen25.log" 2>&1 &

python scripts/judge_v2_ab_compare.py \
    --config_a configs/train_stage1_qwen3.yaml --checkpoint_a checkpoints/stage1_qwen3/best.pt \
    --config_b configs/train_stage2_qwen3.yaml --checkpoint_b checkpoints/stage2_qwen3/best.pt \
    --output "results/judge_v2_stage1_vs_stage2_qwen3_${TIMESTAMP}.json" \
    > "$LOG_DIR/v2_stage1_vs_stage2_qwen3.log" 2>&1 &

# Stage2 vs Stage3
python scripts/judge_v2_ab_compare.py \
    --config_a configs/train_stage2.yaml --checkpoint_a checkpoints/stage2/best.pt \
    --config_b configs/train_stage3.yaml --checkpoint_b checkpoints/stage3/best.pt \
    --output "results/judge_v2_stage2_vs_stage3_qwen25_${TIMESTAMP}.json" \
    > "$LOG_DIR/v2_stage2_vs_stage3_qwen25.log" 2>&1 &

python scripts/judge_v2_ab_compare.py \
    --config_a configs/train_stage2_qwen3.yaml --checkpoint_a checkpoints/stage2_qwen3/best.pt \
    --config_b configs/train_stage3_qwen3.yaml --checkpoint_b checkpoints/stage3_qwen3/best.pt \
    --output "results/judge_v2_stage2_vs_stage3_qwen3_${TIMESTAMP}.json" \
    > "$LOG_DIR/v2_stage2_vs_stage3_qwen3.log" 2>&1 &

wait
echo "  V2评估完成"

echo -e "\n=========================================="
echo "所有评估完成 - $(date)"
echo "结果保存在: results/judge_*_${TIMESTAMP}.json"
echo "日志保存在: $LOG_DIR/"
echo "=========================================="
