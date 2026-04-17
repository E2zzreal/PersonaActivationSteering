#!/bin/bash
# PersonaSteer V2 自动化训练流水线
# 并行训练 Qwen2.5-3B (GPU 0-1) 和 Qwen3-4B (GPU 2-3)
# 每个模型独立完成 Stage1→Stage2→Stage3

set -e

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
PYTHON="python"
LOG_DIR="$PROJECT_ROOT/logs/pipeline_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

echo "=== PersonaSteer V2 自动化训练流水线 ==="
echo "日志目录: $LOG_DIR"
echo "开始时间: $(date)"

# 训练函数：Stage1→Stage2→Stage3
train_model() {
    local MODEL=$1
    local GPU=$2
    local PREFIX=$3

    echo "[$(date)] $MODEL: Stage 1 开始 (GPU $GPU)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train.py \
        --config configs/train_stage1${PREFIX}.yaml \
        > "$LOG_DIR/stage1_${MODEL}.log" 2>&1

    echo "[$(date)] $MODEL: Stage 2 开始 (GPU $GPU)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train.py \
        --config configs/train_stage2${PREFIX}.yaml \
        > "$LOG_DIR/stage2_${MODEL}.log" 2>&1

    echo "[$(date)] $MODEL: Stage 3 开始 (GPU $GPU)"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/train.py \
        --config configs/train_stage3${PREFIX}.yaml \
        > "$LOG_DIR/stage3_${MODEL}.log" 2>&1

    echo "[$(date)] $MODEL: 全部完成"
}

# 并行训练两个模型
train_model "qwen25" "0,1" "" &
PID_QWEN25=$!

train_model "qwen3" "2,3" "_qwen3" &
PID_QWEN3=$!

# 等待两个模型都完成
wait $PID_QWEN25 || { echo "❌ Qwen2.5 训练失败"; exit 1; }
wait $PID_QWEN3 || { echo "❌ Qwen3 训练失败"; exit 1; }

echo -e "\n=== 训练流水线完成 ==="
echo "结束时间: $(date)"
echo "日志目录: $LOG_DIR"