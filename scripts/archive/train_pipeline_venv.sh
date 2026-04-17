#!/bin/bash
# PersonaSteer V2 完整训练流水线（使用虚拟环境）

set -e

MODEL=$1
DEVICE=$2

if [ -z "$MODEL" ] || [ -z "$DEVICE" ]; then
    echo "用法: $0 <model> <device>"
    echo "示例: $0 qwen25 cuda:0"
    echo "示例: $0 qwen3 cuda:2"
    exit 1
fi

# 激活虚拟环境
source /home/kemove/Desktop/PersonaSteer/venv/bin/activate

echo "========================================="
echo "PersonaSteer V2 训练流水线"
echo "模型: $MODEL"
echo "设备: $DEVICE"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "========================================="

# 根据模型选择配置
if [ "$MODEL" = "qwen25" ]; then
    CONFIGS=("configs/train_stage1.yaml" "configs/train_stage2.yaml" "configs/train_stage3.yaml")
    CHECKPOINTS=("checkpoints/stage1_fixed" "checkpoints/stage2_fixed" "checkpoints/stage3_fixed")
    MODEL_NAME="Qwen2.5-3B"
elif [ "$MODEL" = "qwen3" ]; then
    CONFIGS=("configs/train_stage1_qwen3.yaml" "configs/train_stage2_qwen3.yaml" "configs/train_stage3_qwen3.yaml")
    CHECKPOINTS=("checkpoints/stage1_qwen3_fixed" "checkpoints/stage2_qwen3_fixed" "checkpoints/stage3_qwen3_fixed")
    MODEL_NAME="Qwen3-4B"
else
    echo "错误: 未知模型 $MODEL"
    exit 1
fi

# 训练三个阶段
for i in {0..2}; do
    STAGE=$((i+1))
    CONFIG=${CONFIGS[$i]}
    CHECKPOINT_DIR=${CHECKPOINTS[$i]}
    LOG_FILE="logs/train_stage${STAGE}_${MODEL}_fixed.log"
    
    echo ""
    echo "========================================="
    echo "Stage $STAGE / 3"
    echo "配置: $CONFIG"
    echo "输出: $CHECKPOINT_DIR"
    echo "========================================="
    
    mkdir -p $CHECKPOINT_DIR logs
    
    # 加载上一阶段checkpoint
    RESUME_ARG=""
    if [ $STAGE -eq 2 ]; then
        PREV_CHECKPOINT="${CHECKPOINTS[0]}/best.pt"
        if [ -f "$PREV_CHECKPOINT" ]; then
            RESUME_ARG="--resume $PREV_CHECKPOINT"
            echo "加载: $PREV_CHECKPOINT"
        fi
    elif [ $STAGE -eq 3 ]; then
        PREV_CHECKPOINT="${CHECKPOINTS[1]}/best.pt"
        if [ -f "$PREV_CHECKPOINT" ]; then
            RESUME_ARG="--resume $PREV_CHECKPOINT"
            echo "加载: $PREV_CHECKPOINT"
        fi
    fi
    
    # 训练
    python scripts/train.py \
        --config "$CONFIG" \
        --device "$DEVICE" \
        --output "$CHECKPOINT_DIR" \
        $RESUME_ARG \
        2>&1 | tee "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ Stage $STAGE 完成"
    else
        echo "❌ Stage $STAGE 失败"
        exit 1
    fi
done

echo ""
echo "========================================="
echo "✅ 全部训练完成！"
echo "模型: $MODEL_NAME"
echo "Checkpoints:"
for ckpt in "${CHECKPOINTS[@]}"; do
    echo "  - $ckpt"
done
echo "========================================="
