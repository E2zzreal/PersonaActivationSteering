#!/bin/bash
# PersonaSteer V2 完整训练流水线
# 按顺序训练Stage 1-3

set -e  # 遇到错误立即退出

MODEL=$1
DEVICE=$2

if [ -z "$MODEL" ] || [ -z "$DEVICE" ]; then
    echo "用法: $0 <model> <device>"
    echo "示例: $0 qwen25 cuda:0"
    echo "示例: $0 qwen3 cuda:2"
    exit 1
fi

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
    echo "支持: qwen25, qwen3"
    exit 1
fi

echo "========================================="
echo "PersonaSteer V2 训练流水线"
echo "模型: $MODEL_NAME"
echo "设备: $DEVICE"
echo "========================================="

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
    echo "日志: $LOG_FILE"
    echo "========================================="
    
    # 创建输出目录
    mkdir -p $CHECKPOINT_DIR
    mkdir -p logs
    
    # 如果是Stage 2或3，需要加载上一阶段的checkpoint
    RESUME_ARG=""
    if [ $STAGE -eq 2 ]; then
        PREV_CHECKPOINT="${CHECKPOINTS[0]}/best.pt"
        if [ -f "$PREV_CHECKPOINT" ]; then
            RESUME_ARG="--resume $PREV_CHECKPOINT"
            echo "加载上一阶段checkpoint: $PREV_CHECKPOINT"
        else
            echo "警告: 未找到Stage 1 checkpoint，将从头开始训练"
        fi
    elif [ $STAGE -eq 3 ]; then
        PREV_CHECKPOINT="${CHECKPOINTS[1]}/best.pt"
        if [ -f "$PREV_CHECKPOINT" ]; then
            RESUME_ARG="--resume $PREV_CHECKPOINT"
            echo "加载上一阶段checkpoint: $PREV_CHECKPOINT"
        else
            echo "警告: 未找到Stage 2 checkpoint，将从头开始训练"
        fi
    fi
    
    # 开始训练
    echo "开始训练 Stage $STAGE..."
    python3 scripts/train.py \
        --config "$CONFIG" \
        --device "$DEVICE" \
        --output "$CHECKPOINT_DIR" \
        $RESUME_ARG \
        2>&1 | tee "$LOG_FILE"
    
    # 检查训练是否成功
    if [ $? -eq 0 ]; then
        echo "✅ Stage $STAGE 训练完成"
    else
        echo "❌ Stage $STAGE 训练失败"
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
