#!/bin/bash
# 对话生成流水线 - 串行执行确保稳定

set -e

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/generation_pipeline_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "对话生成流水线 - ${TIMESTAMP}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 生成单个模型
generate_model() {
    local name=$1
    local checkpoint=$2
    local config=$3
    local gpu=$4

    local output="results/conversations_${name}_${TIMESTAMP}.json"

    echo "" | tee -a "$LOG_FILE"
    echo "[$name] 开始生成..." | tee -a "$LOG_FILE"
    echo "  Checkpoint: $checkpoint" | tee -a "$LOG_FILE"
    echo "  Config: $config" | tee -a "$LOG_FILE"
    echo "  GPU: $gpu" | tee -a "$LOG_FILE"
    echo "  Output: $output" | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES=$gpu python scripts/generate_all_conversations_fixed.py \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --output "$output" \
        --num_samples 50 \
        --max_new_tokens 150 \
        --gpu 0 \
        2>&1 | tee -a "$LOG_FILE"

    if [ -f "$output" ]; then
        echo "[$name] ✓ 完成" | tee -a "$LOG_FILE"
    else
        echo "[$name] ✗ 失败" | tee -a "$LOG_FILE"
    fi
}

# 模型列表
declare -a MODELS=(
    "baseline:baseline:configs/train_stage1_qwen3.yaml:0"
    "stage1:checkpoints/stage1/best.pt:configs/train_stage1.yaml:0"
    "stage1_qwen3:checkpoints/stage1_qwen3/best.pt:configs/train_stage1_qwen3.yaml:0"
    "exp_gate_init_0:checkpoints/exp_gate_init_0/best.pt:configs/exp_gate_init_0.yaml:0"
    "exp_gate_init_neg1:checkpoints/exp_gate_init_neg1/best.pt:configs/exp_gate_init_neg1.yaml:0"
    "exp_gate_init_neg2:checkpoints/exp_gate_init_neg2/best.pt:configs/exp_gate_init_neg2.yaml:0"
    "exp_gate_init_neg3:checkpoints/exp_gate_init_neg3/best.pt:configs/exp_gate_init_neg3.yaml:0"
    "exp_gate_reg_0.001_lr5e5:checkpoints/exp_gate_reg_0.001_lr5e5/best.pt:configs/exp_gate_reg_0.001_lr5e5.yaml:0"
    "exp_gate_reg_0.01_lr1e4:checkpoints/exp_gate_reg_0.01_lr1e4/best.pt:configs/exp_gate_reg_0.01_lr1e4.yaml:0"
    "exp_gate_reg_0.01_lr3e5:checkpoints/exp_gate_reg_0.01_lr3e5/best.pt:configs/exp_gate_reg_0.01_lr3e5.yaml:0"
    "exp_gate_reg_0.05_lr5e5:checkpoints/exp_gate_reg_0.05_lr5e5/best.pt:configs/exp_gate_reg_0.05_lr5e5.yaml:0"
    "stage3_auto:checkpoints/stage3_auto/best.pt:configs/train_stage3_auto.yaml:0"
    "stage3_gate_init_0:checkpoints/stage3_gate_init_0/best.pt:configs/train_stage3_gate_init_0.yaml:0"
    "stage3_gate_reg_0.01_lr1e4:checkpoints/stage3_gate_reg_0.01_lr1e4/best.pt:configs/train_stage3_gate_reg_0.01_lr1e4.yaml:0"
    "stage3_gate_reg_0.05_lr5e5:checkpoints/stage3_gate_reg_0.05_lr5e5/best.pt:configs/train_stage3_gate_reg_0.05_lr5e5.yaml:0"
)

# 执行生成
total=${#MODELS[@]}
current=0

for model_info in "${MODELS[@]}"; do
    IFS=':' read -r name checkpoint config gpu <<< "$model_info"
    current=$((current + 1))

    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "[$current/$total] 处理: $name" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    generate_model "$name" "$checkpoint" "$config" "$gpu"

done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "生成完成!" | tee -a "$LOG_FILE"
echo "日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
