#!/bin/bash
# 自动化训练+评估流水线

set -e

# 激活虚拟环境
source /home/kemove/Desktop/PersonaSteer/venv/bin/activate

echo "========================================="
echo "PersonaSteer V2 - 自动化流程"
echo "========================================="

# 函数：等待GPU空闲
wait_gpu() {
    local gpu_id=$1
    local threshold=500  # 500MB以下视为空闲

    while true; do
        memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id | tr -d ' ')
        if [ "$memory" -lt "$threshold" ]; then
            echo "[GPU $gpu_id] 空闲 (${memory}MB)"
            break
        fi
        echo "[GPU $gpu_id] 等待中... (${memory}MB)"
        sleep 30
    done
}

# 函数：训练单个Stage
train_stage() {
    local model=$1
    local stage=$2
    local gpu=$3
    local resume=$4

    export CUDA_VISIBLE_DEVICES=$gpu

    local config=""
    local output=""

    if [ "$model" = "qwen25" ]; then
        config="configs/train_stage${stage}.yaml"
        output="checkpoints/stage${stage}_fixed"
    else
        config="configs/train_stage${stage}_qwen3.yaml"
        output="checkpoints/stage${stage}_qwen3_fixed"
    fi

    echo ""
    echo "=== [$model] Stage $stage 训练 ==="
    echo "GPU: $gpu, Config: $config"

    local resume_arg=""
    if [ -n "$resume" ] && [ -f "$resume" ]; then
        resume_arg="--resume $resume"
    fi

    python scripts/train.py \
        --config "$config" \
        --device cuda \
        --output "$output" \
        $resume_arg \
        2>&1 | tee "logs/train_stage${stage}_${model}_auto.log"

    echo "✅ [$model] Stage $stage 完成"
}

# 函数：评估模型
evaluate_model() {
    local model=$1
    local gpu=$2
    local checkpoint_dir=$3

    export CUDA_VISIBLE_DEVICES=$gpu

    echo ""
    echo "=== [$model] 评估 ==="
    echo "GPU: $gpu, Checkpoint: $checkpoint_dir"

    python scripts/complete_comparison_eval.py \
        --model $model \
        --num_samples 50 \
        --device cuda:0 \
        2>&1 | tee "logs/eval_${model}_auto.log"

    echo "✅ [$model] 评估完成"
}

# 主流程
echo ""
echo "=== 第一轮：旧Checkpoint评估（已启动）==="
echo "等待评估完成..."

# 等待GPU 1和GPU 3的评估完成
wait_gpu 1
wait_gpu 3
echo "✅ 旧Checkpoint评估完成"

echo ""
echo "=== 第二轮：等待训练完成并评估新Checkpoint ==="

# Qwen2.5-3B训练流程
echo ""
echo "--- Qwen2.5-3B 训练 ---"
wait_gpu 0  # 等待GPU 0空闲
train_stage "qwen25" 3 0 "checkpoints/stage2_fixed/best.pt"

# Qwen3-4B训练流程
echo ""
echo "--- Qwen3-4B 训练 ---"
wait_gpu 2  # 等待GPU 2空闲
train_stage "qwen3" 2 2 "checkpoints/stage1_qwen3_fixed/best.pt"
train_stage "qwen3" 3 2 "checkpoints/stage2_qwen3_fixed/best.pt"

echo ""
echo "=== 第三轮：新Checkpoint评估 ==="

# 并行评估两个模型的新checkpoint
(
    export CUDA_VISIBLE_DEVICES=0
    evaluate_model "qwen25" 0 "checkpoints/stage3_fixed"
) &

(
    export CUDA_VISIBLE_DEVICES=2
    evaluate_model "qwen3" 2 "checkpoints/stage3_qwen3_fixed"
) &

wait

echo ""
echo "========================================="
echo "✅ 全部流程完成！"
echo ""
echo "结果对比："
echo "  旧checkpoint: logs/eval_qwen25_gpu1.log, logs/eval_qwen3_gpu3.log"
echo "  新checkpoint: logs/eval_qwen25_auto.log, logs/eval_qwen3_auto.log"
echo "========================================="
