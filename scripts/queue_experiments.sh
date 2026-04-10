#!/bin/bash
# 串行实验队列：第一批完成后自动启动第二批

set -e

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
PYTHON="python"
cd "$PROJECT_ROOT"

# 第二批实验：测试gate_reg_weight和learning_rate组合
BATCH2_EXPERIMENTS=(
    "0:gate_reg_0.001_lr5e5:-2.0:0.001:0.00005"
    "1:gate_reg_0.05_lr5e5:-2.0:0.05:0.00005"
    "2:gate_reg_0.01_lr3e5:-2.0:0.01:0.00003"
    "3:gate_reg_0.01_lr1e4:-2.0:0.01:0.0001"
)

echo "=== 等待第一批实验完成 ==="
while pgrep -f "exp_gate_init" > /dev/null; do
    sleep 60
    echo "[$(date +%H:%M)] 第一批实验仍在运行..."
done

echo ""
echo "[$(date)] 第一批实验完成，启动第二批..."
sleep 10

# 启动第二批
for exp in "${BATCH2_EXPERIMENTS[@]}"; do
    IFS=':' read -r gpu_id exp_name gate_bias gate_reg lr <<< "$exp"

    echo ""
    echo "[GPU $gpu_id] 启动: $exp_name"
    echo "  gate_init_bias=$gate_bias, gate_reg_weight=$gate_reg, lr=$lr"

    config_file="configs/exp_${exp_name}.yaml"
    cp configs/train_stage2_qwen3.yaml "$config_file"

    sed -i "s/gate_init_bias: -[0-9.]\+/gate_init_bias: $gate_bias/" "$config_file"
    sed -i "s/gate_reg_weight: [0-9.]\+/gate_reg_weight: $gate_reg/" "$config_file"
    sed -i "s/learning_rate: [0-9.]\+/learning_rate: $lr/" "$config_file"
    sed -i "s|output_dir: checkpoints/stage2_qwen3_v2|output_dir: checkpoints/exp_${exp_name}|" "$config_file"

    CUDA_VISIBLE_DEVICES=$gpu_id nohup $PYTHON scripts/train.py \
        --config "$config_file" \
        > "logs/exp_${exp_name}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

    echo "[GPU $gpu_id] PID: $!"
    sleep 2
done

echo ""
echo "[$(date)] 第二批实验已启动"
