#!/bin/bash
# Gate初始化方案并行实验
# 在4张GPU上测试不同gate_init_bias值

set -e

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
PYTHON="python"
cd "$PROJECT_ROOT"

# 停止当前训练
pkill -9 -f "train.py" 2>/dev/null || true
sleep 3

# 实验方案：测试4个不同的gate_init_bias值
EXPERIMENTS=(
    "0:gate_init_neg1:-1.0"      # sigmoid(-1) ≈ 0.27
    "1:gate_init_neg2:-2.0"      # sigmoid(-2) ≈ 0.12
    "2:gate_init_neg3:-3.0"      # sigmoid(-3) ≈ 0.05
    "3:gate_init_0:0.0"          # sigmoid(0) = 0.50
)

echo "=== Gate初始化并行实验 ==="
echo "开始时间: $(date)"
echo "实验方案: ${#EXPERIMENTS[@]}个"

# 启动并行训练
for exp in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r gpu_id exp_name gate_bias <<< "$exp"

    echo ""
    echo "[GPU $gpu_id] 启动实验: $exp_name (gate_init_bias=$gate_bias)"

    # 创建实验专用配置
    config_file="configs/exp_${exp_name}.yaml"
    cp configs/train_stage2_qwen3.yaml "$config_file"

    # 修改gate_init_bias
    sed -i "s/gate_init_bias: -[0-9.]\+/gate_init_bias: $gate_bias/" "$config_file"

    # 修改output_dir
    sed -i "s|output_dir: checkpoints/stage2_qwen3_v2|output_dir: checkpoints/exp_${exp_name}|" "$config_file"

    # 后台启动训练
    CUDA_VISIBLE_DEVICES=$gpu_id nohup $PYTHON scripts/train.py \
        --config "$config_file" \
        > "logs/exp_${exp_name}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

    echo "[GPU $gpu_id] PID: $!"
    sleep 2
done

echo ""
echo "=== 所有实验已启动 ==="
echo "监控命令: watch -n 10 'nvidia-smi; echo; ps aux | grep train.py | grep -v grep'"
echo "查看日志: tail -f logs/exp_gate_init_*.log"
