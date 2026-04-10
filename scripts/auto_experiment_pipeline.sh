#!/bin/bash
# PersonaSteer 自动化实验流水线
# 1. 监控第一批实验完成
# 2. 启动第二批实验
# 3. 分析结果
# 4. 启动Stage 3

set -e

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
cd "$PROJECT_ROOT"

LOG_FILE="logs/auto_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 第一批实验配置
BATCH1_EXPERIMENTS=(
    "0:gate_init_neg1:-1.0"
    "1:gate_init_neg2:-2.0"
    "2:gate_init_neg3:-3.0"
    "3:gate_init_0:0.0"
)

# 第二批实验配置 (gate_reg_weight + learning_rate)
BATCH2_EXPERIMENTS=(
    "0:gate_reg_0.001_lr5e5:0.001:0.00005"
    "1:gate_reg_0.01_lr1e4:0.01:0.0001"
    "2:gate_reg_0.01_lr3e5:0.01:0.00003"
    "3:gate_reg_0.05_lr5e5:0.05:0.00005"
)

# 检查训练进程是否在运行
check_training_running() {
    pgrep -f "train.py" > /dev/null 2>&1
}

# 获取当前运行的实验数量
count_running_experiments() {
    pgrep -f "train.py" | wc -l
}

# 等待第一批实验完成
wait_for_batch1() {
    log "等待第一批实验完成..."

    while check_training_running; do
        running=$(count_running_experiments)
        log "当前运行中: $running 个实验"

        # 显示进度
        for exp in "${BATCH1_EXPERIMENTS[@]}"; do
            IFS=':' read -r gpu_id exp_name _ <<< "$exp"
            log_file=$(ls -t logs/exp_${exp_name}_*.log 2>/dev/null | head -1)
            if [ -f "$log_file" ]; then
                progress=$(tail -100 "$log_file" | grep -oP "Epoch \d+/\d+" | tail -1)
                loss=$(tail -50 "$log_file" | grep -oP "loss=[\d.]+" | tail -1)
                log "  [$exp_name] $progress, $loss"
            fi
        done

        sleep 300  # 每5分钟检查一次
    done

    log "第一批实验全部完成!"
}

# 启动第二批实验
start_batch2() {
    log "=== 启动第二批实验 (gate_reg_weight + learning_rate) ==="

    for exp in "${BATCH2_EXPERIMENTS[@]}"; do
        IFS=':' read -r gpu_id exp_name gate_reg lr <<< "$exp"

        log "[GPU $gpu_id] 启动实验: $exp_name (gate_reg=$gate_reg, lr=$lr)"

        # 使用现有配置文件
        config_file="configs/exp_${exp_name}.yaml"

        if [ ! -f "$config_file" ]; then
            log "配置文件不存在: $config_file，跳过"
            continue
        fi

        # 后台启动训练
        CUDA_VISIBLE_DEVICES=$gpu_id nohup python scripts/train.py \
            --config "$config_file" \
            > "logs/exp_${exp_name}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

        log "[GPU $gpu_id] PID: $!"
        sleep 2
    done

    log "第二批实验已启动"
}

# 等待第二批实验完成
wait_for_batch2() {
    log "等待第二批实验完成..."

    while check_training_running; do
        running=$(count_running_experiments)
        log "当前运行中: $running 个实验"
        sleep 300
    done

    log "第二批实验全部完成!"
}

# 分析实验结果
analyze_results() {
    log "=== 分析实验结果 ==="

    python3 << 'PYTHON_SCRIPT'
import os
import json
import re
from pathlib import Path

results = {}

# 分析第一批实验
batch1_names = ["gate_init_neg1", "gate_init_neg2", "gate_init_neg3", "gate_init_0"]
batch2_names = ["gate_reg_0.001_lr5e5", "gate_reg_0.01_lr1e4", "gate_reg_0.01_lr3e5", "gate_reg_0.05_lr5e5"]

def parse_log(log_path):
    """解析日志文件提取关键指标"""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # 提取每个epoch结束时的loss
    epoch_losses = []
    for match in re.finditer(r'Epoch (\d+)/\d+.*?best_loss=([0-9.]+|inf)', content):
        epoch = int(match.group(1))
        loss_str = match.group(2)
        loss = float(loss_str) if loss_str != 'inf' else float('inf')
        epoch_losses.append((epoch, loss))

    # 提取最后的best_loss
    best_loss_match = re.search(r'best_loss=([0-9.]+)(?!.*best_loss)', content)
    best_loss = float(best_loss_match.group(1)) if best_loss_match else None

    return {
        'epoch_losses': epoch_losses,
        'best_loss': best_loss
    }

print("\n=== 第一批实验结果 (gate_init_bias) ===")
batch1_results = {}
for name in batch1_names:
    log_files = list(Path('logs').glob(f'exp_{name}_*.log'))
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        result = parse_log(str(latest_log))
        if result:
            batch1_results[name] = result
            print(f"{name}: best_loss={result['best_loss']:.4f}" if result['best_loss'] else f"{name}: 无数据")

print("\n=== 第二批实验结果 (gate_reg + lr) ===")
batch2_results = {}
for name in batch2_names:
    log_files = list(Path('logs').glob(f'exp_{name}_*.log'))
    if log_files:
        latest_log = max(log_files, key=os.path.getmtime)
        result = parse_log(str(latest_log))
        if result:
            batch2_results[name] = result
            print(f"{name}: best_loss={result['best_loss']:.4f}" if result['best_loss'] else f"{name}: 无数据")

# 选择最佳配置
all_results = {**batch1_results, **batch2_results}
valid_results = {k: v for k, v in all_results.items() if v and v.get('best_loss')}

if valid_results:
    best_exp = min(valid_results.items(), key=lambda x: x[1]['best_loss'])
    print(f"\n=== 最佳实验 ===")
    print(f"实验名称: {best_exp[0]}")
    print(f"Best Loss: {best_exp[1]['best_loss']:.4f}")

    # 保存结果
    with open('logs/experiment_analysis.json', 'w') as f:
        json.dump({
            'batch1': {k: v for k, v in batch1_results.items() if v},
            'batch2': {k: v for k, v in batch2_results.items() if v},
            'best_experiment': best_exp[0],
            'best_loss': best_exp[1]['best_loss']
        }, f, indent=2)
    print(f"\n结果已保存到 logs/experiment_analysis.json")
else:
    print("\n警告: 没有找到有效的实验结果")
PYTHON_SCRIPT

}

# 启动Stage 3训练
start_stage3() {
    log "=== 启动 Stage 3 训练 ==="

    # 读取最佳实验配置
    if [ -f "logs/experiment_analysis.json" ]; then
        best_exp=$(python3 -c "import json; print(json.load(open('logs/experiment_analysis.json'))['best_experiment'])")
        log "最佳实验: $best_exp"

        # 查找对应的checkpoint
        checkpoint_dir="checkpoints/exp_${best_exp}"
        if [ -d "$checkpoint_dir" ]; then
            best_checkpoint="${checkpoint_dir}/best.pt"
            log "使用checkpoint: $best_checkpoint"
        else
            log "警告: checkpoint目录不存在，使用默认"
            best_checkpoint="checkpoints/stage2_qwen3/best.pt"
        fi
    else
        log "警告: 未找到分析结果，使用默认配置"
        best_checkpoint="checkpoints/stage2_qwen3/best.pt"
    fi

    # 创建Stage 3配置
    stage3_config="configs/train_stage3_auto.yaml"
    cp configs/train_stage3_qwen3_nodual.yaml "$stage3_config"

    # 更新配置
    sed -i "s|stage2_checkpoint:.*|stage2_checkpoint: $best_checkpoint|" "$stage3_config"
    sed -i "s|output_dir:.*|output_dir: checkpoints/stage3_auto|g" "$stage3_config"

    log "启动Stage 3训练..."
    CUDA_VISIBLE_DEVICES=0 nohup python scripts/train.py \
        --config "$stage3_config" \
        > "logs/stage3_auto_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

    log "Stage 3 训练已启动, PID: $!"
}

# 启动第一批实验
start_batch1() {
    log "=== 启动第一批实验 (gate_init_bias) ==="

    for exp in "${BATCH1_EXPERIMENTS[@]}"; do
        IFS=':' read -r gpu_id exp_name gate_bias <<< "$exp"

        log "[GPU $gpu_id] 启动实验: $exp_name (gate_init_bias=$gate_bias)"

        # 创建实验专用配置
        config_file="configs/exp_${exp_name}.yaml"
        cp configs/train_stage2_qwen3.yaml "$config_file"

        # 修改gate_init_bias
        sed -i "s/gate_init_bias: -[0-9.]\+/gate_init_bias: $gate_bias/" "$config_file"

        # 修改output_dir
        sed -i "s|output_dir: checkpoints/stage2_qwen3_v2|output_dir: checkpoints/exp_${exp_name}|" "$config_file"

        # 后台启动训练
        CUDA_VISIBLE_DEVICES=$gpu_id nohup python scripts/train.py \
            --config "$config_file" \
            > "logs/exp_${exp_name}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

        log "[GPU $gpu_id] PID: $!"
        sleep 2
    done

    log "第一批实验已启动"
}

# 主流程
main() {
    log "=== PersonaSteer 自动化实验流水线启动 ==="

    # 检查第一批实验是否还在运行
    if check_training_running; then
        log "检测到实验正在运行"
        wait_for_batch1
    else
        log "没有运行中的实验，启动第一批实验"
        start_batch1
        wait_for_batch1
    fi

    # 分析第一批结果
    log "分析第一批实验结果..."
    analyze_results

    # 启动第二批实验
    start_batch2
    wait_for_batch2

    # 分析所有结果
    log "分析所有实验结果..."
    analyze_results

    # 启动Stage 3
    start_stage3

    log "=== 自动化流水线完成 ==="
}

main "$@"