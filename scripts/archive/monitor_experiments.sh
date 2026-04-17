#!/bin/bash
# 实验监控脚本 - 实时显示训练进度

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
cd "$PROJECT_ROOT"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           PersonaSteer 实验监控 - $(date '+%Y-%m-%d %H:%M:%S')            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# 检查自动化流水线状态
if pgrep -f "auto_experiment_pipeline.sh" > /dev/null; then
    echo "✓ 自动化流水线运行中"
else
    echo "○ 自动化流水线未运行"
fi

echo ""
echo "┌──────────────────────────────────────────────────────────────────┐"
echo "│                        当前训练进度                               │"
echo "├──────────────────────────────────────────────────────────────────┤"

# 统计运行中的实验
running=$(pgrep -f "train.py" | wc -l)
echo "│ 运行中: $running 个实验                                          │"
echo "├──────────────────────────────────────────────────────────────────┤"

# 显示每个实验的进度
echo "│ 第一批实验 (gate_init_bias):                                     │"
for name in gate_init_neg1 gate_init_neg2 gate_init_neg3 gate_init_0; do
    log_file=$(ls -t logs/exp_${name}_*.log 2>/dev/null | head -1)
    if [ -f "$log_file" ]; then
        epoch=$(tail -200 "$log_file" | grep -oP "Epoch \d+/\d+" | tail -1)
        step=$(tail -200 "$log_file" | grep -oP "\d+/2777" | tail -1)
        loss=$(tail -50 "$log_file" | grep -oP "loss=[\d.]+" | tail -1 | cut -d= -f2)

        # 计算进度百分比
        if [ -n "$step" ]; then
            current=$(echo $step | cut -d/ -f1)
            pct=$(echo "scale=1; $current * 100 / 2777" | bc 2>/dev/null || echo "0")
            printf "│   %-20s %s  Step: %s (%4.1f%%)  Loss: %s\n" "$name" "$epoch" "$step" "$pct" "$loss"
        fi
    fi
done

echo "├──────────────────────────────────────────────────────────────────┤"
echo "│ 第二批实验 (gate_reg + lr):                                      │"
for name in gate_reg_0.001_lr5e5 gate_reg_0.01_lr1e4 gate_reg_0.01_lr3e5 gate_reg_0.05_lr5e5; do
    log_file=$(ls -t logs/exp_${name}_*.log 2>/dev/null | head -1)
    if [ -f "$log_file" ]; then
        epoch=$(tail -200 "$log_file" | grep -oP "Epoch \d+/\d+" | tail -1)
        step=$(tail -200 "$log_file" | grep -oP "\d+/2777" | tail -1)
        loss=$(tail -50 "$log_file" | grep -oP "loss=[\d.]+" | tail -1 | cut -d= -f2)

        if [ -n "$step" ]; then
            current=$(echo $step | cut -d/ -f1)
            pct=$(echo "scale=1; $current * 100 / 2777" | bc 2>/dev/null || echo "0")
            printf "│   %-20s %s  Step: %s (%4.1f%%)  Loss: %s\n" "$name" "$epoch" "$step" "$pct" "$loss"
        else
            printf "│   %-20s 等待中...\n" "$name"
        fi
    else
        printf "│   %-20s 未开始\n" "$name"
    fi
done

echo "├──────────────────────────────────────────────────────────────────┤"
echo "│ GPU 状态:                                                        │"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader | \
    while IFS=, read -r idx mem util temp; do
        printf "│   GPU %s: %s 显存, %s 利用率, %s°C\n" "$idx" "$mem" "$util" "$temp"
    done
echo "└──────────────────────────────────────────────────────────────────┘"

# 显示分析结果
if [ -f "logs/experiment_analysis.json" ]; then
    echo ""
    echo "┌──────────────────────────────────────────────────────────────────┐"
    echo "│ 实验分析结果:                                                    │"
    python3 -c "
import json
try:
    data = json.load(open('logs/experiment_analysis.json'))
    print(f'│   最佳实验: {data.get(\"best_experiment\", \"N/A\")}')
    print(f'│   Best Loss: {data.get(\"best_loss\", \"N/A\"):.4f}' if data.get('best_loss') else '│   Best Loss: N/A')
except:
    print('│   解析失败')
"
    echo "└──────────────────────────────────────────────────────────────────┘"
fi

echo ""
echo "命令:"
echo "  查看日志:     tail -f logs/exp_*.log"
echo "  查看流水线:   tail -f logs/auto_pipeline_*.log"
echo "  持续监控:     watch -n 60 ./scripts/monitor_experiments.sh"