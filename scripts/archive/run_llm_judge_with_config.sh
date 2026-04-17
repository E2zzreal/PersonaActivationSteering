#!/bin/bash
# 使用配置文件的LLM Judge评估脚本

set -e

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
cd "$PROJECT_ROOT"

# 读取配置文件
CONFIG_FILE="configs/api_config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    echo "请先创建配置文件或编辑 configs/api_config.yaml"
    exit 1
fi

# 从配置文件提取API key
API_KEY=$(grep "api_key:" "$CONFIG_FILE" | grep -v "YOUR_" | head -1 | awk '{print $2}' | tr -d '"')
BASE_URL=$(grep "base_url:" "$CONFIG_FILE" | head -1 | awk '{print $2}' | tr -d '"')
MODEL=$(grep "model:" "$CONFIG_FILE" | grep -v "#" | head -1 | awk '{print $2}' | tr -d '"')

if [ -z "$API_KEY" ] || [ "$API_KEY" = "YOUR_BLSC_API_KEY_HERE" ]; then
    echo "错误: API key未配置"
    echo "请编辑 $CONFIG_FILE 并填入实际的API key"
    exit 1
fi

echo "=== API配置已加载 ==="
echo "Base URL: $BASE_URL"
echo "Model: $MODEL"
echo ""

# 导出环境变量
export BLSC_API_KEY="$API_KEY"
export BLSC_BASE_URL="$BASE_URL"

# 定义模型列表
STAGE3_MODELS=(
    "stage3_auto"
    "stage3_gate_init_0"
    "stage3_gate_reg_0.01_lr1e4"
    "stage3_gate_reg_0.05_lr5e5"
)

BASELINE_MODELS=(
    "baseline"
    "stage1"
    "stage2_best"
)

# 函数：评估单个模型
evaluate_model() {
    local model_name=$1
    local conv_file="results/conversations_${model_name}_*.json"
    
    # 查找匹配的文件
    local actual_file=$(ls $conv_file 2>/dev/null | head -1)
    
    if [ -z "$actual_file" ]; then
        echo "⚠️  未找到对话文件: $model_name"
        return 1
    fi
    
    local output_file="results/judge_eval_${model_name}.json"
    
    echo "评估: $model_name"
    echo "  输入: $actual_file"
    echo "  输出: $output_file"
    
    python /tmp/llm_judge_eval.py \
        --conversations "$actual_file" \
        --output "$output_file" \
        --judge_model "$MODEL"
    
    echo "✓ 完成: $model_name"
    echo ""
}

# 主流程
main() {
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║           LLM Judge 评估流程                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # 评估Stage 3模型
    echo "=== Stage 3 模型评估 ==="
    for model in "${STAGE3_MODELS[@]}"; do
        evaluate_model "$model"
    done
    
    # 评估Baseline/Stage1/Stage2模型
    echo "=== Baseline/Stage1/Stage2 模型评估 ==="
    for model in "${BASELINE_MODELS[@]}"; do
        evaluate_model "$model"
    done
    
    # 汇总结果
    echo "=== 生成评估报告 ==="
    python3 << 'SUMMARY'
import json
import glob
from pathlib import Path

print("\n" + "="*70)
print("PersonaSteer 模型评估结果汇总")
print("="*70)

eval_files = sorted(glob.glob("results/judge_eval_*.json"))

if not eval_files:
    print("未找到评估结果文件")
else:
    results = []
    
    for eval_file in eval_files:
        name = Path(eval_file).stem.replace("judge_eval_", "")
        
        with open(eval_file) as f:
            data = json.load(f)
        
        avg = data.get("overall_avg", 0)
        std = data.get("overall_std", 0)
        turns = data.get("total_turns", 0)
        
        results.append((name, avg, std, turns))
        
        print(f"\n{name}:")
        print(f"  平均分: {avg:.2f} ± {std:.2f}")
        print(f"  评估轮数: {turns}")
    
    # 排序并显示
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*70)
    print("🏆 模型排名:")
    print("="*70)
    for i, (name, avg, std, turns) in enumerate(sorted_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i}. {name:30} {avg:.2f} ± {std:.2f}")
    
    print("\n" + "="*70)
    print("📊 性能分析:")
    print("="*70)
    
    # 按类别分组
    categories = {
        "Stage 3": [r for r in sorted_results if "stage3" in r[0]],
        "Stage 2": [r for r in sorted_results if "stage2" in r[0]],
        "Stage 1": [r for r in sorted_results if "stage1" in r[0]],
        "Baseline": [r for r in sorted_results if "baseline" in r[0]]
    }
    
    for cat, models in categories.items():
        if models:
            avg_score = sum(m[1] for m in models) / len(models)
            print(f"{cat}: {avg_score:.2f} (平均)")
    
    print("\n" + "="*70)
    print("✅ 评估完成!")
    print("="*70)

SUMMARY
}

main "$@"
