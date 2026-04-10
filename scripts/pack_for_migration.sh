#!/bin/bash
# 一键打包迁移脚本

set -e

PROJECT_ROOT="/home/kemove/Desktop/PersonaSteer"
OUTPUT_DIR="$HOME/personasteer_migration"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

echo "=== PersonaSteer V2 迁移打包 ==="
echo ""

# 1. 打包代码和配置
echo "[1/3] 打包代码..."
tar -czf "$OUTPUT_DIR/code.tar.gz" \
    --exclude='checkpoints/*' \
    --exclude='results/*' \
    --exclude='logs/*' \
    --exclude='venv/*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    configs/ src/ scripts/ data/ requirements.txt README.md docs/

# 2. 打包模型（可选）
echo "[2/3] 打包模型（可选，按y继续）..."
read -p "是否打包Qwen模型？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    tar -czf "$OUTPUT_DIR/models.tar.gz" \
        ~/.cache/modelscope/Qwen/Qwen3-4B \
        ~/.cache/modelscope/Qwen/Qwen2___5-3B
    echo "✓ 模型已打包"
else
    echo "⊘ 跳过模型打包"
fi

# 3. 打包Stage1 checkpoint（必需）
echo "[3/3] 打包Stage1 checkpoint..."
tar -czf "$OUTPUT_DIR/stage1_checkpoints.tar.gz" \
    checkpoints/stage1_qwen3/ \
    checkpoints/stage1/

echo ""
echo "=== 打包完成 ==="
echo "输出目录: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
echo ""
echo "传输到目标服务器："
echo "  scp $OUTPUT_DIR/*.tar.gz user@target:/path/"
