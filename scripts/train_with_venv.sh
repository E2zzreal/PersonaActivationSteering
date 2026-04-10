#!/bin/bash
# PersonaSteer V2 训练脚本（使用虚拟环境）
# 所有训练都必须在虚拟环境中执行

set -e

# 激活虚拟环境
source /home/kemove/Desktop/PersonaSteer/venv/bin/activate

# 验证虚拟环境
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

# 运行训练
python scripts/train.py "$@"
