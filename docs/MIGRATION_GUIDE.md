# PersonaSteer V2 训练迁移指南

## 前置准备

### 目标服务器要求
- CUDA 11.8+ / 12.x
- Python 3.10+
- 至少4张GPU（24GB显存）
- 磁盘空间：~200GB（模型+checkpoint+数据）

---

## 迁移步骤

### 1. 打包必要文件

在当前服务器执行：

```bash
cd /home/kemove/Desktop/Projects/3-PersonaSteer_V2

# 创建迁移包（排除大文件）
tar -czf personasteer_v2_migration.tar.gz \
    --exclude='checkpoints/*' \
    --exclude='results/*' \
    --exclude='logs/*' \
    --exclude='venv/*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    configs/ \
    src/ \
    scripts/ \
    data/ \
    requirements.txt \
    README.md

echo "迁移包已创建: personasteer_v2_migration.tar.gz"
```

### 2. 传输到目标服务器

```bash
# 方式1: scp
scp personasteer_v2_migration.tar.gz user@target-server:/path/to/destination/

# 方式2: rsync（推荐，支持断点续传）
rsync -avz --progress personasteer_v2_migration.tar.gz user@target-server:/path/to/destination/
```

### 3. 目标服务器配置

```bash
# 解压
cd /path/to/destination
tar -xzf personasteer_v2_migration.tar.gz

# 创建conda环境
conda create -n pytorch python=3.12 -y
conda activate pytorch

# 安装PyTorch（根据CUDA版本）
# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装依赖
pip install -r requirements.txt

# 下载模型（如果目标服务器无法访问modelscope）
# 需要手动传输 ~/.cache/modelscope/Qwen/ 目录
```

### 4. 模型文件迁移（可选）

如果目标服务器无法下载模型：

```bash
# 当前服务器打包模型
tar -czf qwen_models.tar.gz ~/.cache/modelscope/Qwen/

# 传输到目标服务器
scp qwen_models.tar.gz user@target-server:/home/user/

# 目标服务器解压
mkdir -p ~/.cache/modelscope/
tar -xzf qwen_models.tar.gz -C ~/.cache/modelscope/
```

### 5. 继续训练（可选）

如果需要继续当前训练：

```bash
# 打包checkpoint
tar -czf checkpoints_stage1.tar.gz checkpoints/stage1_qwen3/

# 传输并解压到目标服务器相同路径
```

### 6. 启动训练

```bash
cd /path/to/PersonaSteer_V2

# 测试环境
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 启动实验
./scripts/parallel_gate_experiments.sh
```

---

## 配置文件修改

目标服务器上需要修改的配置：

1. **模型路径**（如果不同）：
```bash
# 批量修改configs/*.yaml中的base_model路径
sed -i 's|/home/kemove/.cache/modelscope|/home/newuser/.cache/modelscope|g' configs/*.yaml
```

2. **GPU设备**（如果GPU数量不同）：
```bash
# 修改configs/*.yaml中的device配置
# 例如：device: cuda:0
```

---

## 验证清单

- [ ] PyTorch安装成功，CUDA可用
- [ ] 模型文件存在且可加载
- [ ] 数据文件完整（data/split/*.jsonl）
- [ ] 配置文件路径正确
- [ ] GPU显存充足（nvidia-smi）
- [ ] 测试训练脚本可运行

---

## 最小迁移（仅代码）

如果目标服务器已有模型和环境：

```bash
# 仅传输代码
rsync -avz --exclude='checkpoints' --exclude='results' --exclude='logs' \
    /home/kemove/Desktop/Projects/3-PersonaSteer_V2/ \
    user@target-server:/path/to/destination/
```
