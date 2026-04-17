#!/usr/bin/env python
"""
PersonaSteer 训练脚本
支持三阶段渐进训练
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.aloe_dataset import ALOEDataset
from src.data.collator import PersonaSteerCollator
from src.data.grouped_sampler import PersonalityGroupedSampler
from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig
from src.training.trainer import PersonaSteerTrainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Train PersonaSteer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for checkpoints (overrides config)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data (overrides config)",
    )

    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Path to evaluation data (overrides config)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,  # 默认None，使用配置文件中的值
        help="Device to use for training (overrides config file)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, device: str = None) -> PersonaSteerModel:
    """创建模型实例"""
    from src.models.persona_steer import PersonaSteerConfig
    from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForCausalLM

    logger.info("Creating PersonaSteer model...")

    model_config = config.get("model", {})
    base_model_path = config.get("base_model", "Qwen/Qwen2.5-3B")
    
    # 获取目标设备
    target_device = device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target device: {target_device}")

    # 加载 backbone 配置获取实际 hidden_size
    logger.info(f"Loading backbone config from {base_model_path}")
    backbone_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    actual_layer_dim = backbone_config.hidden_size
    logger.info(f"Backbone hidden_size: {actual_layer_dim}")

    # 使用 PersonaSteerConfig 创建配置
    training_config = config.get("training", {})
    persona_config = PersonaSteerConfig(
        inject_layers=model_config.get("inject_layers", [14, 15, 16, 17, 18]),
        v_dim=model_config.get("v_dim", 1024),
        hidden_dim=model_config.get("hidden_dim", 4096),
        layer_dim=actual_layer_dim,
        gate_hidden_dim=model_config.get("gate_hidden_dim", 256),
        dropout=model_config.get("dropout", 0.1),
        use_layer_embedding=model_config.get("use_layer_embedding", True),
        gate_init_bias=training_config.get("gate_init_bias", -2.0),
        gate_max=training_config.get("gate_max", 1.0),
    )

    # 加载 encoder 和 tokenizer
    # 共享 backbone 和 encoder 的权重，节省显存
    logger.info(f"Loading backbone from {base_model_path}")

    # 解析目标设备
    if target_device.startswith("cuda:"):
        device_map = {"": int(target_device.split(":")[1])}
    else:
        device_map = "auto"
    logger.info(f"Using device_map: {device_map}")

    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        use_cache=False,
        device_map=device_map,
    )

    # Encoder 共享 backbone 的 transformer 层权重
    encoder = backbone.model  # AutoModelForCausalLM.model 就是 AutoModel 的部分
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # 创建模型
    model = PersonaSteerModel(config=persona_config, encoder=encoder)

    # 设置 tokenizer 到 hyper_network
    if hasattr(model, 'hyper_network') and model.hyper_network is not None:
        model.hyper_network._tokenizer = tokenizer
        logger.info("Tokenizer set to hyper_network")

    # backbone 和 encoder 共享权重，直接设置
    model.set_backbone(backbone)
    logger.info("Backbone loaded (encoder shares backbone weights)")

    # 禁用 gradient checkpointing（与 dual loss 冲突）
    if hasattr(backbone, 'gradient_checkpointing_disable'):
        backbone.gradient_checkpointing_disable()
        logger.info("Gradient checkpointing disabled (dual loss mode)")

    # 【修复】确保所有组件在正确设备上
    # backbone 已通过 device_map 加载到目标设备
    # 需要确保 hyper_network 和 injection 也在同一设备
    target_device_obj = torch.device(target_device)

    # 移动 hyper_network 组件（encoder 已在正确设备，只移动其他参数）
    if model.hyper_network is not None:
        # 移动 hyper_network 的可训练参数
        for name, param in model.hyper_network.named_parameters():
            if param.device != target_device_obj:
                param.data = param.data.to(target_device_obj)
        logger.info(f"HyperNetwork parameters moved to {target_device}")

    # 移动 injection 模块
    if model.injection is not None:
        model.injection.to(target_device_obj)
        logger.info(f"Injection module moved to {target_device}")

    logger.info(f"All model components on {target_device}")

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    return model


def create_dataloaders(config: dict, args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None]:
    """创建数据加载器"""
    logger.info("Creating dataloaders...")

    # 加载 tokenizer
    tokenizer_name = config.get("data", {}).get("tokenizer", "Qwen/Qwen3-4B")
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # 数据路径
    train_data_path = args.data or config.get("data", {}).get("train_path", "data/processed/train.jsonl")
    eval_data_path = args.eval_data or config.get("data", {}).get("eval_path", "data/processed/eval.jsonl")

    # 数据集参数
    data_config = config.get("data", {})
    max_turns = data_config.get("max_turns", 6)
    batch_size = data_config.get("batch_size", 4)

    # 创建训练数据集
    train_dataset = ALOEDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        max_turns=max_turns,
    )
    logger.info(f"Training dataset: {len(train_dataset)} samples")

    # Collator
    collator = PersonaSteerCollator(tokenizer, max_turns=max_turns)

    # 判断是否需要 personality-grouped sampling（SCL 需要同 personality 正例对）
    training_cfg = config.get("training", {})
    scl_weight = training_cfg.get("scl_weight", 0.0)
    use_grouped = scl_weight > 0 and batch_size > 1

    if use_grouped:
        logger.info(f"SCL enabled (weight={scl_weight}), using PersonalityGroupedSampler")
        train_sampler = PersonalityGroupedSampler(
            data_path=train_data_path,
            batch_size=batch_size,
            shuffle=True,
            seed=config.get("seed", 42),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collator,
            num_workers=data_config.get("num_workers", 2),
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=data_config.get("num_workers", 2),
            pin_memory=True,
        )

    # 评估数据加载器 (可选)
    eval_loader = None
    if Path(eval_data_path).exists():
        eval_dataset = ALOEDataset(
            data_path=eval_data_path,
            tokenizer=tokenizer,
            max_turns=max_turns,
        )
        logger.info(f"Evaluation dataset: {len(eval_dataset)} samples")

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=data_config.get("num_workers", 2),
            pin_memory=True,
        )

    return train_loader, eval_loader


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # 覆盖配置
    if args.output:
        config["output_dir"] = args.output
    if args.device:
        config["device"] = args.device

    # 设置日志级别
    if config.get("debug", False):
        logging.getLogger().setLevel(logging.DEBUG)

    # 创建模型
    model = create_model(config, args.device)

    # 创建数据加载器
    train_loader, eval_loader = create_dataloaders(config, args)

    # 创建训练器
    logger.info("Creating trainer...")
    trainer_config = config.get("training", {})
    # 确保 output_dir 正确传递（修复 --output 参数不生效的 bug）
    if args.output or config.get("output_dir"):
        trainer_config["output_dir"] = args.output or config["output_dir"]
    trainer = PersonaSteerTrainer(
        model=model,
        config=trainer_config,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=config.get("device", "cuda"),
    )

    # 恢复 checkpoint (如果指定)
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    else:
        current_stage = trainer_config.get("stage", 1)
        # 按 stage 优先级查找前一阶段的 checkpoint
        ckpt_path = None
        if current_stage == 3:
            ckpt_path = config.get("stage2_checkpoint") or config.get("stage1_checkpoint")
            src_label = "Stage 2" if config.get("stage2_checkpoint") else "Stage 1"
        elif current_stage == 2:
            ckpt_path = config.get("stage1_checkpoint")
            src_label = "Stage 1"
        if ckpt_path:
            logger.info(f"Stage {current_stage}: Loading {src_label} checkpoint from {ckpt_path}")
            trainer.load_checkpoint(ckpt_path)

    # 开始训练
    logger.info("Starting training...")
    history = trainer.train()

    logger.info("Training completed!")
    logger.info(f"Best loss: {trainer.best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main()
