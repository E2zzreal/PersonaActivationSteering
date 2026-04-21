#!/usr/bin/env python
"""
PersonaSteer v2: FiLM Injection + Prompt Dropout 训练脚本

三阶段课程训练:
  Phase 1: 100% prompt masked (injection-only) — 逼 HyperNetwork 独立学会人格编码
  Phase 2: 50% prompt included — 学会与 prompt 协作
  Phase 3: 70% prompt included — 面向部署微调
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.aloe_dataset import ALOEDataset
from src.data.collator import PersonaSteerCollator
from src.data.grouped_sampler import PersonalityGroupedSampler
from src.training.trainer import PersonaSteerTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PersonaSteer v2 (FiLM)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--resume_phase", type=int, default=1, help="Resume from phase (1/2/3)")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def create_dataloader(config, tokenizer, prompt_include_rate, scl_weight):
    """根据 prompt_include_rate 创建数据加载器"""
    data_config = config.get("data", {})
    train_path = data_config.get("train_path", "data/big5_cross_persona/train.jsonl")
    batch_size = data_config.get("batch_size", 4)
    max_turns = data_config.get("max_turns", 1)

    dataset = ALOEDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_turns=max_turns,
        prompt_include_rate=prompt_include_rate,
    )
    logger.info(f"Dataset: {len(dataset)} samples, prompt_include_rate={prompt_include_rate}")

    collator = PersonaSteerCollator(tokenizer, max_turns=max_turns)

    use_grouped = scl_weight > 0 and batch_size > 1
    if use_grouped:
        sampler = PersonalityGroupedSampler(
            data_path=train_path,
            batch_size=batch_size,
            shuffle=True,
            seed=config.get("seed", 42),
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=data_config.get("num_workers", 2),
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=data_config.get("num_workers", 2),
            pin_memory=True,
        )
    return loader


def train_phase(model, config, tokenizer, phase_num, phase_config, device, prev_ckpt=None):
    """训练单个 phase"""
    epochs = phase_config["epochs"]
    lr = phase_config["lr"]
    prompt_include_rate = phase_config["prompt_include_rate"]
    scl_weight = phase_config.get("scl_weight", 0.0)

    logger.info(f"{'='*60}")
    logger.info(f"Phase {phase_num}: epochs={epochs}, lr={lr}, "
                f"prompt_include_rate={prompt_include_rate}, scl_weight={scl_weight}")
    logger.info(f"{'='*60}")

    # 创建 dataloader（每个 phase 不同的 prompt_include_rate）
    train_loader = create_dataloader(config, tokenizer, prompt_include_rate, scl_weight)

    # 构建 trainer config
    training_base = config.get("training", {})
    trainer_config = {
        "stage": 1,  # FiLM 不区分 stage，用 1 统一
        "num_epochs": epochs,
        "learning_rate": lr,
        "weight_decay": training_base.get("weight_decay", 0.01),
        "max_grad_norm": training_base.get("max_grad_norm", 1.0),
        "sft_weight": training_base.get("sft_weight", 1.0),
        "scl_weight": scl_weight,
        "v_norm_weight": training_base.get("v_norm_weight", 0.1),
        "v_norm_target": training_base.get("v_norm_target", 5.0),
        "warmup_steps": training_base.get("warmup_steps", 20),
        "use_amp": training_base.get("use_amp", True),
        "use_dual_loss": False,
        "output_dir": str(Path(training_base.get("output_dir", "checkpoints/film_v2")) / f"phase{phase_num}"),
        "save_interval": training_base.get("save_interval", 2),
        "log_interval": training_base.get("log_interval", 10),
        "temperature": training_base.get("temperature", 0.15),
    }

    trainer = PersonaSteerTrainer(
        model=model,
        config=trainer_config,
        train_loader=train_loader,
        device=device,
    )

    # 加载前一 phase 的 checkpoint
    if prev_ckpt and Path(prev_ckpt).exists():
        logger.info(f"Loading checkpoint from previous phase: {prev_ckpt}")
        trainer.load_checkpoint(prev_ckpt)

    history = trainer.train()

    # 保存 phase 最终 checkpoint
    best_ckpt = Path(trainer_config["output_dir"]) / "best.pt"
    logger.info(f"Phase {phase_num} complete. Best checkpoint: {best_ckpt}")

    # 打印 FiLM 统计
    if hasattr(model.injection, 'get_film_stats'):
        stats = model.injection.get_film_stats()
        if stats:
            logger.info("FiLM modulation stats:")
            for k, v in sorted(stats.items()):
                logger.info(f"  {k}: {v:.4f}")

    return str(best_ckpt), history


def main():
    args = parse_args()
    set_seed(args.seed)

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.device:
        config["device"] = args.device

    device = config.get("device", "cuda:0")

    # 创建模型（只创建一次，三个 phase 共享）
    from scripts.train import create_model
    model = create_model(config, device)

    # Tokenizer
    tokenizer_path = config.get("data", {}).get("tokenizer", config.get("base_model"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    logger.info(f"Injection type: {model.config.injection_type}")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    training_config = config.get("training", {})
    phases = [
        (1, training_config.get("phase1", {"epochs": 5, "lr": 5e-5, "prompt_include_rate": 0.0, "scl_weight": 0.0})),
        (2, training_config.get("phase2", {"epochs": 5, "lr": 3e-5, "prompt_include_rate": 0.5, "scl_weight": 0.5})),
        (3, training_config.get("phase3", {"epochs": 3, "lr": 1e-5, "prompt_include_rate": 0.7, "scl_weight": 0.5})),
    ]

    prev_ckpt = args.resume_ckpt
    all_history = {}

    for phase_num, phase_config in phases:
        if phase_num < args.resume_phase:
            # 如果指定了 resume_phase，跳过之前的 phase
            expected_ckpt = Path(training_config.get("output_dir", "checkpoints/film_v2")) / f"phase{phase_num}" / "best.pt"
            if expected_ckpt.exists():
                prev_ckpt = str(expected_ckpt)
                logger.info(f"Skipping Phase {phase_num} (resume_phase={args.resume_phase}), using checkpoint: {prev_ckpt}")
            continue

        prev_ckpt, history = train_phase(
            model, config, tokenizer, phase_num, phase_config, device, prev_ckpt,
        )
        all_history[f"phase{phase_num}"] = history

    logger.info("All phases complete!")
    logger.info(f"Final checkpoint: {prev_ckpt}")


if __name__ == "__main__":
    main()
