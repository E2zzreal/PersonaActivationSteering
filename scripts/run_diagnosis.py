#!/usr/bin/env python
"""PersonaSteer 瓶颈诊断实验

实验 A1: Personality Embedding Table（14 个可学习向量，替换冻结 encoder）
实验 A2: Sentence-BERT 编码器（替换冻结 encoder）
实验 B:  gate_init_bias=0.0（注入强度 12% → 50%）

用法:
  python scripts/run_diagnosis.py --mode A1 --device cuda:0
  python scripts/run_diagnosis.py --mode A2 --device cuda:1
  python scripts/run_diagnosis.py --mode B  --device cuda:3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def build_personality_index(data_path: str) -> dict[str, int]:
    """从训练数据构建 personality → index 映射"""
    personalities = set()
    with open(data_path) as f:
        for line in f:
            if line.strip():
                personalities.add(json.loads(line)["personality"])
    return {p: i for i, p in enumerate(sorted(personalities))}


def setup_embedding_table(model, data_path: str, device: str):
    """实验 A1：用可学习 embedding table 替换 encoder"""
    p2idx = build_personality_index(data_path)
    num_p = len(p2idx)
    # 用 encoder_dim（2560）而非 v_dim（1024），因为 encoder_projector 期望 encoder_dim 输入
    encoder_dim = next(model.hyper_network.encoder.parameters()).shape[-1]

    emb = torch.nn.Embedding(num_p, encoder_dim).to(device)
    torch.nn.init.orthogonal_(emb.weight[:, :min(num_p, encoder_dim)])
    # 缩放到与 encoder 输出类似的 norm（~90）
    with torch.no_grad():
        emb.weight.mul_(90.0 / emb.weight.norm(dim=1, keepdim=True).clamp(min=1e-6))

    def embed_fn(personality_texts: list[str]) -> torch.Tensor:
        indices = torch.tensor([p2idx[p] for p in personality_texts], device=device)
        return emb(indices).float()

    model.hyper_network._personality_embed_fn = embed_fn
    # 注册为模型子模块以使其可训练
    model.hyper_network.personality_emb = emb
    print(f"[A1] Embedding table: {num_p} personalities × {encoder_dim} dim")
    return model


def setup_sbert(model, data_path: str, device: str):
    """实验 A2：用 Sentence-BERT 替换 encoder"""
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    sbert_dim = sbert.get_sentence_embedding_dimension()

    # 投影到 encoder_dim（2560）以与 encoder_projector 兼容
    encoder_dim = next(model.hyper_network.encoder.parameters()).shape[-1]
    projector = torch.nn.Linear(sbert_dim, encoder_dim).to(device)
    torch.nn.init.xavier_normal_(projector.weight)

    def embed_fn(personality_texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            embs = sbert.encode(personality_texts, convert_to_tensor=True, device=device)
        return projector(embs.float())

    model.hyper_network._personality_embed_fn = embed_fn
    model.hyper_network.sbert_projector = projector
    print(f"[A2] Sentence-BERT: {sbert_dim}d → {encoder_dim}d projector")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["A1", "A2", "B"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config_base", default="configs/train_stage1_claude_sft.yaml")
    parser.add_argument("--data", default="data/claude_sft/train.jsonl")
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()

    # 加载基础配置
    cfg = yaml.safe_load(Path(args.config_base).read_text())
    cfg["device"] = args.device
    output_dir = f"checkpoints/diag_{args.mode.lower()}"
    cfg["training"]["output_dir"] = output_dir
    cfg["training"]["num_epochs"] = args.epochs

    if args.mode == "B":
        cfg["training"]["gate_init_bias"] = 0.0  # 50% injection
        print(f"[B] gate_init_bias: -2.0 → 0.0 (injection 12% → 50%)")

    # 写临时配置
    tmp_config = Path(f"/tmp/diag_{args.mode.lower()}_config.yaml")
    tmp_config.write_text(yaml.dump(cfg))

    # 用 train.py 的逻辑创建模型
    from scripts.train import create_model
    model = create_model(cfg, device=args.device)

    if args.mode == "A1":
        model = setup_embedding_table(model, args.data, args.device)
    elif args.mode == "A2":
        model = setup_sbert(model, args.data, args.device)

    # 创建 dataloader + trainer，开始训练
    from scripts.train import create_dataloaders
    from src.training.trainer import PersonaSteerTrainer

    train_loader, eval_loader = create_dataloaders(cfg, argparse.Namespace(
        data=args.data, eval_data=None))

    trainer_config = cfg.get("training", {})
    trainer = PersonaSteerTrainer(
        config=trainer_config,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        device=args.device,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.train()

    print(f"\n[{args.mode}] 训练完成，checkpoint: {output_dir}")
    print(f"下一步：python scripts/audit/eval_new_model.py --checkpoint {output_dir}/best.pt --device {args.device}")


if __name__ == "__main__":
    main()
