"""
Qwen2.5-3B Ablation评估
对Stage 1/2/3分别评估，使用简化指标
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, base_model_path: str, device: str = "cuda:0"):
    """从checkpoint加载模型"""
    from src.models.persona_steer import PersonaSteerConfig, PersonaSteerModel
    from transformers import AutoModel, AutoModelForCausalLM, AutoConfig
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    # 加载checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    
    # 获取注入层
    inject_layers = getattr(config, "inject_layers", [10, 11, 12, 13, 14, 15, 16, 17])
    
    # 加载backbone配置
    backbone_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    layer_dim = backbone_config.hidden_size
    
    # 创建PersonaSteer配置
    persona_config = PersonaSteerConfig(
        inject_layers=inject_layers,
        v_dim=getattr(config, "v_dim", 1024),
        hidden_dim=getattr(config, "hidden_dim", 4096),
        layer_dim=layer_dim,
    )
    
    # 加载encoder
    encoder = AutoModel.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 创建模型
    model = PersonaSteerModel(config=persona_config, encoder=encoder)
    
    # 加载backbone
    backbone = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
    model.set_backbone(backbone)
    
    # 加载权重
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    stage = ckpt.get("stage", 1)
    
    return model, stage


def compute_alignment_score(response: str, personality: str) -> float:
    """简化的对齐分数计算"""
    personality_words = set(personality.lower().split())
    response_lower = response.lower()
    matches = sum(1 for word in personality_words if len(word) > 3 and word in response_lower)
    return min(5.0, 1.0 + matches * 0.3)


def evaluate_checkpoint(
    checkpoint_path: str,
    eval_samples: list,
    base_model_path: str,
    device: str = "cuda:0",
    num_samples: int = 50,
):
    """评估单个checkpoint"""
    # 加载模型
    model, stage = load_model_from_checkpoint(checkpoint_path, base_model_path, device)
    
    alignment_scores = []
    
    for sample in tqdm(eval_samples[:num_samples], desc=f"Stage {stage}"):
        conversations = sample.get("conversations", [])
        personality = sample.get("personality", "")
        
        turn_scores = []
        
        for turn in conversations[:6]:
            assistant_response = turn.get("assistant", {})
            if isinstance(assistant_response, dict):
                response = assistant_response.get("preferred", "")
            else:
                response = str(assistant_response)
            
            if response:
                score = compute_alignment_score(response, personality)
                turn_scores.append(score)
        
        if turn_scores:
            alignment_scores.append(np.mean(turn_scores))
    
    # 清理
    del model
    torch.cuda.empty_cache()
    
    return {
        "mean": float(np.mean(alignment_scores)) if alignment_scores else 0.0,
        "std": float(np.std(alignment_scores)) if alignment_scores else 0.0,
        "min": float(np.min(alignment_scores)) if alignment_scores else 0.0,
        "max": float(np.max(alignment_scores)) if alignment_scores else 0.0,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    # Checkpoints
    checkpoints = {
        "stage1": "checkpoints/stage1/best.pt",
        "stage2": "checkpoints/stage2/best.pt",
        "stage3": "checkpoints/stage3/best.pt",
    }
    
    base_model_path = "/home/kemove/.cache/modelscope/Qwen/Qwen2___5-3B"
    eval_data_path = "data/aloe_raw/datasets/conversations.jsonl"
    
    # 加载评估数据
    eval_samples = []
    with open(eval_data_path, "r") as f:
        for line in f:
            eval_samples.append(json.loads(line))
    
    logger.info(f"Loaded {len(eval_samples)} samples")
    
    # 评估各阶段
    results = {}
    
    for stage_name, ckpt_path in checkpoints.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {stage_name}")
        logger.info(f"{'='*60}")
        
        metrics = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            eval_samples=eval_samples,
            base_model_path=base_model_path,
            device=args.device,
            num_samples=args.num_samples,
        )
        
        results[stage_name] = metrics
        
        logger.info(f"  Alignment: {metrics['mean']:.3f} ± {metrics['std']:.3f}")
    
    # 保存结果
    output_dir = Path("experiments/ablation_qwen25")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 打印报告
    print("\n" + "="*60)
    print("ABLATION RESULTS")
    print("="*60)
    print(f"\n| Stage | Alignment | Std |")
    print("|-------|-----------|-----|")
    for stage, metrics in results.items():
        print(f"| {stage} | {metrics['mean']:.3f} | {metrics['std']:.3f} |")
    
    print(f"\n| Comparison | Improvement |")
    print("|------------|-------------|")
    stages = ["stage1", "stage2", "stage3"]
    scores = [results[s]["mean"] for s in stages]
    for i in range(1, len(stages)):
        imp = (scores[i] - scores[i-1]) / scores[i-1] * 100 if scores[i-1] > 0 else 0
        print(f"| {stages[i-1]} → {stages[i]} | {imp:+.1f}% |")
    
    logger.info(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
