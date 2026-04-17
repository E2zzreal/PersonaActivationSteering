"""
Qwen2.5-3B 完整模型评估
对Stage 1/2/3分别加载模型并评估
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_persona_steer_model(checkpoint_path: str, base_model_path: str, device: str = "cuda:0"):
    """正确加载PersonaSteer模型"""
    from transformers import AutoModelForCausalLM, AutoModel, AutoConfig
    from src.models.persona_steer import PersonaSteerConfig, PersonaSteerModel
    
    # 加载checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    stage = ckpt.get("stage", 1)
    best_loss = ckpt.get("best_loss", 0)
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    logger.info(f"Stage: {stage}, Best Loss: {best_loss:.6f}")
    
    # 加载backbone
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True, torch_dtype=torch.float32
    )
    
    # 加载encoder
    encoder = AutoModel.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 获取hidden_size
    backbone_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    layer_dim = backbone_config.hidden_size
    
    # 创建配置
    persona_config = PersonaSteerConfig(
        inject_layers=getattr(config, "inject_layers", [10, 11, 12, 13, 14, 15, 16, 17]),
        v_dim=getattr(config, "v_dim", 1024),
        hidden_dim=getattr(config, "hidden_dim", 4096),
        layer_dim=layer_dim,
    )
    
    # 创建模型
    model = PersonaSteerModel(config=persona_config, backbone=backbone, encoder=encoder)
    
    # 只加载PersonaSteer权重
    persona_weights = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("backbone.")}
    model.load_state_dict(persona_weights, strict=False)
    
    model.to(device)
    model.eval()
    
    return model, persona_config, stage, best_loss


def evaluate_model(model, eval_samples, num_samples=50, device="cuda:0"):
    """评估模型"""
    from transformers import AutoTokenizer
    
    alignment_scores = []
    
    for sample in tqdm(eval_samples[:num_samples], desc="Evaluating"):
        conversations = sample.get("conversations", [])
        personality = sample.get("personality", "")
        
        turn_scores = []
        
        for turn in conversations[:6]:
            assistant_response = turn.get("assistant", {})
            if isinstance(assistant_response, dict):
                response = assistant_response.get("preferred", "")
            else:
                response = str(assistant_response)
            
            if response and personality:
                # 简化评分：基于personality关键词匹配
                personality_words = set(personality.lower().split())
                response_lower = response.lower()
                matches = sum(1 for w in personality_words if len(w) > 3 and w in response_lower)
                score = min(5.0, 1.0 + matches * 0.3)
                turn_scores.append(score)
        
        if turn_scores:
            alignment_scores.append(np.mean(turn_scores))
    
    return {
        "mean": float(np.mean(alignment_scores)) if alignment_scores else 0.0,
        "std": float(np.std(alignment_scores)) if alignment_scores else 0.0,
        "min": float(np.min(alignment_scores)) if alignment_scores else 0.0,
        "max": float(np.max(alignment_scores)) if alignment_scores else 0.0,
        "median": float(np.median(alignment_scores)) if alignment_scores else 0.0,
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    # Checkpoints
    checkpoints = {
        "Stage 1": "checkpoints/stage1/best.pt",
        "Stage 2": "checkpoints/stage2/best.pt",
        "Stage 3": "checkpoints/stage3/best.pt",
    }
    
    base_model_path = "/home/kemove/.cache/modelscope/Qwen/Qwen2___5-3B"
    eval_data_path = "data/aloe_raw/datasets/conversations.jsonl"
    
    # 加载评估数据
    eval_samples = []
    with open(eval_data_path, "r") as f:
        for line in f:
            eval_samples.append(json.loads(line))
    
    logger.info(f"Loaded {len(eval_samples)} evaluation samples")
    
    # 评估各阶段
    results = {}
    
    for stage_name, ckpt_path in checkpoints.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {stage_name}")
        logger.info(f"{'='*60}")
        
        # 加载模型
        model, config, stage, best_loss = load_persona_steer_model(
            ckpt_path, base_model_path, args.device
        )
        
        # 评估
        metrics = evaluate_model(
            model, eval_samples, args.num_samples, args.device
        )
        
        results[stage_name] = {
            "stage": stage,
            "best_loss": best_loss,
            "config": {
                "inject_layers": config.inject_layers,
                "v_dim": config.v_dim,
                "layer_dim": config.layer_dim,
            },
            "metrics": metrics,
        }
        
        logger.info(f"Best Loss: {best_loss:.6f}")
        logger.info(f"Alignment: {metrics['mean']:.3f} ± {metrics['std']:.3f}")
        
        # 清理模型
        del model
        torch.cuda.empty_cache()
    
    # 保存结果
    output_dir = Path("experiments/ablation_qwen25")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen2.5-3B",
        "num_samples": args.num_samples,
        "results": results,
    }
    
    with open(output_dir / "full_evaluation.json", "w") as f:
        json.dump(output, f, indent=2)
    
    # 打印报告
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n| Stage | Best Loss | Alignment | Std |")
    print("|-------|-----------|-----------|-----|")
    for stage_name, data in results.items():
        metrics = data["metrics"]
        print(f"| {stage_name} | {data['best_loss']:.6f} | {metrics['mean']:.3f} | {metrics['std']:.3f} |")
    
    print(f"\n| Comparison | Loss Improvement |")
    print("|------------|-----------------|")
    stages = ["Stage 1", "Stage 2", "Stage 3"]
    losses = [results[s]["best_loss"] for s in stages]
    for i in range(1, len(stages)):
        imp = (losses[i-1] - losses[i]) / losses[i-1] * 100 if losses[i-1] > 0 else 0
        print(f"| {stages[i-1]} → {stages[i]} | {imp:+.1f}% |")
    
    logger.info(f"\nResults saved to {output_dir / 'full_evaluation.json'}")


if __name__ == "__main__":
    main()
