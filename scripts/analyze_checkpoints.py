"""
Qwen2.5-3B Ablation评估 - 基于Checkpoint指标分析
"""

import json
import torch
from pathlib import Path
from datetime import datetime

def analyze_checkpoint(ckpt_path: str):
    """分析checkpoint中的指标"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    return {
        "stage": ckpt.get("stage", 1),
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", 0),
        "best_loss": ckpt.get("best_loss", 0),
        "config": {
            "inject_layers": getattr(ckpt.get("config"), "inject_layers", []),
            "v_dim": getattr(ckpt.get("config"), "v_dim", 1024),
            "hidden_dim": getattr(ckpt.get("config"), "hidden_dim", 4096),
        }
    }

def main():
    checkpoints = {
        "Stage 1": "checkpoints/stage1/best.pt",
        "Stage 2": "checkpoints/stage2/best.pt",
        "Stage 3": "checkpoints/stage3/best.pt",
    }
    
    print("="*60)
    print("Qwen2.5-3B ABLATION ANALYSIS")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    for name, path in checkpoints.items():
        print(f"\n--- {name} ---")
        metrics = analyze_checkpoint(path)
        results[name] = metrics
        
        print(f"  Stage: {metrics['stage']}")
        print(f"  Best Epoch: {metrics['epoch']}")
        print(f"  Best Loss: {metrics['best_loss']:.6f}")
        print(f"  Inject Layers: {metrics['config']['inject_layers']}")
    
    # 对比分析
    print("\n" + "="*60)
    print("LOSS IMPROVEMENT")
    print("="*60)
    
    losses = [results[s]["best_loss"] for s in checkpoints.keys()]
    stages = list(checkpoints.keys())
    
    print("\n| Stage | Loss | Improvement |")
    print("|-------|------|-------------|")
    
    for i, (name, loss) in enumerate(zip(stages, losses)):
        if i == 0:
            imp = "baseline"
        else:
            imp_pct = (losses[i-1] - loss) / losses[i-1] * 100
            imp = f"{imp_pct:+.1f}%"
        print(f"| {name} | {loss:.6f} | {imp} |")
    
    # 总体改善
    total_improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"\n**Total Improvement: {total_improvement:.1f}%**")
    
    # 训练配置对比
    print("\n" + "="*60)
    print("CONFIGURATION COMPARISON")
    print("="*60)
    
    print("\n| Stage | Inject Layers | Trainable Focus |")
    print("|-------|---------------|-----------------|")
    print("| Stage 1 | [10-17] | HyperNetwork only (Gate frozen) |")
    print("| Stage 2 | [10-17] | HyperNetwork + Gate |")
    print("| Stage 3 | [10-17] | + Contrastive Learning (SCL=0.1) |")
    
    # 保存结果
    output_dir = Path("experiments/ablation_qwen25")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen2.5-3B",
        "checkpoints": results,
        "summary": {
            "total_loss_improvement_pct": total_improvement,
            "stage_losses": dict(zip(stages, losses)),
        }
    }
    
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nResults saved to {output_dir / 'ablation_results.json'}")

if __name__ == "__main__":
    main()
