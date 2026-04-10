#!/usr/bin/env python3
"""比较多个gate实验的结果"""
import json
import glob
from pathlib import Path

# 自动发现所有实验
experiment_dirs = sorted(Path("checkpoints").glob("exp_*"))
experiments = [d.name.replace("exp_", "") for d in experiment_dirs]

print("=== Gate初始化实验结果对比 ===\n")
print(f"发现 {len(experiments)} 个实验\n")

results = []
for exp_name in experiments:
    ckpt_dir = Path(f"checkpoints/exp_{exp_name}")
    if not ckpt_dir.exists():
        continue

    # 读取best.pt的loss
    import torch
    best_pt = ckpt_dir / "best.pt"
    if best_pt.exists():
        ckpt = torch.load(best_pt, map_location='cpu', weights_only=False)
        best_loss = ckpt.get('best_loss', float('inf'))
        epoch = ckpt.get('epoch', 0)

        # 检查gate参数
        gate_bias = None
        for k, v in ckpt['model_state_dict'].items():
            if 'injection.gate.gate_mlp.3.bias' in k:
                gate_bias = v.detach().float().mean().item()
                break

        results.append({
            'name': exp_name,
            'best_loss': best_loss,
            'epoch': epoch,
            'gate_bias_mean': gate_bias,
        })

        print(f"✓ {exp_name}:")
        print(f"  Best Loss: {best_loss:.4f}")
        print(f"  Epoch: {epoch}")
        print(f"  Gate Bias均值: {gate_bias:.4f}" if gate_bias else "  Gate Bias: N/A")
        print()

# 排序并显示最优方案
if results:
    results.sort(key=lambda x: x['best_loss'])
    print("\n=== 排名（按best_loss） ===")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']}: loss={r['best_loss']:.4f}, gate_bias={r['gate_bias_mean']:.4f}")

    print(f"\n🏆 最优方案: {results[0]['name']} (loss={results[0]['best_loss']:.4f})")
