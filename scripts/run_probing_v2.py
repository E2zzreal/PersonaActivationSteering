#!/usr/bin/env python3
"""
增强版Probing实验脚本 V2

功能:
- 多种层选择策略对比
- 完整相关性分析
- 可视化热力图
- 生成训练就绪的配置文件

Usage:
    python scripts/run_probing_v2.py \
        --model Qwen/Qwen3-4B \
        --data data/processed/train.jsonl \
        --num_samples 1000 \
        --output experiments/probing/v2_results \
        --device cuda:0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import yaml

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.probing.head_probing import AttentionHeadProber
from src.probing.attribute_extractor import AttributeExtractor


def load_data(data_path: str, num_samples: int) -> List[dict]:
    """加载ALOE数据样本"""
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError:
                continue
    return samples


def extract_texts_and_attributes(
    samples: List[dict]
) -> Tuple[List[str], Dict[str, List[float]]]:
    """从样本中提取文本和人格属性"""
    extractor = AttributeExtractor()

    texts = []
    attributes = {
        "age": [],
        "gender": [],
        "introversion": [],
        "openness": [],
        "conscientiousness": [],
        "agreeableness": [],
        "neuroticism": [],
    }

    for sample in samples:
        # 合并 profile 和 personality 作为输入文本
        profile = sample.get("profile", "")
        personality = sample.get("personality", "")
        text = f"{profile} {personality}"
        texts.append(text)

        # 提取属性
        attrs = extractor.extract_attributes(sample)
        for key in attributes:
            attributes[key].append(attrs.get(key, 0.5))

    return texts, attributes


def select_layers_top_k(
    correlation_matrix: np.ndarray,
    top_k: int = 8
) -> List[int]:
    """Top-K策略: 选择相关性最高的K层"""
    # 计算每层平均相关性
    layer_scores = np.abs(correlation_matrix).mean(axis=1)
    # 选择最高的K层
    top_indices = np.argsort(layer_scores)[::-1][:top_k]
    return sorted(top_indices.tolist())


def select_layers_continuous(
    correlation_matrix: np.ndarray,
    top_k: int = 8
) -> List[int]:
    """Continuous策略: 选择峰值附近的连续K层"""
    layer_scores = np.abs(correlation_matrix).mean(axis=1)
    num_layers = len(layer_scores)

    # 找到峰值位置
    peak_idx = np.argmax(layer_scores)

    # 从峰值向两边扩展
    selected = [peak_idx]
    left, right = peak_idx - 1, peak_idx + 1

    while len(selected) < top_k:
        left_score = layer_scores[left] if left >= 0 else -1
        right_score = layer_scores[right] if right < num_layers else -1

        if left_score >= right_score and left >= 0:
            selected.append(left)
            left -= 1
        elif right < num_layers:
            selected.append(right)
            right += 1
        else:
            break

    return sorted(selected[:top_k])


def select_layers_early_peak(
    correlation_matrix: np.ndarray,
    top_k: int = 8
) -> List[int]:
    """Early-Peak策略: 只在前50%层中找峰值"""
    num_layers = correlation_matrix.shape[0]
    early_half = correlation_matrix[:num_layers // 2, :]
    return select_layers_continuous(early_half, top_k)


def select_layers_multi_scale(
    correlation_matrix: np.ndarray,
    top_k: int = 8
) -> List[int]:
    """Multi-Scale策略: 前中后各选若干层"""
    num_layers = correlation_matrix.shape[0]

    # 计算每段的相关性
    segment_size = num_layers // 3

    # 从每段选择相关性最高的层
    selected = []
    for segment_idx in range(3):
        start = segment_idx * segment_size
        end = start + segment_size if segment_idx < 2 else num_layers
        segment = correlation_matrix[start:end, :]
        segment_scores = np.abs(segment).mean(axis=1)

        # 每段选2-3层
        num_from_segment = top_k // 3 + (1 if segment_idx < top_k % 3 else 0)
        top_in_segment = np.argsort(segment_scores)[::-1][:num_from_segment]
        selected.extend([start + i for i in top_in_segment])

    return sorted(selected[:top_k])


def select_layers_strategies(
    correlation_matrices: Dict[str, torch.Tensor],
    top_k: int = 8
) -> Dict[str, List[int]]:
    """应用多种层选择策略"""
    # 合并所有属性的相关性矩阵
    all_matrices = []
    for matrix in correlation_matrices.values():
        if isinstance(matrix, torch.Tensor):
            all_matrices.append(matrix.cpu().numpy())
        else:
            all_matrices.append(matrix)

    avg_matrix = np.mean(all_matrices, axis=0)

    strategies = {
        "top": select_layers_top_k(avg_matrix, top_k),
        "continuous": select_layers_continuous(avg_matrix, top_k),
        "early_peak": select_layers_early_peak(avg_matrix, top_k),
        "multi_scale": select_layers_multi_scale(avg_matrix, top_k),
    }

    return strategies


def generate_heatmap(
    correlation_matrix: np.ndarray,
    attr_name: str,
    output_path: str
):
    """生成相关性热力图"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            correlation_matrix.T,
            cmap='RdBu_r',
            center=0,
            xticklabels=5,
            yticklabels=4,
            cbar_kws={'label': 'Spearman Correlation'}
        )
        plt.xlabel('Layer')
        plt.ylabel('Attention Head')
        plt.title(f'Correlation Heatmap: {attr_name}')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except ImportError:
        print(f"[WARN] matplotlib/seaborn not available, skipping heatmap for {attr_name}")


def compute_statistics(
    correlation_matrices: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, Any]]:
    """计算各属性的统计信息"""
    stats = {}

    for attr, matrix in correlation_matrices.items():
        if isinstance(matrix, torch.Tensor):
            matrix_np = matrix.cpu().numpy()
        else:
            matrix_np = matrix

        layer_scores = np.abs(matrix_np).mean(axis=1)

        stats[attr] = {
            "max_correlation": float(matrix_np.max()),
            "min_correlation": float(matrix_np.min()),
            "mean_abs_correlation": float(np.abs(matrix_np).mean()),
            "std_correlation": float(matrix_np.std()),
            "peak_layer": int(np.argmax(layer_scores)),
            "peak_layer_correlation": float(layer_scores.max()),
            "top_3_layers": sorted(
                np.argsort(layer_scores)[::-1][:3].tolist()
            ),
        }

    return stats


def save_layer_config(
    layers: List[int],
    strategy: str,
    output_path: str,
    metadata: Dict = None
):
    """保存层配置到YAML文件"""
    config = {
        "model": {
            "inject_layers": layers,
            "num_inject_layers": len(layers),
        },
        "metadata": {
            "strategy": strategy,
            "source": "probing_v2",
        }
    }

    if metadata:
        config["metadata"].update(metadata)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(
        description="增强版Probing实验 - 确定最优注入层配置"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="模型名称或路径"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/train.jsonl",
        help="ALOE数据路径"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="使用的样本数量"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/probing/v2_results",
        help="输出目录"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="选择的注入层数量"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="运行设备"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批处理大小"
    )
    parser.add_argument(
        "--skip_heatmaps",
        action="store_true",
        help="跳过生成热力图"
    )

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "heatmaps").mkdir(exist_ok=True)
    (output_dir / "layer_configs").mkdir(exist_ok=True)
    (output_dir / "analysis").mkdir(exist_ok=True)

    print("=" * 60)
    print("PersonaSteer Probing Experiment V2")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Step 1: 加载数据
    print("\n[Step 1/5] Loading data...")
    samples = load_data(args.data, args.num_samples)
    print(f"  Loaded {len(samples)} samples")

    # Step 2: 提取文本和属性
    print("\n[Step 2/5] Extracting texts and attributes...")
    texts, attributes = extract_texts_and_attributes(samples)
    print(f"  Extracted {len(texts)} texts")
    print(f"  Attributes: {list(attributes.keys())}")

    # Step 3: 运行Probing
    print("\n[Step 3/5] Running probing (this may take a while)...")
    prober = AttentionHeadProber(model_name=args.model, device=args.device)

    # 先收集激活值（带batch_size）
    activations = prober.collect(texts, batch_size=args.batch_size)
    print(f"  Collected activations: {activations.shape}")

    # 计算每个属性的相关性
    correlation_matrices = {}
    for attr_name, attr_values in attributes.items():
        corr_matrix = prober.compute_spearman(activations, attr_values)
        correlation_matrices[attr_name] = corr_matrix

    print(f"  Computed correlations for {len(correlation_matrices)} attributes")

    # Step 4: 生成热力图
    if not args.skip_heatmaps:
        print("\n[Step 4/5] Generating heatmaps...")
        for attr, matrix in correlation_matrices.items():
            if isinstance(matrix, torch.Tensor):
                matrix_np = matrix.cpu().numpy()
            else:
                matrix_np = matrix

            heatmap_path = output_dir / "heatmaps" / f"correlation_{attr}.png"
            generate_heatmap(matrix_np, attr, str(heatmap_path))
            print(f"  Saved: {heatmap_path}")
    else:
        print("\n[Step 4/5] Skipping heatmaps...")

    # Step 5: 应用层选择策略
    print("\n[Step 5/5] Applying layer selection strategies...")
    selected_layers = select_layers_strategies(correlation_matrices, args.top_k)

    for strategy, layers in selected_layers.items():
        print(f"  {strategy}: {layers}")

        # 保存配置
        config_path = output_dir / "layer_configs" / f"{strategy}_8_layers.yaml"
        save_layer_config(
            layers, strategy, str(config_path),
            metadata={"num_samples": args.num_samples}
        )

    # 计算统计信息
    print("\n[Analysis] Computing statistics...")
    stats = compute_statistics(correlation_matrices)

    for attr, stat in stats.items():
        print(f"  {attr}:")
        print(f"    Peak layer: {stat['peak_layer']} (corr={stat['peak_layer_correlation']:.3f})")
        print(f"    Max correlation: {stat['max_correlation']:.3f}")
        print(f"    Mean |correlation|: {stat['mean_abs_correlation']:.3f}")

    # 保存统计信息
    stats_path = output_dir / "analysis" / "correlation_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {stats_path}")

    # 生成推荐配置
    # 选择peak_layer最靠前的属性对应的策略
    avg_peak_layer = np.mean([s['peak_layer'] for s in stats.values()])

    if avg_peak_layer < 12:
        recommended_strategy = "early_peak"
    elif avg_peak_layer < 24:
        recommended_strategy = "continuous"
    else:
        recommended_strategy = "top"

    recommended_layers = selected_layers[recommended_strategy]

    recommended_config = {
        "inject_layers": recommended_layers,
        "num_inject_layers": len(recommended_layers),
        "recommended_strategy": recommended_strategy,
        "rationale": f"Peak layers average at {avg_peak_layer:.1f}, recommending {recommended_strategy} strategy",
        "statistics_summary": {
            "avg_peak_layer": float(avg_peak_layer),
            "avg_max_correlation": float(np.mean([s['max_correlation'] for s in stats.values()])),
        }
    }

    recommended_path = output_dir / "analysis" / "recommended_config.yaml"
    with open(recommended_path, 'w', encoding='utf-8') as f:
        yaml.dump(recommended_config, f, default_flow_style=False, allow_unicode=True)
    print(f"  Saved: {recommended_path}")

    # 保存完整结果
    results = {
        "config": {
            "model": args.model,
            "num_samples": args.num_samples,
            "top_k": args.top_k,
            "device": args.device,
        },
        "selected_layers": selected_layers,
        "statistics": stats,
        "recommended": recommended_config,
    }

    results_path = output_dir / "v2_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印总结
    print("\n" + "=" * 60)
    print("PROBING COMPLETE")
    print("=" * 60)
    print(f"Recommended inject_layers: {recommended_layers}")
    print(f"Strategy: {recommended_strategy}")
    print(f"Average peak layer: {avg_peak_layer:.1f}")
    print(f"Average max correlation: {np.mean([s['max_correlation'] for s in stats.values()]):.3f}")
    print("=" * 60)
    print(f"\nResults saved to: {args.output}")
    print("\nNext steps:")
    print("  1. Review heatmaps in experiments/probing/v2_results/heatmaps/")
    print("  2. Update configs with recommended inject_layers")
    print("  3. Run training with new layer configuration")


if __name__ == "__main__":
    main()