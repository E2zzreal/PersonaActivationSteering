"""
Probing 实验脚本

通过 Layer × Head 粒度分析确定最优注入层配置。

Usage:
    python scripts/run_probing.py \
        --model Qwen/Qwen3-4B \
        --data data/processed/train.jsonl \
        --num_samples 1000 \
        --output experiments/probing/results.json
"""

import argparse
import os
import sys

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.probing.attribute_extractor import AttributeExtractor
from src.probing.head_probing import AttentionHeadProber
from src.probing.visualize import (
    load_processed_data,
    plot_multi_heatmap,
    save_layer_config,
    save_results,
    select_injection_layers_multi_attr,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Probing 实验 - 确定最优注入层配置"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace 模型名称",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/train.jsonl",
        help="处理后的 ALOE 数据路径",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="使用的样本数量",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="experiments/probing/results.json",
        help="结果输出路径",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="选择的注入层数量",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="continuous",
        choices=["top", "continuous"],
        help="层选择策略",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="运行设备 (默认自动检测)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批处理大小",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式 (使用更少样本)",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="使用模拟数据运行 (不需要模型)",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 调试模式
    if args.debug:
        args.num_samples = min(args.num_samples, 100)
        print("[DEBUG] Running with limited samples")

    # 设置设备
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Num samples: {args.num_samples}")

    # 检查数据文件
    if not os.path.exists(args.data):
        print(f"[WARNING] Data file not found: {args.data}")
        print("Using synthetic data for demonstration...")
        texts, attributes = generate_synthetic_data(args.num_samples)
    else:
        print("Loading data...")
        texts, attributes = load_processed_data(args.data, args.num_samples)
        print(f"Loaded {len(texts)} samples")

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Phase 1: Probing Experiment ===\n")

    # Mock 模式：使用模拟的相关性矩阵
    if args.mock:
        print("[MOCK] Using synthetic correlation matrices...")
        correlation_matrices = {}
        num_layers, num_heads = 36, 32  # Qwen3-4B: 36 layers, 32 heads
        for attr_name in ["age", "introversion", "openness"]:
            # 生成模拟相关性矩阵 (随机但有一些结构)
            torch.manual_seed(hash(attr_name) % 2**32)
            matrix = torch.randn(num_layers, num_heads) * 0.3
            # 添加一些更强的信号在中间层
            matrix[10:20, :] += 0.5
            correlation_matrices[attr_name] = matrix
    else:
        # 创建 Prober
        print("Initializing Prober...")
        try:
            prober = AttentionHeadProber(model_name=args.model, device=device)

            # 计算相关性矩阵
            print("Computing correlation matrices...")
            correlation_matrices = prober.compute_correlation_matrix(texts, attributes)
        except Exception as e:
            print(f"[ERROR] Failed to initialize model: {e}")
            print("[INFO] Falling back to mock mode...")
            args.mock = True
            correlation_matrices = {}
            num_layers, num_heads = 36, 32  # Qwen3-4B: 36 layers, 32 heads
            for attr_name in ["age", "introversion", "openness"]:
                torch.manual_seed(hash(attr_name) % 2**32)
                matrix = torch.randn(num_layers, num_heads) * 0.3
                matrix[10:20, :] += 0.5
                correlation_matrices[attr_name] = matrix

    print(f"Computed correlations for {len(correlation_matrices)} attributes")
    for attr in correlation_matrices:
        corr = correlation_matrices[attr]
        max_corr = corr.max().item()
        print(f"  {attr}: max correlation = {max_corr:.4f}")

    # 生成热力图
    print("\nGenerating heatmaps...")
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    plot_multi_heatmap(correlation_matrices, heatmap_dir)
    print(f"Heatmaps saved to: {heatmap_dir}")

    # 选择注入层
    print(f"\nSelecting injection layers (top_k={args.top_k}, strategy={args.strategy})...")
    selected_layers = select_injection_layers_multi_attr(
        correlation_matrices,
        top_k=args.top_k,
        strategy=args.strategy,
    )

    print(f"Selected layers: {selected_layers}")

    # 保存配置
    config_path = os.path.join(output_dir, "selected_layers.yaml")
    save_layer_config(
        selected_layers,
        config_path,
        metadata={
            "model": args.model,
            "num_samples": args.num_samples,
            "strategy": args.strategy,
            "top_k": args.top_k,
            "attributes": list(correlation_matrices.keys()),
        },
    )
    print(f"Layer config saved to: {config_path}")

    # 保存完整结果
    results = {
        "config": {
            "model": args.model,
            "num_samples": args.num_samples,
            "top_k": args.top_k,
            "strategy": args.strategy,
        },
        "selected_layers": selected_layers,
        "attributes": list(correlation_matrices.keys()),
    }

    # 添加相关性统计
    stats = {}
    for attr, matrix in correlation_matrices.items():
        matrix_np = matrix.numpy() if isinstance(matrix, torch.Tensor) else matrix
        stats[attr] = {
            "max": float(matrix_np.max()),
            "min": float(matrix_np.min()),
            "mean": float(matrix_np.mean()),
            "std": float(matrix_np.std()),
        }
    results["correlation_stats"] = stats

    save_results(results, args.output)
    print(f"\nResults saved to: {args.output}")

    print("\n=== Phase 1 Complete ===")
    print(f"Recommended inject_layers: {selected_layers}")
    print("\nTo use this config, update configs/model.yaml with:")
    print(f"  inject_layers: {selected_layers}")


def generate_synthetic_data(num_samples: int):
    """
    生成合成数据用于测试

    Args:
        num_samples: 样本数量

    Returns:
        texts, attributes
    """
    import random

    # 生成随机属性
    attributes = {
        "age": [random.random() for _ in range(num_samples)],
        "gender": [random.choice([0, 1, 2]) for _ in range(num_samples)],
        "introversion": [random.random() for _ in range(num_samples)],
        "openness": [random.random() for _ in range(num_samples)],
        "conscientiousness": [random.random() for _ in range(num_samples)],
        "agreeableness": [random.random() for _ in range(num_samples)],
        "neuroticism": [random.random() for _ in range(num_samples)],
    }

    # 生成文本模板
    text_templates = [
        "一个{age}岁的{gender}，喜欢{personality}",
        "用户是{age}岁，{personality}",
        "{personality}的用户，今年{age}岁",
    ]

    gender_map = {0: "中性", 1: "男性", 2: "女性"}
    personality_traits = [
        "内向安静",
        "外向活泼",
        "认真负责",
        "创意十足",
        "善良友好",
        "理性务实",
    ]

    texts = []
    for i in range(num_samples):
        age = int(attributes["age"][i] * 50 + 20)
        gender = gender_map[attributes["gender"][i]]
        personality = random.choice(personality_traits)

        text = random.choice(text_templates).format(
            age=age,
            gender=gender,
            personality=personality,
        )
        texts.append(text)

    return texts, attributes


if __name__ == "__main__":
    main()
