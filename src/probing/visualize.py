"""
可视化与层选择模块

生成热力图并自动选择注入层配置。
"""

import json
import os
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml


def plot_heatmap(
    matrix: torch.Tensor | np.ndarray,
    save_path: str,
    title: str = "Correlation Heatmap",
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    """
    绘制 36×32 热力图

    Args:
        matrix: (num_layers, num_heads) 相关性矩阵
        save_path: 保存路径
        title: 图表标题
        figsize: 图表大小
        cmap: 颜色映射
        vmin: 最小值
        vmax: 最大值
    """
    # 转换为 numpy
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热力图
    sns.heatmap(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        xticklabels=8,  # 每8个head显示一个label
        yticklabels=4,  # 每4个layer显示一个label
        ax=ax,
    )

    ax.set_xlabel("Attention Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(title, fontsize=14)

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_multi_heatmap(
    matrices: dict[str, torch.Tensor | np.ndarray],
    save_dir: str,
    num_layers: int = 36,
    num_heads: int = 32,
) -> None:
    """
    绘制多个属性的热力图

    Args:
        matrices: {attr_name: correlation_matrix}
        save_dir: 保存目录
        num_layers: 层数
        num_heads: 头数
    """
    os.makedirs(save_dir, exist_ok=True)

    for attr_name, matrix in matrices.items():
        save_path = os.path.join(save_dir, f"{attr_name}.png")

        # 取每层平均相关性
        if isinstance(matrix, torch.Tensor):
            layer_mean = matrix.mean(dim=1)  # (num_layers,)
            matrix_2d = layer_mean.unsqueeze(1).repeat(1, num_heads).numpy()
        else:
            layer_mean = matrix.mean(axis=1)
            matrix_2d = np.tile(layer_mean.reshape(-1, 1), (1, num_heads))

        plot_heatmap(
            matrix_2d,
            save_path,
            title=f"{attr_name} - Layer × Head Correlation",
        )


def select_injection_layers(
    matrix: torch.Tensor | np.ndarray,
    top_k: int = 8,
    strategy: Literal["top", "continuous"] = "continuous",
) -> list[int]:
    """
    自动选择注入层

    Args:
        matrix: (num_layers, num_heads) 相关性矩阵
        top_k: 选择层数
        strategy: 选择策略
            - "top": 选择相关性最高的 k 层
            - "continuous": 选择相关性峰值附近的连续 k 层

    Returns:
        selected_layers: 选中的层索引列表
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()

    # 计算每层的平均相关性 (取绝对值的均值)
    layer_scores = np.abs(matrix).mean(axis=1)

    if strategy == "top":
        # 选择相关性最高的 k 层
        top_indices = np.argsort(layer_scores)[::-1][:top_k]
        selected_layers = sorted(top_indices.tolist())

    elif strategy == "continuous":
        # 找到相关性最高的峰值位置
        peak_idx = np.argmax(layer_scores)
        num_layers = len(layer_scores)

        # 从峰值向两边扩展，选择连续的层
        selected_layers = []
        half_k = top_k // 2

        # 以峰值为中心，向两边扩展
        left = peak_idx
        right = peak_idx + 1

        selected_layers.append(peak_idx)

        while len(selected_layers) < top_k:
            # 优先选择相关性更高的那一侧
            left_score = layer_scores[left - 1] if left > 0 else -1
            right_score = layer_scores[right] if right < num_layers else -1

            if left_score >= right_score and left > 0:
                left -= 1
                selected_layers.append(left)
            elif right < num_layers:
                selected_layers.append(right)
                right += 1
            else:
                # 已经到达边界，填充剩余位置
                break

        selected_layers = sorted(selected_layers)[:top_k]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return selected_layers


def select_injection_layers_multi_attr(
    matrices: dict[str, torch.Tensor | np.ndarray],
    top_k: int = 8,
    strategy: Literal["top", "continuous"] = "continuous",
) -> list[int]:
    """
    基于多个属性选择注入层

    Args:
        matrices: {attr_name: correlation_matrix}
        top_k: 选择层数
        strategy: 选择策略

    Returns:
        selected_layers: 选中的层索引列表
    """
    # 合并所有属性的相关性 (取平均)
    all_matrices = []
    for matrix in matrices.values():
        if isinstance(matrix, torch.Tensor):
            all_matrices.append(matrix)
        else:
            all_matrices.append(torch.from_numpy(matrix))

    # 计算平均相关性
    avg_matrix = torch.stack(all_matrices).mean(dim=0)

    return select_injection_layers(avg_matrix, top_k=top_k, strategy=strategy)


def save_layer_config(
    selected_layers: list[int],
    config_path: str,
    metadata: dict = None,
) -> None:
    """
    保存层配置到 YAML 文件

    Args:
        selected_layers: 选中的层列表
        config_path: 配置文件路径
        metadata: 附加元数据
    """
    # 确保 layers 是普通 Python int
    layers = [int(x) for x in selected_layers]

    config = {
        "inject_layers": layers,
        "num_layers": len(layers),
    }

    if metadata:
        config["metadata"] = metadata

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_results(
    results: dict,
    output_path: str,
) -> None:
    """
    保存结果到 JSON 文件

    Args:
        results: 结果字典
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 转换 tensor 为 list
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            serializable_results[key] = value.numpy().tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    serializable_results[key][k] = v.numpy().tolist()
                elif isinstance(v, (list, tuple)):
                    serializable_results[key][k] = [
                        int(x) if isinstance(x, (np.integer,)) else x for x in v
                    ]
                else:
                    serializable_results[key][k] = v
        elif isinstance(value, list):
            serializable_results[key] = [
                int(x) if isinstance(x, (np.integer,)) else x for x in value
            ]
        else:
            serializable_results[key] = value

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)


def load_processed_data(
    data_path: str,
    max_samples: int = None,
) -> tuple[list[dict], dict[str, list]]:
    """
    加载处理后的 ALOE 数据

    Args:
        data_path: JSONL 文件路径
        max_samples: 最大样本数

    Returns:
        texts: 文本列表
        attributes: 属性字典 {attr_name: [values]}
    """
    import jsonlines

    samples = []
    with jsonlines.open(data_path) as reader:
        for i, sample in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            samples.append(sample)

    # 提取文本 (使用 profile + personality 作为输入)
    texts = [
        f"{s.get('profile', '')} {s.get('personality', '')}" for s in samples
    ]

    # 提取属性
    from .attribute_extractor import AttributeExtractor

    extractor = AttributeExtractor()
    attributes_list = extractor.extract_batch(samples)

    # 转换为属性字典
    attributes = {}
    for attr in ["age", "gender", "introversion", "openness",
                 "conscientiousness", "agreeableness", "neuroticism"]:
        attributes[attr] = [a[attr] for a in attributes_list]

    return texts, attributes
