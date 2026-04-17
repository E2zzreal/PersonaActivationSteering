#!/usr/bin/env python3
"""
数据集重新划分脚本

修复问题：
1. 训练/验证集personality完全重叠 → 数据泄露
2. 对比学习需要同一personality的多条样本

解决方案：
- 按personality划分数据集
- 训练集：验证集 = 80:20
- 确保验证集的personality不出现在训练集
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import argparse


def load_jsonl(path: Path) -> list[dict]:
    """加载JSONL文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: Path):
    """保存JSONL文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def split_by_personality(
    data: list[dict],
    val_ratio: float = 0.2,
    seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """
    按personality划分数据集
    
    Args:
        data: 原始数据
        val_ratio: 验证集比例
        seed: 随机种子
        
    Returns:
        (train_data, val_data)
    """
    random.seed(seed)
    
    # 按personality分组
    personality_groups = defaultdict(list)
    for item in data:
        personality_groups[item['personality']].append(item)
    
    # 获取所有personality
    personalities = list(personality_groups.keys())
    random.shuffle(personalities)
    
    # 划分personality
    val_size = int(len(personalities) * val_ratio)
    val_personalities = set(personalities[:val_size])
    train_personalities = set(personalities[val_size:])
    
    print(f"=== 数据集划分 ===")
    print(f"总personality数: {len(personalities)}")
    print(f"训练集personality: {len(train_personalities)}")
    print(f"验证集personality: {len(val_personalities)}")
    print(f"重叠检查: {len(train_personalities & val_personalities)} (应为0)")
    
    # 构建数据集
    train_data = []
    val_data = []
    
    for personality, items in personality_groups.items():
        if personality in train_personalities:
            train_data.extend(items)
        else:
            val_data.extend(items)
    
    print(f"\n=== 样本数量 ===")
    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")
    
    # 验证每个personality的样本数
    train_p_counts = defaultdict(int)
    for item in train_data:
        train_p_counts[item['personality']] += 1
    
    print(f"\n=== 训练集personality分布 ===")
    counts = list(train_p_counts.values())
    print(f"最少: {min(counts)} 条")
    print(f"最多: {max(counts)} 条")
    print(f"平均: {sum(counts)/len(counts):.1f} 条")
    
    return train_data, val_data


def main():
    parser = argparse.ArgumentParser(description='重新划分数据集')
    parser.add_argument('--input', type=str, default='data/processed/train.jsonl',
                        help='输入数据文件')
    parser.add_argument('--output-dir', type=str, default='data/split',
                        help='输出目录')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()
    
    # 加载数据
    input_path = Path(args.input)
    print(f"加载数据: {input_path}")
    data = load_jsonl(input_path)
    print(f"总样本数: {len(data)}")
    
    # 划分数据集
    train_data, val_data = split_by_personality(
        data, 
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # 保存数据集
    output_dir = Path(args.output_dir)
    train_path = output_dir / 'train.jsonl'
    val_path = output_dir / 'val.jsonl'
    
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    
    print(f"\n=== 保存完成 ===")
    print(f"训练集: {train_path}")
    print(f"验证集: {val_path}")


if __name__ == '__main__':
    main()
