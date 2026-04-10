"""
ALOE评估指标计算示例
====================

根据ALOE论文，计算三个核心指标：
1. AL(K)_AVG - Average Alignment Score
2. N-IR - Normalized Improvement Ratio  
3. N-R² - Normalized Coefficient of Determination

"""

import numpy as np
from typing import List, Dict
import json

# ==========================================
# 1. AL(K)_AVG - 平均对齐分数
# ==========================================

def compute_alignment_score(
    response: str,
    personality: str,
    profile: str = ""
) -> float:
    """
    计算单个回复的对齐分数
    
    使用LLM Judge评估回复与personality的匹配度
    
    Args:
        response: 模型生成的回复
        personality: 用户人格描述
        profile: 用户画像（可选）
    
    Returns:
        float: 对齐分数 (1-5分)
    """
    # 这里用LLM Judge（如GPT-4）评分
    # 实际实现见 src/evaluation/llm_judge.py
    
    # 示例：基于关键词的简化计算
    personality_words = personality.lower().split()
    response_lower = response.lower()
    
    # 计算personality关键词在回复中的体现
    matches = sum(1 for word in personality_words if word in response_lower)
    score = min(5.0, 1.0 + matches * 0.5)
    
    return score


def compute_al_k_avg(
    conversations: List[Dict],
    personality: str,
    profile: str = ""
) -> float:
    """
    计算平均对齐分数 AL(K)_AVG
    
    公式: AL(K)_AVG = (1/K) * Σ AL(k)
    其中 K 是对话轮数, AL(k) 是第k轮的对齐分数
    
    Args:
        conversations: 对话列表 [{"user": str, "assistant": str}, ...]
        personality: 人格描述
        profile: 用户画像
    
    Returns:
        float: 平均对齐分数
    """
    scores = []
    
    for turn in conversations:
        assistant_response = turn.get("assistant", "")
        score = compute_alignment_score(
            response=assistant_response,
            personality=personality,
            profile=profile
        )
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0


# ==========================================
# 2. N-IR - 归一化改进比率
# ==========================================

def compute_n_ir(
    current_scores: List[float],
    baseline_scores: List[float]
) -> float:
    """
    计算归一化改进比率 N-IR
    
    衡量模型相对于baseline的改进程度
    
    公式: N-IR = (AVG_baseline - AVG_current) / AVG_baseline
    
    Args:
        current_scores: 当前模型的各轮对齐分数
        baseline_scores: baseline模型的各轮对齐分数
    
    Returns:
        float: 改进比率 (-1到1, 正值表示改进)
    """
    if not baseline_scores:
        return 0.0
    
    current_avg = np.mean(current_scores)
    baseline_avg = np.mean(baseline_scores)
    
    # 改进 = (baseline - current) / baseline
    # 注意：这里假设baseline是未训练的随机模型
    n_ir = (baseline_avg - current_avg) / baseline_avg
    
    return float(n_ir)


def compute_progressive_n_ir(
    turn_scores: List[float]
) -> float:
    """
    计算渐进改进比率
    
    衡量模型在多轮对话中逐步改进对齐的能力
    
    公式: N-IR = Σ (AL(k) - AL(1)) / (AL(K) - AL(1))
    
    Args:
        turn_scores: 各轮的对齐分数 [AL(1), AL(2), ..., AL(K)]
    
    Returns:
        float: 渐进改进比率
    """
    if len(turn_scores) < 2:
        return 0.0
    
    al_1 = turn_scores[0]  # 第一轮分数
    al_k = turn_scores[-1]  # 最后一轮分数
    
    if al_k == al_1:
        return 0.0
    
    # 计算累积改进
    improvements = [(score - al_1) for score in turn_scores[1:]]
    total_improvement = sum(improvements)
    
    # 归一化
    n_ir = total_improvement / (len(turn_scores) - 1) / (al_k - al_1)
    
    return float(n_ir)


# ==========================================
# 3. N-R² - 归一化决定系数
# ==========================================

def compute_n_r2(
    predicted_scores: List[float],
    ground_truth_scores: List[float]
) -> float:
    """
    计算归一化决定系数 N-R²
    
    衡量模型预测与人工标注的一致性和稳定性
    
    公式: N-R² = (R² - R²_min) / (R²_max - R²_min)
    
    Args:
        predicted_scores: 模型预测的对齐分数
        ground_truth_scores: 人工标注的对齐分数
    
    Returns:
        float: 归一化R² (0-1)
    """
    if len(predicted_scores) != len(ground_truth_scores):
        raise ValueError("预测分数和真实分数长度不一致")
    
    if len(predicted_scores) < 2:
        return 0.0
    
    # 计算皮尔逊相关系数
    correlation = np.corrcoef(predicted_scores, ground_truth_scores)[0, 1]
    
    # R² 是相关系数的平方
    r_squared = correlation ** 2
    
    # 归一化到 0-1 范围
    # 假设 R²_min = 0, R²_max = 1
    n_r2 = r_squared  # 已经在0-1范围内
    
    return float(n_r2)


def compute_stability_n_r2(
    repeated_scores: List[List[float]]
) -> float:
    """
    计算稳定性N-R²
    
    衡量模型多次运行的一致性
    
    Args:
        repeated_scores: 多次运行的分数列表
            [[run1_scores], [run2_scores], ...]
    
    Returns:
        float: 稳定性系数
    """
    if len(repeated_scores) < 2:
        return 0.0
    
    # 计算各次运行的均值
    means = [np.mean(scores) for scores in repeated_scores]
    
    # 计算方差
    variance = np.var(means)
    
    # 归一化
    mean_of_means = np.mean(means)
    n_r2 = 1.0 - (variance / (mean_of_means ** 2)) if mean_of_means != 0 else 0.0
    
    return float(max(0.0, n_r2))


# ==========================================
# 使用示例
# ==========================================

if __name__ == "__main__":
    # 示例数据
    personality = "enthusiastic, full of energy and passion"
    profile = "34-year-old designer who loves hiking"
    
    conversations = [
        {
            "user": "I just got back from a hike!",
            "assistant": "That sounds AMAZING! What trail did you explore?"
        },
        {
            "user": "It was a mountain trail with waterfalls",
            "assistant": "Wow! I can imagine the thrill! Waterfalls are breathtaking!"
        },
    ]
    
    # 1. 计算AL(K)_AVG
    al_k_avg = compute_al_k_avg(conversations, personality, profile)
    print(f"AL(K)_AVG: {al_k_avg:.3f}")
    
    # 2. 计算N-IR
    baseline_scores = [2.5, 2.8, 3.0, 2.7]  # 随机模型的分数
    current_scores = [4.0, 4.2, 4.5, 4.3]   # 训练后模型的分数
    n_ir = compute_n_ir(current_scores, baseline_scores)
    print(f"N-IR: {n_ir:.3f}")
    print(f"  解释: 改进了 {n_ir*100:.1f}%")
    
    # 3. 计算N-R²
    predicted = [4.0, 4.2, 4.5, 4.3]
    ground_truth = [4.1, 4.0, 4.6, 4.2]  # 人工标注
    n_r2 = compute_n_r2(predicted, ground_truth)
    print(f"N-R²: {n_r2:.3f}")
    print(f"  解释: 与人工标注一致性 {n_r2*100:.1f}%")
