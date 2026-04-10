#!/usr/bin/env python3
"""
模拟LLM Judge评估（用于测试，无需API）
基于自动指标进行模拟评分
"""
import json
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import re

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_personality_match(response, personality):
    """分析回复与人格的匹配度（基于关键词）"""
    response_lower = response.lower()
    personality_lower = personality.lower()

    # 提取人格关键词
    keywords = []
    # 常见人格特质词
    trait_patterns = [
        r'\b(independent|independent)\b',
        r'\b(empathetic|empathy|understanding)\b',
        r'\b(creative|creativity|innovative)\b',
        r'\b(organized|organized|methodical)\b',
        r'\b(outgoing|social|friendly)\b',
        r'\b(reflective|thoughtful|deep)\b',
        r'\b(ambitious|driven|goal-oriented)\b',
        r'\b(humble|modest)\b',
        r'\b(supportive|helpful)\b',
        r'\b(adaptable|flexible)\b',
    ]

    matches = 0
    for pattern in trait_patterns:
        if re.search(pattern, personality_lower):
            if re.search(pattern, response_lower):
                matches += 1

    # AI身份检测
    ai_phrases = [
        "i'm an ai", "i am an ai", "as an ai", "language model",
        "don't have feelings", "don't have emotions", "i don't have personal"
    ]
    ai_detected = any(phrase in response_lower for phrase in ai_phrases)

    # 基础分数
    base_score = 3.0

    # 根据匹配度调整
    if matches >= 3:
        base_score = 4.5
    elif matches >= 1:
        base_score = 3.5

    # 检测到AI身份则扣分
    if ai_detected:
        base_score = max(1.0, base_score - 2.0)

    # 回复长度惩罚（太短或太长）
    word_count = len(response.split())
    if word_count < 10:
        base_score = max(1.0, base_score - 0.5)

    return min(5.0, max(1.0, base_score))


def evaluate_conversations(conversations_file, output_file):
    print(f"评估文件: {conversations_file}")

    with open(conversations_file) as f:
        data = json.load(f)

    results = []
    all_scores = []

    for sample in tqdm(data, desc="评估中"):
        personality = sample.get("personality", "")
        conversations = sample.get("conversation", [])

        sample_scores = []

        for i in range(0, len(conversations), 2):
            if i + 1 < len(conversations):
                user_msg = conversations[i].get("content", "")
                response = conversations[i + 1].get("content", "")

                if response and personality:
                    score = analyze_personality_match(response, personality)
                    sample_scores.append(score)
                    all_scores.append(score)

        results.append({
            "user_id": sample.get("user_id"),
            "scores": sample_scores,
            "avg_score": np.mean(sample_scores) if sample_scores else 0
        })

    avg_score = np.mean(all_scores) if all_scores else 0

    with open(output_file, 'w') as f:
        json.dump({
            "detailed_results": results,
            "average_score": avg_score,
            "overall_std": np.std(all_scores) if all_scores else 0,
            "total_turns": len(all_scores),
            "note": "模拟评估（基于关键词匹配）"
        }, f, indent=2)

    return avg_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    avg_score = evaluate_conversations(args.conversations, args.output)
    print(f"\n平均分数: {avg_score:.2f}")


if __name__ == "__main__":
    main()
