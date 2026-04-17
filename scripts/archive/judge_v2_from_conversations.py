#!/usr/bin/env python3
"""
Judge V2 - 从预生成对话文件进行A/B对比评估
"""
import argparse
import json
import logging
import os
import time
import numpy as np
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AB_COMPARE_PROMPT = """你是一个专业的对话质量评估专家。请比较以下两个模型的对话，判断哪个更好地符合用户人格特征。

**用户画像**：{profile}
**人格特征**：{personality}

**模型A的对话**：
{dialogue_a}

**模型B的对话**：
{dialogue_b}

**评分标准**：
- 哪个模型的回复更符合人格特征？
- 哪个模型的风格更一致？

请只输出一个字母：A 或 B（表示哪个模型更好）"""


def judge_ab(client, judge_model, conv_a, conv_b, profile, personality, max_retries=3):
    if client is None:
        return np.random.choice(["A", "B"])

    dialogue_a = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conv_a)
    dialogue_b = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conv_b)
    prompt = AB_COMPARE_PROMPT.format(profile=profile, personality=personality, dialogue_a=dialogue_a, dialogue_b=dialogue_b)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=judge_model, messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=10,
            )
            result = resp.choices[0].message.content.strip().upper()
            if result in ["A", "B"]:
                return result
            return np.random.choice(["A", "B"])
        except Exception as e:
            logger.warning(f"Judge failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"All retries failed, random choice")
                return np.random.choice(["A", "B"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations_a", required=True, help="模型A的对话文件")
    parser.add_argument("--conversations_b", required=True, help="模型B的对话文件")
    parser.add_argument("--model_a_name", required=True, help="模型A名称")
    parser.add_argument("--model_b_name", required=True, help="模型B名称")
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge_model", default="Claude-Sonnet-4.6")
    args = parser.parse_args()

    # Init judge client
    client = None
    if OPENAI_AVAILABLE:
        api_key = os.environ.get("BLSC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key, base_url="https://llmapi.blsc.cn")
            logger.info(f"Judge: {args.judge_model}")

    # Load conversations
    with open(args.conversations_a, encoding="utf-8") as f:
        samples_a = json.load(f)
    with open(args.conversations_b, encoding="utf-8") as f:
        samples_b = json.load(f)

    # Match by user_id
    samples_b_dict = {s["user_id"]: s for s in samples_b}
    logger.info(f"Model A: {len(samples_a)} samples, Model B: {len(samples_b)} samples")

    per_sample = []
    a_wins = 0
    b_wins = 0

    for sample_a in samples_a:
        user_id = sample_a["user_id"]
        if user_id not in samples_b_dict:
            continue

        sample_b = samples_b_dict[user_id]
        result = judge_ab(client, args.judge_model, sample_a["conversation"], sample_b["conversation"],
                         sample_a.get("profile", ""), sample_a.get("personality", ""))

        if result == "A":
            a_wins += 1
        else:
            b_wins += 1

        per_sample.append({
            "user_id": user_id,
            "winner": result,
        })
        logger.info(f"  user={user_id} winner={result}")

    total = a_wins + b_wins
    results = {
        "method": "v2_ab_compare",
        "model_a": args.model_a_name,
        "model_b": args.model_b_name,
        "conversations_a": args.conversations_a,
        "conversations_b": args.conversations_b,
        "judge_model": args.judge_model,
        "total_comparisons": total,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "a_win_rate": a_wins / total if total > 0 else 0,
        "b_win_rate": b_wins / total if total > 0 else 0,
        "per_sample": per_sample,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results: A wins {a_wins}/{total} ({a_wins/total*100:.1f}%), B wins {b_wins}/{total} ({b_wins/total*100:.1f}%)")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
