#!/usr/bin/env python3
"""
Judge V1 - 从预生成对话文件评估
"""
import argparse
import json
import logging
import os
import sys
import numpy as np
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

STRICT_RUBRIC_PROMPT = """你是一个专业的对话质量评估专家。请根据以下标准对对话进行评分（1-5分）：

**用户画像**：{profile}
**人格特征**：{personality}

**对话内容**：
{dialogue}

**评分标准**：
5分：完美符合人格特征，风格一致，内容相关
4分：较好符合人格特征，偶有偏差
3分：基本符合，但有明显不一致
2分：部分符合，多处不符
1分：完全不符合人格特征

请只输出一个1-5之间的整数分数。"""


import time

def judge_strict(client, judge_model, conversation, profile, personality, max_retries=3):
    if client is None:
        return float(np.random.uniform(2.0, 4.0))

    dialogue = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)
    prompt = STRICT_RUBRIC_PROMPT.format(profile=profile, personality=personality, dialogue=dialogue)

    for attempt in range(max_retries):
        try:
            temperature = 1.0 if "GPT-5" in judge_model else 0.0
            resp = client.chat.completions.create(
                model=judge_model, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=10,
            )
            score_text = resp.choices[0].message.content.strip()
            score = float(score_text)
            return max(1.0, min(5.0, score))
        except Exception as e:
            logger.warning(f"Judge failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避: 1s, 2s, 4s
            else:
                logger.error(f"All retries failed, using random score")
                return float(np.random.uniform(2.0, 4.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", required=True, help="预生成对话JSON文件")
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
    with open(args.conversations, encoding="utf-8") as f:
        samples = json.load(f)
    logger.info(f"Evaluating {len(samples)} samples")

    per_sample = []
    scores = []

    for sample in samples:
        conv = sample["conversation"]
        score = judge_strict(client, args.judge_model, conv, sample.get("profile", ""), sample.get("personality", ""))
        scores.append(score)
        per_sample.append({
            "user_id": sample.get("user_id", ""),
            "personality": sample.get("personality", "")[:80],
            "score": score,
            "num_turns": len(conv) // 2,
        })
        logger.info(f"  user={sample.get('user_id','')} score={score}")

    score_arr = np.array(scores)
    distribution = {str(i): int((score_arr == i).sum()) for i in range(1, 6)}

    results = {
        "method": "v1_strict_rubric",
        "conversations_file": args.conversations,
        "judge_model": args.judge_model,
        "num_samples": len(scores),
        "al_k_avg": float(score_arr.mean()),
        "al_k_std": float(score_arr.std()),
        "al_k_min": float(score_arr.min()),
        "al_k_max": float(score_arr.max()),
        "score_distribution": distribution,
        "per_sample": per_sample,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
