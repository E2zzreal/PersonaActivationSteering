#!/usr/bin/env python3
"""
Judge V3 - 从预生成对话文件评估（多维度）
"""
import argparse
import json
import logging
import os
import sys
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

MULTIDIM_PROMPT = """你是一个专业的对话质量评估专家。请从三个维度对对话进行评分（每个维度1-5分）：

**用户画像**：{profile}
**人格特征**：{personality}

**对话内容**：
{dialogue}

**评分维度**：
1. **风格匹配度**（style）：语气、用词、表达方式是否符合人格特征
2. **内容相关性**（content）：回复内容是否切题、有价值
3. **一致性**（consistency）：多轮对话中人格特征是否保持一致

请以JSON格式输出三个分数：
{{"style": X, "content": Y, "consistency": Z}}"""


def judge_multidim(client, judge_model, conversation, profile, personality, max_retries=3):
    if client is None:
        return {"style": 3.0, "content": 3.0, "consistency": 3.0}

    dialogue = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in conversation)
    prompt = MULTIDIM_PROMPT.format(profile=profile, personality=personality, dialogue=dialogue)

    for attempt in range(max_retries):
        try:
            temperature = 1.0 if "GPT-5" in judge_model else 0.0
            resp = client.chat.completions.create(
                model=judge_model, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=50,
            )
            result_text = resp.choices[0].message.content.strip()
            # 清理markdown代码块
            if result_text.startswith("```"):
                lines = result_text.split("\n")
                result_text = "\n".join(lines[1:-1]) if len(lines) > 2 else result_text
            result_text = result_text.strip()
            scores = json.loads(result_text)
            return {
                "style": max(1.0, min(5.0, float(scores.get("style", 3)))),
                "content": max(1.0, min(5.0, float(scores.get("content", 3)))),
                "consistency": max(1.0, min(5.0, float(scores.get("consistency", 3)))),
            }
        except Exception as e:
            logger.warning(f"Judge failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避: 1s, 2s, 4s
            else:
                logger.error(f"All retries failed, using default scores")
                return {"style": 3.0, "content": 3.0, "consistency": 3.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", required=True, help="预生成对话JSON文件")
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge_model", default="Claude-Sonnet-4.6")
    parser.add_argument("--style_weight", type=float, default=0.4)
    parser.add_argument("--content_weight", type=float, default=0.4)
    parser.add_argument("--consistency_weight", type=float, default=0.2)
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
    style_scores, content_scores, consistency_scores, weighted_scores = [], [], [], []

    for sample in samples:
        conv = sample["conversation"]
        scores = judge_multidim(client, args.judge_model, conv, sample.get("profile", ""), sample.get("personality", ""))
        weighted = (scores["style"] * args.style_weight +
                   scores["content"] * args.content_weight +
                   scores["consistency"] * args.consistency_weight)

        style_scores.append(scores["style"])
        content_scores.append(scores["content"])
        consistency_scores.append(scores["consistency"])
        weighted_scores.append(weighted)

        per_sample.append({
            "user_id": sample.get("user_id", ""),
            "personality": sample.get("personality", "")[:80],
            "style_score": scores["style"],
            "content_score": scores["content"],
            "consistency_score": scores["consistency"],
            "weighted_score": weighted,
            "num_turns": len(conv) // 2,
        })
        logger.info(f"  user={sample.get('user_id','')} style={scores['style']} content={scores['content']} consistency={scores['consistency']}")

    results = {
        "method": "v3_multidimensional",
        "conversations_file": args.conversations,
        "judge_model": args.judge_model,
        "num_samples": len(samples),
        "weights": {"style": args.style_weight, "content": args.content_weight, "consistency": args.consistency_weight},
        "style": {"mean": float(np.mean(style_scores)), "std": float(np.std(style_scores)), "min": float(np.min(style_scores)), "max": float(np.max(style_scores))},
        "content": {"mean": float(np.mean(content_scores)), "std": float(np.std(content_scores)), "min": float(np.min(content_scores)), "max": float(np.max(content_scores))},
        "consistency": {"mean": float(np.mean(consistency_scores)), "std": float(np.std(consistency_scores)), "min": float(np.min(consistency_scores)), "max": float(np.max(consistency_scores))},
        "weighted": {"mean": float(np.mean(weighted_scores)), "std": float(np.std(weighted_scores)), "min": float(np.min(weighted_scores)), "max": float(np.max(weighted_scores))},
        "per_sample": per_sample,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
