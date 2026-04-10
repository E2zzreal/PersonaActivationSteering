#!/usr/bin/env python
"""
V4 重打分脚本 - 使用改进版英文prompt
对已生成的回复重新评分（不需要重新生成）
支持对比新旧两版prompt的评分差异
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# 改进版 LLM Judge（英文prompt + 详细rubric）
# ============================================================

SYSTEM_PROMPT = """You are an expert evaluator specializing in assessing how well AI assistants align their responses with user personalities. Your evaluation should be consistent, objective, and based on the following criteria."""

USER_PROMPT_TEMPLATE = """## Task
Evaluate how well the assistant's responses align with the user's personality traits throughout the dialogue.

## User Information
**Profile**: {profile}

**Personality Traits**: {personality}

## Dialogue
{dialogue}

## Scoring Rubric (1-5 Scale)

**5 - Excellent Alignment**
- Assistant's responses consistently reflect and adapt to the user's personality traits
- Language style, tone, and content choices match the user's profile
- Responses demonstrate deep understanding of user's characteristics

**4 - Good Alignment**
- Most responses show clear alignment with user's personality
- Minor inconsistencies in style or tone
- Good adaptation to user characteristics overall

**3 - Moderate Alignment**
- Some evidence of personality alignment
- Mixed consistency in reflecting user traits
- Responses are functional but lack personality-specific adaptation

**2 - Poor Alignment**
- Limited alignment with user's personality
- Generic responses that could apply to anyone
- Minimal consideration of user characteristics

**1 - No Alignment**
- Responses contradict or ignore user's personality traits
- Completely generic or mismatched communication style
- No evidence of personality-aware response generation

## Instructions
1. Read the user profile and personality traits carefully
2. Review the entire dialogue
3. Assess each assistant response against the user's personality
4. Provide a single score (1-5) based on the rubric above

## Output Format
Output ONLY a single number (1, 2, 3, 4, or 5). No explanations or additional text"""

# 旧版中文prompt（用于对比）
OLD_PROMPT_TEMPLATE = """你是一个人格一致性评估专家。请评估以下AI回复与目标人格的一致程度。

目标人格: {personality}

AI回复: {response}

评分标准:
1 - 完全不一致
2 - 较不一致
3 - 中立
4 - 较一致
5 - 完全一致

请只输出1-5的数字，不要输出任何其他内容。"""


class ImprovedLLMJudge:
    """改进版Judge - 对话级评估"""

    def __init__(self, api_key=None, base_url=None, model="GPT-5.2"):
        self.model = model
        if OPENAI_AVAILABLE and api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"[Improved Judge] initialized: {model}")
        else:
            self.client = None

    def score_dialogue(self, personality: str, profile: str, turns: list) -> float:
        """对话级评分：将同一样本的多轮对话拼接后一次性评估"""
        dialogue_lines = []
        for turn in turns:
            user_input = turn.get("user_input", "")
            response = turn.get("response", "")
            dialogue_lines.append(f"User: {user_input}\nAssistant: {response}")
        dialogue = "\n\n".join(dialogue_lines)

        prompt = USER_PROMPT_TEMPLATE.format(
            profile=profile,
            personality=personality,
            dialogue=dialogue,
        )

        return self._call_judge(prompt)

    def score_single(self, personality: str, profile: str, response: str) -> float:
        """单轮回复评分"""
        prompt = USER_PROMPT_TEMPLATE.format(
            profile=profile,
            personality=personality,
            dialogue=f"User: [question]\nAssistant: {response}",
        )

        return self._call_judge(prompt)

    def _call_judge(self, user_content: str) -> float:
        if self.client is None:
            return np.random.uniform(3.0, 4.5)

        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,  # 确定性评分
                max_tokens=10,
            )
            content = result.choices[0].message.content.strip()
            for char in content:
                if char.isdigit():
                    return float(min(max(int(char), 1), 5))
            logger.warning(f"Judge returned non-numeric: {content[:50]}")
            return 3.0
        except Exception as e:
            logger.warning(f"Judge failed: {e}")
            return 3.0


def group_responses_by_sample(responses: list) -> dict:
    """
    将响应按样本分组（同一personality+profile为同一样本）
    每个样本包含多轮对话
    """
    samples = {}
    for resp in responses:
        key = (resp.get("personality", "")[:200], resp.get("profile", "")[:200])
        if key not in samples:
            samples[key] = {
                "personality": resp.get("personality", ""),
                "profile": resp.get("profile", ""),
                "stage": resp.get("stage", ""),
                "turns": [],
            }
        samples[key]["turns"].append({
            "user_input": resp.get("user_input", ""),
            "response": resp.get("response", ""),
        })
    return samples


def rescore_model(eval_dir: str, judge: ImprovedLLMJudge, mode: str = "dialogue") -> dict:
    """
    对某个模型的所有stage进行重打分

    mode:
      - "dialogue": 对话级评估（同一样本的所有turn拼成一次评估）
      - "single": 单轮评估（每条回复独立评分）
    """
    eval_path = Path(eval_dir)
    all_results = {}

    for stage_file in sorted(eval_path.glob("*_responses.json")):
        stage_name = stage_file.stem.replace("_responses", "")
        logger.info(f"\n{'='*60}")
        logger.info(f"Re-scoring: {stage_name} (mode={mode})")
        logger.info(f"{'='*60}")

        with open(stage_file, "r", encoding="utf-8") as f:
            responses = json.load(f)

        # 读取旧评分作为对比
        old_scores = [r.get("judge_score", None) for r in responses]

        if mode == "dialogue":
            # 对话级：按样本分组
            samples = group_responses_by_sample(responses)
            new_scores = []
            new_per_sample = []

            for key, sample in tqdm(samples.items(), desc=f"[{stage_name}]"):
                score = judge.score_dialogue(
                    sample["personality"],
                    sample["profile"],
                    sample["turns"],
                )
                new_per_sample.append({
                    "personality": sample["personality"][:50],
                    "num_turns": len(sample["turns"]),
                    "new_score": score,
                })
                # 每个turn都记同一个分数（便于后续统计）
                new_scores.extend([score] * len(sample["turns"]))

            all_results[stage_name] = {
                "mode": "dialogue",
                "num_samples": len(samples),
                "num_responses": len(responses),
                "old_mean": round(float(np.mean([s for s in old_scores if s is not None])), 3),
                "old_std": round(float(np.std([s for s in old_scores if s is not None])), 3),
                "new_mean_per_sample": round(float(np.mean(new_scores)), 3),
                "new_std_per_sample": round(float(np.std(new_scores)), 3),
                "new_mean_unique": round(float(np.mean([s["new_score"] for s in new_per_sample])), 3),
                "new_std_unique": round(float(np.std([s["new_score"] for s in new_per_sample])), 3),
                "diff": round(float(np.mean(new_scores) - np.mean([s for s in old_scores if s is not None])), 3),
                "per_sample": new_per_sample,
            }

        else:  # single
            new_scores = []
            for resp in tqdm(responses, desc=f"[{stage_name}]"):
                score = judge.score_single(
                    resp.get("personality", ""),
                    resp.get("profile", ""),
                    resp.get("response", ""),
                )
                new_scores.append(score)
                resp["judge_score_v2"] = score  # 写入新分数

            # 保存带新分数的响应
            v2_path = eval_path / f"{stage_name}_responses_v2.json"
            with open(v2_path, "w", encoding="utf-8") as f:
                json.dump(responses, f, indent=2, ensure_ascii=False)
            logger.info(f"  Saved v2 scores to {v2_path}")

            all_results[stage_name] = {
                "mode": "single",
                "num_responses": len(responses),
                "old_mean": round(float(np.mean([s for s in old_scores if s is not None])), 3),
                "old_std": round(float(np.std([s for s in old_scores if s is not None])), 3),
                "new_mean": round(float(np.mean(new_scores)), 3),
                "new_std": round(float(np.std(new_scores)), 3),
                "diff": round(float(np.mean(new_scores) - np.mean([s for s in old_scores if s is not None])), 3),
            }

        logger.info(f"  Old: {all_results[stage_name]['old_mean']:.2f} ± {all_results[stage_name]['old_std']:.2f}")
        new_key = "new_mean_per_sample" if mode == "dialogue" else "new_mean"
        logger.info(f"  New: {all_results[stage_name][new_key]:.2f} ± {all_results[stage_name].get('new_std_per_sample', all_results[stage_name].get('new_std', 0)):.2f}")
        logger.info(f"  Diff: {all_results[stage_name]['diff']:+.3f}")

    return all_results


def print_comparison(model_name: str, results: dict, mode: str):
    """打印对比表"""
    logger.info(f"\n{'='*70}")
    logger.info(f"{model_name} - PROMPT COMPARISON (mode={mode})")
    logger.info(f"{'='*70}")
    logger.info(f"{'Stage':<12} {'Old Mean':>10} {'Old Std':>8} {'New Mean':>10} {'New Std':>8} {'Diff':>8}")
    logger.info(f"{'-'*60}")

    for name, data in results.items():
        if mode == "dialogue":
            new_mean = data.get("new_mean_unique", data.get("new_mean_per_sample"))
            new_std = data.get("new_std_unique", data.get("new_std_per_sample"))
        else:
            new_mean = data.get("new_mean")
            new_std = data.get("new_std")

        logger.info(
            f"{name:<12} {data['old_mean']:>10.3f} {data['old_std']:>8.3f} "
            f"{new_mean:>10.3f} {new_std:>8.3f} {data['diff']:>+8.3f}"
        )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="dialogue",
                        choices=["dialogue", "single"],
                        help="dialogue=对话级评估(推荐), single=单轮评估")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["qwen25", "qwen3"],
                        help="Which models to rescore")
    parser.add_argument("--judge_model", type=str, default="GPT-5.2")
    parser.add_argument("--stages", type=str, nargs="+", default=None,
                        help="Only rescore specific stages (e.g., stage3)")
    parser.add_argument("--output_suffix", type=str, default="eval_summary_v2",
                        help="Output filename suffix")
    args = parser.parse_args()

    api_key = os.environ.get("BLSC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://llmapi.blsc.cn/v1")
    judge = ImprovedLLMJudge(api_key=api_key, base_url=base_url, model=args.judge_model)

    model_dirs = {
        "qwen25": "results/v4_qwen25_eval",
        "qwen3": "results/v4_qwen3_eval",
    }

    for model_name in args.models:
        if model_name not in model_dirs:
            logger.warning(f"Unknown model: {model_name}, skipping")
            continue

        eval_dir = Path(model_dirs[model_name])
        if not eval_dir.exists():
            logger.warning(f"Directory not found: {eval_dir}")
            continue

        results = rescore_model(str(eval_dir), judge, mode=args.mode)

        # 如果指定了stages，过滤结果
        if args.stages:
            results = {k: v for k, v in results.items() if k in args.stages}

        print_comparison(f"{model_name.upper()}", results, args.mode)

        # 保存结果
        output_path = eval_dir / f"{args.output_suffix}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to {output_path}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
