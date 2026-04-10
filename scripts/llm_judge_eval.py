#!/usr/bin/env python3
"""
LLM Judge评估脚本
评估对话的人格一致性
"""
import json
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("错误: 需要安装 openai 包")
    sys.exit(1)


class LLMJudge:
    def __init__(self, api_key=None, base_url=None, model="GPT-4o-mini"):
        self.model = model
        if api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            print(f"LLM Judge 初始化成功: {model}")
        else:
            raise ValueError("需要API key")

    def evaluate_response(self, response, personality, user_msg):
        prompt = f"""评估回复与人格的一致性。

人格描述: {personality}

用户消息: {user_msg}

助手回复: {response}

评分标准 (1-5分):
1分 = 完全不一致，回复与描述的人格完全不符
2分 = 大部分不一致，回复偶尔体现人格特征
3分 = 中立，回复部分体现人格特征
4分 = 大部分一致，回复较好体现人格特征
5分 = 完全一致，回复完美体现人格特征

请只输出一个数字(1-5):"""

        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是专业的对话评估专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=10,
            )
            content = result.choices[0].message.content.strip()
            for char in content:
                if char.isdigit():
                    return float(min(max(int(char), 1), 5))
            return 3.0
        except Exception as e:
            print(f"API错误: {e}")
            return 3.0


def evaluate_conversations(conversations_file, output_file, judge):
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
                    score = judge.evaluate_response(response, personality, user_msg)
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
            "total_turns": len(all_scores)
        }, f, indent=2)

    return avg_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge_model", default="GPT-4o-mini")
    args = parser.parse_args()

    api_key = os.environ.get("BLSC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("BLSC_BASE_URL", "https://llmapi.blsc.cn")

    if not api_key:
        print("错误: 未找到API key，请设置 BLSC_API_KEY 环境变量")
        sys.exit(1)

    judge = LLMJudge(api_key=api_key, base_url=base_url, model=args.judge_model)

    avg_score = evaluate_conversations(args.conversations, args.output, judge)
    print(f"\n平均分数: {avg_score:.2f}")


if __name__ == "__main__":
    main()
