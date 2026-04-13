#!/usr/bin/env python
"""对训练集 gold vs baseline 做 A/B 审计。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

PROMPT = """你是一个人格一致性与回复质量评估专家。请比较两个候选回复，判断哪个更适合作为训练监督目标。

请重点考虑：
1. 是否更符合目标人格
2. 是否更自然、像真实对话回复
3. 是否更具体、更有帮助
4. 是否避免机械、模板化表达

目标人格：
{personality}

用户画像：
{profile}

用户输入：
{user_input}

候选A（训练集 gold 回复）：
{gold_response}

候选B（baseline 回复）：
{baseline_response}

请输出 JSON，格式如下：
{{
  "winner": "A" 或 "B" 或 "tie",
  "reason": "一句简短理由"
}}
"""


def parse_response(text: str):
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
    except Exception:
        pass
    return {'winner': 'tie', 'reason': text[:200]}


def main():
    parser = argparse.ArgumentParser(description='Judge 训练集监督质量 A/B 审计')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='GPT-5.2')
    args = parser.parse_args()

    api_key = os.environ.get('BLSC_API_KEY') or os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('OPENAI_BASE_URL', 'https://llmapi.blsc.cn/v1')
    client = OpenAI(api_key=api_key, base_url=base_url)

    items = json.loads(Path(args.input).read_text(encoding='utf-8'))
    results = []
    win = {'A': 0, 'B': 0, 'tie': 0}

    for item in items:
        prompt = PROMPT.format(**item)
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=1.0,
            max_tokens=200,
        )
        content = resp.choices[0].message.content or ''
        parsed = parse_response(content)
        winner = parsed.get('winner', 'tie')
        if winner not in win:
            winner = 'tie'
        win[winner] += 1
        results.append({
            **item,
            'judge_winner': winner,
            'judge_reason': parsed.get('reason', ''),
            'raw_judge_output': content,
        })

    summary = {
        'total': len(results),
        'gold_win': win['A'],
        'baseline_win': win['B'],
        'tie': win['tie'],
        'gold_win_rate': round(win['A'] / len(results), 4) if results else 0.0,
        'baseline_win_rate': round(win['B'] / len(results), 4) if results else 0.0,
        'results': results,
    }

    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
