#!/usr/bin/env python
"""对训练集 gold vs baseline 做 A/B 审计（极简可用版）。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

PROMPT = """请比较两个候选回复，判断哪个更适合作为人格化对话训练的监督目标。

判断标准：
1. 更符合目标人格
2. 更自然，更像真实对话
3. 更具体、更贴合用户输入

目标人格：{personality}
用户输入：{user_input}

A：{gold_response}

B：{baseline_response}

如果 A 更好，只输出 A
如果 B 更好，只输出 B
如果差不多，只输出 TIE
不要输出任何其他内容。
"""


def normalize_winner(text: str) -> str:
    content = (text or '').strip().upper()
    if content.startswith('A'):
        return 'A'
    if content.startswith('B'):
        return 'B'
    return 'TIE'


def main():
    parser = argparse.ArgumentParser(description='Judge 训练集监督质量 A/B 审计')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='GPT-5.2')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get('BLSC_API_KEY') or os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('OPENAI_BASE_URL', 'https://llmapi.blsc.cn/v1')
    client = OpenAI(api_key=api_key, base_url=base_url)

    items = json.loads(Path(args.input).read_text(encoding='utf-8'))
    if args.limit is not None:
        items = items[:args.limit]

    results = []
    win = {'A': 0, 'B': 0, 'TIE': 0}

    for idx, item in enumerate(items, start=1):
        prompt = PROMPT.format(**item)
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=1.0,
            max_tokens=5,
        )
        content = resp.choices[0].message.content or ''
        winner = normalize_winner(content)
        win[winner] += 1
        results.append({
            **item,
            'judge_winner': winner,
            'raw_judge_output': content,
        })
        print(f'[{idx}/{len(items)}] winner={winner}')
        sys.stdout.flush()

    summary = {
        'total': len(results),
        'gold_win': win['A'],
        'baseline_win': win['B'],
        'tie': win['TIE'],
        'gold_win_rate': round(win['A'] / len(results), 4) if results else 0.0,
        'baseline_win_rate': round(win['B'] / len(results), 4) if results else 0.0,
        'results': results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
