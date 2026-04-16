#!/usr/bin/env python
"""教师人格区分能力审计：评审候选样本。"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

PAIR_JUDGE_PROMPT = """你在评估两个候选回复，判断哪个更符合给定人格，并判断它们的人格差异是否足够明显。

目标人格：
{persona_text}

用户输入：
{user_input}

候选 A：
{candidate_a}

候选 B：
{candidate_b}

请输出 JSON，字段如下：
{{
  "preferred": "A" 或 "B" 或 "TIE",
  "difference_strength": "strong" 或 "weak",
  "avg_collapse_flag": true 或 false,
  "failure_tags": ["标签1", "标签2"],
  "judge_note": "一句简短说明"
}}

判断标准：
1. 哪个更符合目标人格
2. 是否自然、像真实聊天
3. 是否有人格差异，而不是平均化中性回复
4. 是否有 thinking / planning / 草稿感

只输出 JSON，不要输出其他内容。
"""

LEAK_PATTERNS = [
    'okay,', 'i need to', 'let me think', 'the user is', 'i should', 'my response should'
]


def detect_leak(text: str) -> bool:
    low = (text or '').lower()
    return any(p in low for p in LEAK_PATTERNS)


def safe_json_loads(text: str) -> dict:
    text = (text or '').strip()
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        text = text[start:end+1]
    return json.loads(text)


def call_with_retry(client: OpenAI, model: str, messages: list, timeout: int = 60, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
                max_tokens=300,
                timeout=timeout,
            )
            return resp.choices[0].message.content or '{}'
        except Exception as e:
            wait = 2 ** attempt
            print(f'  [warn] attempt {attempt}/{max_retries} failed: {e}. retrying in {wait}s...')
            if attempt == max_retries:
                raise
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description='运行教师人格区分能力审计（评审阶段）')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='GPT-5.2')
    parser.add_argument('--base_url', default=None)
    parser.add_argument('--api_env', default='BLSC_API_KEY')
    parser.add_argument('--timeout', type=int, default=60, help='单次 API 调用超时秒数')
    args = parser.parse_args()

    api_key = os.environ.get(args.api_env) or os.environ.get('OPENAI_API_KEY')
    base_url = args.base_url or os.environ.get('OPENAI_BASE_URL')
    if not api_key or not base_url:
        _cfg_path = Path('configs/api_config.yaml')
        if _cfg_path.exists():
            import yaml
            _cfg = yaml.safe_load(_cfg_path.read_text())
            _default = _cfg.get('default', 'blsc')
            _api_cfg = _cfg.get(_default, {})
            api_key = api_key or _api_cfg.get('api_key')
            base_url = base_url or _api_cfg.get('base_url')
    client = OpenAI(api_key=api_key, base_url=base_url)

    items = json.loads(Path(args.input).read_text(encoding='utf-8'))
    grouped = defaultdict(list)
    for item in items:
        grouped[(item['sample_id'], item['persona_id'])].append(item)

    # 断点续传：加载已有结果
    output_path = Path(args.output)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding='utf-8'))
        if isinstance(existing, dict):
            results = existing.get('results', [])
        else:
            results = existing
        done_keys = {(r['sample_id'], r['persona_id']) for r in results}
        print(f'断点续传：已有 {len(results)} 条，跳过已完成项')
    else:
        results = []
        done_keys = set()

    for key, pair in grouped.items():
        if len(pair) != 2:
            continue
        sample_id, persona_id = key
        if key in done_keys:
            print(f'skip {sample_id} {persona_id} (already done)')
            continue
        a, b = pair[0], pair[1]
        prompt = PAIR_JUDGE_PROMPT.format(
            persona_text=a['persona_text'],
            user_input=a['user_input'],
            candidate_a=a['response'],
            candidate_b=b['response'],
        )
        content = call_with_retry(
            client, args.model,
            [{'role': 'user', 'content': prompt}],
            timeout=args.timeout,
        )
        parsed = safe_json_loads(content)
        result = {
            'sample_id': a['sample_id'],
            'persona_id': a['persona_id'],
            'persona_text': a['persona_text'],
            'user_input': a['user_input'],
            'candidate_a': a['response'],
            'candidate_b': b['response'],
            'leak_flag_a': detect_leak(a['response']),
            'leak_flag_b': detect_leak(b['response']),
            'judge_model': args.model,
            'preferred': parsed.get('preferred', 'TIE'),
            'difference_strength': parsed.get('difference_strength', 'weak'),
            'avg_collapse_flag': bool(parsed.get('avg_collapse_flag', False)),
            'failure_tags': parsed.get('failure_tags', []),
            'judge_note': parsed.get('judge_note', ''),
            'raw_judge_output': content,
        }
        results.append(result)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'judged {a["sample_id"]} {a["persona_id"]}')

    total = len(results)
    leak_count = sum(1 for x in results if x['leak_flag_a'] or x['leak_flag_b'])
    strong_count = sum(1 for x in results if x['difference_strength'] == 'strong')
    collapse_count = sum(1 for x in results if x['avg_collapse_flag'])
    summary = {
        'total_pairs': total,
        'leak_pair_rate': round(leak_count / total, 4) if total else 0.0,
        'difference_strong_rate': round(strong_count / total, 4) if total else 0.0,
        'avg_collapse_rate': round(collapse_count / total, 4) if total else 0.0,
        'results': results,
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
