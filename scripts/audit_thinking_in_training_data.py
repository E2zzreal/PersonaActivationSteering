#!/usr/bin/env python
"""审计训练数据中的伪思考样本。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.thinking_leak import detect_patterns

CN_HINTS = ["用户", "我需要", "我应该", "首先", "接下来"]
EN_HINTS = ["Okay,", "I need to", "I should", "Let me think", "the user is"]


def detect_language(text: str) -> str:
    has_cn = any('\u4e00' <= ch <= '\u9fff' for ch in text)
    has_en = any(('a' <= ch.lower() <= 'z') for ch in text)
    if has_cn and has_en:
        return 'mixed'
    if has_cn:
        return 'zh'
    if has_en:
        return 'en'
    return 'other'


def extract_turns(path: Path, sample_limit: int | None):
    count = 0
    with path.open('r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if sample_limit is not None and count >= sample_limit:
                break
            if not line.strip():
                continue
            sample = json.loads(line)
            conv = sample.get('conversations', [])
            for i in range(0, len(conv) - 1, 2):
                user_msg = conv[i]
                asst_msg = conv[i + 1]
                if user_msg.get('role') != 'user' or asst_msg.get('role') != 'assistant':
                    continue
                yield {
                    'sample_index': line_idx,
                    'turn_index': i // 2,
                    'user_id': sample.get('user_id', f'u{line_idx}'),
                    'personality': sample.get('personality', ''),
                    'profile': sample.get('profile', ''),
                    'assistant_text': asst_msg.get('content', ''),
                }
            count += 1


def main():
    parser = argparse.ArgumentParser(description='审计训练数据中的伪思考样本')
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', default='train')
    parser.add_argument('--sample_limit', type=int, default=None)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_turns = 0
    thinking_like_turns = 0
    by_pattern = Counter()
    by_language = Counter()
    by_turn_index = Counter()
    examples = []

    for item in extract_turns(path, args.sample_limit):
        total_turns += 1
        text = item['assistant_text']
        patterns = detect_patterns(text)
        if not patterns:
            extra = []
            for hint in EN_HINTS + CN_HINTS:
                if hint in text:
                    extra.append(hint)
            patterns = extra
        language = detect_language(text)
        by_language[language] += 1
        if patterns:
            thinking_like_turns += 1
            by_turn_index[item['turn_index']] += 1
            for p in patterns:
                by_pattern[p] += 1
            if len(examples) < 50:
                examples.append({
                    'sample_index': item['sample_index'],
                    'turn_index': item['turn_index'],
                    'user_id': item['user_id'],
                    'language': language,
                    'patterns': patterns,
                    'preview': text[:240],
                })

    result = {
        'data': str(path),
        'split': args.split,
        'total_turns': total_turns,
        'thinking_like_turns': thinking_like_turns,
        'thinking_like_ratio': round(thinking_like_turns / total_turns, 4) if total_turns else 0.0,
        'by_pattern': dict(by_pattern.most_common()),
        'by_language': dict(by_language),
        'by_turn_index': dict(sorted(by_turn_index.items())),
        'top_examples': examples,
    }

    (output_dir / 'thinking_audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    md = [
        '# 训练数据伪思考审计结果',
        '',
        f'- 数据文件：`{path}`',
        f'- split：`{args.split}`',
        f'- 总 turn 数：`{total_turns}`',
        f'- 命中伪思考 turn 数：`{thinking_like_turns}`',
        f'- 命中比例：`{result["thinking_like_ratio"]}`',
        '',
        '## 模式统计',
    ]
    for key, value in by_pattern.most_common(20):
        md.append(f'- `{key}`: {value}')
    md.append('')
    md.append('## 语言分布')
    for key, value in by_language.items():
        md.append(f'- `{key}`: {value}')
    md.append('')
    md.append('## 轮次分布')
    for key, value in sorted(by_turn_index.items()):
        md.append(f'- `turn_{key}`: {value}')
    md.append('')
    md.append('## 样本摘录')
    for ex in examples[:10]:
        md.append(f"- sample={ex['sample_index']} turn={ex['turn_index']} lang={ex['language']} patterns={ex['patterns']} preview={ex['preview']}")

    (output_dir / 'thinking_audit.md').write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
