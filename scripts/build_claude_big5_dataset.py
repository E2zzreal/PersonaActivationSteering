#!/usr/bin/env python
"""
从 generate_parallel_dialogues.py 的输出构建高质量 Big Five 训练集

提取 claude_response，附加 Big5 分数（用已有的 personality → Big5 映射），
输出与 ALOEDataset 兼容的 JSONL 文件。

最终数据集构成：
  dialogues_train.json（claude_response）× 2777 条 + Big5 映射
  + Big Five 原始数据 × 471 条
  = ~3248 条高质量训练数据

用法：
  python scripts/build_claude_big5_dataset.py
  python scripts/build_claude_big5_dataset.py --max_turns 3 --output data/big5_cross_persona/train_claude_big5.jsonl
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialogues', default='results/parallel_dialogues/dialogues_train.json',
                        help='generate_parallel_dialogues.py 的输出')
    parser.add_argument('--big5_map', default='data/big5_cross_persona/aloe_personality_big5_map.json',
                        help='personality → Big5 映射（57个ALOE personality）')
    parser.add_argument('--big5_orig', default='data/big5_cross_persona/train.jsonl',
                        help='Big Five 原始数据（16个设计personality）')
    parser.add_argument('--output', default='data/big5_cross_persona/train_claude_big5.jsonl',
                        help='输出路径')
    parser.add_argument('--max_turns', type=int, default=1,
                        help='每条对话取前 N 轮（1=单轮，与当前训练设置一致）')
    parser.add_argument('--skip_empty', action='store_true', default=True,
                        help='跳过空回复')
    args = parser.parse_args()

    # 加载 Big5 映射
    big5_map = json.loads(Path(args.big5_map).read_text())
    print(f'Big5 映射: {len(big5_map)} 个 personality')

    # 加载并行对话
    dialogues_path = Path(args.dialogues)
    if not dialogues_path.exists():
        print(f'ERROR: {dialogues_path} 不存在')
        return
    dialogues = json.loads(dialogues_path.read_text())
    print(f'并行对话: {len(dialogues)} 条')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_no_map = 0
    skipped_empty = 0

    with open(out_path, 'w', encoding='utf-8') as fout:
        # Step 1: 从 dialogues_train 提取 claude_response
        for d in dialogues:
            personality = d.get('personality', '')
            big5_scores = big5_map.get(personality)

            if big5_scores is None:
                skipped_no_map += 1
                continue

            turns = d.get('turns', [])[:args.max_turns]
            conversations = []
            for t in turns:
                user_input = t.get('user_input', '').strip()
                claude_resp = t.get('claude_response', '').strip()
                if not user_input or not claude_resp:
                    skipped_empty += 1
                    continue
                conversations.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": claude_resp},
                ])

            if not conversations:
                continue

            item = {
                'persona_id': f"aloe_{d.get('user_id', '')}",
                'persona_name': f"ALOE-Claude",
                'big5_scores': big5_scores,
                'personality': personality,
                'profile': d.get('profile', ''),
                'user_input': conversations[0]['content'],
                'user_id': d.get('user_id', ''),
                'conversations': conversations,
            }
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            converted += 1

        # Step 2: 追加 Big Five 原始数据（设计 OCEAN 空间）
        big5_orig_count = 0
        if Path(args.big5_orig).exists():
            with open(args.big5_orig) as f:
                for line in f:
                    if line.strip():
                        fout.write(line)
                        big5_orig_count += 1

    print(f'\n=== 构建完成 ===')
    print(f'dialogues_train (claude): {converted} 条')
    print(f'Big Five 原始:            {big5_orig_count} 条')
    print(f'合计:                     {converted + big5_orig_count} 条')
    print(f'跳过(无Big5映射):          {skipped_no_map} 条')
    print(f'跳过(空回复):              {skipped_empty} 条')
    print(f'输出: {out_path}')

    # 简单质量检查
    data = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    personas = Counter(d['personality'][:40] for d in data)
    print(f'\n唯一 personality: {len(personas)}')

    import numpy as np
    scores = np.array([d['big5_scores'] for d in data])
    for i, dim in enumerate(['O', 'C', 'E', 'A', 'N']):
        print(f'  {dim}: mean={scores[:,i].mean():.2f}  std={scores[:,i].std():.2f}  range=[{scores[:,i].min():.1f},{scores[:,i].max():.1f}]')


if __name__ == '__main__':
    main()
