#!/usr/bin/env python
"""
将 ALOE 数据集的 57 个 personality 映射到 Big Five 5D 分数
然后为全部 2777 条 ALOE 数据添加 big5_scores 字段

步骤:
  1. 提取 57 个唯一 personality 描述
  2. 用 LLM 对每个 personality 打 Big Five 分 [O,C,E,A,N] ∈ [-1,1]
  3. 将分数写回每条 ALOE 数据，输出兼容 Big Five 格式的 JSONL

用法:
  python scripts/convert_aloe_to_big5.py
"""
import json, re, sys, time, yaml, argparse
from pathlib import Path
from openai import OpenAI

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

BIG5_PROMPT = """Rate this personality description on the Big Five (OCEAN) dimensions.
Each dimension should be scored from -1.0 (very low) to +1.0 (very high), in increments of 0.1.

Personality: {personality}

Dimensions:
- O (Openness): creativity, curiosity, intellectual interests
- C (Conscientiousness): organization, discipline, reliability
- E (Extraversion): sociability, energy, assertiveness
- A (Agreeableness): warmth, cooperation, empathy
- N (Neuroticism): anxiety, emotional instability, sensitivity

Reply ONLY in this exact format:
O=X.X C=X.X E=X.X A=X.X N=X.X"""


def parse_big5(text):
    """从 LLM 回复中提取 Big Five 分数"""
    scores = {}
    for dim in ['O', 'C', 'E', 'A', 'N']:
        m = re.search(rf'{dim}\s*=\s*([+-]?\d+\.?\d*)', text)
        if m:
            val = float(m.group(1))
            scores[dim] = max(-1.0, min(1.0, val))
    if len(scores) == 5:
        return [scores['O'], scores['C'], scores['E'], scores['A'], scores['N']]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aloe_path', default='/home/kemove/Desktop/PersonaSteer/data/split/train.jsonl')
    parser.add_argument('--output', default='data/big5_cross_persona/aloe_big5.jsonl')
    parser.add_argument('--mapping_output', default='data/big5_cross_persona/aloe_personality_big5_map.json')
    parser.add_argument('--api_config', default='configs/api_config.yaml')
    parser.add_argument('--max_turns', type=int, default=1, help='每条取几轮对话(1=单轮)')
    args = parser.parse_args()

    # API
    cfg = yaml.safe_load(Path(args.api_config).read_text())
    api_cfg = cfg.get(cfg.get('default', 'blsc'), {})
    client = OpenAI(api_key=api_cfg['api_key'], base_url=api_cfg['base_url'])

    # 加载 ALOE 数据
    aloe_data = []
    with open(args.aloe_path) as f:
        for line in f:
            if line.strip():
                aloe_data.append(json.loads(line))
    print(f'加载 {len(aloe_data)} 条 ALOE 数据')

    # Step 1: 提取唯一 personality
    unique_personalities = {}
    for d in aloe_data:
        p = d.get('personality', '')
        if p and p not in unique_personalities:
            unique_personalities[p] = None
    print(f'唯一 personality: {len(unique_personalities)}')

    # Step 2: 检查是否已有映射
    mapping_path = Path(args.mapping_output)
    if mapping_path.exists():
        existing = json.loads(mapping_path.read_text())
        for p, scores in existing.items():
            if p in unique_personalities:
                unique_personalities[p] = scores
        done = sum(1 for v in unique_personalities.values() if v is not None)
        print(f'已有映射: {done}/{len(unique_personalities)}')

    # Step 3: 对未映射的 personality 调用 LLM
    todo = [p for p, v in unique_personalities.items() if v is None]
    print(f'需映射: {len(todo)} 个 personality')

    for i, personality in enumerate(todo):
        try:
            resp = client.chat.completions.create(
                model='GPT-5.4',
                messages=[{"role": "user", "content": BIG5_PROMPT.format(personality=personality[:500])}],
                max_tokens=100, temperature=0.0, timeout=30)
            raw = resp.choices[0].message.content or ''
            scores = parse_big5(raw)
            if scores:
                unique_personalities[personality] = scores
                print(f'  [{i+1}/{len(todo)}] O={scores[0]:+.1f} C={scores[1]:+.1f} E={scores[2]:+.1f} A={scores[3]:+.1f} N={scores[4]:+.1f} | {personality[:60]}')
            else:
                print(f'  [{i+1}/{len(todo)}] PARSE FAILED: {raw[:80]}')
        except Exception as e:
            print(f'  [{i+1}/{len(todo)}] ERROR: {e}')
        time.sleep(0.3)

    # 保存映射
    mapped = {p: v for p, v in unique_personalities.items() if v is not None}
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapped, indent=2, ensure_ascii=False))
    print(f'\n映射保存: {len(mapped)}/{len(unique_personalities)} → {mapping_path}')

    if len(mapped) < len(unique_personalities):
        unmapped = [p[:60] for p, v in unique_personalities.items() if v is None]
        print(f'未映射 personality: {unmapped}')
        print('请重新运行以补全映射')
        return

    # Step 4: 转换全部 ALOE 数据
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0

    with open(out_path, 'w', encoding='utf-8') as fout:
        for d in aloe_data:
            personality = d.get('personality', '')
            scores = unique_personalities.get(personality)
            if scores is None:
                skipped += 1
                continue

            convs = d.get('conversations', [])
            # 取前 max_turns 轮
            turns_convs = []
            for j in range(0, min(len(convs), args.max_turns * 2), 2):
                if j + 1 < len(convs):
                    turns_convs.extend([convs[j], convs[j + 1]])

            if not turns_convs:
                skipped += 1
                continue

            user_input = turns_convs[0]['content'] if turns_convs else ''

            item = {
                'persona_id': f"aloe_{d.get('user_id', '')}",
                'persona_name': f"ALOE-{personality[:30]}",
                'big5_scores': scores,
                'personality': personality,
                'profile': f"Big Five: O={scores[0]:+.1f} C={scores[1]:+.1f} E={scores[2]:+.1f} A={scores[3]:+.1f} N={scores[4]:+.1f}",
                'user_input': user_input,
                'user_id': d.get('user_id', ''),
                'conversations': turns_convs,
            }
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            converted += 1

    print(f'\n转换完成: {converted} 条写入 {out_path} (跳过 {skipped})')


if __name__ == '__main__':
    main()
