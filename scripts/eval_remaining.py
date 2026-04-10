#!/usr/bin/env python3
"""评估剩余模型的脚本"""

import json
import yaml
import sys
import os
from tqdm import tqdm
from openai import OpenAI
from collections import Counter

# 加载配置
with open('configs/api_config.yaml') as f:
    config = yaml.safe_load(f)

api_key = config['blsc']['api_key']
base_url = config['blsc']['base_url']
model = config['blsc']['model']

client = OpenAI(api_key=api_key, base_url=base_url)

def evaluate_conversation(user_msg, assistant_msg, personality):
    """评估单个对话回合"""
    prompt = f"""请评估以下对话中助手回复与用户人格特征的对齐程度。

人格特征: {personality}

用户: {user_msg}
助手: {assistant_msg}

评分标准:
1分 - 完全不符: 回复与人格特征明显矛盾
2分 - 部分符合: 回复偶有体现人格特征但不稳定
3分 - 基本符合: 回复大部分体现人格特征
4分 - 很好符合: 回复很好地体现人格特征

请只返回一个数字(1-4):"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的对话评估专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=1.0,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip()
        # 提取数字
        for char in content:
            if char in '1234':
                return int(char)
        return 2  # 默认
    except Exception as e:
        print(f"API错误: {e}")
        return None

def clean_thinking(text):
    """清理thinking标记"""
    if '' in text or '</think>' in text:
        # 处理 Qwen3 格式
        if '</think>' in text:
            parts = text.split('</think>')
            if len(parts) > 1:
                return parts[-1].strip()
    return text

def evaluate_model(name, input_file, output_file):
    """评估单个模型"""
    if os.path.exists(output_file):
        print(f"跳过 {name}: 结果已存在")
        return None

    print(f"\n评估: {name}")
    with open(input_file) as f:
        conversations = json.load(f)

    results = []
    scores_all = []

    for item in tqdm(conversations, desc=f"评估 {name}"):
        gen_convs = item.get('generated_conversations', [])
        personality = item.get('personality', '')
        user_id = item.get('user_id', '')

        turn_scores = []
        # 评估每个回合
        for conv in gen_convs:
            user_msg = conv.get('user', '')
            assistant_msg = conv.get('assistant', '')

            # 清理thinking标记
            assistant_msg = clean_thinking(assistant_msg)

            if user_msg and assistant_msg:
                score = evaluate_conversation(user_msg, assistant_msg, personality)
                if score:
                    turn_scores.append(score)

        if turn_scores:
            avg_score = sum(turn_scores) / len(turn_scores)
            results.append({
                'user_id': user_id,
                'scores': turn_scores,
                'avg_score': avg_score
            })
            scores_all.extend(turn_scores)

    # 保存结果
    output_data = {
        'detailed_results': results,
        'average_score': sum(scores_all) / len(scores_all) if scores_all else 0,
        'total_turns': len(scores_all)
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    avg = output_data['average_score']
    print(f"平均分数: {avg:.2f}")
    print(f"✓ 完成: {name}")
    return avg

if __name__ == '__main__':
    # 评估剩余的Stage 3模型
    models = [
        ("stage3_gate_reg_0.01_lr1e4", "results/conversations_stage3_gate_reg_0.01_lr1e4_20260405_165459.json", "results/judge_eval_stage3_gate_reg_0.01_lr1e4.json"),
        ("stage3_gate_reg_0.05_lr5e5", "results/conversations_stage3_gate_reg_0.05_lr5e5_20260405_165459.json", "results/judge_eval_stage3_gate_reg_0.05_lr5e5.json"),
    ]

    for name, inp, out in models:
        # 删除之前错误的结果
        if os.path.exists(out):
            os.remove(out)
        evaluate_model(name, inp, out)

    print("\n=== Stage 3 评估完成 ===")