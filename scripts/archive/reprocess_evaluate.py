#!/usr/bin/env python3
"""
后处理重新评估脚本
对已生成的回复进行思考过程过滤，然后重新评分
"""

import json
import re
import os
import yaml
from pathlib import Path
from datetime import datetime
from openai import OpenAI


def filter_thinking_process(response: str) -> str:
    """移除思考过程，保留实际回复"""
    if not response:
        return response

    original = response

    # 模式1: 移除开头的思考过程段落（直到第一个实际回复）
    thinking_start_patterns = [
        # English patterns
        (r'^Okay,.*?(?=\n\n|\n[A-Z])', re.DOTALL | re.IGNORECASE),
        (r'^Let me (think|start|first|analyze)[^。\.]*[。\.]\s*', re.IGNORECASE),
        (r'^I (need to|should|will)[^。\.]*[。\.]\s*', re.IGNORECASE),
        (r'^First,?\s*I[^。\.]*[。\.]\s*', re.IGNORECASE),
        (r'^Next,?\s*I[^。\.]*[。\.]\s*', re.IGNORECASE),

        # Chinese patterns
        (r'^首先，[^。]*[。]\s*', 0),
        (r'^接下来，[^。]*[。]\s*', 0),
        (r'^然后，[^。]*[。]\s*', 0),
        (r'^嗯，用户[^。]*[。]\s*', 0),
        (r'^好的，用户[^。]*[。]\s*', 0),
        (r'^我需要[^。]*[。]\s*', 0),
    ]

    filtered = response
    for pattern, flags in thinking_start_patterns:
        filtered = re.sub(pattern, '', filtered, flags=flags)

    # 模式2: 移除整段思考过程（多行）
    thinking_block_patterns = [
        # "Okay, the user... I need to..." 整段
        r'Okay, the user is[^.]*\.[^.]*\.[^.]*\.',
        r'Okay, the user[^.]*\.[^.]*\.',
        # 中文思考段落
        r'首先，我需要[^。]*[。][^。]*[。]',
        r'用户可能[^。]*[。][^。]*[。]',
    ]

    for pattern in thinking_block_patterns:
        filtered = re.sub(pattern, '', filtered, flags=re.DOTALL)

    # 模式3: 移除"用户:"之后的内容（这是模型继续生成对话的问题）
    filtered = re.sub(r'\n*用户:.*$', '', filtered, flags=re.DOTALL)
    filtered = re.sub(r'\n*User:.*$', '', filtered, flags=re.DOTALL)

    # 清理多余空白
    filtered = filtered.strip()

    # 如果过滤后太短，尝试提取最后有意义的句子
    if len(filtered) < 15 and len(original) > 30:
        # 尝试找到第一个完整的句子
        sentences = re.split(r'(?<=[。\.!?！？])\s*', original)
        for sent in reversed(sentences):
            sent = sent.strip()
            # 跳过明显的思考过程句子
            if len(sent) > 15 and not any(
                keyword in sent.lower()
                for keyword in ['okay, the user', 'i need to', '首先', '接下来', '嗯，用户', '好的，用户']
            ):
                filtered = sent
                break

    return filtered if filtered else original


def load_api_config():
    """加载API配置"""
    config_path = Path(__file__).parent.parent / "configs" / "api_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    default_api = config.get('default', 'blsc')
    api_config = config[default_api]

    return api_config


def llm_judge_score(client: OpenAI, model: str, personality: str, response: str) -> float:
    """使用LLM Judge对回复进行人格一致性评分"""

    prompt = f"""你是一个专业的人格一致性评估专家。

请评估以下回复是否符合给定的人格特质描述。

人格特质描述：
{personality}

回复内容：
{response}

评分标准（1-5分）：
1分 - 完全不符合人格特质，回复与描述的人格特征严重矛盾
2分 - 大部分不符合，存在明显的性格不一致
3分 - 部分符合，有一些人格特征体现但不明显
4分 - 大部分符合，较好地体现了人格特质
5分 - 完全符合，回复很好地体现了描述的人格特征

请只输出一个数字（1-5），不要输出其他内容。"""

    try:
        result = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=10,
        )

        score_text = result.choices[0].message.content.strip()
        # 提取数字
        match = re.search(r'[1-5]', score_text)
        if match:
            return float(match.group())
        return 3.0
    except Exception as e:
        print(f"  API调用错误: {e}")
        return 3.0


def main():
    # 加载API配置
    api_config = load_api_config()
    client = OpenAI(
        api_key=api_config['api_key'],
        base_url=api_config['base_url'],
    )
    model = api_config['model']

    # 原始结果目录
    original_dir = Path(__file__).parent.parent / "results" / "eval_20260411_163906"

    # 新结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = Path(__file__).parent.parent / "results" / f"reprocess_{timestamp}"
    new_dir.mkdir(parents=True, exist_ok=True)

    print(f"原始结果目录: {original_dir}")
    print(f"新结果目录: {new_dir}")
    print(f"使用API: {model}")
    print("=" * 60)

    # 处理所有response文件
    summary = {}

    for json_file in sorted(original_dir.glob("*_responses.json")):
        config_name = json_file.stem.replace("_responses", "")
        print(f"\n处理: {config_name}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        new_data = []
        scores = []

        for i, item in enumerate(data):
            original_response = item['response']
            filtered_response = filter_thinking_process(original_response)

            # 重新评分
            print(f"  样本 {i+1}/{len(data)}: ", end="", flush=True)
            new_score = llm_judge_score(
                client, model,
                item['personality'],
                filtered_response
            )
            print(f"原分={item['score']:.1f} -> 新分={new_score:.1f}")

            new_item = {
                'personality': item['personality'],
                'user_input': item['user_input'],
                'original_response': original_response,
                'filtered_response': filtered_response,
                'original_score': item['score'],
                'new_score': new_score,
                'response_changed': original_response != filtered_response,
            }
            new_data.append(new_item)
            scores.append(new_score)

        # 保存新结果
        output_file = new_dir / f"{config_name}_reprocessed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

        # 汇总统计
        summary[config_name] = {
            'count': len(scores),
            'avg_original': sum(d['original_score'] for d in new_data) / len(new_data),
            'avg_new': sum(scores) / len(scores),
            'responses_changed': sum(1 for d in new_data if d['response_changed']),
        }
        print(f"  平均分: {summary[config_name]['avg_original']:.2f} -> {summary[config_name]['avg_new']:.2f}")
        print(f"  修改了 {summary[config_name]['responses_changed']}/{len(new_data)} 个回复")

    # 保存汇总
    summary_file = new_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"结果保存在: {new_dir}")

    # 打印对比表
    print("\n评分对比汇总:")
    print("-" * 60)
    print(f"{'配置':<25} {'原平均分':>10} {'新平均分':>10} {'变化':>10}")
    print("-" * 60)
    for name, stats in sorted(summary.items()):
        diff = stats['avg_new'] - stats['avg_original']
        print(f"{name:<25} {stats['avg_original']:>10.2f} {stats['avg_new']:>10.2f} {diff:>+10.2f}")


if __name__ == "__main__":
    main()