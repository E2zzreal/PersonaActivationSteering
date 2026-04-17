#!/usr/bin/env python3
"""
PersonaSteer 训练前后效果对比展示
生成对比报告和可视化
"""

import torch
import json
from pathlib import Path
from datetime import datetime

# 示例对话测试用例
TEST_CASES = [
    {
        "personality": {
            "age": "young",
            "gender": "female", 
            "openness": 0.8,
            "conscientiousness": 0.6,
            "extraversion": 0.9,
            "agreeableness": 0.7,
            "neuroticism": 0.3
        },
        "input": "你好，今天天气怎么样？",
        "expected": "活泼、年轻、外向的风格"
    },
    {
        "personality": {
            "age": "old",
            "gender": "male",
            "openness": 0.4,
            "conscientiousness": 0.9,
            "extraversion": 0.3,
            "agreeableness": 0.8,
            "neuroticism": 0.2
        },
        "input": "你好，今天天气怎么样？",
        "expected": "成熟、稳重、内向的风格"
    },
    {
        "personality": {
            "age": "young",
            "gender": "male",
            "openness": 0.9,
            "conscientiousness": 0.4,
            "extraversion": 0.7,
            "agreeableness": 0.5,
            "neuroticism": 0.6
        },
        "input": "你对人工智能有什么看法？",
        "expected": "开放、创新、有深度的回答"
    }
]

def compare_responses(model, tokenizer, test_cases, device="cuda"):
    """对比不同personality的生成效果"""
    results = []
    
    for case in test_cases:
        # 生成回答
        # TODO: 实际推理代码
        
        result = {
            "input": case["input"],
            "personality": case["personality"],
            "expected_style": case["expected"],
            # "response": response,
            # "metrics": metrics
        }
        results.append(result)
    
    return results

def generate_report(results, output_path):
    """生成对比报告"""
    report = f"""
# PersonaSteer V2 效果对比报告

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 测试用例

"""
    
    for i, result in enumerate(results, 1):
        report += f"""
### 案例 {i}

**输入**: {result['input']}

**Personality**: 
- Age: {result['personality']['age']}
- Gender: {result['personality']['gender']}
- Openness: {result['personality']['openness']}
- Extraversion: {result['personality']['extraversion']}

**预期风格**: {result['expected']}

---

"""
    
    return report

if __name__ == "__main__":
    print("效果对比脚本准备就绪")
    print("训练完成后运行:")
    print("  python scripts/compare_results.py --checkpoint checkpoints/stage3_qwen3/best.pt")
