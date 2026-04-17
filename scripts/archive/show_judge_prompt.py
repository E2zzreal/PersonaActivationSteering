"""
PersonaSteer模型生成回复 + LLM Judge评估
输出提示词供用户确认
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_judge_prompt(response: str, personality: str, profile: str = "") -> str:
    """
    构建LLM Judge评估提示词
    
    Args:
        response: 模型生成的回复
        personality: 用户人格描述
        profile: 用户画像（可选）
    
    Returns:
        str: 完整的评估提示词
    """
    prompt = f"""You are an expert evaluator for persona-aligned dialogue systems.

Your task is to evaluate how well the assistant's response aligns with the given personality.

**Evaluation Criteria:**

1. **Personality Consistency** (1-5 points)
   - Does the response reflect the personality traits?
   - Are the language style and tone consistent with the personality?

2. **Emotional Intensity** (1-5 points)
   - Does the response show appropriate emotional intensity?
   - For enthusiastic personality: Should use strong emotional words
   - For calm personality: Should be more measured

3. **Language Style** (1-5 points)
   - Does the vocabulary match the personality?
   - For young personality: May use informal language
   - For professional personality: Should be formal

4. **Context Awareness** (1-5 points)
   - Does the response show understanding of the user's situation?
   - Is the response relevant to the conversation?

**Input Information:**

**Personality**: {personality}

**User Profile**: {profile if profile else "Not provided"}

**Assistant Response**: {response}

**Instructions:**

1. Analyze the response against each criterion
2. Provide a score (1-5) for each criterion
3. Calculate the overall alignment score (average)
4. Output ONLY a single number (1-5) representing the overall alignment

**Output Format:**
- Output ONLY a single number between 1 and 5
- 1 = Completely misaligned
- 2 = Mostly misaligned
- 3 = Neutral
- 4 = Mostly aligned
- 5 = Perfectly aligned

Your evaluation:"""

    return prompt


def build_generation_prompt(user_input: str, personality: str, profile: str = "") -> str:
    """
    构建模型生成回复的提示词
    
    Args:
        user_input: 用户输入
        personality: 用户人格描述
        profile: 用户画像
    
    Returns:
        str: 生成提示词
    """
    prompt = f"""You are an AI assistant with a specific personality.

**Your Personality**: {personality}

**Your Profile**: {profile if profile else "Not provided"}

**Instructions:**
1. Respond naturally as someone with this personality
2. Match your language style to the personality traits
3. Show appropriate emotional intensity
4. Be helpful while staying in character

**User**: {user_input}

**Assistant**: """

    return prompt


def main():
    # 示例数据
    sample_data = {
        "personality": "enthusiastic, full of energy and passion, loves outdoor activities",
        "profile": "34-year-old freelance designer who loves hiking",
        "user_input": "Hey there! How's your day going? I just got back from a hike and I'm feeling pretty energized! Do you enjoy spending time outdoors?",
        "model_response": "Hey! That sounds AMAZING! I totally get the thrill of being outdoors - there's NOTHING quite like the rush of a great hike! What trail did you explore? I'm always looking for new adventures!"
    }
    
    print("="*70)
    print("LLM Judge评估提示词示例")
    print("="*70)
    
    # 1. 生成提示词
    print("\n【1. 模型生成回复提示词】")
    print("-"*70)
    generation_prompt = build_generation_prompt(
        sample_data["user_input"],
        sample_data["personality"],
        sample_data["profile"]
    )
    print(generation_prompt)
    
    # 2. 评估提示词
    print("\n" + "="*70)
    print("【2. LLM Judge评估提示词】")
    print("-"*70)
    judge_prompt = build_judge_prompt(
        sample_data["model_response"],
        sample_data["personality"],
        sample_data["profile"]
    )
    print(judge_prompt)
    
    print("\n" + "="*70)
    print("【3. 评估流程说明】")
    print("-"*70)
    print("""
评估流程:
1. 加载PersonaSteer模型（Stage 1/2/3）
2. 对于每个样本:
   a. 输入: user_input + personality + profile
   b. 模型生成: response (使用PersonaSteer生成)
   c. LLM Judge评估: response vs personality
   d. 记录评分

评分模型:
- GPT-5.2 (如果GPT-5不可用)
- Claude-Sonnet-4.5

输出:
- 各Stage的平均评分
- 评分标准差
- Stage间改善对比
    """)
    
    print("\n" + "="*70)
    print("【4. 确认信息】")
    print("-"*70)
    print(f"""
请确认以下信息是否正确:

1. 评估模型: GPT-5.2, Claude-Sonnet-4.5
2. 评估样本数: 10个样本（可调整）
3. 评估内容: 模型生成的回复（非数据集preferred）
4. 输出格式: 单个评分数字(1-5)

确认后请回复"确认执行"开始评估。
如需修改，请告知需要调整的部分。
    """)


if __name__ == "__main__":
    main()
