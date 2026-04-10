"""
正确的评估流程：模型生成回复 + LLM Judge评分

关键差异：
- Stage 1模型生成 → 评估Stage 1的回复
- Stage 2模型生成 → 评估Stage 2的回复
- Stage 3模型生成 → 评估Stage 3的回复

这样才能看出不同Stage的改善
"""

import torch
import json
from transformers import AutoTokenizer

def generate_response(model, tokenizer, user_input, personality, profile, device="cuda:0"):
    """
    使用PersonaSteer模型生成回复
    
    这个函数会：
    1. 输入用户问题 + personality
    2. 模型生成回复
    3. 返回生成的文本
    """
    # 构建输入
    prompt = f"""You are an AI assistant with this personality: {personality}
User Profile: {profile}

User: {user_input}

Assistant:"""
    
    # 编码
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取Assistant回复部分
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    return response


def correct_evaluation_flow():
    """
    正确的评估流程
    """
    print("正确的评估流程：")
    print()
    print("1. 加载Stage 1模型")
    print("   → 对每个样本，让模型生成回复")
    print("   → LLM Judge评分")
    print("   → 记录分数: S1_score")
    print()
    print("2. 加载Stage 2模型")
    print("   → 对每个样本，让模型生成回复")
    print("   → LLM Judge评分")
    print("   → 记录分数: S2_score")
    print()
    print("3. 加载Stage 3模型")
    print("   → 对每个样本，让模型生成回复")
    print("   → LLM Judge评分")
    print("   → 记录分数: S3_score")
    print()
    print("4. 对比：S1 vs S2 vs S3")
    print("   → 这时才能看出不同Stage的改善！")
    print()
    print("=" * 60)
    print("为什么之前没有差异？")
    print("=" * 60)
    print()
    print("之前评估的是：")
    print("  Stage 1: preferred回复 → 评分")
    print("  Stage 2: preferred回复 → 评分  ← 同样的文本！")
    print("  Stage 3: preferred回复 → 评分  ← 同样的文本！")
    print()
    print("应该评估的是：")
    print("  Stage 1: Stage1模型生成的回复 → 评分")
    print("  Stage 2: Stage2模型生成的回复 → 评分")
    print("  Stage 3: Stage3模型生成的回复 → 评分")


if __name__ == "__main__":
    correct_evaluation_flow()
