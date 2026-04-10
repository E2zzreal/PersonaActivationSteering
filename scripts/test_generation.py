#!/usr/bin/env python3
"""测试生成质量 - 确认是否解决重复循环问题"""

import torch
import sys
sys.path.insert(0, '/home/kemove/Desktop/Projects/3-PersonaSteer_V2')

from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig
from transformers import AutoTokenizer
import json

def test_generation(checkpoint_path, base_model_path, num_samples=5):
    """测试生成质量"""
    print("=" * 60)
    print("生成质量测试")
    print("=" * 60)
    
    # 加载模型
    print(f"\n加载模型: {base_model_path}")
    config = PersonaSteerConfig(
        backbone_model_name=base_model_path,
        encoder_model_name=base_model_path,
        v_dim=1024,
        inject_layers=[10, 11, 12, 13, 14, 15, 16, 17],
    )
    
    model = PersonaSteerModel(config)
    
    # 加载checkpoint
    print(f"加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    device = torch.device('cuda:0')
    model.to(device)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model.hyper_network.set_tokenizer(tokenizer)
    
    # 测试样本
    test_inputs = [
        "Hey there! How's it going?",
        "Hi! I hope you're having a great day.",
        "Hello! What are you up to today?",
        "Hey! Nice to meet you.",
        "Hi there! How have you been lately?",
    ]
    
    personalities = [
        "You are a friendly and enthusiastic person.",
        "You are a calm and thoughtful individual.",
        "You are a creative and imaginative person.",
        "You are a professional and polite individual.",
        "You are a curious and adventurous person.",
    ]
    
    print(f"\n生成{num_samples}个样本...\n")
    
    for i, (user_input, personality) in enumerate(zip(test_inputs[:num_samples], personalities[:num_samples])):
        print(f"样本 {i+1}:")
        print(f"Personality: {personality}")
        print(f"User: {user_input}")
        
        # 编码输入
        input_ids = tokenizer.encode(user_input, return_tensors='pt').to(device)
        
        # 初始化v_prev
        v_prev = torch.zeros(1, config.v_dim).to(device)
        
        # 生成
        with torch.no_grad():
            generated_ids, v_t = model.generate(
                input_ids=input_ids,
                v_prev=v_prev,
                user_texts=[user_input],
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
            )
        
        # 解码
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # 检查重复
        words = generated_text.split()
        if len(words) > 3:
            # 检查是否重复
            last_3 = words[-3:]
            all_same = all(w == last_3[0] for w in last_3)
            if all_same:
                print(f"⚠️ 检测到重复: ... {' '.join(last_3)}")
        
        print(f"Generated: {generated_text}")
        print("-" * 60)
    
    print("\n✅ 测试完成")

if __name__ == "__main__":
    test_generation(
        checkpoint_path="/home/kemove/Desktop/Projects/3-PersonaSteer_V2/checkpoints/stage1/best.pt",
        base_model_path="/home/kemove/.cache/modelscope/Qwen/Qwen2___5-3B",
        num_samples=5,
    )
