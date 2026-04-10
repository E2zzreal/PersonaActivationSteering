#!/usr/bin/env python3
"""测试各 stage 生成质量，检查重复问题"""
import sys
sys.path.insert(0, '/home/kemove/Desktop/Projects/3-PersonaSteer_V2')

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig

def test_stage(config_path, ckpt_path, stage_name):
    print(f"\n{'='*70}")
    print(f"【{stage_name}】")
    print('='*70)

    cfg = yaml.safe_load(open(config_path))
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model'], trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        cfg['base_model'], trust_remote_code=True, torch_dtype=torch.float16, device_map='cuda:0')

    ps_config = PersonaSteerConfig(
        inject_layers=cfg['model']['inject_layers'],
        v_dim=cfg['model']['v_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        layer_dim=cfg['model']['layer_dim'],
    )

    model = PersonaSteerModel(config=ps_config, backbone=backbone, encoder=backbone.model)
    model._tokenizer = tokenizer

    if ckpt_path != "baseline":
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"✓ 加载检查点: {ckpt_path.split('/')[-1]}")
    else:
        print("✓ 使用基础模型（无干预）")

    model.eval()

    personality = "He is empathetic, creative, and enthusiastic."
    queries = [
        "Hey, how are you?",
        "Tell me about yourself.",
    ]

    with torch.no_grad():
        for i, query in enumerate(queries, 1):
            print(f"\n[Query {i}]: {query}")

            messages = [{"role": "user", "content": query}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(text, return_tensors='pt').to('cuda:0')

            outputs = model.generate(
                input_ids=input_ids,
                personality_text=personality,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
            )

            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            print(f"[Response]: {response}")

            # 检查重复
            words = response.split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                print(f"[重复检测] 唯一词比例: {unique_ratio:.2%}")
                if unique_ratio < 0.5:
                    print("⚠️  检测到高重复率")

if __name__ == "__main__":
    stages = [
        ("configs/train_stage1_qwen3.yaml", "baseline", "Baseline (无干预)"),
        ("configs/train_stage1_qwen3.yaml", "checkpoints/stage1_qwen3/best.pt", "Stage1 (HyperNet)"),
        ("configs/train_stage2_qwen3.yaml", "checkpoints/stage2_qwen3/best.pt", "Stage2 (+ SCL)"),
        ("configs/train_stage3_qwen3.yaml", "checkpoints/stage3_qwen3/best.pt", "Stage3 (+ Gate)"),
    ]

    for config, ckpt, name in stages:
        try:
            test_stage(config, ckpt, name)
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
        finally:
            torch.cuda.empty_cache()
