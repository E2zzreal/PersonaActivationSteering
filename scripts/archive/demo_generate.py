#!/usr/bin/env python
"""为每个checkpoint生成2条对话样本，用于直观检查模型输出质量"""

import json
import sys
import torch
import yaml
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, AutoTokenizer

# 两个测试样本（不同personality）
SAMPLES = [
    {
        "personality": "He is easy-going, taking life as it comes. He is articulate, able to express himself clearly. He is attentive, always listening to others. He is driven, constantly striving to achieve his goals. He is creative, full of innovative ideas. He is humble, never bragging about his achievements. He is reflective, always thinking deeply about things. He is supportive, always there to lend a helping hand.",
        "user_msg": "Hey there! How's it going? I've been pretty busy lately with some exciting plans. What about you?",
    },
    {
        "personality": "She is visionary, always looking towards the future. She is compassionate, feeling deeply for others. She is articulate, clearly expressing her thoughts. She is practical, focusing on realistic solutions. She is dependable, consistently reliable. She is imaginative, thinking creatively. She is sincere, always genuine. She is reflective, often contemplating her actions.",
        "user_msg": "Hey there! I've just returned from an amazing research trip. You wouldn't believe the underwater life I encountered. Have you ever tried scuba diving?",
    },
]

# 所有要测试的checkpoint
CHECKPOINTS = [
    ("Qwen2.5 Baseline", None, "configs/train_stage1.yaml"),
    ("Qwen2.5 Stage1", "checkpoints/stage1/best.pt", "configs/train_stage1.yaml"),
    ("Qwen2.5 Stage2", "checkpoints/stage2/best.pt", "configs/train_stage2.yaml"),
    ("Qwen2.5 Stage3", "checkpoints/stage3/best.pt", "configs/train_stage3.yaml"),
    ("Qwen3 Baseline", None, "configs/train_stage1_qwen3.yaml"),
    ("Qwen3 Stage1", "checkpoints/stage1_qwen3/best.pt", "configs/train_stage1_qwen3.yaml"),
    ("Qwen3 Stage2", "checkpoints/stage2_qwen3/best.pt", "configs/train_stage2_qwen3.yaml"),
    ("Qwen3 Stage3", "checkpoints/stage3_qwen3/best.pt", "configs/train_stage3_qwen3.yaml"),
]


def load_model(config_path, checkpoint_path, device):
    """正确加载模型（encoder + backbone，与evaluate_fixed.py一致）"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    base_model_path = config.get("base_model")
    model_config = config.get("model", {})

    # 创建PersonaSteer配置
    backbone_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    persona_config = PersonaSteerConfig(
        inject_layers=model_config.get("inject_layers", [10, 11, 12, 13, 14, 15, 16, 17]),
        v_dim=model_config.get("v_dim", 1024),
        hidden_dim=model_config.get("hidden_dim", 4096),
        layer_dim=backbone_config.hidden_size,
        gate_hidden_dim=model_config.get("gate_hidden_dim", 256),
    )

    # 加载encoder (CPU, float16)
    encoder = AutoModel.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # 创建模型（只传encoder）
    model = PersonaSteerModel(config=persona_config, encoder=encoder)

    # 设置v_norm_clip和tokenizer
    if model.hyper_network is not None:
        model.hyper_network.v_norm_clip = model_config.get("v_norm_clip", 10.0)
        model.hyper_network._tokenizer = tokenizer

    # 加载backbone (AutoModelForCausalLM)
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.set_backbone(backbone)

    # 加载checkpoint
    if checkpoint_path:
        print(f"  Loading checkpoint: {checkpoint_path}", flush=True)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)

    # backbone和hyper_network放GPU，encoder留CPU
    model.backbone.to(device)
    if model.hyper_network is not None:
        model.hyper_network.to(device)
    model.injection.to(device)
    model.eval()

    # 【V4修复】Baseline模式（无checkpoint）时禁用注入
    if checkpoint_path is None:
        model.baseline_mode = True

    return model, tokenizer


def generate_response(model, tokenizer, personality, user_msg, device, max_new_tokens=120):
    """生成一条回复"""
    messages = [{"role": "user", "content": user_msg}]

    try:
        result = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(result, 'input_ids'):
            inputs = result.input_ids.to(device)
        else:
            inputs = result.to(device)
    except Exception as e:
        return f"[ERROR] tokenization: {e}"

    try:
        with torch.no_grad():
            v_dim = 1024
            v_prev = torch.zeros(1, v_dim, dtype=torch.float32, device=device)

            outputs, v_t = model.generate(
                input_ids=inputs,
                v_prev=v_prev,
                user_texts=[personality],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            new_tokens = outputs[0][inputs.shape[-1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()
    except Exception as e:
        import traceback
        return f"[ERROR] generate: {e}\n{traceback.format_exc()}"


def main():
    device = "cuda:0"
    results = {}

    for ckpt_name, ckpt_path, config_path in CHECKPOINTS:
        print(f"\n{'='*80}")
        print(f"  {ckpt_name}")
        print(f"{'='*80}", flush=True)

        try:
            # 强制清理GPU
            torch.cuda.empty_cache()
            import gc
            gc.collect()

            model, tokenizer = load_model(config_path, ckpt_path, device)

            ckpt_results = []
            for i, sample in enumerate(SAMPLES):
                response = generate_response(
                    model, tokenizer, sample['personality'],
                    sample['user_msg'], device
                )
                ckpt_results.append(response)
                print(f"\n--- Sample {i+1} ---")
                print(f"Personality: {sample['personality'][:80]}...")
                print(f"User: {sample['user_msg']}")
                print(f"Assistant: {response}")
                print(flush=True)

            results[ckpt_name] = ckpt_results

            # 释放显存
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"  [FAILED] {e}", flush=True)
            torch.cuda.empty_cache()
            import traceback
            traceback.print_exc()

    # 输出对比汇总
    print(f"\n\n{'='*100}")
    print("  八个Checkpoint生成结果对比")
    print(f"{'='*100}")
    
    for ckpt_name, _, _ in CHECKPOINTS:
        if ckpt_name in results:
            print(f"\n{'─'*80}")
            print(f"  【{ckpt_name}】")
            print(f"{'─'*80}")
            for i, resp in enumerate(results[ckpt_name]):
                print(f"  Sample {i+1}: {resp[:200]}{'...' if len(resp)>200 else ''}")
        else:
            print(f"\n{'─'*80}")
            print(f"  【{ckpt_name}】 — FAILED")
            print(f"{'─'*80}")


if __name__ == "__main__":
    main()
