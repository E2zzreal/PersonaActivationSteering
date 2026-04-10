#!/usr/bin/env python3
"""
批量生成所有模型的对话（修复版）
修复问题：
1. 使用正确的配置文件（匹配训练时使用的模型）
2. 添加系统提示词支持角色扮演
3. 修复baseline配置问题
"""
import argparse
import json
import logging
import sys
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.persona_steer import PersonaSteerModel, PersonaSteerConfig
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, config):
    """加载模型"""
    base_model_path = config["base_model"]
    backbone_cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model_config = config.get("model", {})

    # 获取注入层配置
    inject_layers = model_config.get("inject_layers", [14, 15, 16, 17, 18])
    if isinstance(inject_layers, str):
        inject_layers = eval(inject_layers)

    persona_config = PersonaSteerConfig(
        inject_layers=inject_layers,
        v_dim=model_config.get("v_dim", 1024),
        hidden_dim=model_config.get("hidden_dim", 4096),
        layer_dim=backbone_cfg.hidden_size,
        gate_hidden_dim=model_config.get("gate_hidden_dim", 256),
    )

    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True, use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = PersonaSteerModel(config=persona_config, encoder=backbone.model)
    if hasattr(model, "hyper_network") and model.hyper_network:
        model.hyper_network._tokenizer = tokenizer
    model.set_backbone(backbone)

    if checkpoint_path != "baseline":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model_state = model.state_dict()

        # 加载匹配的参数
        loaded_count = 0
        for key, val in state_dict.items():
            if key in model_state and model_state[key].shape == val.shape:
                model_state[key] = val
                loaded_count += 1

        model.load_state_dict(model_state)
        logger.info(f"Loaded checkpoint: {loaded_count}/{len(state_dict)} parameters matched")
    else:
        logger.info("Using baseline (no checkpoint loaded)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer


def generate_conversation(model, tokenizer, sample, max_new_tokens=150, use_system_prompt=True):
    """生成对话"""
    device = next(model.parameters()).device
    v_t = torch.zeros(1, model.v_dim).to(device)
    conversation = []

    personality = sample.get("personality", "")

    for turn in sample.get("conversations", [])[:8]:
        if turn.get("role") != "user" or len([t for t in conversation if t["role"] == "user"]) >= 4:
            continue
        user_text = turn.get("content", "")

        # 构建消息，添加系统提示词
        messages = []
        if use_system_prompt and personality:
            messages.append({
                "role": "system",
                "content": f"You are role-playing as a person with the following personality: {personality}. Respond naturally as this person would."
            })
        messages.append({"role": "user", "content": user_text})

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs, v_t = model.generate(
                input_ids=input_ids,
                v_prev=v_t,
                personality_texts=[personality],
                user_query_texts=[user_text],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )

        assistant_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        # 清理回复
        assistant_text = assistant_text.strip()
        if assistant_text.startswith("<think>"):
            assistant_text = assistant_text[assistant_text.find("\n"):].strip()

        conversation.append({"role": "user", "content": user_text})
        conversation.append({"role": "assistant", "content": assistant_text})

    return conversation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default="data/processed/eval.jsonl")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_system_prompt", action="store_true", help="禁用系统提示词")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loading model: {args.checkpoint}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Base model: {config.get('base_model', 'N/A')}")

    model, tokenizer = load_model(args.checkpoint, config)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    samples = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    samples = samples[:args.num_samples]
    logger.info(f"Generating {len(samples)} conversations")

    results = []
    for sample in tqdm(samples, desc="Generating"):
        conv = generate_conversation(
            model, tokenizer, sample, args.max_new_tokens,
            use_system_prompt=not args.no_system_prompt
        )
        results.append({
            "user_id": sample.get("user_id", ""),
            "personality": sample.get("personality", ""),
            "profile": sample.get("profile", ""),
            "conversation": conv,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {args.output}")

    # 验证输出
    if results:
        first_conv = results[0].get("conversation", [])
        logger.info(f"首样本对话轮数: {len(first_conv)}")
        if first_conv:
            logger.info(f"首样本首回复: {first_conv[1].get('content', '')[:100]}...")


if __name__ == "__main__":
    main()
