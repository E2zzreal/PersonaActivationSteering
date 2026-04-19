#!/usr/bin/env python
"""PersonaSteer DPO 训练脚本

只训练 HyperNetwork，backbone 保持冻结。
DPO loss = -log σ(β * (log π(y_w|x) - log π_ref(y_w|x)) - β * (log π(y_l|x) - log π_ref(y_l|x)))

用法:
  python scripts/train_dpo.py \
    --config configs/train_dpo_claude.yaml \
    --sft_checkpoint checkpoints/stage1_claude_sft/best.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.persona_steer import PersonaSteerConfig, PersonaSteerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Dataset ────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        logger.info(f"Loaded {len(self.data)} DPO samples from {data_path}")

    def _encode_sample(self, personality: str, profile: str,
                       prompt: list[dict], response: str) -> dict:
        """将 prompt + response 编码为 input_ids 和 labels（prompt 部分 mask 为 -100）"""
        messages = [
            {"role": "system",
             "content": f"/no_think\n你的人格特征是：{personality}\n\n你的个人简介：{profile}"}
        ]
        messages.extend(prompt)

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        full_text = prompt_text + response + self.tokenizer.eos_token
        input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

        # 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        personality = item["personality"]
        profile = item.get("profile", "")
        prompt = item["prompt"]

        chosen = self._encode_sample(personality, profile, prompt, item["chosen"])
        rejected = self._encode_sample(personality, profile, prompt, item["rejected"])

        return {
            "personality": personality,
            "profile": profile,
            "chosen_input_ids": chosen["input_ids"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_labels": rejected["labels"],
        }


def collate_fn(batch, pad_id: int):
    def pad(seqs):
        max_len = max(len(s) for s in seqs)
        return torch.tensor(
            [s + [pad_id] * (max_len - len(s)) for s in seqs], dtype=torch.long)

    return {
        "personalities": [b["personality"] for b in batch],
        "profiles": [b["profile"] for b in batch],
        "chosen_input_ids": pad([b["chosen_input_ids"] for b in batch]),
        "chosen_labels": pad([b["chosen_labels"] for b in batch]),
        "rejected_input_ids": pad([b["rejected_input_ids"] for b in batch]),
        "rejected_labels": pad([b["rejected_labels"] for b in batch]),
    }


# ── Log prob computation ───────────────────────────────────────────────

def compute_logprobs(model: PersonaSteerModel, input_ids: torch.Tensor,
                     labels: torch.Tensor, personalities: list[str],
                     profiles: list[str]) -> torch.Tensor:
    """计算每个样本的 sum log p(y|x)，仅对 labels != -100 的 token 求和。

    使用 PersonaSteerModel.forward() 保持完整计算图（梯度流回 HyperNetwork）。
    user_query_texts 用 personality 代替空字符串，避免 Qwen3 seq_len=0。
    """
    device = input_ids.device
    batch_size = input_ids.size(0)
    v_prev = torch.zeros(batch_size, model.v_dim, device=device)

    logits, _, _ = model(
        input_ids=input_ids,
        v_prev=v_prev,
        personality_texts=personalities,
        user_query_texts=personalities,  # 不能传空字符串，Qwen3 tokenizer 会编码为 0 token
    )  # (batch, seq, vocab)

    # shift: logits[i] predicts token[i+1]
    shift_logits = logits[:, :-1, :]   # (batch, seq-1, vocab)
    shift_labels = labels[:, 1:]        # (batch, seq-1)

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)  # (batch, seq-1, vocab)
    token_log_probs = log_probs.gather(
        2, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # (batch, seq-1)

    mask = (shift_labels != -100).float()
    sum_log_probs = (token_log_probs * mask).sum(dim=-1)  # (batch,)
    return sum_log_probs


# ── Model loading ──────────────────────────────────────────────────────

def load_model(sft_checkpoint: str, base_model_path: str, device: str,
               inject_layers: list[int]) -> tuple[PersonaSteerModel, AutoTokenizer]:
    logger.info(f"Loading backbone from {base_model_path}")
    dev_id = int(device.split(":")[1]) if ":" in device else 0
    backbone_cfg = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map={"": dev_id}, use_cache=False,
        attn_implementation="eager",  # 禁用 flash_attention varlen 模式，避免 seq_len=0
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    config = PersonaSteerConfig(
        inject_layers=inject_layers,
        v_dim=1024, hidden_dim=4096,
        layer_dim=backbone_cfg.hidden_size,
        gate_hidden_dim=256, gate_init_bias=-2.0, gate_max=1.0,
    )
    model = PersonaSteerModel(config=config, backbone=backbone, encoder=backbone.model)
    model.hyper_network._tokenizer = tokenizer

    # Load SFT weights
    dev_obj = torch.device(device)
    ckpt = torch.load(sft_checkpoint, map_location="cpu", weights_only=False)
    sd = model.state_dict()
    loaded = 0
    for k, v in ckpt.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v.to(sd[k].dtype)
            loaded += 1
    model.load_state_dict(sd, strict=False)

    # Move HyperNetwork + injection to device
    for p in model.hyper_network.parameters():
        if p.device != dev_obj:
            p.data = p.data.to(dev_obj)
    model.injection.to(dev_obj)

    logger.info(f"Loaded {loaded} tensors from {sft_checkpoint}")
    return model, tokenizer


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft_checkpoint", required=True,
                        help="Stage1 SFT checkpoint (用作初始模型和参考模型)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = args.device or cfg.get("device", "cuda:0")

    inject_layers = cfg.get("model", {}).get(
        "inject_layers", [4, 5, 6, 7, 8, 9, 10, 11])
    beta = cfg.get("training", {}).get("beta", 0.1)
    lr = cfg.get("training", {}).get("learning_rate", 5e-5)
    num_epochs = cfg.get("training", {}).get("num_epochs", 3)
    batch_size = cfg.get("data", {}).get("batch_size", 2)
    max_length = cfg.get("data", {}).get("max_length", 512)
    output_dir = Path(cfg.get("training", {}).get("output_dir", "checkpoints/dpo"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_interval = cfg.get("training", {}).get("log_interval", 20)
    base_model = cfg.get("base_model", "/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B")

    # Load model (trainable) + reference model (frozen copy of HyperNetwork)
    logger.info("Loading training model...")
    model, tokenizer = load_model(args.sft_checkpoint, base_model, device, inject_layers)

    logger.info("Creating reference model (frozen HyperNetwork copy)...")
    ref_hyper = copy.deepcopy(model.hyper_network)
    ref_injection = copy.deepcopy(model.injection)
    for p in ref_hyper.parameters():
        p.requires_grad_(False)
    for p in ref_injection.parameters():
        p.requires_grad_(False)

    # Only HyperNetwork params are trainable
    trainable = [p for p in model.hyper_network.parameters() if p.requires_grad]
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    # Dataset
    data_path = cfg.get("data", {}).get("train_path", "data/claude_dpo/train.jsonl")
    dataset = DPODataset(data_path, tokenizer, max_length=max_length)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, pad_id), num_workers=2)

    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.hyper_network.train()
        model.injection.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            dev = torch.device(device)
            chosen_ids = batch["chosen_input_ids"].to(dev)
            chosen_lbl = batch["chosen_labels"].to(dev)
            rejected_ids = batch["rejected_input_ids"].to(dev)
            rejected_lbl = batch["rejected_labels"].to(dev)
            personalities = batch["personalities"]
            profiles = batch["profiles"]

            with torch.amp.autocast('cuda'):
                # Model log probs
                lp_chosen = compute_logprobs(
                    model, chosen_ids, chosen_lbl, personalities, profiles)
                lp_rejected = compute_logprobs(
                    model, rejected_ids, rejected_lbl, personalities, profiles)

                # Reference log probs (swap HyperNetwork temporarily)
                model.hyper_network, ref_hyper = ref_hyper, model.hyper_network
                model.injection, ref_injection = ref_injection, model.injection
                with torch.no_grad():
                    ref_lp_chosen = compute_logprobs(
                        model, chosen_ids, chosen_lbl, personalities, profiles)
                    ref_lp_rejected = compute_logprobs(
                        model, rejected_ids, rejected_lbl, personalities, profiles)
                model.hyper_network, ref_hyper = ref_hyper, model.hyper_network
                model.injection, ref_injection = ref_injection, model.injection

                # DPO loss
                chosen_rewards = beta * (lp_chosen - ref_lp_chosen)
                rejected_rewards = beta * (lp_rejected - ref_lp_rejected)
                loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            # Reward margin for monitoring
            margin = (chosen_rewards - rejected_rewards).mean().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", margin=f"{margin:.3f}")

            if global_step % log_interval == 0:
                logger.info(
                    f"[step {global_step}] loss={loss.item():.4f} "
                    f"margin={margin:.3f} "
                    f"chosen_r={chosen_rewards.mean().item():.3f} "
                    f"rejected_r={rejected_rewards.mean().item():.3f}")

        avg_loss = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        ckpt_path = output_dir / f"epoch_{epoch}.pt"
        # Save only HyperNetwork + injection weights
        save_dict = {k: v for k, v in model.state_dict().items()
                     if "hyper_network" in k or "injection" in k}
        torch.save(save_dict, ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(save_dict, output_dir / "best.pt")
            logger.info(f"  → best model saved (loss={best_loss:.4f})")

    logger.info(f"DPO training done. Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
