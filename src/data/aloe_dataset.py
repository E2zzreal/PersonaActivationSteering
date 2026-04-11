"""
ALOE 数据集的 PyTorch Dataset
支持多轮对话加载，最大轮次限制
"""

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class ALOEDataset(Dataset):
    """
    ALOE 对话数据集加载器

    负责从预处理后的 JSONL 文件加载多轮对话数据，
    并进行 tokenization。每个样本包含用户画像和多轮对话。

    重要：对于 Qwen3 等支持 thinking 模式的模型，本类会自动禁用
    thinking 标记（enable_thinking=False），确保训练数据格式与
    推理时一致，避免模型学习输出思考过程。

    Args:
        data_path: JSONL 数据文件路径
        tokenizer: 分词器对象（支持 enable_thinking 参数）
        max_turns: 最大对话轮次数
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Any,
        max_turns: int = 10,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.data = self._load_data()

    def _load_data(self) -> list[dict]:
        """加载 JSONL 格式数据"""
        samples = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        获取单个样本

        Returns:
            dict: 包含以下键的字典:
                - user_id: 用户ID
                - profile: 用户画像文本
                - personality: 人格描述
                - turns: 对话轮次列表，每个元素包含:
                    - user_text: 用户消息文本
                    - input_ids: 完整序列 token ids (user + assistant)
                    - labels: 标签序列，user 部分为 -100
        """
        sample = self.data[idx]
        turns = []

        # 展开对话轮次 (user-assistant 交替)
        conversations = sample.get("conversations", [])
        for i in range(0, len(conversations) - 1, 2):
            if i // 2 >= self.max_turns:
                break

            user_msg = conversations[i]
            asst_msg = conversations[i + 1]

            if "content" not in user_msg or "content" not in asst_msg:
                continue

            # 验证 role 字段，防止 system 消息或角色错位导致 labels 构建错误
            if user_msg.get("role") != "user" or asst_msg.get("role") != "assistant":
                continue

            user_content = user_msg["content"]
            asst_content = asst_msg["content"]

            # 使用 chat template 构建完整序列
            # 【修复】添加 enable_thinking=False 以匹配推理时的行为
            # 这确保训练数据不包含 thinking 标记，避免模型学习输出思考过程
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": asst_content}
            ]

            try:
                full_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=False
                )
            except TypeError:
                # 非 Qwen3 tokenizer 不支持 enable_thinking 参数
                full_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )

            # Tokenize 完整序列
            input_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

            # 计算 user 部分长度（用于构建 labels）
            try:
                user_only = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                user_only = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False, add_generation_prompt=True
                )
            user_ids = self.tokenizer.encode(user_only, add_special_tokens=False)

            # labels: user 部分为 -100，assistant 部分为真实 token
            labels = [-100] * len(user_ids) + input_ids[len(user_ids):]

            turns.append({
                "user_text": user_content,
                "input_ids": input_ids,
                "labels": labels,
            })

        return {
            "user_id": sample.get("user_id", f"u{idx}"),
            "profile": sample.get("profile", ""),
            "personality": sample.get("personality", ""),
            "turns": turns,
        }
