"""
动态批处理器，支持变长对话
"""

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


class PersonaSteerCollator:
    """
    动态批处理器

    负责将变长对话 batch 统一 padding 到相同长度，
    便于批量计算损失。

    Args:
        tokenizer: 分词器对象
        pad_to_multiple_of: padding 到该值的倍数
    """

    def __init__(
        self,
        tokenizer: Any,
        pad_to_multiple_of: int = 8,
        max_turns: int = 10,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_turns = max_turns

    def __call__(self, batch: list[dict]) -> dict:
        """
        对 batch 进行 padding 处理

        Args:
            batch: ALOEDataset 返回的样本列表

        Returns:
            dict: 包含以下键的字典:
                - input_ids: (num_turns, batch, seq_len) padded input ids
                - labels: (num_turns, batch, seq_len) padded labels
                - attention_mask: (num_turns, batch, seq_len) attention mask
                - user_texts: list[list[str]] 每轮用户消息文本
                - user_ids: list[str] 用户ID列表
                - num_turns: list[int] 每样本的对话轮次数
        """
        batch_size = len(batch)

        # 获取每样本的最大轮次数
        max_turns = max(len(sample["turns"]) for sample in batch)
        max_turns = min(max_turns, self.max_turns)  # 使用配置的 max_turns

        # 统计每样本的实际轮次数
        num_turns = [min(len(sample["turns"]), max_turns) for sample in batch]

        # 收集所有 tokenized 数据用于 padding
        all_input_ids = []
        all_labels = []
        all_user_texts = []

        for sample in batch:
            turn_input_ids = []
            turn_labels = []
            turn_user_texts = []

            for turn_idx in range(max_turns):
                if turn_idx < len(sample["turns"]):
                    turn = sample["turns"][turn_idx]
                    turn_input_ids.append(torch.tensor(turn["input_ids"], dtype=torch.long))
                    turn_labels.append(torch.tensor(turn["labels"], dtype=torch.long))
                    turn_user_texts.append(turn["user_text"])
                else:
                    # Padding 用空 token
                    turn_input_ids.append(torch.tensor([self.pad_token_id], dtype=torch.long))
                    turn_labels.append(torch.tensor([-100], dtype=torch.long))
                    turn_user_texts.append("")

            all_input_ids.append(turn_input_ids)
            all_labels.append(turn_labels)
            all_user_texts.append(turn_user_texts)

        # 对每个轮次分别进行 padding
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for turn_idx in range(max_turns):
            # 收集当前轮次的所有序列
            turn_ids = [all_input_ids[b][turn_idx] for b in range(batch_size)]
            turn_lbls = [all_labels[b][turn_idx] for b in range(batch_size)]

            # Padding - input_ids 和 labels 现在长度相同
            padded_turn_ids = pad_sequence(
                turn_ids,
                batch_first=True,
                padding_value=self.pad_token_id,
            )
            padded_turn_lbls = pad_sequence(
                turn_lbls,
                batch_first=True,
                padding_value=-100,
            )

            # Attention mask
            attention_mask = (padded_turn_ids != self.pad_token_id).long()

            padded_input_ids.append(padded_turn_ids)
            padded_labels.append(padded_turn_lbls)
            padded_attention_mask.append(attention_mask)

        # 统一序列长度
        max_seq_len = max(t.size(1) for t in padded_input_ids)

        if self.pad_to_multiple_of > 0:
            max_seq_len = (
                (max_seq_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # 将所有 turn padding 到相同长度
        for i in range(len(padded_input_ids)):
            seq_len = padded_input_ids[i].size(1)
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                padded_input_ids[i] = torch.nn.functional.pad(
                    padded_input_ids[i],
                    (0, pad_len),
                    value=self.pad_token_id,
                )
                padded_labels[i] = torch.nn.functional.pad(
                    padded_labels[i],
                    (0, pad_len),
                    value=-100,
                )
                padded_attention_mask[i] = torch.nn.functional.pad(
                    padded_attention_mask[i],
                    (0, pad_len),
                    value=0,
                )

        return {
            "input_ids": torch.stack(padded_input_ids),  # (num_turns, batch, seq_len)
            "labels": torch.stack(padded_labels),  # (num_turns, batch, seq_len)
            "attention_mask": torch.stack(padded_attention_mask),  # (num_turns, batch, seq_len)
            "user_texts": all_user_texts,  # list[list[str]]
            "user_ids": [sample["user_id"] for sample in batch],  # list[str]
            "personalities": [sample["personality"] for sample in batch],  # list[str] - 【新增】
            "profiles": [sample["profile"] for sample in batch],  # list[str] - 【新增】
            "num_turns": num_turns,  # list[int]
        }
