#!/usr/bin/env python
"""从并行对话数据构建 SFT / DPO 训练集。

输入：generate_parallel_dialogues.py 的输出（三源并行多轮对话 JSON）
输出：与 data/split/train.jsonl 格式兼容的 JSONL 文件

用法示例：
  # 1) 用 Claude 回复构建 SFT 数据
  python scripts/build_claude_sft_data.py \
    --input results/parallel_dialogues/dialogues_full.json \
    --output data/claude_sft/train.jsonl \
    --source claude

  # 2) 同时构建 DPO 偏好对（preferred=claude, rejected=qwen3）
  python scripts/build_claude_sft_data.py \
    --input results/parallel_dialogues/dialogues_full.json \
    --output data/claude_sft/train.jsonl \
    --source claude \
    --build_dpo \
    --dpo_rejected qwen3 \
    --dpo_output data/claude_dpo/train.jsonl

  # 3) 与原始 ALOE 数据合并
  python scripts/build_claude_sft_data.py \
    --input results/parallel_dialogues/dialogues_full.json \
    --output data/claude_sft/train.jsonl \
    --source claude \
    --merge_with data/split/train.jsonl \
    --merge_output data/merged_sft/train.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_parallel_dialogues(path: Path) -> list[dict]:
    """加载并行对话 JSON。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("dialogues", data.get("items", []))
    return data


def build_sft_sample(conv: dict, source: str) -> dict | None:
    """将一条并行对话转换为 ALOE SFT 格式。

    Args:
        conv: 并行对话条目
        source: 使用哪个来源的回复 ("claude", "qwen3", "aloe")

    Returns:
        与 data/split/train.jsonl 格式一致的 dict，或 None（跳过无效条目）
    """
    resp_key = f"{source}_response"
    conversations = []

    for turn in conv.get("turns", []):
        user_input = turn.get("user_input", "")
        response = turn.get(resp_key, "")
        if not user_input or not response:
            continue
        conversations.append({"role": "user", "content": user_input})
        conversations.append({"role": "assistant", "content": response})

    if len(conversations) < 2:
        return None

    return {
        "user_id": conv.get("user_id", conv.get("conv_id", "")),
        "profile": conv.get("profile", ""),
        "personality": conv.get("personality", ""),
        "conversations": conversations,
    }


def build_dpo_sample(
    conv: dict, preferred_source: str, rejected_source: str
) -> list[dict]:
    """将一条并行对话转换为 per-turn DPO 偏好对。

    每个对话轮次生成一条 DPO 样本：
    {
        "user_id", "profile", "personality",
        "prompt": 包含历史对话 + 当前用户输入,
        "chosen": preferred 回复,
        "rejected": rejected 回复
    }
    """
    pref_key = f"{preferred_source}_response"
    rej_key = f"{rejected_source}_response"
    samples = []
    history: list[dict] = []

    for turn in conv.get("turns", []):
        user_input = turn.get("user_input", "")
        chosen = turn.get(pref_key, "")
        rejected = turn.get(rej_key, "")

        if not user_input or not chosen or not rejected:
            # 将 chosen 加入 history 继续（如果有的话）
            if user_input and chosen:
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": chosen})
            continue

        # 构建 prompt：历史对话 + 当前用户输入
        prompt_messages = list(history) + [{"role": "user", "content": user_input}]

        samples.append({
            "user_id": conv.get("user_id", ""),
            "profile": conv.get("profile", ""),
            "personality": conv.get("personality", ""),
            "prompt": prompt_messages,
            "chosen": chosen,
            "rejected": rejected,
        })

        # 用 chosen 回复推进历史
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": chosen})

    return samples


def write_jsonl(data: list[dict], path: Path) -> None:
    """写入 JSONL 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_stats(data: list[dict], label: str) -> None:
    """打印数据集统计信息。"""
    n = len(data)
    if n == 0:
        print(f"[{label}] 空数据集")
        return

    n_turns = [len(d.get("conversations", [])) // 2 for d in data]
    personalities = Counter(d.get("personality", "")[:60] for d in data)

    # 计算回复长度
    resp_lens = []
    for d in data:
        convs = d.get("conversations", [])
        for i in range(1, len(convs), 2):
            resp_lens.append(len(convs[i].get("content", "")))

    print(f"[{label}]")
    print(f"  样本数: {n}")
    print(f"  对话轮次: min={min(n_turns)}, max={max(n_turns)}, "
          f"mean={sum(n_turns)/len(n_turns):.1f}")
    if resp_lens:
        avg_words = sum(len(r.split()) for r in []) or 0  # placeholder
        print(f"  回复长度(字符): min={min(resp_lens)}, max={max(resp_lens)}, "
              f"mean={sum(resp_lens)/len(resp_lens):.0f}")
    print(f"  唯一 personality: {len(personalities)}")


def main():
    parser = argparse.ArgumentParser(
        description="从并行对话数据构建 SFT / DPO 训练集"
    )
    parser.add_argument(
        "--input",
        default="results/parallel_dialogues/dialogues_full.json",
        help="并行对话 JSON 文件路径",
    )
    parser.add_argument(
        "--output",
        default="data/claude_sft/train.jsonl",
        help="SFT 输出 JSONL 路径",
    )
    parser.add_argument(
        "--source",
        default="claude",
        choices=["claude", "qwen3", "aloe"],
        help="用哪个来源的回复作为 SFT 标准答案",
    )
    # DPO 相关
    parser.add_argument("--build_dpo", action="store_true", help="同时构建 DPO 偏好对")
    parser.add_argument(
        "--dpo_rejected",
        default="qwen3",
        choices=["qwen3", "aloe"],
        help="DPO rejected 来源",
    )
    parser.add_argument(
        "--dpo_output",
        default="data/claude_dpo/train.jsonl",
        help="DPO 输出路径",
    )
    # 合并
    parser.add_argument("--merge_with", default=None, help="与已有 JSONL 合并")
    parser.add_argument("--merge_output", default=None, help="合并后输出路径")
    # 过滤
    parser.add_argument(
        "--min_turns", type=int, default=1, help="最少对话轮次（低于此数的对话被跳过）"
    )
    parser.add_argument(
        "--min_response_len",
        type=int,
        default=10,
        help="最短回复字符数（过短的轮次被跳过）",
    )
    args = parser.parse_args()

    # 加载并行对话
    dialogues = load_parallel_dialogues(Path(args.input))
    print(f"加载 {len(dialogues)} 条并行对话")

    # 构建 SFT 数据
    sft_data = []
    skipped = 0
    for conv in dialogues:
        sample = build_sft_sample(conv, args.source)
        if sample is None:
            skipped += 1
            continue
        # 过滤
        n_turns = len(sample["conversations"]) // 2
        if n_turns < args.min_turns:
            skipped += 1
            continue
        sft_data.append(sample)

    write_jsonl(sft_data, Path(args.output))
    print(f"\nSFT 数据集 ({args.source}):")
    print(f"  有效: {len(sft_data)}, 跳过: {skipped}")
    print_stats(sft_data, f"SFT-{args.source}")
    print(f"  输出: {args.output}")

    # 构建 DPO 数据
    if args.build_dpo:
        dpo_data = []
        for conv in dialogues:
            pairs = build_dpo_sample(conv, args.source, args.dpo_rejected)
            dpo_data.extend(pairs)

        write_jsonl(dpo_data, Path(args.dpo_output))
        print(f"\nDPO 偏好对 (preferred={args.source}, rejected={args.dpo_rejected}):")
        print(f"  样本数: {len(dpo_data)}")
        print(f"  输出: {args.dpo_output}")

    # 合并
    if args.merge_with and args.merge_output:
        existing = []
        merge_path = Path(args.merge_with)
        if merge_path.exists():
            with open(merge_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing.append(json.loads(line))
        merged = existing + sft_data
        write_jsonl(merged, Path(args.merge_output))
        print(f"\n合并数据集:")
        print(f"  原始: {len(existing)} + 新增: {len(sft_data)} = {len(merged)}")
        print(f"  输出: {args.merge_output}")

    print("\n完成。")


if __name__ == "__main__":
    main()
