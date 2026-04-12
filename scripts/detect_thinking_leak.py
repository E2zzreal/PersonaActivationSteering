#!/usr/bin/env python
"""
检测评估输出中的思考过程泄露。

用途：
1. 扫描单个 responses.json 或目录下的多个 responses.json
2. 统计每个文件的总样本数、泄露样本数、泄露率
3. 输出 JSON 汇总，供 baseline 门槛判定使用
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LEAK_PATTERNS = [
    re.compile(r"\bOkay,?\b", re.IGNORECASE),
    re.compile(r"\bI need to\b", re.IGNORECASE),
    re.compile(r"\bthe user is\b", re.IGNORECASE),
    re.compile(r"\bI should\b", re.IGNORECASE),
    re.compile(r"\blet me think\b", re.IGNORECASE),
    re.compile(r"\bmy response should\b", re.IGNORECASE),
]

TEXT_KEYS = ["response", "assistant_response", "output", "text", "content"]


def extract_text(item: dict) -> str:
    for key in TEXT_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(item.get("conversation"), list):
        texts = []
        for msg in item["conversation"]:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    texts.append(content)
        return "\n".join(texts)
    return ""


def detect_leak(text: str) -> list[str]:
    matches = []
    for pattern in LEAK_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches


def analyze_file(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if isinstance(data.get("responses"), list):
            items = data["responses"]
        else:
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        items = []

    leak_count = 0
    records = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        text = extract_text(item)
        matched = detect_leak(text)
        if matched:
            leak_count += 1
        records.append(
            {
                "index": index,
                "has_leak": bool(matched),
                "matched_patterns": matched,
                "preview": text[:160],
            }
        )

    total = len(records)
    leak_rate = (leak_count / total) if total else 0.0
    return {
        "file": str(path),
        "total": total,
        "leak_count": leak_count,
        "leak_rate": round(leak_rate, 4),
        "records": records,
    }


def collect_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(target.rglob("*responses.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="检测评估输出中的思考泄露")
    parser.add_argument("target", type=str, help="responses.json 文件或结果目录")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    target = Path(args.target)
    files = collect_files(target)
    summary = {"target": str(target), "files": [], "overall_total": 0, "overall_leak_count": 0, "overall_leak_rate": 0.0}

    for file in files:
        result = analyze_file(file)
        summary["files"].append(result)
        summary["overall_total"] += result["total"]
        summary["overall_leak_count"] += result["leak_count"]

    if summary["overall_total"]:
        summary["overall_leak_rate"] = round(summary["overall_leak_count"] / summary["overall_total"], 4)

    output = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
