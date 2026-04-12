#!/usr/bin/env python
"""
基于评分和思考泄露率执行 baseline 门槛判定。

输入：
- summary.json（包含各配置 mean_score）
- thinking_leak_summary.json（包含各 responses 文件泄露率）

输出：
- 每个候选配置相对于 baseline 的 pass/fail 判定
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_leak_map(leak_summary: dict) -> dict:
    mapping = {}
    for item in leak_summary.get("files", []):
        file_path = Path(item["file"])
        stem = file_path.stem.replace("_responses", "")
        mapping[stem] = {
            "file": str(file_path),
            "leak_rate": item.get("leak_rate", 0.0),
            "leak_count": item.get("leak_count", 0),
            "total": item.get("total", 0),
        }
    return mapping


def find_baseline_reference(summary: dict, preferred: str | None):
    if preferred and preferred in summary:
        return preferred, summary[preferred]
    candidates = [key for key in summary if key.startswith("baseline_")]
    if not candidates:
        raise ValueError("未在 summary 中找到 baseline 项")
    best_key = max(candidates, key=lambda key: summary[key].get("mean_score", float("-inf")))
    return best_key, summary[best_key]


def main() -> None:
    parser = argparse.ArgumentParser(description="执行 baseline 门槛判定")
    parser.add_argument("--summary", required=True, help="summary.json 路径")
    parser.add_argument("--leak", required=True, help="thinking_leak_summary.json 路径")
    parser.add_argument("--baseline", default=None, help="指定 baseline key，例如 baseline_stage1")
    parser.add_argument("--max_leak_multiplier", type=float, default=1.5, help="允许相对 baseline 的最大泄露倍数")
    parser.add_argument("--min_score_margin", type=float, default=0.0, help="需要超过 baseline 的最小分数差")
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    summary = load_json(Path(args.summary))
    leak_summary = load_json(Path(args.leak))
    leak_map = normalize_leak_map(leak_summary)

    baseline_key, baseline_info = find_baseline_reference(summary, args.baseline)
    baseline_score = baseline_info.get("mean_score", 0.0)
    baseline_leak = leak_map.get(baseline_key, {}).get("leak_rate", 0.0)
    leak_threshold = baseline_leak * args.max_leak_multiplier

    results = {
        "baseline": {
            "key": baseline_key,
            "mean_score": baseline_score,
            "leak_rate": baseline_leak,
            "leak_threshold": round(leak_threshold, 4),
            "max_leak_multiplier": args.max_leak_multiplier,
            "min_score_margin": args.min_score_margin,
        },
        "candidates": {},
    }

    for key, info in summary.items():
        if key == baseline_key:
            continue
        score = info.get("mean_score", 0.0)
        leak_rate = leak_map.get(key, {}).get("leak_rate", None)

        reasons = []
        status = "pass"

        if score < baseline_score + args.min_score_margin:
            status = "fail_score"
            reasons.append(
                f"score {score:.3f} < baseline {baseline_score:.3f} + margin {args.min_score_margin:.3f}"
            )

        if leak_rate is None:
            if status == "pass":
                status = "fail_missing_leak"
            reasons.append("missing leak statistics")
        elif leak_rate > leak_threshold:
            if status == "pass":
                status = "fail_leak"
            else:
                status = f"{status}+fail_leak"
            reasons.append(
                f"leak_rate {leak_rate:.4f} > threshold {leak_threshold:.4f}"
            )

        results["candidates"][key] = {
            "status": status,
            "mean_score": score,
            "score_delta_vs_baseline": round(score - baseline_score, 4),
            "leak_rate": leak_rate,
            "reasons": reasons,
        }

    output = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
