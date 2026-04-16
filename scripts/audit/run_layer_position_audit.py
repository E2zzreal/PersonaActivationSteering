#!/usr/bin/env python3
"""B1：注入层位置归因最小对照实验驱动脚本。"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_GROUPS = {
    "mid_8_15": list(range(8, 16)),
    "early_4_11": list(range(4, 12)),
    "late_12_19": list(range(12, 20)),
}


def layers_to_arg(layers: list[int]) -> str:
    return ",".join(str(x) for x in layers)


def build_command(args: argparse.Namespace, group_name: str, layers: list[int], run_dir: Path) -> list[str]:
    command = [
        "python",
        "scripts/v4_eval_qwen3.py",
        "--device", args.device,
        "--judge_model", args.judge_model,
        "--output_dir", str(run_dir),
        "--stages", args.stage,
        "--num_samples", str(args.num_samples),
        "--max_turns_per_sample", str(args.max_turns_per_sample),
        "--stage3_checkpoint", args.checkpoint,
        "--stage3_inject_layers", layers_to_arg(layers),
    ]
    if args.base_model:
        command.extend(["--base_model", args.base_model])
    if args.skip_judge:
        command.append("--skip_judge")
    return command


def summarize_run(run_dir: Path, stage: str, layers: list[int]) -> dict:
    summary_path = run_dir / "eval_summary.json"
    responses_path = run_dir / f"{stage}_responses.json"
    result = {
        "run_dir": str(run_dir),
        "stage": stage,
        "inject_layers": layers,
        "eval_summary_exists": summary_path.exists(),
        "responses_exists": responses_path.exists(),
    }
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        stage_summary = data.get(stage, {})
        result["score"] = stage_summary.get("avg_score")
        result["leak_rate"] = stage_summary.get("thinking_leak_rate")
        result["clean_leak_rate"] = stage_summary.get("clean_thinking_leak_rate")
        result["num_responses"] = stage_summary.get("num_responses")
    return result


def main():
    parser = argparse.ArgumentParser(description="运行 B1 注入层位置归因最小对照实验")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--judge_model", default="GPT-5.2")
    parser.add_argument("--stage", default="stage3", choices=["stage1", "stage2", "stage3"])
    parser.add_argument("--checkpoint", default="checkpoints/stage3_qwen3_v2/best.pt")
    parser.add_argument("--base_model", default="/home/kemove/Desktop/PersonaSteer/Qwen/Qwen3-4B")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--max_turns_per_sample", type=int, default=3)
    parser.add_argument("--output_root", default="results/layer_position_audit")
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument(
        "--groups_json",
        default=None,
        help="可选：JSON 字符串，格式如 '{\"mid_8_15\":[8,9,10,11,12,13,14,15]}'",
    )
    parser.add_argument("--print_only", action="store_true", help="仅打印命令，不实际执行")
    args = parser.parse_args()

    groups = DEFAULT_GROUPS
    if args.groups_json:
        groups = json.loads(args.groups_json)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.output_root) / f"{args.stage}_{timestamp}"
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": timestamp,
        "stage": args.stage,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "judge_model": args.judge_model,
        "groups": groups,
        "runs": [],
    }

    for group_name, layers in groups.items():
        run_dir = root / group_name
        command = build_command(args, group_name, layers, run_dir)
        run_record = {
            "group": group_name,
            "inject_layers": layers,
            "output_dir": str(run_dir),
            "command": command,
        }
        print(f"\n=== {group_name} ===")
        print(" ".join(command))
        if not args.print_only:
            subprocess.run(command, check=True)
            run_record["summary"] = summarize_run(run_dir, args.stage, layers)
        manifest["runs"].append(run_record)
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n已写出：{root / 'manifest.json'}")


if __name__ == "__main__":
    main()
