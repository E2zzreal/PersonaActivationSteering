#!/usr/bin/env python
"""
Judge 对比分析脚本
汇总三个版本（V1严格评分/V2 A/B对比/V3多维度）的评估结果，生成对比报告
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyze_v1(data):
    """V1: 严格评分标准"""
    if "error" in data:
        return {"method": "v1_strict_rubric", "error": data["error"]}

    dist = data.get("score_distribution", {})
    total = sum(dist.values())

    return {
        "method": "v1_strict_rubric",
        "checkpoint": data.get("checkpoint", ""),
        "num_samples": data.get("num_samples", 0),
        "al_k_avg": data.get("al_k_avg", 0.0),
        "al_k_std": data.get("al_k_std", 0.0),
        "score_distribution": dist,
        "low_score_rate": round((dist.get("1", 0) + dist.get("2", 0)) / total, 4) if total > 0 else 0,
        "high_score_rate": round((dist.get("4", 0) + dist.get("5", 0)) / total, 4) if total > 0 else 0,
    }


def analyze_v2(data):
    """V2: A/B 对比"""
    if "error" in data:
        return {"method": "v2_ab_comparison", "error": data["error"]}

    total = data.get("num_samples", 0)
    win_rate = data.get("injected_win_rate", 0.0)

    return {
        "method": "v2_ab_comparison",
        "checkpoint_injected": data.get("checkpoint_injected", ""),
        "checkpoint_baseline": data.get("checkpoint_baseline", ""),
        "num_samples": total,
        "wins_injected": data.get("wins_injected", 0),
        "wins_baseline": data.get("wins_baseline", 0),
        "ties": data.get("ties", 0),
        "injected_win_rate": win_rate,
        "baseline_win_rate": data.get("baseline_win_rate", 0.0),
        "tie_rate": data.get("tie_rate", 0.0),
    }


def analyze_v3(data):
    """V3: 多维度评分"""
    if "error" in data:
        return {"method": "v3_multidimensional", "error": data["error"]}

    return {
        "method": "v3_multidimensional",
        "checkpoint": data.get("checkpoint", ""),
        "num_samples": data.get("num_samples", 0),
        "style_mean": data.get("style", {}).get("mean", 0.0),
        "style_std": data.get("style", {}).get("std", 0.0),
        "content_mean": data.get("content", {}).get("mean", 0.0),
        "content_std": data.get("content", {}).get("std", 0.0),
        "consistency_mean": data.get("consistency", {}).get("mean", 0.0),
        "consistency_std": data.get("consistency", {}).get("std", 0.0),
        "weighted_mean": data.get("weighted", {}).get("mean", 0.0),
        "weighted_std": data.get("weighted", {}).get("std", 0.0),
    }


def compare_methods(v1_result, v2_result, v3_result):
    """生成跨方法对比分析"""
    analysis = []

    # V1 分析：区分度
    if "error" not in v1_result:
        low_rate = v1_result.get("low_score_rate", 0)
        high_rate = v1_result.get("high_score_rate", 0)
        std = v1_result.get("al_k_std", 0)

        if std > 0.8:
            analysis.append(f"✓ V1严格评分：标准差={std:.3f}，区分度良好")
        else:
            analysis.append(f"✗ V1严格评分：标准差={std:.3f}，区分度仍然不足")

        if low_rate > 0.2:
            analysis.append(f"✓ V1严格评分：低分段(1-2分)占比={low_rate:.1%}，成功压制了虚高")
        else:
            analysis.append(f"✗ V1严格评分：低分段占比={low_rate:.1%}，可能仍存在评分虚高")

    # V2 分析：胜率
    if "error" not in v2_result:
        win_rate = v2_result.get("injected_win_rate", 0)
        tie_rate = v2_result.get("tie_rate", 0)

        if win_rate > 0.6:
            analysis.append(f"✓ V2 A/B对比：注入模型胜率={win_rate:.1%}，显著优于baseline")
        elif win_rate > 0.5:
            analysis.append(f"△ V2 A/B对比：注入模型胜率={win_rate:.1%}，略优于baseline")
        else:
            analysis.append(f"✗ V2 A/B对比：注入模型胜率={win_rate:.1%}，未超过baseline")

        if tie_rate > 0.5:
            analysis.append(f"△ V2 A/B对比：平局率={tie_rate:.1%}，judge难以区分差异")

    # V3 分析：维度分解
    if "error" not in v3_result:
        style = v3_result.get("style_mean", 0)
        content = v3_result.get("content_mean", 0)
        consist = v3_result.get("consistency_mean", 0)

        max_dim = max(style, content, consist)
        min_dim = min(style, content, consist)

        dims = {"风格(style)": style, "内容(content)": content, "一致性(consistency)": consist}
        best_dim = max(dims, key=dims.get)
        worst_dim = min(dims, key=dims.get)

        analysis.append(f"V3多维度：最强维度={best_dim}({max_dim:.3f})，最弱维度={worst_dim}({min_dim:.3f})")

        if max_dim - min_dim > 0.5:
            analysis.append(f"✓ V3多维度：维度差异明显，可针对性优化")
        else:
            analysis.append(f"△ V3多维度：维度差异较小，注入效果均匀")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Compare three judge evaluation results")
    parser.add_argument("--v1", type=str, default="results/judge_v1_strict.json")
    parser.add_argument("--v2", type=str, default="results/judge_v2_ab.json")
    parser.add_argument("--v3", type=str, default="results/judge_v3_multidim.json")
    parser.add_argument("--output", type=str, default="results/judge_comparison_report.json")
    args = parser.parse_args()

    # 加载结果
    v1_data, v2_data, v3_data = {}, {}, {}

    if Path(args.v1).exists():
        v1_data = load_json(args.v1)
        print(f"Loaded V1: {args.v1}")
    else:
        v1_data = {"error": f"File not found: {args.v1}"}
        print(f"[WARNING] V1 result not found: {args.v1}")

    if Path(args.v2).exists():
        v2_data = load_json(args.v2)
        print(f"Loaded V2: {args.v2}")
    else:
        v2_data = {"error": f"File not found: {args.v2}"}
        print(f"[WARNING] V2 result not found: {args.v2}")

    if Path(args.v3).exists():
        v3_data = load_json(args.v3)
        print(f"Loaded V3: {args.v3}")
    else:
        v3_data = {"error": f"File not found: {args.v3}"}
        print(f"[WARNING] V3 result not found: {args.v3}")

    # 分析各方法
    v1_result = analyze_v1(v1_data)
    v2_result = analyze_v2(v2_data)
    v3_result = analyze_v3(v3_data)

    # 对比分析
    analysis = compare_methods(v1_result, v2_result, v3_result)

    # 汇总报告
    report = {
        "v1_strict_rubric": v1_result,
        "v2_ab_comparison": v2_result,
        "v3_multidimensional": v3_result,
        "cross_method_analysis": analysis,
    }

    # 输出
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Judge Evaluation Comparison Report")
    print(f"{'='*60}")

    if "error" not in v1_result:
        print(f"\n[V1 严格评分]")
        print(f"  AL(K)_AVG = {v1_result['al_k_avg']:.3f} ± {v1_result['al_k_std']:.3f}")
        print(f"  分布: {v1_result['score_distribution']}")
        print(f"  低分段(1-2)占比: {v1_result['low_score_rate']:.1%}")
        print(f"  高分段(4-5)占比: {v1_result['high_score_rate']:.1%}")

    if "error" not in v2_result:
        print(f"\n[V2 A/B对比]")
        print(f"  注入模型胜率: {v2_result['injected_win_rate']:.1%}")
        print(f"  Baseline胜率: {v2_result['baseline_win_rate']:.1%}")
        print(f"  平局率: {v2_result['tie_rate']:.1%}")
        print(f"  样本数: {v2_result['num_samples']}")

    if "error" not in v3_result:
        print(f"\n[V3 多维度评分]")
        print(f"  风格(style): {v3_result['style_mean']:.3f} ± {v3_result['style_std']:.3f}")
        print(f"  内容(content): {v3_result['content_mean']:.3f} ± {v3_result['content_std']:.3f}")
        print(f"  一致性(consistency): {v3_result['consistency_mean']:.3f} ± {v3_result['consistency_std']:.3f}")
        print(f"  加权综合: {v3_result['weighted_mean']:.3f} ± {v3_result['weighted_std']:.3f}")

    print(f"\n[跨方法对比分析]")
    for line in analysis:
        print(f"  {line}")

    print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    main()
