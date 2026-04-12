#!/usr/bin/env python
"""分析 gate 与思考泄露的相关性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_items(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def avg(values):
    return round(mean(values), 6) if values else None


def main():
    parser = argparse.ArgumentParser(description='分析 gate 与泄露的相关性')
    parser.add_argument('--input', required=True, help='responses.json 路径')
    parser.add_argument('--output', required=True, help='summary.json 输出路径')
    args = parser.parse_args()

    items = load_items(Path(args.input))
    leak_items = [x for x in items if x.get('leak_detected')]
    non_leak_items = [x for x in items if not x.get('leak_detected')]
    gate_items = [x for x in items if x.get('gate_values')]

    def layer_means(rows):
        if not rows:
            return []
        width = len(rows[0]['gate_values'])
        result = []
        for idx in range(width):
            vals = [row['gate_values'][idx] for row in rows if row.get('gate_values')]
            result.append(round(mean(vals), 6) if vals else None)
        return result

    summary = {
        'total': len(items),
        'gate_item_count': len(gate_items),
        'leak_count': len(leak_items),
        'non_leak_count': len(non_leak_items),
        'leak_ratio': round(len(leak_items) / len(items), 6) if items else 0.0,
        'gate_mean_all': avg([x['gate_mean'] for x in gate_items if x.get('gate_mean') is not None]),
        'gate_mean_leak': avg([x['gate_mean'] for x in leak_items if x.get('gate_mean') is not None]),
        'gate_mean_non_leak': avg([x['gate_mean'] for x in non_leak_items if x.get('gate_mean') is not None]),
        'gate_max_leak': avg([x['gate_max'] for x in leak_items if x.get('gate_max') is not None]),
        'gate_max_non_leak': avg([x['gate_max'] for x in non_leak_items if x.get('gate_max') is not None]),
        'layer_mean_leak': layer_means([x for x in leak_items if x.get('gate_values')]),
        'layer_mean_non_leak': layer_means([x for x in non_leak_items if x.get('gate_values')]),
    }

    Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
