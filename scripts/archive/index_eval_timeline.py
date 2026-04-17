#!/usr/bin/env python
"""索引评估结果时间线，辅助判定当前权威口径。"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

RESULTS = Path('results')
items = []
for path in RESULTS.iterdir():
    if not path.is_dir():
        continue
    if path.name.startswith('archive_'):
        continue
    files = sorted([p for p in path.rglob('*') if p.is_file()])
    if not files:
        continue
    latest = max(files, key=lambda p: p.stat().st_mtime)
    items.append({
        'dir': str(path),
        'file_count': len(files),
        'latest_file': str(latest),
        'latest_mtime': datetime.fromtimestamp(latest.stat().st_mtime).isoformat(timespec='seconds'),
    })
items.sort(key=lambda x: x['latest_mtime'], reverse=True)
print(json.dumps({'generated_at': datetime.now().isoformat(timespec='seconds'), 'results': items}, indent=2, ensure_ascii=False))
