from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.config import load_yaml


def _count_images(root: Path) -> int:
    if not root.exists():
        return 0
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    count = 0
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description='Check local availability of calibration datasets.')
    parser.add_argument('--config', default='configs/calibration_datasets.yaml')
    parser.add_argument('--out', default='reports/calibration_sources_status.json')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rows = []
    for item in cfg.get('calibration_datasets', []):
        local_dir = Path(str(item.get('local_dir', '')))
        images = _count_images(local_dir)
        rows.append(
            {
                'id': item.get('id'),
                'name': item.get('name'),
                'type': item.get('type', 'manual'),
                'url': item.get('url'),
                'local_dir': str(local_dir),
                'exists': local_dir.exists(),
                'images': images,
                'ready': images > 0,
            }
        )

    report = {
        'total_sources': len(rows),
        'ready_sources': sum(1 for x in rows if x['ready']),
        'rows': rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
