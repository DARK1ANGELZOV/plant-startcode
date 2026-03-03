from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from utils.config import load_yaml


def _run(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    text = (proc.stdout or '') + (proc.stderr or '')
    return proc.returncode, text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description='Fetch calibration sources from kaggle/git when supported.')
    parser.add_argument('--config', default='configs/calibration_datasets.yaml')
    parser.add_argument('--source-ids', nargs='*', default=[])
    parser.add_argument('--out', default='reports/calibration_fetch_report.json')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    selected = {x.strip() for x in args.source_ids if x.strip()}

    results = []
    for item in cfg.get('calibration_datasets', []):
        sid = str(item.get('id', '')).strip()
        if selected and sid not in selected:
            continue

        local_dir = Path(str(item.get('local_dir', '')))
        source_type = str(item.get('type', 'manual')).lower()
        local_dir.parent.mkdir(parents=True, exist_ok=True)

        result = {
            'id': sid,
            'type': source_type,
            'local_dir': str(local_dir),
            'status': 'skipped',
            'message': '',
        }

        if args.skip_existing and local_dir.exists() and any(local_dir.rglob('*')):
            result['status'] = 'exists'
            result['message'] = 'Skipped existing local directory.'
            results.append(result)
            continue

        if source_type == 'git':
            repo = str(item.get('git_repo', '')).strip()
            if not repo:
                result['status'] = 'error'
                result['message'] = 'Missing git_repo in config.'
            elif shutil.which('git') is None:
                result['status'] = 'error'
                result['message'] = 'git is not installed.'
            else:
                if local_dir.exists() and any(local_dir.iterdir()):
                    code, text = _run(['git', '-C', str(local_dir), 'pull'])
                else:
                    code, text = _run(['git', 'clone', '--depth', '1', repo, str(local_dir)])
                result['status'] = 'ok' if code == 0 else 'error'
                result['message'] = text[-2000:]

        elif source_type == 'kaggle':
            slug = str(item.get('kaggle_slug', '')).strip()
            if not slug:
                result['status'] = 'error'
                result['message'] = 'Missing kaggle_slug in config.'
            elif shutil.which('kaggle') is None:
                result['status'] = 'error'
                result['message'] = 'kaggle CLI is not installed.'
            else:
                local_dir.mkdir(parents=True, exist_ok=True)
                code, text = _run(['kaggle', 'datasets', 'download', '-d', slug, '-p', str(local_dir), '--unzip'])
                result['status'] = 'ok' if code == 0 else 'error'
                result['message'] = text[-2000:]

        else:
            result['status'] = 'manual'
            result['message'] = f"Download manually from {item.get('url', '')} into {local_dir}."

        results.append(result)

    report = {
        'total': len(results),
        'ok': sum(1 for r in results if r['status'] == 'ok'),
        'manual': sum(1 for r in results if r['status'] == 'manual'),
        'errors': sum(1 for r in results if r['status'] == 'error'),
        'results': results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
