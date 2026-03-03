from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Workaround for Windows environments where a compatible torch build is installed
# in a short custom site-packages path (e.g. C:\ptpkgs).
_extra_site = os.getenv('AGRO_EXTRA_SITE_PACKAGES', r'C:\ptpkgs').strip()
if _extra_site and os.path.isdir(_extra_site) and _extra_site not in sys.path:
    sys.path.insert(0, _extra_site)

from ultralytics import YOLO


def _run(cmd: list[str], cwd: str = '.') -> None:
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _extract_seg_metrics(val_result: Any) -> dict[str, float]:
    seg = getattr(val_result, 'seg', None)
    return {
        'map50': float(getattr(seg, 'map50', 0.0) or 0.0),
        'map50_95': float(getattr(seg, 'map', 0.0) or 0.0),
        'precision': float(getattr(seg, 'mp', 0.0) or 0.0),
        'recall': float(getattr(seg, 'mr', 0.0) or 0.0),
    }


def _rank_models(models: list[Path], data_yaml: Path, imgsz: int, device: str) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []
    for path in models:
        if not path.exists():
            continue
        try:
            model = YOLO(str(path))
            val = model.val(
                data=str(data_yaml),
                split='val',
                plots=False,
                save_json=False,
                batch=2,
                imgsz=int(imgsz),
                device=device,
            )
            metrics = _extract_seg_metrics(val)
            ranking.append(
                {
                    'weights': str(path).replace('\\', '/'),
                    **metrics,
                }
            )
        except Exception as exc:
            ranking.append(
                {
                    'weights': str(path).replace('\\', '/'),
                    'error': str(exc),
                    'map50': 0.0,
                    'map50_95': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                }
            )

    ranking.sort(
        key=lambda x: (
            float(x.get('map50', 0.0)),
            float(x.get('map50_95', 0.0)),
            float(x.get('recall', 0.0)),
            float(x.get('precision', 0.0)),
        ),
        reverse=True,
    )
    return ranking


def _update_app_weights(app_yaml: Path, weights_path: str) -> None:
    payload = yaml.safe_load(app_yaml.read_text(encoding='utf-8')) or {}
    if not isinstance(payload, dict):
        payload = {}
    model_cfg = payload.setdefault('model', {})
    model_cfg['weights'] = weights_path
    app_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Auto pipeline: HF dataset build -> filtering/balance -> training -> ranking.')
    parser.add_argument('--prepare-out', default='data/hf_multisource_raw_auto', type=str)
    parser.add_argument('--balanced-out', default='data/hf_multisource_balanced_auto', type=str)
    parser.add_argument('--train-config', default='configs/train_final_cpu.yaml', type=str)
    parser.add_argument('--run-name', default='auto_hf_train', type=str)
    parser.add_argument('--report', default='reports/auto_hf_pipeline_report.json', type=str)
    parser.add_argument('--ranking-report', default='reports/auto_hf_model_ranking.json', type=str)
    parser.add_argument('--app-config', default='configs/app.yaml', type=str)
    parser.add_argument('--update-app-config', action='store_true')
    parser.add_argument('--imgsz', default=512, type=int)
    parser.add_argument('--device', default='cpu', type=str)

    # Prepare dataset sources
    parser.add_argument('--chronoroot-max', default=600, type=int)
    parser.add_argument('--plantorgans-max', default=1800, type=int)
    parser.add_argument('--include-weak-100crops', action='store_true')
    parser.add_argument('--weak-100crops-max', default=700, type=int)
    parser.add_argument('--include-plantseg-lesions', action='store_true')
    parser.add_argument('--plantseg-max', default=250, type=int)
    parser.add_argument('--val-ratio', default=0.15, type=float)
    parser.add_argument('--img-max-side', default=1280, type=int)
    parser.add_argument('--min-area', default=64.0, type=float)
    parser.add_argument('--seed', default=42, type=int)

    # Filtering / balancing
    parser.add_argument('--min-blur', default=35.0, type=float)
    parser.add_argument('--min-brightness', default=30.0, type=float)
    parser.add_argument('--max-brightness', default=230.0, type=float)
    parser.add_argument('--min-area-ratio', default=0.0003, type=float)
    parser.add_argument('--max-area-ratio', default=0.92, type=float)
    parser.add_argument('--min-instances-per-image', default=1, type=int)
    parser.add_argument('--class-ratio-max', default=2.2, type=float)
    parser.add_argument('--min-kept-images', default=160, type=int)

    args = parser.parse_args()

    prepare_out = Path(args.prepare_out)
    balanced_out = Path(args.balanced_out)
    report_path = Path(args.report)
    ranking_path = Path(args.ranking_report)
    app_config = Path(args.app_config)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    auto_model_name = f'best_auto_{timestamp}.pt'
    auto_onnx_name = f'best_auto_{timestamp}.onnx'

    # 1) HF dataset build
    cmd_prepare = [
        sys.executable,
        '-m',
        'training.prepare_multisource_dataset',
        '--out',
        str(prepare_out),
        '--chronoroot-max',
        str(args.chronoroot_max),
        '--plantorgans-max',
        str(args.plantorgans_max),
        '--val-ratio',
        str(args.val_ratio),
        '--img-max-side',
        str(args.img_max_side),
        '--min-area',
        str(args.min_area),
        '--seed',
        str(args.seed),
    ]
    if args.include_weak_100crops:
        cmd_prepare.extend(['--include-weak-100crops', '--weak-100crops-max', str(args.weak_100crops_max)])
    if args.include_plantseg_lesions:
        cmd_prepare.extend(['--include-plantseg-lesions', '--plantseg-max', str(args.plantseg_max)])
    _run(cmd_prepare)

    raw_yaml = prepare_out / 'dataset.yaml'
    if not raw_yaml.exists():
        raise FileNotFoundError(f'Prepared dataset yaml not found: {raw_yaml}')

    # 2) Filter + balance
    cmd_balance = [
        sys.executable,
        '-m',
        'training.filter_balance_yoloseg',
        '--data-yaml',
        str(raw_yaml),
        '--out',
        str(balanced_out),
        '--min-blur',
        str(args.min_blur),
        '--min-brightness',
        str(args.min_brightness),
        '--max-brightness',
        str(args.max_brightness),
        '--min-area-ratio',
        str(args.min_area_ratio),
        '--max-area-ratio',
        str(args.max_area_ratio),
        '--min-instances-per-image',
        str(args.min_instances_per_image),
        '--class-ratio-max',
        str(args.class_ratio_max),
        '--min-kept-images',
        str(args.min_kept_images),
    ]
    _run(cmd_balance)

    balanced_yaml = balanced_out / 'dataset.yaml'
    if not balanced_yaml.exists():
        raise FileNotFoundError(f'Balanced dataset yaml not found: {balanced_yaml}')

    # 3) Train
    cmd_train = [
        sys.executable,
        '-m',
        'training.train_maximum',
        '--config',
        str(args.train_config),
        '--stage1-data',
        str(balanced_yaml),
        '--stage2-data',
        str(balanced_yaml),
        '--name',
        str(args.run_name),
    ]
    _run(cmd_train)

    # Snapshot latest trained best_max artifacts
    models_dir = Path('models')
    best_max_pt = models_dir / 'best_max.pt'
    best_max_onnx = models_dir / 'best_max.onnx'
    snap_pt = models_dir / auto_model_name
    snap_onnx = models_dir / auto_onnx_name
    if best_max_pt.exists():
        shutil.copy2(best_max_pt, snap_pt)
    if best_max_onnx.exists():
        shutil.copy2(best_max_onnx, snap_onnx)

    # 4) Rank models
    candidate_models = [
        p
        for p in [
            models_dir / 'best.pt',
            models_dir / 'best_length_boost_quick.pt',
            models_dir / 'best_max.pt',
            snap_pt,
        ]
        if p.exists()
    ]
    ranking = _rank_models(
        models=candidate_models,
        data_yaml=balanced_yaml,
        imgsz=int(args.imgsz),
        device=str(args.device),
    )
    ranking_payload = {
        'data_yaml': str(balanced_yaml.resolve()),
        'ranking': ranking,
    }
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_path.write_text(json.dumps(ranking_payload, indent=2, ensure_ascii=True), encoding='utf-8')

    winner = ranking[0]['weights'] if ranking else ''
    if args.update_app_config and winner and app_config.exists():
        # Keep relative path in config if candidate lives in ./models
        winner_path = Path(winner)
        rel = str(winner_path).replace('\\', '/')
        if winner_path.is_absolute():
            try:
                rel = str(winner_path.relative_to(Path.cwd())).replace('\\', '/')
            except ValueError:
                rel = str(winner_path).replace('\\', '/')
        _update_app_weights(app_yaml=app_config, weights_path=rel)

    report_payload = {
        'prepare_output': str(prepare_out.resolve()),
        'balanced_output': str(balanced_out.resolve()),
        'raw_dataset_yaml': str(raw_yaml.resolve()),
        'balanced_dataset_yaml': str(balanced_yaml.resolve()),
        'train_config': str(Path(args.train_config).resolve()),
        'run_name': args.run_name,
        'model_snapshot_pt': str(snap_pt.resolve()) if snap_pt.exists() else '',
        'model_snapshot_onnx': str(snap_onnx.resolve()) if snap_onnx.exists() else '',
        'ranking_report': str(ranking_path.resolve()),
        'winner': winner,
        'app_config_updated': bool(args.update_app_config and winner),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=True), encoding='utf-8')

    print(json.dumps(report_payload, indent=2, ensure_ascii=True))
    print(json.dumps(ranking_payload, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
