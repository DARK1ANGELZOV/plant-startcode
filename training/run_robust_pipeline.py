from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from training.build_multidomain_manifest import build_manifest
from training.compare_architectures import compare_architectures
from utils.config import load_yaml


def _run_cmd(cmd: list[str], cwd: str = '.') -> None:
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _resolve_plant_data_yaml(cfg: dict[str, Any]) -> str:
    datasets = cfg.get('datasets', {})
    plants_cfg = datasets.get('plants_local', {})
    root = Path(str(plants_cfg.get('root', 'data/hf_multisource_medium')))
    data_yaml = root / 'dataset.yaml'
    if data_yaml.exists():
        return str(data_yaml)
    return 'data/hf_multisource_medium/dataset.yaml'


def _resolve_adverse_data_yaml(cfg: dict[str, Any]) -> str:
    datasets = cfg.get('datasets', {})
    for key in ('acdc', 'weatherproof', 'raidar'):
        ds = datasets.get(key, {})
        root = Path(str(ds.get('root', '')))
        data_yaml = root / 'dataset.yaml'
        if data_yaml.exists():
            return str(data_yaml)
    return ''


def main() -> None:
    parser = argparse.ArgumentParser(description='End-to-end robust training pipeline orchestrator.')
    parser.add_argument('--config', default='configs/robust_train.yaml')
    parser.add_argument('--train-manifest', default='data/robust/train_manifest.jsonl')
    parser.add_argument('--val-manifest', default='data/robust/val_manifest.jsonl')
    parser.add_argument('--output-dir', default='models/robust')
    parser.add_argument('--max-images-per-domain', type=int, default=50000)
    parser.add_argument('--skip-deeplab', action='store_true')
    parser.add_argument('--skip-yolo', action='store_true')
    parser.add_argument('--skip-benchmark', action='store_true')
    parser.add_argument('--cpu-only-benchmark', action='store_true')
    parser.add_argument('--name', default='robust_full')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_info = build_manifest(
        config_path=args.config,
        output_jsonl=args.train_manifest,
        split='train',
        max_images_per_domain=args.max_images_per_domain,
    )
    val_info = build_manifest(
        config_path=args.config,
        output_jsonl=args.val_manifest,
        split='val',
        max_images_per_domain=max(2000, args.max_images_per_domain // 4),
    )

    if int(val_info.get('total_rows', 0)) == 0:
        print('Validation manifest is empty for split=val, using split=train fallback.')
        val_info = build_manifest(
            config_path=args.config,
            output_jsonl=args.val_manifest,
            split='train',
            max_images_per_domain=max(2000, args.max_images_per_domain // 4),
        )

    pipeline_report: dict[str, Any] = {
        'train_manifest': train_info,
        'val_manifest': val_info,
        'runs': {},
    }

    plant_data_yaml = _resolve_plant_data_yaml(cfg)
    adverse_data_yaml = _resolve_adverse_data_yaml(cfg)

    if not args.skip_deeplab:
        _run_cmd(
            [
                sys.executable,
                '-m',
                'training.train_deeplab_robust',
                '--config',
                args.config,
                '--train-manifest',
                args.train_manifest,
                '--val-manifest',
                args.val_manifest,
                '--output',
                args.output_dir,
            ]
        )
        pipeline_report['runs']['deeplab'] = 'completed'
    else:
        pipeline_report['runs']['deeplab'] = 'skipped'

    if not args.skip_yolo:
        yolo_cmd = [
            sys.executable,
            '-m',
            'training.train_yolo_robust',
            '--config',
            args.config,
            '--pretrain-data',
            plant_data_yaml,
            '--plant-data',
            plant_data_yaml,
            '--name',
            args.name,
            '--output',
            args.output_dir,
        ]
        if adverse_data_yaml:
            yolo_cmd.extend(['--adverse-data', adverse_data_yaml])
        _run_cmd(yolo_cmd)
        pipeline_report['runs']['yolo'] = 'completed'
    else:
        pipeline_report['runs']['yolo'] = 'skipped'

    yolo_report = str((output_dir / 'yolo_metrics.json').resolve())
    deeplab_report = str((output_dir / 'deeplab_metrics.json').resolve())
    best_report = str((output_dir / 'best_architecture.json').resolve())

    best = compare_architectures(
        yolo_report_path=yolo_report,
        deeplab_report_path=deeplab_report,
        output_path=best_report,
    )
    pipeline_report['best_architecture'] = best

    if not args.skip_benchmark:
        bench_cmd = [
            sys.executable,
            '-m',
            'training.benchmark_robust',
            '--config',
            args.config,
            '--architecture',
            'auto',
            '--output',
            str(output_dir / 'benchmark.json'),
        ]
        if args.cpu_only_benchmark:
            bench_cmd.append('--cpu-only')
        _run_cmd(bench_cmd)
        pipeline_report['runs']['benchmark'] = 'completed'
    else:
        pipeline_report['runs']['benchmark'] = 'skipped'

    pipeline_path = output_dir / 'pipeline_report.json'
    pipeline_path.write_text(json.dumps(pipeline_report, indent=2, ensure_ascii=True), encoding='utf-8')
    print(json.dumps(pipeline_report, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
