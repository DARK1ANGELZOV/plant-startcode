from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env or os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    return int(proc.returncode), proc.stdout


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Unified runner for points 1,2,3,6,5: golden500, 2-stage class-aware train, compact replies, strict gate blocker.'
    )
    p.add_argument('--project-root', type=str, default='.')
    p.add_argument('--python-exe', type=str, default=sys.executable)

    # Point 1: hard-negative + golden
    p.add_argument('--skip-point1', action='store_true')
    p.add_argument('--model', type=str, default='models/best_max.pt')
    p.add_argument('--data-root', type=str, default='data/hf_multisource_mega10')
    p.add_argument('--hard-out', type=str, default='data/hard_mined_rootstem_rs_v2')
    p.add_argument('--golden-out', type=str, default='data/golden_rootstem_500')
    p.add_argument('--golden-target', type=int, default=500)

    # Point 2+3: class-aware 2-stage train
    p.add_argument('--skip-point23', action='store_true')
    p.add_argument('--boost-out', type=str, default='data/hf_multisource_rootstem_boost')
    p.add_argument('--hardmix-out', type=str, default='data/hf_multisource_hardmix')
    p.add_argument('--train-config', type=str, default='configs/train_rootstem_2stage_classaware.yaml')
    p.add_argument('--train-name', type=str, default='points123_stage')
    p.add_argument('--oversample-rs-factor', type=int, default=3)

    # Point 6: compact+reliable reply validation
    p.add_argument('--skip-point6', action='store_true')

    # Point 5: strict gate release blocker
    p.add_argument('--skip-point5', action='store_true')
    p.add_argument('--benchmark-data', type=str, default='data/hf_multisource_mega10_fast/dataset.yaml')
    p.add_argument('--strict-n', type=int, default=30)
    p.add_argument('--strict-threshold', type=float, default=0.90)
    p.add_argument('--lightweight-gate', action='store_true')
    p.add_argument('--auto-deploy', action='store_true')
    p.add_argument('--fail-on-block', action='store_true')

    p.add_argument('--report-out', type=str, default='reports/points_1_2_3_6_5_summary.json')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    py = args.python_exe
    report: dict[str, Any] = {
        'timestamp': datetime.now(UTC).isoformat(),
        'params': vars(args),
        'steps': {},
        'artifacts': {},
    }

    models_dir = root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = (root / args.model).resolve()
    if not baseline_path.exists():
        raise FileNotFoundError(f'Baseline model not found: {baseline_path}')
    baseline_backup = models_dir / f'best_max_baseline_{datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}.pt'
    shutil.copy2(baseline_path, baseline_backup)
    report['artifacts']['baseline_backup'] = str(baseline_backup)

    # Point 1: golden 500 with hard-negatives
    if not args.skip_point1:
        cmd = [
            py,
            '-m',
            'training.run_hardneg_golden_pipeline',
            '--model',
            args.model,
            '--data-root',
            args.data_root,
            '--hard-out',
            args.hard_out,
            '--golden-out',
            args.golden_out,
            '--golden-target',
            str(int(args.golden_target)),
            '--golden-max',
            str(max(500, int(args.golden_target))),
            '--review-out',
            f"{args.golden_out}_review",
            '--report-out',
            'reports/point1_hardneg_golden500_summary.json',
        ]
        rc, out = _run(cmd, cwd=root)
        report['steps']['point1_hardneg_golden500'] = {'returncode': rc, 'tail': out[-8000:]}
        if rc != 0:
            raise RuntimeError('Point 1 failed: hard-negative + golden pipeline.')

    # Point 2+3: build class-aware datasets + 2-stage training
    candidate_path = models_dir / f'best_candidate_points123_{datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}.pt'
    if not args.skip_point23:
        cmd_boost = [
            py,
            '-m',
            'training.build_rootstem_boost_dataset',
            '--out',
            args.boost_out,
            '--mega10',
            args.data_root,
            '--gold',
            args.golden_out,
            '--oversample-rs-factor',
            str(max(1, int(args.oversample_rs_factor))),
        ]
        rc, out = _run(cmd_boost, cwd=root)
        report['steps']['point23_build_boost_dataset'] = {'returncode': rc, 'tail': out[-8000:]}
        if rc != 0:
            raise RuntimeError('Point 2/3 failed: build_rootstem_boost_dataset.')

        cmd_hardmix = [
            py,
            '-m',
            'training.build_hardmix_from_mined',
            '--out',
            args.hardmix_out,
            '--hard',
            args.hard_out,
            '--mega-fast',
            'data/hf_multisource_mega10_fast',
            '--gold',
            args.golden_out,
        ]
        rc, out = _run(cmd_hardmix, cwd=root)
        report['steps']['point23_build_hardmix_dataset'] = {'returncode': rc, 'tail': out[-8000:]}
        if rc != 0:
            raise RuntimeError('Point 2/3 failed: build_hardmix_from_mined.')

        cmd_train = [
            py,
            '-m',
            'training.train_maximum',
            '--config',
            args.train_config,
            '--stage1-data',
            f'{args.boost_out}/dataset.yaml',
            '--stage2-data',
            f'{args.hardmix_out}/dataset.yaml',
            '--name',
            args.train_name,
            '--no-plots',
        ]
        rc, out = _run(cmd_train, cwd=root)
        report['steps']['point23_train_classaware_2stage'] = {'returncode': rc, 'tail': out[-12000:]}
        if rc != 0:
            raise RuntimeError('Point 2/3 failed: 2-stage class-aware training.')

        trained_best = models_dir / 'best_max.pt'
        if not trained_best.exists():
            raise FileNotFoundError('Expected models/best_max.pt after train_maximum.')
        shutil.copy2(trained_best, candidate_path)
        report['artifacts']['candidate_model'] = str(candidate_path)
        # Restore baseline until strict gate approves candidate.
        shutil.copy2(baseline_backup, models_dir / 'best_max.pt')

    # Point 6: compact reply strictness tests
    if not args.skip_point6:
        cmd_tests = [py, '-m', 'pytest', 'tests/test_insight_service.py', 'tests/test_insight_service_realism.py', '-q']
        rc, out = _run(cmd_tests, cwd=root)
        report['steps']['point6_compact_reply_tests'] = {'returncode': rc, 'tail': out[-8000:]}
        if rc != 0:
            raise RuntimeError('Point 6 failed: insight reply tests.')

    # Point 5: strict quality gate blocker before release
    if not args.skip_point5:
        candidate_for_gate = candidate_path if candidate_path.exists() else (models_dir / 'best_max.pt')
        cmd_gate = [
            py,
            '-m',
            'training.release_guard_strict',
            '--baseline-model',
            str(baseline_backup),
            '--candidate-model',
            str(candidate_for_gate),
            '--benchmark-data',
            args.benchmark_data,
            '--strict-n',
            str(int(args.strict_n)),
            '--strict-threshold',
            str(float(args.strict_threshold)),
            '--report-out',
            'reports/point5_release_guard_report.json',
        ]
        if args.lightweight_gate:
            cmd_gate.append('--lightweight-gate')
        if args.auto_deploy:
            cmd_gate.append('--auto-deploy')
        if args.fail_on_block:
            cmd_gate.append('--fail-on-block')

        rc, out = _run(cmd_gate, cwd=root)
        report['steps']['point5_strict_gate_release_blocker'] = {'returncode': rc, 'tail': out[-12000:]}
        if rc != 0 and args.fail_on_block:
            raise RuntimeError('Point 5 failed: release guard blocked candidate.')

    out_path = root / args.report_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
