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

# Workaround for Windows environments where a compatible torch build is installed
# in a short custom site-packages path (e.g. C:\ptpkgs).
_extra_site = os.getenv('AGRO_EXTRA_SITE_PACKAGES', r'C:\ptpkgs').strip()
if _extra_site and os.path.isdir(_extra_site) and _extra_site not in sys.path:
    sys.path.insert(0, _extra_site)

from ultralytics import YOLO


def _run(cmd: list[str], env: dict[str, str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    return int(proc.returncode), proc.stdout


def _eval_model(model_path: Path, data_yaml: Path, imgsz: int, batch: int, workers: int) -> dict[str, Any]:
    model = YOLO(str(model_path))
    val = model.val(
        data=str(data_yaml),
        split='val',
        imgsz=int(imgsz),
        batch=int(batch),
        workers=int(workers),
        device='cpu',
        plots=False,
        save_json=False,
    )
    seg = getattr(val, 'seg', None)
    maps = list(getattr(seg, 'maps', [])) if seg is not None else []
    iou = {model.names.get(i, str(i)): float(v) for i, v in enumerate(maps)}
    return {
        'iou_per_class': iou,
        'precision': float(getattr(seg, 'mp', 0.0)) if seg is not None else 0.0,
        'recall': float(getattr(seg, 'mr', 0.0)) if seg is not None else 0.0,
        'mAP50': float(getattr(seg, 'map50', 0.0)) if seg is not None else 0.0,
        'mAP50_95': float(getattr(seg, 'map', 0.0)) if seg is not None else 0.0,
    }


def _run_strict_gate(
    *,
    python_exe: str,
    model_path: Path,
    out_json: Path,
    n: int,
    threshold: float,
    lightweight: bool,
    cwd: Path,
) -> tuple[int, dict[str, Any], str]:
    cmd = [
        python_exe,
        '-m',
        'tools.strict_photo_quality_gate',
        '--n',
        str(int(n)),
        '--strict-pass-threshold',
        str(float(threshold)),
        '--out',
        str(out_json),
        '--fail-on-threshold',
    ]
    if lightweight:
        cmd.append('--lightweight')
    env = os.environ.copy()
    env['MODEL_WEIGHTS'] = str(model_path)
    rc, stdout = _run(cmd, env=env, cwd=cwd)
    payload: dict[str, Any] = {}
    if out_json.exists():
        payload = json.loads(out_json.read_text(encoding='utf-8'))
    return rc, payload, stdout


def _safe_ratio(x: float, y: float) -> float:
    if abs(y) < 1e-12:
        return 0.0
    return float(x / y)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Release guard: strict gate + metric floors before deploying a candidate model.')
    p.add_argument('--project-root', type=str, default='.')
    p.add_argument('--python-exe', type=str, default=sys.executable)
    p.add_argument('--baseline-model', type=str, default='models/best_max.pt')
    p.add_argument('--candidate-model', type=str, required=True)
    p.add_argument('--benchmark-data', type=str, default='data/hf_multisource_mega10_fast/dataset.yaml')
    p.add_argument('--eval-imgsz', type=int, default=512)
    p.add_argument('--eval-batch', type=int, default=1)
    p.add_argument('--eval-workers', type=int, default=0)
    p.add_argument('--strict-n', type=int, default=30)
    p.add_argument('--strict-threshold', type=float, default=0.90)
    p.add_argument('--lightweight-gate', action='store_true')
    p.add_argument('--recall-floor-ratio', type=float, default=0.90)
    p.add_argument('--map50-floor-ratio', type=float, default=0.90)
    p.add_argument('--strict-rate-floor-ratio', type=float, default=0.95)
    p.add_argument('--skip-baseline-gate', action='store_true')
    p.add_argument('--auto-deploy', action='store_true')
    p.add_argument('--report-out', type=str, default='reports/release_guard_strict.json')
    p.add_argument('--fail-on-block', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    baseline_model = (root / args.baseline_model).resolve()
    candidate_model = (root / args.candidate_model).resolve()
    benchmark_data = (root / args.benchmark_data).resolve()

    if not baseline_model.exists():
        raise FileNotFoundError(f'Baseline model not found: {baseline_model}')
    if not candidate_model.exists():
        raise FileNotFoundError(f'Candidate model not found: {candidate_model}')
    if not benchmark_data.exists():
        raise FileNotFoundError(f'Benchmark data yaml not found: {benchmark_data}')

    report: dict[str, Any] = {
        'timestamp': datetime.now(UTC).isoformat(),
        'params': vars(args),
        'models': {
            'baseline': str(baseline_model),
            'candidate': str(candidate_model),
        },
    }

    baseline_metrics = _eval_model(
        model_path=baseline_model,
        data_yaml=benchmark_data,
        imgsz=int(args.eval_imgsz),
        batch=int(args.eval_batch),
        workers=int(args.eval_workers),
    )
    candidate_metrics = _eval_model(
        model_path=candidate_model,
        data_yaml=benchmark_data,
        imgsz=int(args.eval_imgsz),
        batch=int(args.eval_batch),
        workers=int(args.eval_workers),
    )
    report['metrics'] = {
        'baseline': baseline_metrics,
        'candidate': candidate_metrics,
    }

    baseline_gate = {
        'quality_gate_passed': True,
        'strict_sample_pass_rate': None,
    }
    if not args.skip_baseline_gate:
        out_baseline = root / 'reports' / f'strict_gate_baseline_{datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}.json'
        rc, payload, tail = _run_strict_gate(
            python_exe=args.python_exe,
            model_path=baseline_model,
            out_json=out_baseline,
            n=int(args.strict_n),
            threshold=float(args.strict_threshold),
            lightweight=bool(args.lightweight_gate),
            cwd=root,
        )
        baseline_gate = {
            'returncode': rc,
            'report': str(out_baseline),
            'quality_gate_passed': bool(payload.get('quality_gate_passed', False)),
            'strict_sample_pass_rate': payload.get('strict_sample_pass_rate'),
            'global_checks_pass_rate': payload.get('global_checks_pass_rate'),
            'critical_checks_pass_rate': payload.get('critical_checks_pass_rate'),
            'tail': tail[-4000:],
        }

    out_candidate = root / 'reports' / f'strict_gate_candidate_{datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}.json'
    rc, payload, tail = _run_strict_gate(
        python_exe=args.python_exe,
        model_path=candidate_model,
        out_json=out_candidate,
        n=int(args.strict_n),
        threshold=float(args.strict_threshold),
        lightweight=bool(args.lightweight_gate),
        cwd=root,
    )
    candidate_gate = {
        'returncode': rc,
        'report': str(out_candidate),
        'quality_gate_passed': bool(payload.get('quality_gate_passed', False)),
        'strict_sample_pass_rate': payload.get('strict_sample_pass_rate'),
        'global_checks_pass_rate': payload.get('global_checks_pass_rate'),
        'critical_checks_pass_rate': payload.get('critical_checks_pass_rate'),
        'tail': tail[-4000:],
    }
    report['strict_gate'] = {
        'baseline': baseline_gate,
        'candidate': candidate_gate,
    }

    baseline_recall = float(baseline_metrics.get('recall', 0.0))
    baseline_map50 = float(baseline_metrics.get('mAP50', 0.0))
    candidate_recall = float(candidate_metrics.get('recall', 0.0))
    candidate_map50 = float(candidate_metrics.get('mAP50', 0.0))
    baseline_strict_rate = float(baseline_gate.get('strict_sample_pass_rate') or 0.0)
    candidate_strict_rate = float(candidate_gate.get('strict_sample_pass_rate') or 0.0)

    checks = {
        'candidate_gate_passed': bool(candidate_gate.get('quality_gate_passed', False)),
        'recall_floor_ok': candidate_recall >= baseline_recall * float(args.recall_floor_ratio),
        'map50_floor_ok': candidate_map50 >= baseline_map50 * float(args.map50_floor_ratio),
        'strict_rate_floor_ok': (
            True
            if args.skip_baseline_gate
            else candidate_strict_rate >= baseline_strict_rate * float(args.strict_rate_floor_ratio)
        ),
    }
    report['checks'] = checks
    report['ratios'] = {
        'recall_ratio': _safe_ratio(candidate_recall, baseline_recall),
        'map50_ratio': _safe_ratio(candidate_map50, baseline_map50),
        'strict_rate_ratio': _safe_ratio(candidate_strict_rate, baseline_strict_rate) if baseline_strict_rate > 0 else None,
    }

    allow_release = all(bool(v) for v in checks.values())
    report['release_allowed'] = bool(allow_release)

    deployed_to = None
    if allow_release and bool(args.auto_deploy):
        dst = root / 'models' / 'best_max.pt'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_model, dst)
        deployed_to = str(dst)
    report['deployed_to'] = deployed_to

    report_path = root / args.report_out
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if bool(args.fail_on_block) and (not allow_release):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
