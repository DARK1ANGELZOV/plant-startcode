from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from ultralytics import YOLO


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    return int(proc.returncode), proc.stdout


def evaluate_model(model_path: Path, data_yaml: Path, imgsz: int, batch: int, workers: int) -> dict:
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


def safe_pct(before: float, after: float) -> float | None:
    if abs(before) < 1e-12:
        return None
    return (after - before) / before * 100.0


def resolve_best_from_run(name: str) -> Path | None:
    p = Path(f'C:/Users/Dark_Angel/runs/segment/runs/segment/{name}_stage1/weights/best.pt')
    return p if p.exists() else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Root/stem recall cycle: hard-negative mining + hardmix + train + eval.')
    p.add_argument('--project-root', type=str, default='.')
    p.add_argument('--python-exe', type=str, default=sys.executable)

    p.add_argument('--data-root', type=str, default='data/hf_multisource_mega10')
    p.add_argument('--hard-out', type=str, default='data/hard_mined_rootstem_api')
    p.add_argument('--golden-out', type=str, default='data/golden_candidates_rootstem')
    p.add_argument('--hardmix-out', type=str, default='data/hf_multisource_hardmix')

    p.add_argument('--mine-train-limit', type=int, default=260)
    p.add_argument('--mine-val-limit', type=int, default=90)
    p.add_argument('--max-hard-per-split', type=int, default=500)
    p.add_argument('--golden-target', type=int, default=400)
    p.add_argument('--min-class-conf', type=float, default=0.55)
    p.add_argument('--min-trust', type=float, default=0.72)

    p.add_argument('--cap-mega-train', type=int, default=900)
    p.add_argument('--cap-mega-val', type=int, default=220)
    p.add_argument('--cap-pack-train', type=int, default=700)
    p.add_argument('--cap-pack-val', type=int, default=90)
    p.add_argument('--cap-gold-train', type=int, default=420)
    p.add_argument('--cap-gold-val', type=int, default=72)
    p.add_argument('--repeat-hard-train', type=int, default=3)

    p.add_argument('--train-config', type=str, default='configs/train_rootstem_hardneg_quick.yaml')
    p.add_argument('--train-name', type=str, default='rootstem_recall_cycle')
    p.add_argument('--stage2-data', type=str, default='')

    p.add_argument('--baseline-model', type=str, default='models/best_max.pt')
    p.add_argument('--candidate-model', type=str, default='models/best_rootstem_recall_cycle.pt')
    p.add_argument('--benchmark-data', type=str, default='data/hf_multisource_mega10_fast/dataset.yaml')
    p.add_argument('--eval-imgsz', type=int, default=512)
    p.add_argument('--eval-batch', type=int, default=1)
    p.add_argument('--eval-workers', type=int, default=0)

    p.add_argument('--report-out', type=str, default='reports/rootstem_recall_cycle_summary.json')
    p.add_argument('--skip-train', action='store_true')
    p.add_argument('--skip-mine', action='store_true')
    p.add_argument('--skip-hardmix', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()

    report: dict = {
        'timestamp': datetime.now(UTC).isoformat(),
        'steps': {},
        'artifacts': {},
    }

    baseline_metrics = evaluate_model(
        model_path=(root / args.baseline_model),
        data_yaml=(root / args.benchmark_data),
        imgsz=int(args.eval_imgsz),
        batch=int(args.eval_batch),
        workers=int(args.eval_workers),
    )
    report['baseline_metrics'] = baseline_metrics

    if not args.skip_mine:
        mine_cmd = [
            args.python_exe,
            '-m',
            'training.mine_hard_negatives_via_api',
            '--data-root',
            args.data_root,
            '--out',
            args.hard_out,
            '--golden-out',
            args.golden_out,
            '--train-limit',
            str(args.mine_train_limit),
            '--val-limit',
            str(args.mine_val_limit),
            '--max-hard-per-split',
            str(args.max_hard_per_split),
            '--golden-target',
            str(args.golden_target),
            '--min-class-conf',
            str(args.min_class_conf),
            '--min-trust',
            str(args.min_trust),
        ]
        rc, out = run_cmd(mine_cmd, cwd=root)
        report['steps']['mine'] = {'returncode': rc, 'tail': out[-4000:]}

    if not args.skip_hardmix:
        hardmix_cmd = [
            args.python_exe,
            '-m',
            'training.build_hardmix_from_mined',
            '--out',
            args.hardmix_out,
            '--hard',
            args.hard_out,
            '--mega-fast',
            'data/hf_multisource_mega10_fast',
            '--pack',
            'data/roboflow_upload/plant_2f4ay_pack',
            '--gold',
            args.golden_out,
            '--cap-mega-train',
            str(args.cap_mega_train),
            '--cap-mega-val',
            str(args.cap_mega_val),
            '--cap-pack-train',
            str(args.cap_pack_train),
            '--cap-pack-val',
            str(args.cap_pack_val),
            '--cap-gold-train',
            str(args.cap_gold_train),
            '--cap-gold-val',
            str(args.cap_gold_val),
            '--repeat-hard-train',
            str(args.repeat_hard_train),
        ]
        rc, out = run_cmd(hardmix_cmd, cwd=root)
        report['steps']['hardmix'] = {'returncode': rc, 'tail': out[-4000:]}

    train_failed = False
    if not args.skip_train:
        train_cmd = [
            args.python_exe,
            '-m',
            'training.train_maximum',
            '--config',
            args.train_config,
            '--stage1-data',
            f"{args.hardmix_out}/dataset.yaml",
            '--name',
            args.train_name,
        ]
        if args.stage2_data:
            train_cmd.extend(['--stage2-data', args.stage2_data])

        rc, out = run_cmd(train_cmd, cwd=root)
        report['steps']['train'] = {'returncode': rc, 'tail': out[-7000:]}
        train_failed = rc != 0

    run_best = resolve_best_from_run(args.train_name)
    if run_best is not None:
        candidate_path = root / args.candidate_model
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_best, candidate_path)
        report['artifacts']['candidate_model'] = str(candidate_path)

    candidate_metrics = None
    if (root / args.candidate_model).exists():
        candidate_metrics = evaluate_model(
            model_path=(root / args.candidate_model),
            data_yaml=(root / args.benchmark_data),
            imgsz=int(args.eval_imgsz),
            batch=int(args.eval_batch),
            workers=int(args.eval_workers),
        )
        report['candidate_metrics'] = candidate_metrics

    if candidate_metrics is not None:
        report['delta_percent'] = {
            'precision': safe_pct(float(baseline_metrics.get('precision', 0.0)), float(candidate_metrics.get('precision', 0.0))),
            'recall': safe_pct(float(baseline_metrics.get('recall', 0.0)), float(candidate_metrics.get('recall', 0.0))),
            'mAP50': safe_pct(float(baseline_metrics.get('mAP50', 0.0)), float(candidate_metrics.get('mAP50', 0.0))),
            'mAP50_95': safe_pct(float(baseline_metrics.get('mAP50_95', 0.0)), float(candidate_metrics.get('mAP50_95', 0.0))),
            'root_iou': safe_pct(
                float((baseline_metrics.get('iou_per_class') or {}).get('root', 0.0)),
                float((candidate_metrics.get('iou_per_class') or {}).get('root', 0.0)),
            ),
            'stem_iou': safe_pct(
                float((baseline_metrics.get('iou_per_class') or {}).get('stem', 0.0)),
                float((candidate_metrics.get('iou_per_class') or {}).get('stem', 0.0)),
            ),
            'leaves_iou': safe_pct(
                float((baseline_metrics.get('iou_per_class') or {}).get('leaves', 0.0)),
                float((candidate_metrics.get('iou_per_class') or {}).get('leaves', 0.0)),
            ),
        }

    report['status'] = 'warning_train_failed' if train_failed else 'ok'

    out = root / args.report_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
