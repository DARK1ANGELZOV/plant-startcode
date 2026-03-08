from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return int(proc.returncode), proc.stdout


def _collect_label_stats(dataset_root: Path) -> dict[str, Any]:
    class_counts = {0: 0, 1: 0, 2: 0}
    split_counts = {'train': 0, 'val': 0}
    for split in ('train', 'val'):
        lbl_dir = dataset_root / 'labels' / split
        if not lbl_dir.exists():
            continue
        files = sorted(lbl_dir.glob('*.txt'))
        split_counts[split] = len(files)
        for lbl in files:
            text = lbl.read_text(encoding='utf-8', errors='ignore').strip()
            if not text:
                continue
            for row in text.splitlines():
                parts = row.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                if cid in class_counts:
                    class_counts[cid] += 1

    nonzero = [v for v in class_counts.values() if v > 0]
    imbalance = (max(nonzero) / min(nonzero)) if nonzero and min(nonzero) > 0 else None
    return {
        'split_label_files': split_counts,
        'class_instances': {
            'root': int(class_counts[0]),
            'stem': int(class_counts[1]),
            'leaves': int(class_counts[2]),
        },
        'class_imbalance_max_div_min': float(imbalance) if imbalance is not None else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-negative + golden(300-500) + manual-review-pack pipeline.")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--python-exe", type=str, default=sys.executable)

    parser.add_argument("--model", type=str, default="models/best_max.pt")
    parser.add_argument("--data-root", type=str, default="data/hf_multisource_mega10")
    parser.add_argument("--hard-out", type=str, default="data/hard_mined_rootstem_rs_v2")
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--val-limit", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--predict-conf", type=float, default=0.01)
    parser.add_argument("--presence-conf", type=float, default=0.10)
    parser.add_argument("--max-det", type=int, default=120)

    parser.add_argument("--golden-out", type=str, default="data/golden_rootstem_500")
    parser.add_argument("--golden-target", type=int, default=500)
    parser.add_argument("--golden-min", type=int, default=300)
    parser.add_argument("--golden-max", type=int, default=500)
    parser.add_argument("--min-class-instances", type=int, default=80)
    parser.add_argument("--max-class-imbalance", type=float, default=12.0)

    parser.add_argument("--review-out", type=str, default="data/golden_rootstem_500_review")
    parser.add_argument("--review-limit", type=int, default=0)
    parser.add_argument("--skip-mine", action="store_true")
    parser.add_argument("--skip-review-pack", action="store_true")
    parser.add_argument("--report-out", type=str, default="reports/hardneg_golden_pipeline_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()

    summary: dict = {
        "timestamp": datetime.now(UTC).isoformat(),
        "params": vars(args),
        "steps": {},
    }

    if not args.skip_mine:
        mine_cmd = [
            args.python_exe,
            "-m",
            "training.mine_hard_negatives_rs",
            "--model",
            args.model,
            "--data-root",
            args.data_root,
            "--out",
            args.hard_out,
            "--train-limit",
            str(args.train_limit),
            "--val-limit",
            str(args.val_limit),
            "--imgsz",
            str(args.imgsz),
            "--predict-conf",
            str(args.predict_conf),
            "--presence-conf",
            str(args.presence_conf),
            "--max-det",
            str(args.max_det),
        ]
        rc, out = run_cmd(mine_cmd, cwd=root)
        summary["steps"]["mine_hardneg"] = {"returncode": rc, "tail": out[-12000:]}
        if rc != 0:
            raise RuntimeError("Hard-negative mining failed.")

    build_cmd = [
        args.python_exe,
        "-m",
        "training.build_golden_from_hard_mining",
        "--report",
        f"{args.hard_out}/mining_report.json",
        "--hard-root",
        args.hard_out,
        "--out",
        args.golden_out,
        "--target",
        str(args.golden_target),
        "--min-target",
        str(args.golden_min),
        "--max-target",
        str(args.golden_max),
        "--strict-min",
    ]
    rc, out = run_cmd(build_cmd, cwd=root)
    summary["steps"]["build_golden"] = {"returncode": rc, "tail": out[-12000:]}
    if rc != 0:
        raise RuntimeError("Golden build failed.")

    golden_root = root / args.golden_out
    golden_stats = _collect_label_stats(golden_root)
    summary['golden_stats'] = golden_stats
    cls = golden_stats.get('class_instances', {})
    min_per_class = int(args.min_class_instances)
    if any(int(cls.get(k, 0)) < min_per_class for k in ('root', 'stem', 'leaves')):
        raise RuntimeError(
            f"Golden coverage is too low: {cls}. "
            f"Expected >= {min_per_class} instances per class."
        )
    imbalance = golden_stats.get('class_imbalance_max_div_min')
    if imbalance is not None and float(imbalance) > float(args.max_class_imbalance):
        raise RuntimeError(
            f"Golden class imbalance is too high: {imbalance:.3f} > {float(args.max_class_imbalance):.3f}"
        )

    if not args.skip_review_pack:
        review_cmd = [
            args.python_exe,
            "-m",
            "training.prepare_manual_review_pack",
            "--golden-root",
            args.golden_out,
            "--out",
            args.review_out,
        ]
        if int(args.review_limit) > 0:
            review_cmd.extend(["--limit", str(args.review_limit)])

        rc, out = run_cmd(review_cmd, cwd=root)
        summary["steps"]["prepare_review_pack"] = {"returncode": rc, "tail": out[-12000:]}
        if rc != 0:
            raise RuntimeError("Review pack build failed.")

    report_path = root / args.report_out
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
