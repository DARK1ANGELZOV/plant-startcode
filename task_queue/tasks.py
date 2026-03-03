from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_subprocess(command: list[str], cwd: str | None = None) -> dict:
    proc = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return {
        'returncode': proc.returncode,
        'stdout_tail': '\n'.join(proc.stdout.splitlines()[-80:]),
        'stderr_tail': '\n'.join(proc.stderr.splitlines()[-80:]),
    }


def run_finetune_job(
    config_path: str,
    data_yaml: str,
    epochs: int,
    batch: int,
    imgsz: int,
    name: str,
    project_root: str,
) -> dict:
    cmd = [
        sys.executable,
        '-m',
        'training.train_yolo_seg',
        '--config',
        config_path,
        '--data',
        data_yaml,
        '--epochs',
        str(epochs),
        '--batch',
        str(batch),
        '--imgsz',
        str(imgsz),
        '--name',
        name,
    ]
    return run_subprocess(cmd, cwd=project_root)


def run_blind_eval_job(
    data_yaml: str,
    split: str,
    max_images: int,
    iou_sla: float,
    project_root: str,
) -> dict:
    cmd = [
        sys.executable,
        '-m',
        'training.evaluate_blind',
        '--data',
        data_yaml,
        '--split',
        split,
        '--max-images',
        str(max_images),
        '--iou-sla',
        str(iou_sla),
    ]
    return run_subprocess(cmd, cwd=project_root)
