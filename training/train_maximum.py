from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Workaround for Windows environments where a compatible torch build is installed
# in a short custom site-packages path (e.g. C:\ptpkgs).
_extra_site = os.getenv('AGRO_EXTRA_SITE_PACKAGES', r'C:\ptpkgs').strip()
if _extra_site and os.path.isdir(_extra_site) and _extra_site not in sys.path:
    sys.path.insert(0, _extra_site)

from ultralytics import YOLO

from training.train_yolo_seg import (
    extract_metrics,
    patch_ultralytics_polars,
    polars_available,
    resolve_best_weights_path,
)
from utils.config import load_yaml
from utils.seed import set_global_seed

try:
    import torch
except Exception:
    torch = None


def resolve_device(device_cfg: Any) -> str | int:
    if device_cfg not in (None, 'auto'):
        return device_cfg
    if torch is not None and torch.cuda.is_available():
        return 0
    return 'cpu'


def train_stage(
    model_source: str,
    stage_cfg: dict[str, Any],
    data_yaml: str,
    name: str,
    has_polars: bool,
) -> Path:
    model = YOLO(model_source)

    train_args = {
        'data': data_yaml,
        'epochs': int(stage_cfg['epochs']),
        'batch': int(stage_cfg['batch']),
        'imgsz': int(stage_cfg['imgsz']),
        'device': resolve_device(stage_cfg.get('device', 'auto')),
        'workers': int(stage_cfg.get('workers', 4)),
        'amp': bool(stage_cfg.get('amp', True)),
        'project': stage_cfg.get('project', 'runs/segment'),
        'name': name,
        'patience': int(stage_cfg.get('patience', 20)),
        'optimizer': stage_cfg.get('optimizer', 'auto'),
        'lr0': float(stage_cfg.get('lr0', 0.003)),
        'lrf': float(stage_cfg.get('lrf', 0.05)),
        'weight_decay': float(stage_cfg.get('weight_decay', 0.0005)),
        'warmup_epochs': float(stage_cfg.get('warmup_epochs', 3.0)),
        'mosaic': float(stage_cfg.get('mosaic', 0.5)),
        'mixup': float(stage_cfg.get('mixup', 0.1)),
        'hsv_h': float(stage_cfg.get('hsv_h', 0.015)),
        'hsv_s': float(stage_cfg.get('hsv_s', 0.6)),
        'hsv_v': float(stage_cfg.get('hsv_v', 0.35)),
        'degrees': float(stage_cfg.get('degrees', 5.0)),
        'translate': float(stage_cfg.get('translate', 0.1)),
        'scale': float(stage_cfg.get('scale', 0.2)),
        'fliplr': float(stage_cfg.get('fliplr', 0.5)),
        'flipud': float(stage_cfg.get('flipud', 0.0)),
        'cache': stage_cfg.get('cache', 'ram'),
        'seed': int(stage_cfg.get('seed', 42)),
        'val': True,
        'plots': has_polars,
        'overlap_mask': False,
        'single_cls': False,
    }

    print(f'\\nTraining stage: {name}')
    print(json.dumps(train_args, indent=2, ensure_ascii=True))

    model.train(**train_args)
    best_path = resolve_best_weights_path(
        project=str(train_args['project']),
        name=name,
        model=model,
    )
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Maximum training pipeline for multi-source plant segmentation.')
    parser.add_argument('--config', default='configs/train_max.yaml', type=str)
    parser.add_argument('--stage1-data', default='data/hf_multisource_yoloseg/dataset.yaml', type=str)
    parser.add_argument('--stage2-data', default='', type=str)
    parser.add_argument('--name', default='max_pipeline', type=str)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get('seed', 42))
    set_global_seed(seed)
    patch_ultralytics_polars()
    has_polars = polars_available()

    stage1_cfg = cfg['stage1']
    stage2_cfg = cfg.get('stage2', {})
    export_cfg = cfg.get('export', {})

    best_stage1 = train_stage(
        model_source=stage1_cfg.get('model', 'yolo11s-seg.pt'),
        stage_cfg=stage1_cfg,
        data_yaml=args.stage1_data,
        name=f"{args.name}_stage1",
        has_polars=has_polars,
    )

    final_best = best_stage1
    if args.stage2_data and Path(args.stage2_data).exists() and stage2_cfg.get('enabled', True):
        best_stage2 = train_stage(
            model_source=str(best_stage1),
            stage_cfg=stage2_cfg,
            data_yaml=args.stage2_data,
            name=f"{args.name}_stage2",
            has_polars=has_polars,
        )
        final_best = best_stage2

    final_model = YOLO(str(final_best))
    val_data = args.stage2_data if args.stage2_data and Path(args.stage2_data).exists() else args.stage1_data
    val_result = final_model.val(data=val_data, split='val', plots=has_polars, save_json=True)

    metrics = extract_metrics(val_result, final_model.names)
    out_dir = final_best.parent.parent
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding='utf-8')
    print('Final metrics:')
    print(json.dumps(metrics, indent=2, ensure_ascii=True))

    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_best, models_dir / 'best_max.pt')

    onnx_path: Path | None = None
    try:
        exported = final_model.export(
            format='onnx',
            dynamic=bool(export_cfg.get('dynamic', True)),
            simplify=bool(export_cfg.get('simplify', True)),
            imgsz=int(stage2_cfg.get('imgsz', stage1_cfg.get('imgsz', 640))),
        )
        onnx_path = Path(str(exported))
    except Exception as exc:
        print(f'WARNING: ONNX export failed: {exc}')

    if onnx_path is not None and onnx_path.exists():
        shutil.copy2(onnx_path, models_dir / 'best_max.onnx')
        print('Saved: models/best_max.pt and models/best_max.onnx')
    else:
        print('Saved: models/best_max.pt (ONNX export skipped)')


if __name__ == '__main__':
    main()
