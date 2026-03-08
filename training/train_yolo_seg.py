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

from utils.config import load_train_config
from utils.seed import set_global_seed

try:
    import torch
except Exception:
    torch = None


def patch_ultralytics_polars() -> None:
    try:
        import polars  # noqa: F401
        return
    except Exception:
        pass

    try:
        from ultralytics.engine.trainer import BaseTrainer
    except Exception:
        return

    if getattr(BaseTrainer, '_agro_polars_patch', False):
        return

    def _read_results_csv(self):
        csv_path = getattr(self, 'csv', None)
        if csv_path is None:
            return {}
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            return {str(k): [float(x) for x in v] for k, v in df.to_dict(orient='list').items()}
        except Exception:
            return {}

    BaseTrainer.read_results_csv = _read_results_csv
    BaseTrainer._agro_polars_patch = True


def polars_available() -> bool:
    try:
        import polars  # noqa: F401

        return True
    except Exception:
        return False


def resolve_device(device_cfg: Any) -> str | int:
    if device_cfg not in (None, 'auto'):
        return device_cfg
    if torch is not None and torch.cuda.is_available():
        return 0
    return 'cpu'


def extract_metrics(val_result: Any, class_names: dict[int, str] | dict[str, str]) -> dict[str, Any]:
    seg = getattr(val_result, 'seg', None)
    iou_per_class: dict[str, float] = {}

    maps = list(getattr(seg, 'maps', [])) if seg is not None else []
    for idx, value in enumerate(maps):
        class_name = class_names.get(idx, str(idx)) if isinstance(class_names, dict) else str(idx)
        iou_per_class[str(class_name)] = float(value)

    metrics = {
        'iou_per_class': iou_per_class,
        'precision': float(getattr(seg, 'mp', 0.0)) if seg is not None else 0.0,
        'recall': float(getattr(seg, 'mr', 0.0)) if seg is not None else 0.0,
        'mAP50': float(getattr(seg, 'map50', 0.0)) if seg is not None else 0.0,
        'mAP50_95': float(getattr(seg, 'map', 0.0)) if seg is not None else 0.0,
        'confusion_matrix': str(Path(val_result.save_dir) / 'confusion_matrix.png')
        if hasattr(val_result, 'save_dir')
        else '',
    }
    return metrics


def resolve_best_weights_path(project: str, name: str, model: YOLO) -> Path:
    candidates = []
    trainer = getattr(model, 'trainer', None)
    if trainer is not None and getattr(trainer, 'save_dir', None):
        candidates.append(Path(trainer.save_dir) / 'weights' / 'best.pt')

    candidates.extend(
        [
            Path(project) / name / 'weights' / 'best.pt',
            Path.home() / project / name / 'weights' / 'best.pt',
            Path.home() / project / project / name / 'weights' / 'best.pt',
        ]
    )

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        'best.pt not found. Checked paths: ' + ', '.join(str(p) for p in candidates)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Train YOLO-seg model for Agro AI System.')
    parser.add_argument('--config', default='configs/train.yaml', type=str)
    parser.add_argument('--data', default=None, type=str)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--batch', default=None, type=int)
    parser.add_argument('--imgsz', default=None, type=int)
    parser.add_argument('--name', default=None, type=str)
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    train_cfg = cfg['train']
    export_cfg = cfg.get('export', {})

    if args.data:
        train_cfg['data_yaml'] = args.data
    if args.epochs is not None:
        train_cfg['epochs'] = args.epochs
    if args.batch is not None:
        train_cfg['batch'] = args.batch
    if args.imgsz is not None:
        train_cfg['imgsz'] = args.imgsz
    if args.name:
        train_cfg['name'] = args.name

    set_global_seed(int(train_cfg.get('seed', 42)))
    patch_ultralytics_polars()
    has_polars = polars_available()

    model = YOLO(train_cfg.get('model', 'yolo11n-seg.pt'))

    train_args = {
        'data': train_cfg['data_yaml'],
        'epochs': int(train_cfg['epochs']),
        'batch': int(train_cfg['batch']),
        'imgsz': int(train_cfg['imgsz']),
        'workers': int(train_cfg.get('workers', 4)),
        'device': resolve_device(train_cfg.get('device', 'auto')),
        'amp': bool(train_cfg.get('amp', True)),
        'project': train_cfg.get('project', 'runs/segment'),
        'name': train_cfg.get('name', 'arugula_wheat_seg'),
        'patience': int(train_cfg.get('patience', 15)),
        'save': bool(train_cfg.get('save', True)),
        'save_period': int(train_cfg.get('save_period', -1)),
        'cache': train_cfg.get('cache', 'ram'),
        'optimizer': train_cfg.get('optimizer', 'auto'),
        'lr0': float(train_cfg.get('lr0', 0.005)),
        'lrf': float(train_cfg.get('lrf', 0.05)),
        'weight_decay': float(train_cfg.get('weight_decay', 0.0005)),
        'warmup_epochs': float(train_cfg.get('warmup_epochs', 3.0)),
        'mosaic': float(train_cfg.get('mosaic', 0.6)),
        'mixup': float(train_cfg.get('mixup', 0.15)),
        'hsv_h': float(train_cfg.get('hsv_h', 0.015)),
        'hsv_s': float(train_cfg.get('hsv_s', 0.7)),
        'hsv_v': float(train_cfg.get('hsv_v', 0.4)),
        'degrees': float(train_cfg.get('degrees', 7.5)),
        'translate': float(train_cfg.get('translate', 0.1)),
        'scale': float(train_cfg.get('scale', 0.2)),
        'shear': float(train_cfg.get('shear', 2.0)),
        'fliplr': float(train_cfg.get('fliplr', 0.5)),
        'flipud': float(train_cfg.get('flipud', 0.0)),
        'copy_paste': float(train_cfg.get('copy_paste', 0.0)),
        'close_mosaic': int(train_cfg.get('close_mosaic', 10)),
        'box': float(train_cfg.get('box', 7.5)),
        'cls': float(train_cfg.get('cls', 0.5)),
        'dfl': float(train_cfg.get('dfl', 1.5)),
        'cos_lr': bool(train_cfg.get('cos_lr', False)),
        'dropout': float(train_cfg.get('dropout', 0.0)),
        'fraction': float(train_cfg.get('fraction', 1.0)),
        'val': bool(train_cfg.get('val', True)),
        'plots': bool(train_cfg.get('plots', False) and has_polars),
        'seed': int(train_cfg.get('seed', 42)),
        'overlap_mask': bool(train_cfg.get('overlap_mask', False)),
        'mask_ratio': int(train_cfg.get('mask_ratio', 4)),
        'single_cls': bool(train_cfg.get('single_cls', False)),
    }

    print('Starting training with params:')
    print(json.dumps(train_args, indent=2))

    model.train(**train_args)

    best_path = resolve_best_weights_path(
        project=str(train_args['project']),
        name=str(train_args['name']),
        model=model,
    )
    run_dir = best_path.parent.parent

    final_model = YOLO(str(best_path))
    val_result = final_model.val(
        data=train_args['data'],
        split='val',
        plots=has_polars,
        save_json=True,
    )

    metrics = extract_metrics(val_result, final_model.names)
    metrics_path = run_dir / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding='utf-8')
    print('Validation metrics:')
    print(json.dumps(metrics, indent=2))

    onnx_path = final_model.export(
        format='onnx',
        dynamic=bool(export_cfg.get('onnx_dynamic', True)),
        simplify=bool(export_cfg.get('simplify', True)),
        imgsz=int(train_cfg['imgsz']),
    )
    print(f'ONNX exported to: {onnx_path}')

    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, models_dir / 'best.pt')

    onnx_src = Path(str(onnx_path))
    if onnx_src.exists():
        shutil.copy2(onnx_src, models_dir / 'best.onnx')

    print('Artifacts copied to models/: best.pt and best.onnx')


if __name__ == '__main__':
    main()
