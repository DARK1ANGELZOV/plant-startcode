from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from training.train_yolo_seg import (
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


def extract_yolo_seg_metrics(val_result: Any, class_names: dict[int, str] | dict[str, str]) -> dict[str, Any]:
    seg = getattr(val_result, 'seg', None)
    maps = list(getattr(seg, 'maps', [])) if seg is not None else []

    per_class = {}
    for idx, value in enumerate(maps):
        name = class_names.get(idx, str(idx)) if isinstance(class_names, dict) else str(idx)
        per_class[str(name)] = float(value)

    miou = float(np.mean(maps)) if maps else 0.0
    return {
        'miou': miou,
        'precision': float(getattr(seg, 'mp', 0.0)) if seg is not None else 0.0,
        'recall': float(getattr(seg, 'mr', 0.0)) if seg is not None else 0.0,
        'map50': float(getattr(seg, 'map50', 0.0)) if seg is not None else 0.0,
        'map50_95': float(getattr(seg, 'map', 0.0)) if seg is not None else 0.0,
        'per_class_iou': per_class,
    }


def _apply_corruption(img: np.ndarray, name: str) -> np.ndarray:
    name = name.lower()
    if name == 'gaussian_blur':
        return cv2.GaussianBlur(img, (5, 5), 0)
    if name == 'motion_blur':
        k = 9
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        return cv2.filter2D(img, -1, kernel)
    if name == 'noise':
        noise = np.random.normal(0, 18, size=img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if name == 'fog':
        h, w = img.shape[:2]
        y = np.linspace(-1, 1, h).reshape(h, 1)
        x = np.linspace(-1, 1, w).reshape(1, w)
        dist = np.sqrt(x * x + y * y)
        mask = np.clip(1.0 - dist, 0.0, 1.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=15)
        fog = np.full_like(img, 230)
        out = img.astype(np.float32) * (1 - 0.35 * mask[..., None]) + fog.astype(np.float32) * (0.35 * mask[..., None])
        return np.clip(out, 0, 255).astype(np.uint8)
    if name == 'rain':
        out = img.copy()
        h, w = out.shape[:2]
        layer = np.zeros_like(out)
        for _ in range(900):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            x2 = min(w - 1, x + 2)
            y2 = min(h - 1, y + np.random.randint(6, 16))
            cv2.line(layer, (x, y), (x2, y2), (190, 190, 190), 1)
        return cv2.addWeighted(out, 1.0, layer, 0.25, 0)
    if name == 'jpeg':
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 28])
        if not ok:
            return img
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return dec if dec is not None else img
    return img


def build_corrupted_val_dataset(data_yaml: str, corruption: str, work_dir: Path) -> Path:
    payload = yaml.safe_load(Path(data_yaml).read_text(encoding='utf-8')) or {}
    root = Path(payload.get('path', Path(data_yaml).parent))
    if not root.is_absolute():
        root = (Path(data_yaml).parent / root).resolve()

    val_rel = Path(payload.get('val', 'images/val'))
    val_dir = root / val_rel
    labels_dir = root / 'labels' / val_rel.name

    out_root = work_dir / f'corrupted_{corruption}'
    out_images = out_root / 'images' / 'val'
    out_labels = out_root / 'labels' / 'val'
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(val_dir.rglob('*')):
        if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        corr = _apply_corruption(image, corruption)
        cv2.imwrite(str(out_images / img_path.name), corr)

        lbl = labels_dir / f'{img_path.stem}.txt'
        if lbl.exists():
            shutil.copy2(lbl, out_labels / lbl.name)

    out_yaml = {
        'path': str(out_root.resolve()).replace('\\', '/'),
        'train': 'images/val',
        'val': 'images/val',
        'names': payload.get('names', {0: 'root', 1: 'stem', 2: 'leaves'}),
    }
    out_yaml_path = out_root / 'dataset.yaml'
    out_yaml_path.write_text(yaml.safe_dump(out_yaml, sort_keys=False), encoding='utf-8')
    return out_yaml_path


def train_stage(
    model_source: str,
    data_yaml: str,
    stage_cfg: dict[str, Any],
    stage_name: str,
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
        'project': stage_cfg.get('project', 'runs/robust/yolo'),
        'name': stage_name,
        'patience': int(stage_cfg.get('patience', 10)),
        'optimizer': stage_cfg.get('optimizer', 'auto'),
        'lr0': float(stage_cfg.get('lr0', 0.002)),
        'lrf': float(stage_cfg.get('lrf', 0.05)),
        'weight_decay': float(stage_cfg.get('weight_decay', 0.0005)),
        'warmup_epochs': float(stage_cfg.get('warmup_epochs', 2.0)),
        'mosaic': float(stage_cfg.get('mosaic', 0.5)),
        'mixup': float(stage_cfg.get('mixup', 0.15)),
        'hsv_h': float(stage_cfg.get('hsv_h', 0.015)),
        'hsv_s': float(stage_cfg.get('hsv_s', 0.7)),
        'hsv_v': float(stage_cfg.get('hsv_v', 0.4)),
        'degrees': float(stage_cfg.get('degrees', 8.0)),
        'translate': float(stage_cfg.get('translate', 0.1)),
        'scale': float(stage_cfg.get('scale', 0.2)),
        'shear': float(stage_cfg.get('shear', 2.0)),
        'fliplr': float(stage_cfg.get('fliplr', 0.5)),
        'flipud': float(stage_cfg.get('flipud', 0.0)),
        'cache': stage_cfg.get('cache', 'ram'),
        'seed': int(stage_cfg.get('seed', 42)),
        'val': True,
        'plots': has_polars,
        'overlap_mask': False,
        'single_cls': False,
    }
    model.train(**train_args)
    return resolve_best_weights_path(project=str(train_args['project']), name=stage_name, model=model)


def main() -> None:
    parser = argparse.ArgumentParser(description='Robust YOLO-seg progressive training and robustness validation.')
    parser.add_argument('--config', default='configs/robust_train.yaml')
    parser.add_argument('--pretrain-data', default='data/hf_multisource_medium/dataset.yaml')
    parser.add_argument('--plant-data', default='data/hf_multisource_medium/dataset.yaml')
    parser.add_argument('--adverse-data', default='')
    parser.add_argument('--name', default='yolo_robust_pipeline')
    parser.add_argument('--output', default='models/robust')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_global_seed(int(cfg.get('seed', 42)))
    patch_ultralytics_polars()
    has_polars = polars_available()

    ycfg = cfg.get('architectures', {}).get('yolo_seg', {})
    stage_base = {
        'epochs': int(ycfg.get('epochs', 25)),
        'batch': int(ycfg.get('batch', 4)),
        'imgsz': int(ycfg.get('imgsz', 1024)),
        'project': str(ycfg.get('project', 'runs/robust/yolo')),
        'device': cfg.get('hardware', {}).get('device', 'auto'),
        'workers': cfg.get('hardware', {}).get('num_workers', 4),
        'amp': cfg.get('hardware', {}).get('amp', True),
        'cache': 'ram',
        'mosaic': 0.6,
        'mixup': 0.15,
    }

    model_source = str(ycfg.get('model', 'yolo11s-seg.pt'))

    stage1 = dict(stage_base)
    stage1['epochs'] = max(1, int(stage_base['epochs'] * 0.4))
    best1 = train_stage(
        model_source,
        args.pretrain_data,
        stage1,
        f'{args.name}_stage1_pretrain',
        has_polars=has_polars,
    )

    stage2 = dict(stage_base)
    stage2['epochs'] = max(1, int(stage_base['epochs'] * 0.35))
    stage2['mosaic'] = 0.45
    best2 = train_stage(
        str(best1),
        args.plant_data,
        stage2,
        f'{args.name}_stage2_plant',
        has_polars=has_polars,
    )

    stage3 = dict(stage_base)
    stage3['epochs'] = max(1, int(stage_base['epochs'] * 0.25))
    stage3['mosaic'] = 0.2
    stage3['mixup'] = 0.1
    final_data = args.adverse_data if args.adverse_data and Path(args.adverse_data).exists() else args.plant_data
    best3 = train_stage(
        str(best2),
        final_data,
        stage3,
        f'{args.name}_stage3_adverse',
        has_polars=has_polars,
    )

    model = YOLO(str(best3))
    clean_val = model.val(data=args.plant_data, split='val', plots=has_polars, save_json=True)
    clean_metrics = extract_yolo_seg_metrics(clean_val, model.names)

    corr_results = {}
    with tempfile.TemporaryDirectory(prefix='robust_yolo_') as td:
        work = Path(td)
        for corr in cfg.get('validation', {}).get('corruption_tests', []):
            corr_yaml = build_corrupted_val_dataset(args.plant_data, corr, work)
            val_res = model.val(data=str(corr_yaml), split='val', plots=False, save_json=False)
            corr_results[corr] = extract_yolo_seg_metrics(val_res, model.names)

    drops = {k: max(0.0, clean_metrics['miou'] - v['miou']) for k, v in corr_results.items()}
    mean_drop = float(np.mean(list(drops.values()))) if drops else 0.0
    robust_score = max(0.0, clean_metrics['miou'] * (1.0 - mean_drop))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_pt = output_dir / 'yolo_robust_best.pt'
    shutil.copy2(best3, target_pt)

    onnx_path = model.export(
        format='onnx',
        dynamic=bool(cfg.get('export', {}).get('onnx_dynamic', True)),
        simplify=True,
        imgsz=int(ycfg.get('imgsz', 1024)),
    )

    if bool(cfg.get('export', {}).get('tensorrt', True)) and torch is not None and torch.cuda.is_available():
        try:
            model.export(format='engine', imgsz=int(ycfg.get('imgsz', 1024)))
        except Exception:
            pass

    report = {
        'best_checkpoint': str(target_pt.resolve()),
        'onnx_path': str(onnx_path),
        'clean': clean_metrics,
        'corrupted': corr_results,
        'robustness': {
            'drop_by_corruption': drops,
            'mean_drop': mean_drop,
            'robustness_score': robust_score,
        },
        'stages': {
            'stage1': str(best1),
            'stage2': str(best2),
            'stage3': str(best3),
        },
    }

    report_path = output_dir / 'yolo_metrics.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
