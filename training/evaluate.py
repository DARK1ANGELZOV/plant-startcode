from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def extract_metrics(val_result: Any, class_names: dict[int, str]) -> dict[str, Any]:
    seg = getattr(val_result, 'seg', None)
    maps = list(getattr(seg, 'maps', [])) if seg is not None else []

    iou = {}
    for idx, value in enumerate(maps):
        iou[class_names.get(idx, str(idx))] = float(value)

    return {
        'iou_per_class': iou,
        'precision': float(getattr(seg, 'mp', 0.0)) if seg is not None else 0.0,
        'recall': float(getattr(seg, 'mr', 0.0)) if seg is not None else 0.0,
        'mAP50': float(getattr(seg, 'map50', 0.0)) if seg is not None else 0.0,
        'mAP50_95': float(getattr(seg, 'map', 0.0)) if seg is not None else 0.0,
        'confusion_matrix': str(Path(val_result.save_dir) / 'confusion_matrix.png')
        if hasattr(val_result, 'save_dir')
        else '',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate YOLO-seg model.')
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data', required=True, type=str)
    args = parser.parse_args()

    model = YOLO(args.model)
    val_result = model.val(data=args.data, split='val', plots=True, save_json=True)
    metrics = extract_metrics(val_result, model.names)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
