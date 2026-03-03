from __future__ import annotations

import argparse
import asyncio
import json

from services.evaluation_service import BlindEvaluationService
from services.model_service import ModelService
from utils.config import load_app_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Blind evaluation with per-class SLA checks.')
    p.add_argument('--data', required=True, help='Path to YOLO dataset yaml')
    p.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    p.add_argument('--max-images', type=int, default=200)
    p.add_argument('--iou-sla', type=float, default=0.5)
    p.add_argument('--conf', type=float, default=0.05)
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--max-det', type=int, default=200)
    p.add_argument('--config', default='configs/app.yaml')
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    cfg = load_app_config(args.config)

    model_service = ModelService(cfg)
    await model_service.load()
    if not model_service.is_loaded():
        raise RuntimeError('Model not loaded.')

    class_colors = {k: tuple(v) for k, v in cfg.get('inference', {}).get('class_colors', {}).items()}
    evaluator = BlindEvaluationService(model_service=model_service, class_colors=class_colors)

    report = await evaluator.evaluate(
        data_yaml=args.data,
        split=args.split,
        max_images=args.max_images,
        iou_sla=args.iou_sla,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
