from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


class BlindEvaluationService:
    def __init__(self, model_service, class_colors: dict[str, tuple[int, int, int]]) -> None:
        self.model_service = model_service
        self.class_colors = class_colors

    @staticmethod
    def _resolve_split_path(data_cfg: dict[str, Any], split: str, root: Path, data_yaml_parent: Path) -> Path:
        split_path = data_cfg.get(split)
        if not split_path:
            raise ValueError(f'Split "{split}" is not defined in dataset yaml.')

        p = Path(split_path)
        if p.is_absolute():
            return p

        for candidate in (p, root / p, data_yaml_parent / p, Path.cwd() / p):
            if candidate.exists():
                return candidate.resolve()
        return (root / p).resolve()

    @staticmethod
    def _parse_names(data_cfg: dict[str, Any]) -> dict[int, str]:
        names_cfg = data_cfg.get('names', {})
        if isinstance(names_cfg, dict):
            return {int(k): str(v) for k, v in names_cfg.items()}
        if isinstance(names_cfg, list):
            return {idx: str(name) for idx, name in enumerate(names_cfg)}
        raise ValueError('Dataset yaml names field is invalid.')

    @staticmethod
    def _label_path_from_image(image_path: Path) -> Path:
        parts = list(image_path.parts)
        try:
            idx = parts.index('images')
            parts[idx] = 'labels'
        except ValueError:
            pass
        return Path(*parts).with_suffix('.txt')

    @staticmethod
    def _load_gt_masks(label_path: Path, img_w: int, img_h: int, id_to_name: dict[int, str]) -> dict[str, np.ndarray]:
        masks: dict[str, np.ndarray] = {name: np.zeros((img_h, img_w), dtype=np.uint8) for name in id_to_name.values()}

        if not label_path.exists():
            return masks

        for raw in label_path.read_text(encoding='utf-8').splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue

            class_id = int(float(parts[0]))
            class_name = id_to_name.get(class_id, f'class_{class_id}')
            coords = [float(x) for x in parts[1:]]
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue

            pts = []
            for i in range(0, len(coords), 2):
                x = int(np.clip(coords[i] * img_w, 0, img_w - 1))
                y = int(np.clip(coords[i + 1] * img_h, 0, img_h - 1))
                pts.append([x, y])

            if len(pts) >= 3:
                cv2.fillPoly(masks[class_name], [np.array(pts, dtype=np.int32)], 1)

        return masks

    @staticmethod
    def _summarize(stats: dict[str, dict[str, float]], iou_sla: float) -> dict[str, Any]:
        per_class: dict[str, Any] = {}
        macro = {'iou': [], 'precision': [], 'recall': [], 'f1': []}

        for class_name, s in stats.items():
            inter = s['intersection']
            union = s['union']
            pred = s['pred']
            gt = s['gt']

            iou = inter / union if union > 0 else 0.0
            precision = inter / pred if pred > 0 else 0.0
            recall = inter / gt if gt > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            per_class[class_name] = {
                'iou': round(iou, 6),
                'precision': round(precision, 6),
                'recall': round(recall, 6),
                'f1': round(f1, 6),
                'sla_pass': bool(iou >= iou_sla),
            }
            macro['iou'].append(iou)
            macro['precision'].append(precision)
            macro['recall'].append(recall)
            macro['f1'].append(f1)

        return {
            'per_class': per_class,
            'macro': {
                'iou': round(float(np.mean(macro['iou'])) if macro['iou'] else 0.0, 6),
                'precision': round(float(np.mean(macro['precision'])) if macro['precision'] else 0.0, 6),
                'recall': round(float(np.mean(macro['recall'])) if macro['recall'] else 0.0, 6),
                'f1': round(float(np.mean(macro['f1'])) if macro['f1'] else 0.0, 6),
            },
        }

    async def evaluate(
        self,
        data_yaml: str,
        split: str = 'val',
        max_images: int = 200,
        iou_sla: float = 0.5,
        conf: float = 0.05,
        iou: float = 0.5,
        max_det: int = 200,
    ) -> dict[str, Any]:
        data_path = Path(data_yaml)
        if not data_path.exists():
            raise ValueError(f'Dataset yaml not found: {data_yaml}')

        cfg = yaml.safe_load(data_path.read_text(encoding='utf-8')) or {}
        raw_root = Path(cfg.get('path', data_path.parent))
        if raw_root.is_absolute():
            root = raw_root
        elif raw_root.exists():
            root = raw_root.resolve()
        elif (data_path.parent / raw_root).exists():
            root = (data_path.parent / raw_root).resolve()
        else:
            root = (data_path.parent / raw_root).resolve()

        split_dir = self._resolve_split_path(cfg, split, root, data_path.parent)
        if not split_dir.exists():
            raise ValueError(f'Split directory not found: {split_dir}')

        id_to_name = self._parse_names(cfg)
        image_paths = [p for p in sorted(split_dir.rglob('*')) if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
        image_paths = image_paths[: max(1, max_images)]

        stats = {
            name: {'intersection': 0.0, 'union': 0.0, 'pred': 0.0, 'gt': 0.0}
            for name in id_to_name.values()
        }

        for image_path in image_paths:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            gt_masks = self._load_gt_masks(self._label_path_from_image(image_path), w, h, id_to_name)

            pred = await self.model_service.predict(
                image_path=str(image_path),
                class_colors=self.class_colors,
                overlay_alpha=0.35,
                conf=conf,
                iou=iou,
                max_det=max_det,
            )
            pred_masks = {name: np.zeros((h, w), dtype=np.uint8) for name in id_to_name.values()}
            for det in pred.get('detections', []):
                name = str(det.class_name)
                if name not in pred_masks:
                    continue
                pred_masks[name] = np.maximum(pred_masks[name], (det.mask > 0).astype(np.uint8))

            for class_name in stats.keys():
                gt_m = gt_masks.get(class_name, np.zeros((h, w), dtype=np.uint8))
                pr_m = pred_masks.get(class_name, np.zeros((h, w), dtype=np.uint8))
                inter = float(np.logical_and(gt_m > 0, pr_m > 0).sum())
                union = float(np.logical_or(gt_m > 0, pr_m > 0).sum())
                stats[class_name]['intersection'] += inter
                stats[class_name]['union'] += union
                stats[class_name]['pred'] += float((pr_m > 0).sum())
                stats[class_name]['gt'] += float((gt_m > 0).sum())

        summary = self._summarize(stats, iou_sla=iou_sla)
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'dataset_yaml': str(data_path),
            'split': split,
            'images_evaluated': len(image_paths),
            'iou_sla': iou_sla,
            **summary,
        }

        out_path = Path('reports') / f"blind_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(__import__('json').dumps(report, ensure_ascii=True, indent=2), encoding='utf-8')
        report['report_path'] = str(out_path)
        return report
