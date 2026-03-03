from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from inference.predictor import Detection


def _draw_legend(
    image: np.ndarray,
    class_colors: dict[str, tuple[int, int, int]],
) -> None:
    x = 15
    y = 20
    for class_name, color in class_colors.items():
        cv2.rectangle(image, (x, y), (x + 18, y + 18), color, thickness=-1)
        cv2.putText(
            image,
            class_name,
            (x + 26, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24


def _collect_names(result: Any) -> dict[int, str]:
    names = getattr(result, 'names', {})
    if isinstance(names, dict):
        out = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    return {}


def _class_count_from_names(names: dict[int, str]) -> int:
    if not names:
        return 3
    return max(1, max(names.keys()) + 1)


def run_ensemble_inference(
    models: list[Any],
    image_path: str,
    class_colors: dict[str, tuple[int, int, int]],
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    max_det: int = 200,
    overlay_alpha: float = 0.45,
    device: str = 'cpu',
) -> dict[str, Any]:
    if not models:
        raise ValueError('No models were provided for ensemble inference.')

    per_model_results = []
    names_map: dict[int, str] = {}
    image = None

    for model in models:
        rs = model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            retina_masks=True,
            verbose=False,
            device=device,
        )
        if not rs:
            continue
        r = rs[0]
        per_model_results.append(r)
        if image is None:
            image = r.orig_img.copy()
        names_map.update(_collect_names(r))

    if not per_model_results or image is None:
        raise RuntimeError('Ensemble inference returned empty result set.')

    h, w = image.shape[:2]
    n_classes = _class_count_from_names(names_map)
    score_maps = np.zeros((n_classes, h, w), dtype=np.float32)

    for r in per_model_results:
        boxes = r.boxes
        masks = r.masks
        if boxes is None or masks is None or masks.data is None:
            continue
        cls_array = boxes.cls.cpu().numpy().astype(int)
        conf_array = boxes.conf.cpu().numpy().astype(np.float32)
        mask_array = masks.data.cpu().numpy().astype(np.uint8)

        for idx, mask in enumerate(mask_array):
            if idx >= len(cls_array) or idx >= len(conf_array):
                continue
            cls_id = int(cls_array[idx])
            if cls_id < 0 or cls_id >= n_classes:
                continue
            score_maps[cls_id] += (mask > 0).astype(np.float32) * float(conf_array[idx])

    sum_scores = np.sum(score_maps, axis=0)
    class_idx = np.argmax(score_maps, axis=0).astype(np.int32)
    confidence_map = np.max(score_maps, axis=0)
    uncertainty_map = 1.0 - (confidence_map / (sum_scores + 1e-6))
    uncertainty_map = np.clip(uncertainty_map, 0.0, 1.0)

    detections: list[Detection] = []
    overlay = image.copy()
    instance_id = 0
    min_area = 20

    for cls_id in range(n_classes):
        cls_mask = np.logical_and(class_idx == cls_id, sum_scores > 0).astype(np.uint8)
        if cls_mask.sum() < min_area:
            continue

        num_labels, labels = cv2.connectedComponents(cls_mask)
        class_name = names_map.get(cls_id, f'class_{cls_id}')
        color = class_colors.get(class_name, (255, 255, 255))
        for lbl in range(1, num_labels):
            comp = (labels == lbl).astype(np.uint8)
            if int(comp.sum()) < min_area:
                continue
            ys, xs = np.where(comp > 0)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            conf_val = float(np.mean(confidence_map[comp > 0]) / max(1, len(per_model_results)))

            colored = np.zeros_like(overlay, dtype=np.uint8)
            colored[comp > 0] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, overlay_alpha, 0.0)

            detections.append(
                Detection(
                    instance_id=instance_id,
                    class_id=int(cls_id),
                    class_name=class_name,
                    confidence=float(np.clip(conf_val, 0.0, 1.0)),
                    bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
                    mask=comp,
                )
            )
            instance_id += 1

    _draw_legend(overlay, class_colors)
    return {
        'image': image,
        'overlay': overlay,
        'detections': detections,
        'uncertainty_map': uncertainty_map,
        'confidence_map': confidence_map,
    }
