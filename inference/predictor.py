from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class Detection:
    instance_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    mask: np.ndarray


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


def run_yolo_inference(
    model: Any,
    image_path: str,
    class_colors: dict[str, tuple[int, int, int]],
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 640,
    max_det: int = 200,
    overlay_alpha: float = 0.45,
    device: str = 'cpu',
) -> dict[str, Any]:
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        retina_masks=True,
        verbose=False,
        device=device,
    )
    if not results:
        raise RuntimeError('YOLO inference returned empty result set.')

    result = results[0]
    image = result.orig_img.copy()
    overlay = image.copy()

    names_map = result.names if isinstance(result.names, dict) else {}
    boxes = result.boxes
    masks = result.masks

    cls_array = boxes.cls.cpu().numpy().astype(int) if boxes is not None else np.array([])
    conf_array = boxes.conf.cpu().numpy() if boxes is not None else np.array([])
    bbox_array = boxes.xyxy.cpu().numpy() if boxes is not None else np.zeros((0, 4), dtype=float)

    mask_array = (
        masks.data.cpu().numpy().astype(np.uint8)
        if masks is not None and masks.data is not None
        else np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)
    )

    detections: list[Detection] = []
    for idx, mask in enumerate(mask_array):
        class_id = int(cls_array[idx]) if idx < len(cls_array) else -1
        class_name = names_map.get(class_id, f'class_{class_id}')
        confidence = float(conf_array[idx]) if idx < len(conf_array) else 0.0
        bbox = bbox_array[idx].tolist() if idx < len(bbox_array) else [0.0, 0.0, 0.0, 0.0]

        binary_mask = (mask > 0).astype(np.uint8)
        color = class_colors.get(class_name, (255, 255, 255))

        colored = np.zeros_like(overlay, dtype=np.uint8)
        colored[binary_mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, overlay_alpha, 0.0)

        detections.append(
            Detection(
                instance_id=idx,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox_xyxy=[float(v) for v in bbox],
                mask=binary_mask,
            )
        )

    _draw_legend(overlay, class_colors)
    return {
        'image': image,
        'overlay': overlay,
        'detections': detections,
    }
