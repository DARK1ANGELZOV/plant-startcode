from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from inference.predictor import Detection
from utils.schemas import RobustnessPoint, RobustnessResponse


class RobustnessService:
    def __init__(self, model_service, class_colors: dict[str, tuple[int, int, int]], overlay_alpha: float = 0.45) -> None:
        self.model_service = model_service
        self.class_colors = class_colors
        self.overlay_alpha = float(overlay_alpha)

    @staticmethod
    def _detections_to_class_masks(
        detections: list[Detection],
        shape: tuple[int, int],
    ) -> dict[str, np.ndarray]:
        h, w = shape
        out: dict[str, np.ndarray] = {}
        for det in detections:
            name = str(det.class_name)
            mask = (det.mask > 0).astype(np.uint8)
            if name not in out:
                out[name] = np.zeros((h, w), dtype=np.uint8)
            out[name] = np.maximum(out[name], mask)
        return out

    @staticmethod
    def _miou(base: dict[str, np.ndarray], cur: dict[str, np.ndarray], classes: list[str]) -> float:
        vals = []
        for cls in classes:
            a = base.get(cls)
            b = cur.get(cls)
            if a is None and b is None:
                vals.append(1.0)
                continue
            if a is None:
                a = np.zeros_like(b, dtype=np.uint8)
            if b is None:
                b = np.zeros_like(a, dtype=np.uint8)
            inter = float(np.logical_and(a > 0, b > 0).sum())
            union = float(np.logical_or(a > 0, b > 0).sum())
            vals.append(inter / union if union > 0 else 1.0)
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _blur(img: np.ndarray, level: int) -> np.ndarray:
        k = max(3, level * 2 + 1)
        return cv2.GaussianBlur(img, (k, k), 0)

    @staticmethod
    def _noise(img: np.ndarray, level: int) -> np.ndarray:
        std = 4.0 + level * 3.5
        noise = np.random.normal(0, std, size=img.shape).astype(np.float32)
        out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def _brightness_drop(img: np.ndarray, level: int) -> np.ndarray:
        factor = max(0.2, 1.0 - level * 0.07)
        out = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def _contrast_distortion(img: np.ndarray, level: int) -> np.ndarray:
        alpha = 1.0 + (level - 5) * 0.08
        beta = (level - 5) * 2.0
        out = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return out

    async def _predict_image(self, image_bgr: np.ndarray) -> list[Detection]:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), image_bgr)
            result = await self.model_service.predict(
                image_path=str(tmp_path),
                class_colors=self.class_colors,
                overlay_alpha=self.overlay_alpha,
            )
            return result['detections']
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    async def evaluate(self, image_bgr: np.ndarray, run_id: str) -> RobustnessResponse:
        if image_bgr is None:
            raise ValueError('Input image is empty.')

        base_dets = await self._predict_image(image_bgr)
        classes = sorted({str(d.class_name) for d in base_dets}) or ['root', 'stem', 'leaves']
        base_masks = self._detections_to_class_masks(base_dets, image_bgr.shape[:2])

        stress_ops: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
            'blur': self._blur,
            'noise': self._noise,
            'brightness_drop': self._brightness_drop,
            'contrast_distortion': self._contrast_distortion,
        }

        curves: dict[str, list[RobustnessPoint]] = {}
        all_vals: list[float] = []

        for name, fn in stress_ops.items():
            pts: list[RobustnessPoint] = []
            for level in range(1, 11):
                stressed = fn(image_bgr, level)
                dets = await self._predict_image(stressed)
                cur_masks = self._detections_to_class_masks(dets, image_bgr.shape[:2])
                miou = self._miou(base_masks, cur_masks, classes)
                score = max(0.0, 100.0 * miou)
                pts.append(RobustnessPoint(level=level, miou=round(miou, 6), score=round(score, 4)))
                all_vals.append(miou)
            curves[name] = pts

        robustness_score = float(np.mean(all_vals) * 100.0) if all_vals else 0.0
        return RobustnessResponse(
            run_id=run_id,
            baseline_instances=len(base_dets),
            curves=curves,
            robustness_score=round(robustness_score, 4),
            files={},
        )
