from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from inference.predictor import Detection


class PlantCVService:
    """Optional PlantCV-powered post-analysis.

    If PlantCV is unavailable, service falls back to lightweight mask analytics.
    """

    def __init__(self) -> None:
        self.available = False
        self._pcv = None
        try:
            from plantcv import plantcv as pcv  # type: ignore

            self._pcv = pcv
            self.available = True
        except Exception:
            self.available = False
            self._pcv = None

    @staticmethod
    def _union_mask(detections: list[Detection], shape: tuple[int, int]) -> np.ndarray:
        h, w = int(shape[0]), int(shape[1])
        union = np.zeros((h, w), dtype=np.uint8)
        for det in detections:
            union = np.maximum(union, (det.mask > 0).astype(np.uint8))
        return union

    def analyze(self, image_bgr: np.ndarray, detections: list[Detection]) -> dict[str, Any]:
        h, w = image_bgr.shape[:2]
        img_area = max(1, int(h) * int(w))
        union = self._union_mask(detections, (h, w))
        plant_area_px = int(union.sum())
        payload: dict[str, Any] = {
            'available': bool(self.available),
            'source': 'plantcv' if self.available else 'fallback',
            'plant_area_px': plant_area_px,
            'plant_area_ratio': round(float(plant_area_px) / float(img_area), 6),
            'notes': [],
        }

        # Lightweight metrics that always work.
        if plant_area_px > 0:
            contours, _ = cv2.findContours((union * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(cnt)
                hull_area = float(cv2.contourArea(hull))
                if hull_area > 0:
                    solidity = float(plant_area_px) / float(hull_area)
                    payload['mask_solidity'] = round(solidity, 6)
        else:
            payload['notes'].append('No plant mask detected.')

        # Optional PlantCV features.
        if not self.available or self._pcv is None:
            return payload

        try:
            # Saturation channel threshold often separates green biomass from background.
            s_channel = self._pcv.rgb2gray_hsv(rgb_img=image_bgr, channel='s')
            thr = self._pcv.threshold.binary(
                gray_img=s_channel,
                threshold=60,
                max_value=255,
                object_type='light',
            )
            thr_mask = (thr > 0).astype(np.uint8)
            payload['plantcv_threshold_area_px'] = int(thr_mask.sum())
            payload['plantcv_threshold_ratio'] = round(float(thr_mask.sum()) / float(img_area), 6)
            payload['notes'].append('PlantCV threshold features extracted.')
        except Exception as exc:
            payload['notes'].append(f'PlantCV runtime fallback: {exc.__class__.__name__}')
        return payload
