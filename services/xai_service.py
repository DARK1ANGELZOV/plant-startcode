from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inference.predictor import Detection
from utils.schemas import ExplainabilityArtifacts


class XAIService:
    @staticmethod
    def _normalize_to_u8(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        lo = float(arr.min()) if arr.size else 0.0
        hi = float(arr.max()) if arr.size else 0.0
        if hi - lo < 1e-8:
            return np.zeros(arr.shape, dtype=np.uint8)
        out = (arr - lo) / (hi - lo)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    def generate(
        self,
        image_bgr: np.ndarray,
        detections: list[Detection],
        run_dir: Path,
        uncertainty_map: np.ndarray | None = None,
    ) -> ExplainabilityArtifacts:
        h, w = image_bgr.shape[:2]
        conf_map = np.zeros((h, w), dtype=np.float32)
        attn_map = np.zeros((h, w), dtype=np.float32)

        for det in detections:
            mask = (det.mask > 0).astype(np.float32)
            conf_map += mask * float(det.confidence)
            attn_map += mask

        conf_u8 = self._normalize_to_u8(conf_map)
        attn_u8 = self._normalize_to_u8(attn_map)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        edges = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        edges = np.abs(edges)
        edges_u8 = self._normalize_to_u8(edges)

        grad_proxy = cv2.addWeighted(conf_u8, 0.7, edges_u8, 0.3, 0.0)

        conf_heat = cv2.applyColorMap(conf_u8, cv2.COLORMAP_TURBO)
        attn_heat = cv2.applyColorMap(attn_u8, cv2.COLORMAP_VIRIDIS)
        grad_heat = cv2.applyColorMap(grad_proxy, cv2.COLORMAP_JET)

        conf_overlay = cv2.addWeighted(image_bgr, 0.55, conf_heat, 0.45, 0.0)
        attn_overlay = cv2.addWeighted(image_bgr, 0.55, attn_heat, 0.45, 0.0)
        grad_overlay = cv2.addWeighted(image_bgr, 0.55, grad_heat, 0.45, 0.0)

        xai_dir = run_dir / 'xai'
        xai_dir.mkdir(parents=True, exist_ok=True)
        conf_path = xai_dir / 'confidence_map.png'
        attn_path = xai_dir / 'attention_heatmap.png'
        grad_path = xai_dir / 'gradcam.png'

        cv2.imwrite(str(conf_path), conf_overlay)
        cv2.imwrite(str(attn_path), attn_overlay)
        cv2.imwrite(str(grad_path), grad_overlay)

        uncertainty_path: str | None = None
        notes = ['Grad-CAM image is proxy-based for YOLO heads (activation + edge evidence).']
        if uncertainty_map is not None and uncertainty_map.size:
            unc_u8 = self._normalize_to_u8(uncertainty_map)
            unc_heat = cv2.applyColorMap(unc_u8, cv2.COLORMAP_INFERNO)
            unc_overlay = cv2.addWeighted(image_bgr, 0.55, unc_heat, 0.45, 0.0)
            unc_path = xai_dir / 'uncertainty_map.png'
            cv2.imwrite(str(unc_path), unc_overlay)
            uncertainty_path = str(unc_path)

        return ExplainabilityArtifacts(
            confidence_map=str(conf_path),
            attention_heatmap=str(attn_path),
            gradcam=str(grad_path),
            uncertainty_map=uncertainty_path,
            notes=notes,
        )
