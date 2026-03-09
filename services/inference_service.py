from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inference.predictor import Detection
from morphometry.analysis import analyze_mask, path_length_between_points
from services.active_learning_service import ActiveLearningService
from services.disease_service import DiseaseService
from services.phi_service import PHIService
from services.plantcv_service import PlantCVService
from services.report_service import ReportService
from services.storage_service import StorageService
from services.xai_service import XAIService
from utils.errors import ModelNotLoadedError
from utils.image_io import decode_image_bytes
from utils.schemas import PredictResponse


logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        model_service,
        calibrator,
        storage: StorageService,
        reporter: ReportService,
        recommender,
        config: dict[str, Any],
        history_service=None,
        disease_service: DiseaseService | None = None,
        phi_service: PHIService | None = None,
        plantcv_service: PlantCVService | None = None,
        xai_service: XAIService | None = None,
        active_learning_service: ActiveLearningService | None = None,
    ) -> None:
        self.model_service = model_service
        self.calibrator = calibrator
        self.storage = storage
        self.reporter = reporter
        self.recommender = recommender
        self.config = config
        self.history_service = history_service
        self.disease_service = disease_service or DiseaseService()
        self.phi_service = phi_service or PHIService(
            config.get('morphometry', {}).get('recommendation_thresholds', {})
        )
        self.plantcv_service = plantcv_service or PlantCVService()
        self.xai_service = xai_service or XAIService()
        al_cfg = config.get('active_learning', {})
        self.active_learning_service = active_learning_service or ActiveLearningService(
            root_dir=str(al_cfg.get('root_dir', 'data/active_learning')),
            low_conf_threshold=float(al_cfg.get('low_conf_threshold', 0.12)),
        )

    @staticmethod
    def _largest_components(mask_u8: np.ndarray, keep: int = 3) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n <= 1:
            return mask_u8
        areas = []
        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            areas.append((area, idx))
        areas.sort(reverse=True)
        keep_ids = {idx for _, idx in areas[:max(1, keep)]}
        out = np.zeros_like(mask_u8)
        for idx in keep_ids:
            out[labels == idx] = 1
        return out

    @staticmethod
    def _cleanup_mask(
        mask_u8: np.ndarray,
        *,
        min_area: int = 20,
        keep: int = 3,
        k_open: int = 3,
        k_close: int = 5,
    ) -> np.ndarray:
        out = (mask_u8 > 0).astype(np.uint8)
        if int(out.sum()) == 0:
            return out
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
        out = InferenceService._largest_components(out, keep=keep)
        if min_area <= 1:
            return out

        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        if n <= 1:
            return out if int(out.sum()) >= int(min_area) else np.zeros_like(out)
        clean = np.zeros_like(out)
        for idx in range(1, n):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area >= int(min_area):
                clean[labels == idx] = 1
        return clean

    @staticmethod
    def _skeletonize(mask_u8: np.ndarray) -> np.ndarray:
        src = ((mask_u8 > 0).astype(np.uint8) * 255)
        skel = np.zeros_like(src)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            opened = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
            temp = cv2.subtract(src, opened)
            eroded = cv2.erode(src, kernel)
            skel = cv2.bitwise_or(skel, temp)
            src = eroded
            if cv2.countNonZero(src) == 0:
                break
        return (skel > 0).astype(np.uint8)

    @staticmethod
    def _mask_bbox(mask_u8: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask_u8 > 0)
        if len(xs) == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    @staticmethod
    def _component_at_point(mask_u8: np.ndarray, point_xy: tuple[int, int]) -> np.ndarray:
        if int(mask_u8.sum()) == 0:
            return mask_u8
        x, y = int(point_xy[0]), int(point_xy[1])
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8.astype(np.uint8), connectivity=8)
        if n <= 1:
            return mask_u8
        if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
            target = int(labels[y, x])
            if target > 0:
                out = np.zeros_like(mask_u8, dtype=np.uint8)
                out[labels == target] = 1
                return out
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.zeros_like(mask_u8, dtype=np.uint8)
        out[labels == largest] = 1
        return out

    @staticmethod
    def _detection_major_axis_px(det: Detection) -> float:
        x1, y1, x2, y2 = det.bbox_xyxy
        bbox_w = max(1.0, float(x2) - float(x1))
        bbox_h = max(1.0, float(y2) - float(y1))
        bbox_major_px = max(bbox_w, bbox_h)
        mask = (det.mask > 0).astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return float(bbox_major_px)
        diag = math.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
        return float(max(bbox_major_px, diag))

    @staticmethod
    def _merge_class_mask(detections: list[Detection], class_name: str, min_confidence: float = 0.0) -> np.ndarray | None:
        masks = [
            (d.mask > 0).astype(np.uint8)
            for d in detections
            if d.class_name == class_name and float(d.confidence) >= float(min_confidence)
        ]
        if not masks:
            return None
        merged = masks[0].copy()
        for m in masks[1:]:
            merged = np.maximum(merged, m)
        return merged

    @staticmethod
    def _point_lowest(mask_u8: np.ndarray) -> tuple[int, int] | None:
        ys, xs = np.where(mask_u8 > 0)
        if len(xs) == 0:
            return None
        idx = int(np.argmax(ys))
        return int(xs[idx]), int(ys[idx])

    @staticmethod
    def _point_topmost(mask_u8: np.ndarray) -> tuple[int, int] | None:
        ys, xs = np.where(mask_u8 > 0)
        if len(xs) == 0:
            return None
        idx = int(np.argmin(ys))
        return int(xs[idx]), int(ys[idx])

    @staticmethod
    def _nearest_point_between_masks(src_mask: np.ndarray, dst_mask: np.ndarray) -> tuple[int, int] | None:
        src_pts = np.argwhere(src_mask > 0)
        dst_pts = np.argwhere(dst_mask > 0)
        if len(src_pts) == 0 or len(dst_pts) == 0:
            return None
        best_src = None
        best_d2 = float('inf')
        for sy, sx in src_pts:
            diff = dst_pts - np.array([sy, sx], dtype=np.int32)
            d2 = np.min(np.sum(diff * diff, axis=1))
            if d2 < best_d2:
                best_d2 = float(d2)
                best_src = (int(sx), int(sy))
        return best_src

    @staticmethod
    def _mean_confidence(detections: list[Detection], class_name: str) -> float:
        vals = [float(d.confidence) for d in detections if d.class_name == class_name]
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def _resolve_camera_id(self, camera_id: str, source_type: str) -> str:
        raw_camera = str(camera_id or 'default').strip() or 'default'
        if raw_camera != 'default':
            return raw_camera

        cal_cfg = self.config.get('calibration', {})
        profile_map = cal_cfg.get('default_camera_profiles', {}) or {}
        src_key = str(source_type or 'unknown').strip().lower()
        mapped = profile_map.get(src_key) or profile_map.get('default')
        if isinstance(mapped, str) and mapped.strip():
            return mapped.strip()
        return raw_camera

    def _detections_trustworthy(self, detections: list[Detection], image_shape: tuple[int, int]) -> bool:
        if not detections:
            return False

        h, w = int(image_shape[0]), int(image_shape[1])
        img_area = max(1, h * w)
        min_conf = float(self.config.get('inference', {}).get('min_trust_confidence', 0.12))
        min_area_ratio = float(self.config.get('inference', {}).get('min_trust_area_ratio', 0.0015))
        min_major_ratio = float(self.config.get('inference', {}).get('min_trust_major_axis_ratio', 0.06))

        max_conf = max(float(d.confidence) for d in detections)
        total_area = float(sum(int((d.mask > 0).sum()) for d in detections))
        max_major_axis = max(self._detection_major_axis_px(d) for d in detections)

        if max_conf < min_conf:
            return False
        if (total_area / img_area) < min_area_ratio:
            return False
        if max_major_axis < (min(h, w) * min_major_ratio):
            return False
        return True

    def _estimate_adaptive_scale_from_priors(
        self,
        detections: list[Detection],
        crop: str,
        current_scale: float,
    ) -> float | None:
        adaptive_cfg = self.config.get('morphometry', {}).get('adaptive_scale', {})
        if adaptive_cfg.get('enabled', True) is False:
            return None

        priors_cfg = adaptive_cfg.get('priors_mm', {})
        default_priors = {'root': 45.0, 'stem': 35.0, 'leaves': 80.0}
        crop_priors = priors_cfg.get(crop, priors_cfg.get('default', default_priors))

        min_conf = float(adaptive_cfg.get('min_confidence', 0.22))
        min_length_px = float(adaptive_cfg.get('min_length_px', 30.0))
        min_scale = float(adaptive_cfg.get('mm_per_px_min', 0.05))
        max_scale = float(adaptive_cfg.get('mm_per_px_max', 0.35))
        max_relative_delta = float(adaptive_cfg.get('max_relative_delta', 1.8))

        candidates: list[tuple[float, float]] = []
        for det in detections:
            prior_len = float(crop_priors.get(det.class_name, 0.0))
            if prior_len <= 0.0:
                continue
            if float(det.confidence) < min_conf:
                continue
            length_px = self._detection_major_axis_px(det)
            if length_px < min_length_px:
                continue
            est = prior_len / length_px
            if math.isfinite(est):
                candidates.append((est, max(0.001, float(det.confidence))))

        if not candidates:
            longest_px = max((self._detection_major_axis_px(d) for d in detections), default=0.0)
            generic_len = float(adaptive_cfg.get('generic_target_length_mm', 80.0))
            if longest_px >= max(120.0, min_length_px * 2.0):
                candidates.append((generic_len / longest_px, 0.35))

        if not candidates:
            return None

        vals = np.asarray([c[0] for c in candidates], dtype=np.float64)
        weights = np.asarray([c[1] for c in candidates], dtype=np.float64)
        order = np.argsort(vals)
        vals = vals[order]
        weights = weights[order]
        cum = np.cumsum(weights)
        median_idx = int(np.searchsorted(cum, cum[-1] * 0.5))
        est_scale = float(vals[min(median_idx, len(vals) - 1)])

        est_scale = float(np.clip(est_scale, min_scale, max_scale))
        if current_scale > 0 and math.isfinite(current_scale):
            est_scale = float(np.clip(est_scale, current_scale / max_relative_delta, current_scale * max_relative_delta))
        return est_scale

    def _heuristic_predict(
        self,
        image_bgr: np.ndarray,
        class_colors: dict[str, tuple[int, int, int]],
        overlay_alpha: float,
        min_conf_for_emit: float = 0.08,
    ) -> dict[str, Any]:
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        hch, sch, vch = cv2.split(hsv)

        sat_mask = (sch > 28) & (vch > 22)
        green_mask = (hch >= 25) & (hch <= 95)
        red_mask = (hch <= 16) | (hch >= 160)
        plant_mask = ((green_mask | red_mask | sat_mask) & sat_mask).astype(np.uint8)
        plant_mask = self._cleanup_mask(plant_mask, min_area=max(48, int(0.0007 * h * w)), keep=5)

        if int(plant_mask.sum()) < int(0.002 * h * w):
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            _, plant_mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            plant_mask = self._cleanup_mask(plant_mask, min_area=max(64, int(0.0012 * h * w)), keep=3, k_open=5, k_close=7)

        x1, y1, x2, y2 = self._mask_bbox(plant_mask)
        if x2 <= x1 or y2 <= y1:
            # Fallback tiny center blob to keep pipeline alive.
            plant_mask = np.zeros((h, w), dtype=np.uint8)
            cx, cy = w // 2, h // 2
            cv2.circle(plant_mask, (cx, cy), max(6, min(h, w) // 12), 1, -1)
            x1, y1, x2, y2 = self._mask_bbox(plant_mask)

        dist = cv2.distanceTransform((plant_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
        ay, ax = np.unravel_index(int(np.argmax(dist)), dist.shape)
        anchor_xy = (int(ax), int(ay))
        plant_mask = self._component_at_point(plant_mask, anchor_xy)

        bw = max(1, x2 - x1 + 1)
        bh = max(1, y2 - y1 + 1)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        bright = gray >= np.percentile(gray, 62)
        non_green = (plant_mask > 0) & (~green_mask) & bright & (sch < 95)
        root_candidate = non_green.astype(np.uint8)
        if int(root_candidate.sum()) < 20:
            root_candidate = (((plant_mask > 0) & (~green_mask) & (gray >= np.percentile(gray, 52))).astype(np.uint8))

        lower_half = np.zeros_like(root_candidate, dtype=np.uint8)
        lower_half[min(h - 1, anchor_xy[1]):, :] = 1
        root_mask = self._cleanup_mask((root_candidate & lower_half).astype(np.uint8), min_area=max(18, int(0.00025 * h * w)), keep=4, k_open=2, k_close=3)

        stem_radius = max(5, int(0.04 * min(bw, bh)))
        stem_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(stem_mask, anchor_xy, stem_radius, 1, -1)
        stem_mask = self._cleanup_mask((stem_mask & plant_mask).astype(np.uint8), min_area=max(12, int(0.00015 * h * w)), keep=2, k_open=2, k_close=3)
        if int(stem_mask.sum()) < 20:
            stem_mask = np.zeros((h, w), dtype=np.uint8)
            ax, ay = anchor_xy
            stem_mask[max(0, ay - 10):min(h, ay + 11), max(0, ax - 10):min(w, ax + 11)] = 1
            stem_mask = self._cleanup_mask((stem_mask & plant_mask).astype(np.uint8), min_area=max(10, int(0.0001 * h * w)), keep=1, k_open=2, k_close=3)

        leaves_mask = ((green_mask.astype(np.uint8)) & plant_mask).astype(np.uint8)
        leaves_mask[root_mask > 0] = 0
        leaves_mask[stem_mask > 0] = 0
        leaves_mask = self._cleanup_mask(leaves_mask, min_area=max(24, int(0.0003 * h * w)), keep=5)
        if int(leaves_mask.sum()) < 15:
            upper_part = np.zeros_like(plant_mask)
            upper_part[:max(1, anchor_xy[1]), :] = 1
            leaves_mask = self._cleanup_mask(((plant_mask > 0) & (upper_part > 0)).astype(np.uint8), min_area=max(18, int(0.00025 * h * w)), keep=4)

        plant_pixels = int(plant_mask.sum())
        green_pixels = int(((green_mask.astype(np.uint8) > 0) & (plant_mask > 0)).sum())
        green_fraction = float(green_pixels) / float(max(1, plant_pixels))
        root_fraction = float(int(root_mask.sum())) / float(max(1, plant_pixels))
        leaves_fraction = float(int(leaves_mask.sum())) / float(max(1, plant_pixels))

        # Domain-aware adjustment:
        # root-lab scenes are usually low-green, while shoot scenes are high-green.
        if green_fraction < 0.18:
            # Root-lab scenes often contain thin bright structures on dark background.
            # Use line-like extraction + skeleton to avoid inflating root area to full plant blob.
            gray_f = gray.astype(np.float32)
            top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8))
            line_score = np.clip(0.65 * top_hat.astype(np.float32) + 0.35 * gray_f, 0.0, 255.0).astype(np.uint8)
            thr = int(np.clip(np.percentile(line_score[plant_mask > 0], 62) if int((plant_mask > 0).sum()) > 0 else 96, 40, 220))
            thin_candidate = ((line_score >= thr) & (plant_mask > 0)).astype(np.uint8)
            thin_candidate = self._cleanup_mask(
                thin_candidate,
                min_area=max(18, int(0.0002 * h * w)),
                keep=8,
                k_open=2,
                k_close=2,
            )
            if int(thin_candidate.sum()) > 0:
                sk = self._skeletonize(thin_candidate)
                root_mask = cv2.dilate(sk, np.ones((3, 3), np.uint8), iterations=1)
                root_mask = self._cleanup_mask(
                    root_mask,
                    min_area=max(16, int(0.00012 * h * w)),
                    keep=8,
                    k_open=1,
                    k_close=2,
                )
            else:
                root_mask = self._cleanup_mask(
                    root_candidate.astype(np.uint8),
                    min_area=max(16, int(0.00012 * h * w)),
                    keep=6,
                    k_open=2,
                    k_close=2,
                )

            leaves_mask = np.zeros_like(plant_mask, dtype=np.uint8)
            root_fraction = float(int(root_mask.sum())) / float(max(1, plant_pixels))
            leaves_fraction = 0.0
        elif green_fraction > 0.55 and leaves_fraction < 0.2:
            leaves_mask = self._cleanup_mask((green_mask.astype(np.uint8) & plant_mask).astype(np.uint8), min_area=max(24, int(0.0003 * h * w)), keep=5)
            leaves_fraction = float(int(leaves_mask.sum())) / float(max(1, plant_pixels))

        if int(root_mask.sum()) < 12:
            root_mask = np.zeros((h, w), dtype=np.uint8)
            ax, ay = anchor_xy
            root_mask[max(0, ay):min(h, ay + max(10, bh // 6)), max(0, ax - 4):min(w, ax + 5)] = 1
            root_mask = self._cleanup_mask((root_mask & plant_mask).astype(np.uint8), min_area=max(10, int(0.0001 * h * w)), keep=1, k_open=2, k_close=3)
            root_fraction = float(int(root_mask.sum())) / float(max(1, plant_pixels))

        # Enforce disjoint masks to reduce area inflation.
        stem_mask[root_mask > 0] = 0
        leaves_mask[root_mask > 0] = 0
        leaves_mask[stem_mask > 0] = 0
        stem_mask = self._cleanup_mask(stem_mask, min_area=max(10, int(0.0001 * h * w)), keep=2, k_open=2, k_close=3)
        leaves_mask = self._cleanup_mask(leaves_mask, min_area=max(18, int(0.0002 * h * w)), keep=5, k_open=2, k_close=3)

        # Keep fallback geometry conservative to avoid inflated areas on hard backgrounds.
        def _shrink_mask(mask: np.ndarray, max_area: int, keep: int = 3) -> np.ndarray:
            out = (mask > 0).astype(np.uint8)
            area = int(out.sum())
            if area <= max_area:
                return out

            kernel = np.ones((3, 3), np.uint8)
            for _ in range(12):
                out = cv2.erode(out, kernel, iterations=1)
                out = self._cleanup_mask(
                    out,
                    min_area=max(8, int(0.00008 * h * w)),
                    keep=keep,
                    k_open=1,
                    k_close=2,
                )
                area = int(out.sum())
                if area == 0 or area <= max_area:
                    break
            return out

        cap_cfg = self.config.get('inference', {}).get('heuristic_area_caps', {})
        cap_enabled = bool(cap_cfg.get('enabled', True))
        if cap_enabled and plant_pixels > 0:
            max_root_frac = float(cap_cfg.get('max_root_fraction_of_plant', 0.38))
            max_stem_frac = float(cap_cfg.get('max_stem_fraction_of_plant', 0.18))
            max_leaves_frac = float(cap_cfg.get('max_leaves_fraction_of_plant', 0.90))
            max_root_img_frac = float(cap_cfg.get('max_root_fraction_of_image', 0.08))
            max_stem_img_frac = float(cap_cfg.get('max_stem_fraction_of_image', 0.05))
            max_leaves_img_frac = float(cap_cfg.get('max_leaves_fraction_of_image', 0.85))
            root_line_thickness_px = float(cap_cfg.get('root_line_thickness_px', 8.0))
            stem_line_thickness_px = float(cap_cfg.get('stem_line_thickness_px', 12.0))
            max_root_frac = float(np.clip(max_root_frac, 0.08, 0.95))
            max_stem_frac = float(np.clip(max_stem_frac, 0.04, 0.60))
            max_leaves_frac = float(np.clip(max_leaves_frac, 0.30, 0.98))
            max_root_img_frac = float(np.clip(max_root_img_frac, 0.01, 0.6))
            max_stem_img_frac = float(np.clip(max_stem_img_frac, 0.005, 0.3))
            max_leaves_img_frac = float(np.clip(max_leaves_img_frac, 0.05, 0.98))
            root_line_thickness_px = float(np.clip(root_line_thickness_px, 2.0, 28.0))
            stem_line_thickness_px = float(np.clip(stem_line_thickness_px, 3.0, 36.0))

            root_area = int(root_mask.sum())
            stem_area = int(stem_mask.sum())
            leaves_area = int(leaves_mask.sum())

            if green_fraction < 0.18:
                root_limit_frac = max(max_root_frac, 0.55)
            elif green_fraction < 0.30:
                root_limit_frac = max_root_frac
            else:
                root_limit_frac = min(max_root_frac, 0.24)

            img_pixels = int(max(1, h * w))
            root_limit = int(max(16, min(plant_pixels * root_limit_frac, img_pixels * max_root_img_frac)))
            stem_limit = int(max(12, min(plant_pixels * max_stem_frac, img_pixels * max_stem_img_frac)))
            leaves_limit = int(max(20, min(plant_pixels * max_leaves_frac, img_pixels * max_leaves_img_frac)))

            # Additional geometric caps for line-like organs on lab/root scenes.
            if root_area > 0 and green_fraction < 0.35:
                rx1, ry1, rx2, ry2 = self._mask_bbox(root_mask)
                r_major = max(1, max(rx2 - rx1 + 1, ry2 - ry1 + 1))
                root_line_limit = int(max(24, r_major * root_line_thickness_px))
                root_limit = min(root_limit, root_line_limit)

            if stem_area > 0 and green_fraction < 0.35:
                sx1, sy1, sx2, sy2 = self._mask_bbox(stem_mask)
                s_major = max(1, max(sx2 - sx1 + 1, sy2 - sy1 + 1))
                stem_line_limit = int(max(18, s_major * stem_line_thickness_px))
                stem_limit = min(stem_limit, stem_line_limit)

            # Root in fallback should remain line-like, not dominate whole plant blob.
            if root_area > root_limit:
                sk = self._skeletonize(root_mask)
                root_mask = cv2.dilate(sk, np.ones((3, 3), np.uint8), iterations=1)
                root_mask = self._cleanup_mask(root_mask, min_area=max(10, int(0.0001 * h * w)), keep=6, k_open=1, k_close=2)
                if int(root_mask.sum()) > root_limit:
                    root_mask = _shrink_mask(root_mask, root_limit, keep=6)

            if stem_area > stem_limit:
                stem_mask = _shrink_mask(stem_mask, stem_limit, keep=2)

            if leaves_area > leaves_limit:
                leaves_mask = _shrink_mask(leaves_mask, leaves_limit, keep=5)

        root_fraction = float(int(root_mask.sum())) / float(max(1, plant_pixels))
        stem_fraction = float(int(stem_mask.sum())) / float(max(1, plant_pixels))
        leaves_fraction = float(int(leaves_mask.sum())) / float(max(1, plant_pixels))

        conf_root = float(np.clip(0.16 + 0.72 * root_fraction + 0.12 * (1.0 - green_fraction), 0.14, 0.9))
        conf_stem = float(np.clip(0.14 + 0.72 * stem_fraction + 0.08 * (1.0 - abs(root_fraction - leaves_fraction)), 0.12, 0.86))
        conf_leaves = float(np.clip(0.15 + 0.70 * leaves_fraction + 0.18 * green_fraction, 0.12, 0.92))

        detections: list[Detection] = []
        min_area_for_emit = max(12, int(0.00012 * h * w))
        for idx, (cls, mask, confv) in enumerate(
            [
                ('root', root_mask, conf_root),
                ('stem', stem_mask, conf_stem),
                ('leaves', leaves_mask, conf_leaves),
            ]
        ):
            area = int((mask > 0).sum())
            if area < min_area_for_emit or float(confv) < float(min_conf_for_emit):
                continue
            bx1, by1, bx2, by2 = self._mask_bbox(mask)
            if bx2 <= bx1 or by2 <= by1:
                continue
            detections.append(
                Detection(
                    instance_id=idx,
                    class_id=idx,
                    class_name=cls,
                    confidence=round(confv, 5),
                    bbox_xyxy=[float(bx1), float(by1), float(bx2), float(by2)],
                    mask=(mask > 0).astype(np.uint8),
                )
            )

        overlay = image_bgr.copy()
        for det in detections:
            color = class_colors.get(det.class_name, (255, 255, 255))
            color_layer = np.zeros_like(overlay, dtype=np.uint8)
            color_layer[det.mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1.0, color_layer, overlay_alpha, 0.0)

        return {
            'image': image_bgr,
            'overlay': overlay,
            'detections': detections,
            'fallback_mode': True,
            'confidence_by_class': {
                'root': round(conf_root, 4),
                'stem': round(conf_stem, 4),
                'leaves': round(conf_leaves, 4),
            },
        }

    @staticmethod
    def _measurement_trust_score(
        measurements: list[dict[str, Any]],
        *,
        calibration_reliable: bool,
        logic_checks: dict[str, Any],
        inference_mode: str,
    ) -> dict[str, Any]:
        if not measurements:
            return {
                'score': 0.0,
                'level': 'low',
                'factors': {
                    'mean_confidence': 0.0,
                    'reliable_ratio': 0.0,
                    'logic_ok': 0.0,
                    'mode_factor': 0.0,
                    'calibration_factor': 0.0,
                },
            }

        conf_vals = [float(m.get('confidence', 0.0)) for m in measurements]
        mean_conf = float(np.clip(np.mean(conf_vals) if conf_vals else 0.0, 0.0, 1.0))
        reliable_ratio = float(np.mean([1.0 if bool(m.get('reliable', False)) else 0.0 for m in measurements]))
        logic_ok = 1.0 if bool(logic_checks.get('passed', True)) else max(0.35, 1.0 - 0.18 * len(logic_checks.get('issues', [])))
        mode_factor_map = {
            'model': 1.0,
            'model_low_confidence_kept': 0.72,
            'model_assisted_no_detections': 0.66,
            'heuristic_fallback': 0.52,
            'model_no_detections': 0.38,
            'model_low_confidence': 0.34,
            'model_unavailable': 0.22,
        }
        mode_factor = float(mode_factor_map.get(str(inference_mode), 0.5))
        calibration_factor = 1.0 if calibration_reliable else 0.78

        base = (
            0.45 * mean_conf
            + 0.25 * reliable_ratio
            + 0.20 * logic_ok
            + 0.10 * mode_factor
        )
        score = float(np.clip(100.0 * base * calibration_factor, 0.0, 99.9))
        if score >= 75.0:
            level = 'high'
        elif score >= 45.0:
            level = 'medium'
        else:
            level = 'low'

        return {
            'score': round(score, 2),
            'level': level,
            'factors': {
                'mean_confidence': round(mean_conf, 5),
                'reliable_ratio': round(reliable_ratio, 5),
                'logic_ok': round(logic_ok, 5),
                'mode_factor': round(mode_factor, 5),
                'calibration_factor': round(calibration_factor, 5),
            },
        }

    @staticmethod
    def _empty_result(image_bgr: np.ndarray) -> dict[str, Any]:
        return {
            'image': image_bgr,
            'overlay': image_bgr.copy(),
            'detections': [],
            'fallback_mode': False,
            'confidence_by_class': {},
        }

    @staticmethod
    def _image_quality(image_bgr: np.ndarray) -> dict[str, Any]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(gray.mean())
        contrast = float(gray.std())

        if blur_score >= 130:
            sharp_label = 'Изображение четкое'
        elif blur_score >= 55:
            sharp_label = 'Изображение умеренно четкое'
        else:
            sharp_label = 'Есть заметное размытие'

        if mean_brightness < 75:
            light_label = 'Освещение ниже нормы'
        elif mean_brightness > 195:
            light_label = 'Освещение избыточное'
        else:
            light_label = 'Нормальное освещение'

        if contrast < 30:
            contrast_label = 'Низкий контраст'
        elif contrast > 85:
            contrast_label = 'Высокий контраст'
        else:
            contrast_label = 'Сбалансированный контраст'

        return {
            'blur_score': round(blur_score, 4),
            'brightness': round(mean_brightness, 4),
            'contrast': round(contrast, 4),
            'notes': [sharp_label, contrast_label, light_label],
        }

    def _adaptive_inference_params(
        self,
        image_quality: dict[str, Any],
        conf: float | None,
        iou: float | None,
        max_det: int | None,
    ) -> tuple[float | None, float | None, int | None, dict[str, Any]]:
        adaptive_cfg = self.config.get('inference', {}).get('adaptive_params', {})
        enabled = bool(adaptive_cfg.get('enabled', True))
        if not enabled:
            return conf, iou, max_det, {'enabled': False, 'applied': False, 'reasons': []}

        respect_manual = bool(adaptive_cfg.get('respect_manual_overrides', True))
        manual_overrides = any(x is not None for x in (conf, iou, max_det))
        if respect_manual and manual_overrides:
            return conf, iou, max_det, {
                'enabled': True,
                'applied': False,
                'reasons': ['manual_override'],
            }

        model_cfg = self.config.get('model', {})
        base_conf = float(model_cfg.get('conf', 0.08) if conf is None else conf)
        base_iou = float(model_cfg.get('iou', 0.5) if iou is None else iou)
        base_max_det = int(model_cfg.get('max_det', 200) if max_det is None else max_det)

        blur_score = float(image_quality.get('blur_score', 0.0))
        brightness = float(image_quality.get('brightness', 0.0))
        contrast = float(image_quality.get('contrast', 0.0))

        reasons: list[str] = []
        if blur_score < float(adaptive_cfg.get('blur_threshold', 55.0)):
            reasons.append('blur')
        if brightness < float(adaptive_cfg.get('low_light_threshold', 75.0)):
            reasons.append('low_light')
        if brightness > float(adaptive_cfg.get('overexposed_threshold', 210.0)):
            reasons.append('overexposed')
        if contrast < float(adaptive_cfg.get('low_contrast_threshold', 30.0)):
            reasons.append('low_contrast')

        severity = len(reasons)
        if severity == 0:
            return base_conf, base_iou, base_max_det, {
                'enabled': True,
                'applied': False,
                'reasons': [],
                'severity': 0,
                'params': {
                    'conf': round(base_conf, 5),
                    'iou': round(base_iou, 5),
                    'max_det': int(base_max_det),
                },
            }

        hard = severity >= int(adaptive_cfg.get('hard_severity_threshold', 2))
        conf_mult = float(adaptive_cfg.get('conf_hard_multiplier', 0.72 if hard else 0.9))
        iou_mult = float(adaptive_cfg.get('iou_hard_multiplier', 0.88 if hard else 0.96))
        max_det_mult = float(adaptive_cfg.get('max_det_hard_multiplier', 1.5 if hard else 1.2))
        if not hard:
            conf_mult = float(adaptive_cfg.get('conf_mild_multiplier', conf_mult))
            iou_mult = float(adaptive_cfg.get('iou_mild_multiplier', iou_mult))
            max_det_mult = float(adaptive_cfg.get('max_det_mild_multiplier', max_det_mult))

        tuned_conf = float(np.clip(
            base_conf * conf_mult,
            float(adaptive_cfg.get('min_conf', 0.01)),
            float(adaptive_cfg.get('max_conf', 0.35)),
        ))
        tuned_iou = float(np.clip(
            base_iou * iou_mult,
            float(adaptive_cfg.get('min_iou', 0.35)),
            float(adaptive_cfg.get('max_iou', 0.7)),
        ))
        tuned_max_det = int(np.clip(
            int(round(base_max_det * max_det_mult)),
            int(adaptive_cfg.get('min_max_det', 20)),
            int(adaptive_cfg.get('max_max_det', 400)),
        ))

        return tuned_conf, tuned_iou, tuned_max_det, {
            'enabled': True,
            'applied': True,
            'severity': int(severity),
            'mode': 'hard' if hard else 'mild',
            'reasons': reasons,
            'base': {
                'conf': round(base_conf, 5),
                'iou': round(base_iou, 5),
                'max_det': int(base_max_det),
            },
            'params': {
                'conf': round(tuned_conf, 5),
                'iou': round(tuned_iou, 5),
                'max_det': int(tuned_max_det),
            },
        }

    def _apply_metric_logic_checks(
        self,
        measurements: list[dict[str, Any]],
        detections: list[Detection],
        image_shape: tuple[int, int],
    ) -> dict[str, Any]:
        if not measurements:
            return {'passed': True, 'issues': [], 'adjustments': 0}

        issues: list[str] = []
        adjustments = 0
        h, _ = image_shape[:2]

        stem_rows = [m for m in measurements if m.get('class_name') == 'stem']
        root_rows = [m for m in measurements if m.get('class_name') == 'root']

        stem_avg_len = float(np.mean([float(m.get('length_px', 0.0)) for m in stem_rows])) if stem_rows else 0.0
        root_avg_len = float(np.mean([float(m.get('length_px', 0.0)) for m in root_rows])) if root_rows else 0.0

        if stem_avg_len > 0 and stem_avg_len < (0.03 * float(h)):
            issues.append('Стебель занимает слишком малую долю кадра; проверьте качество сегментации.')

        stem_det_axes = [self._detection_major_axis_px(d) for d in detections if d.class_name == 'stem']
        root_det_axes = [self._detection_major_axis_px(d) for d in detections if d.class_name == 'root']
        stem_axis = float(np.mean(stem_det_axes)) if stem_det_axes else 0.0
        root_axis = float(np.mean(root_det_axes)) if root_det_axes else 0.0

        if stem_rows:
            for row in stem_rows:
                row_len = float(row.get('length_px', 0.0))
                if stem_axis >= 10.0 and row_len < 0.25 * stem_axis:
                    row['length_px'] = round(stem_axis, 5)
                    if row.get('length_mm') is not None:
                        mm_per_px = float(row['length_mm']) / max(1e-6, row_len)
                        row['length_mm'] = round(stem_axis * mm_per_px, 5)
                    adjustments += 1
            if adjustments > 0:
                issues.append('Обнаружена логическая несогласованность длины стебля; выполнен пересчет по геометрии маски.')

        if stem_axis > (1.2 * max(1e-6, root_axis)) and stem_avg_len < (0.8 * root_avg_len):
            issues.append('Длины root/stem противоречат визуальной доминанте; проверьте условия съемки и маски.')

        return {'passed': len(issues) == 0, 'issues': issues, 'adjustments': adjustments}

    async def run_single(
        self,
        image_bytes: bytes,
        image_name: str,
        crop: str = 'Unknown',
        calibration_bytes: bytes | None = None,
        camera_id: str = 'default',
        tenant_id: str = 'default',
        source_type: str = 'unknown',
        conf: float | None = None,
        iou: float | None = None,
        max_det: int | None = None,
        use_ensemble: bool | None = None,
    ) -> PredictResponse:
        run_id, run_dir = self.storage.create_run_dir()

        input_path = run_dir / image_name
        self.storage.save_bytes(input_path, image_bytes)

        main_img = decode_image_bytes(image_bytes)
        if main_img is None:
            raise ValueError('Cannot decode input image.')

        calibration_image = None
        if calibration_bytes:
            calibration_image = decode_image_bytes(calibration_bytes)

        requested_camera_id = str(camera_id or 'default')
        resolved_camera_id = self._resolve_camera_id(requested_camera_id, source_type)
        # Avoid leaking stale cache scale into arbitrary photos when camera_id/profile is generic.
        calibration_cfg = self.config.get('calibration', {})
        allow_default_cache = bool(self.config.get('calibration', {}).get('use_cache_for_default_camera', False))
        allow_default_profile_cache = bool(calibration_cfg.get('allow_default_profile_cache', True))
        use_cache = calibration_image is not None or resolved_camera_id != 'default' or allow_default_cache
        scale_mm_per_px, scale_source = self.calibrator.get_scale(
            calibration_image if calibration_image is not None else main_img,
            camera_id=resolved_camera_id,
            use_cache=use_cache,
        )
        metric_policy = self.config.get('morphometry', {}).get('metric_policy', {})
        strict_scale_required = bool(metric_policy.get('strict_scale_required', True))
        valid_scale_sources = {
            str(x) for x in metric_policy.get('valid_scale_sources', ['chessboard', 'charuco', 'cache_scene', 'cache_scene_near'])
        }
        allow_estimated_mm_when_unreliable = bool(metric_policy.get('allow_estimated_mm_when_unreliable', False))
        estimated_scale_sources = {
            str(x)
            for x in metric_policy.get(
                'estimated_scale_sources',
                [
                    'fallback',
                    'cache',
                    'cache+adaptive_prior',
                    'adaptive_prior',
                    'auto_profile',
                    'cache+auto_profile',
                    'cache+auto_profile+adaptive_prior',
                ],
            )
        }
        min_reliable_conf = float(metric_policy.get('min_confidence_for_reliable_measurements', 0.7))
        allow_adaptive_scale = bool(self.config.get('morphometry', {}).get('adaptive_scale', {}).get('enabled', False))
        auto_profile_cfg = self.config.get('calibration', {}).get('auto_profile', {})
        auto_profile_enabled = bool(auto_profile_cfg.get('enabled', False)) and (not strict_scale_required)

        cache_like_source = str(scale_source).startswith('cache')
        cache_entry_validated = True
        if cache_like_source:
            cache_entry_validated = bool(self.calibrator.is_cache_scale_validated(resolved_camera_id))
        cache_scene_matched = str(scale_source).startswith('cache_scene')
        cache_is_conditioned = bool(
            use_cache
            and cache_entry_validated
            and (
                cache_scene_matched
                or (
                    source_type != 'unknown'
                    and (resolved_camera_id != 'default' or allow_default_profile_cache)
                )
            )
        )
        calibration_reliable = (scale_source in valid_scale_sources) and ((not cache_like_source) or cache_is_conditioned)
        image_quality = self._image_quality(main_img)
        adaptive_conf, adaptive_iou, adaptive_max_det, adaptive_info = self._adaptive_inference_params(
            image_quality=image_quality,
            conf=conf,
            iou=iou,
            max_det=max_det,
        )

        class_colors = {
            k: tuple(v)
            for k, v in self.config['inference'].get('class_colors', {}).items()
        }
        overlay_alpha = float(self.config['inference'].get('overlay_alpha', 0.45))
        allow_heuristic_fallback = bool(self.config.get('inference', {}).get('allow_heuristic_fallback', False))
        assist_on_no_detections = bool(self.config.get('inference', {}).get('assist_on_no_detections', True))
        strict_model_mode = bool(self.config.get('inference', {}).get('strict_model_mode', True))
        heuristic_emit_conf = max(
            0.08,
            float(self.config.get('inference', {}).get('min_confidence_for_measurements', 0.03)),
        )
        if adaptive_conf is not None:
            heuristic_emit_conf = max(0.06, min(heuristic_emit_conf, float(adaptive_conf)))

        fallback_mode = False
        inference_mode = 'model'
        try:
            yolo_result = await self.model_service.predict(
                image_path=str(input_path),
                class_colors=class_colors,
                overlay_alpha=overlay_alpha,
                conf=adaptive_conf,
                iou=adaptive_iou,
                max_det=adaptive_max_det,
                use_ensemble=use_ensemble,
            )
        except (ModelNotLoadedError, RuntimeError, ValueError) as exc:
            if allow_heuristic_fallback:
                logger.warning('Model prediction failed, using heuristic fallback: %s', exc)
                yolo_result = self._heuristic_predict(
                    image_bgr=main_img,
                    class_colors=class_colors,
                    overlay_alpha=overlay_alpha,
                    min_conf_for_emit=heuristic_emit_conf,
                )
                fallback_mode = True
                inference_mode = 'heuristic_fallback'
            else:
                logger.error('Model prediction failed, strict model mode active: %s', exc)
                yolo_result = self._empty_result(main_img)
                inference_mode = 'model_unavailable'

        detections = yolo_result.get('detections') or []
        if detections and (not self._detections_trustworthy(detections, main_img.shape[:2])):
            if allow_heuristic_fallback:
                logger.warning('Model detections are not trustworthy, switching to heuristic fallback.')
                yolo_result = self._heuristic_predict(
                    image_bgr=main_img,
                    class_colors=class_colors,
                    overlay_alpha=overlay_alpha,
                    min_conf_for_emit=heuristic_emit_conf,
                )
                fallback_mode = True
                inference_mode = 'heuristic_fallback'
            elif strict_model_mode:
                logger.warning('Model detections are low-trust, dropping detections in strict mode.')
                yolo_result = self._empty_result(main_img)
                inference_mode = 'model_low_confidence'
            else:
                logger.warning('Model detections are low-trust, keeping raw model output.')
                inference_mode = 'model_low_confidence_kept'
        elif (not detections) and inference_mode == 'model':
            if assist_on_no_detections:
                assisted = self._heuristic_predict(
                    image_bgr=main_img,
                    class_colors=class_colors,
                    overlay_alpha=overlay_alpha,
                    min_conf_for_emit=heuristic_emit_conf,
                )
                assisted_detections = assisted.get('detections') or []
                if assisted_detections:
                    yolo_result = assisted
                    fallback_mode = True
                    inference_mode = 'model_assisted_no_detections'
                    logger.info('No model detections, heuristic assist generated %s candidate masks.', len(assisted_detections))
                else:
                    inference_mode = 'model_no_detections'
            else:
                inference_mode = 'model_no_detections'

        if (not calibration_reliable) and (not strict_scale_required) and allow_adaptive_scale:
            adaptive_scale = self._estimate_adaptive_scale_from_priors(
                detections=yolo_result['detections'],
                crop=crop,
                current_scale=scale_mm_per_px,
            )
            if adaptive_scale is not None:
                scale_mm_per_px = float(adaptive_scale)
                if scale_source == 'fallback':
                    scale_source = 'adaptive_prior'
                elif scale_source == 'cache':
                    scale_source = 'cache+adaptive_prior'
                elif scale_source == 'auto_profile':
                    scale_source = 'auto_profile+adaptive_prior'
                elif scale_source == 'cache+auto_profile':
                    scale_source = 'cache+auto_profile+adaptive_prior'

        has_numeric_scale = (
            scale_mm_per_px is not None
            and math.isfinite(float(scale_mm_per_px))
            and float(scale_mm_per_px) > 0.0
        )
        mm_conversion_possible = bool(
            calibration_reliable
            or (
                allow_estimated_mm_when_unreliable
                and has_numeric_scale
                and str(scale_source) in estimated_scale_sources
            )
        )
        mm_estimated = bool(mm_conversion_possible and (not calibration_reliable))

        overlay_path = run_dir / 'overlay.png'
        self.storage.save_image(overlay_path, yolo_result['overlay'])

        save_masks = bool(self.config['inference'].get('save_masks', True))
        min_mask_area_px = int(self.config.get('morphometry', {}).get('min_mask_area_px', 0))
        min_mask_area_by_class = self.config.get('morphometry', {}).get('min_mask_area_px_by_class', {})
        min_measure_conf = float(self.config.get('inference', {}).get('min_confidence_for_measurements', 0.03))
        measurements: list[dict[str, Any]] = []
        masks_dir = run_dir / 'masks'
        detections_for_metrics = yolo_result['detections']

        merged_root_mask = self._merge_class_mask(detections_for_metrics, 'root')
        merged_stem_mask = self._merge_class_mask(detections_for_metrics, 'stem')
        merged_leaves_mask = self._merge_class_mask(detections_for_metrics, 'leaves')

        root_bottom_global = self._point_lowest(merged_root_mask) if merged_root_mask is not None else None
        root_to_stem_global = (
            self._nearest_point_between_masks(merged_root_mask, merged_stem_mask)
            if merged_root_mask is not None and merged_stem_mask is not None
            else None
        )
        stem_to_root_global = (
            self._nearest_point_between_masks(merged_stem_mask, merged_root_mask)
            if merged_stem_mask is not None and merged_root_mask is not None
            else None
        )
        leaf_tip_global = None
        if merged_leaves_mask is not None:
            leaf_tip_global = self._point_topmost(merged_leaves_mask)
        elif merged_stem_mask is not None:
            leaf_tip_global = self._point_topmost(merged_stem_mask)

        for det in detections_for_metrics:
            if float(det.confidence) < min_measure_conf:
                continue
            morph = analyze_mask(det.mask, scale_mm_per_px)
            cls_min_area = int(min_mask_area_by_class.get(det.class_name, min_mask_area_px))
            if det.class_name in {'root', 'stem'}:
                cls_min_area = min(cls_min_area, max(20, min_mask_area_px // 4))
            if morph.area_px < cls_min_area:
                continue
            x1, y1, x2, y2 = det.bbox_xyxy
            bbox_w = max(1.0, float(x2) - float(x1))
            bbox_h = max(1.0, float(y2) - float(y1))
            bbox_major_px = max(bbox_w, bbox_h)
            length_px = float(morph.length_px)
            det_mask = (det.mask > 0).astype(np.uint8)
            measurement_reliable = float(det.confidence) >= min_reliable_conf

            if det.class_name == 'root':
                root_start = root_bottom_global or self._point_lowest(det_mask)
                root_end = root_to_stem_global or self._point_topmost(det_mask)
                if root_start is not None and root_end is not None:
                    geodesic = path_length_between_points(det_mask, root_start, root_end)
                    if geodesic > 0.0:
                        length_px = float(geodesic)

            if det.class_name == 'stem':
                stem_start = stem_to_root_global or self._point_lowest(det_mask)
                stem_end = leaf_tip_global or self._point_topmost(det_mask)
                stem_path_mask = det_mask
                if merged_leaves_mask is not None:
                    stem_path_mask = np.maximum(stem_path_mask, merged_leaves_mask)
                if stem_start is not None and stem_end is not None:
                    geodesic = path_length_between_points(stem_path_mask, stem_start, stem_end)
                    if geodesic > 0.0:
                        length_px = float(geodesic)

            if length_px <= 0.0 or (bbox_major_px >= 6.0 and length_px < 0.25 * bbox_major_px):
                length_px = bbox_major_px

            # Real metric conversion depends on scale validity; confidence affects trust, not conversion itself.
            mm_allowed = mm_conversion_possible
            record = {
                'instance_id': det.instance_id,
                'crop': crop,
                'class_name': det.class_name,
                'confidence': round(det.confidence, 5),
                'area_px': morph.area_px,
                'area_mm2': round(morph.area_mm2, 5) if mm_allowed else None,
                'length_px': round(length_px, 5),
                'length_mm': round(length_px * scale_mm_per_px, 5) if mm_allowed else None,
                'reliable': bool(measurement_reliable),
            }
            measurements.append(record)

            if save_masks:
                mask_path = masks_dir / f'mask_{det.instance_id}_{det.class_name}.png'
                self.storage.save_image(mask_path, (det.mask * 255).astype(np.uint8))

        logic_checks = self._apply_metric_logic_checks(
            measurements=measurements,
            detections=detections_for_metrics,
            image_shape=main_img.shape[:2],
        )

        if auto_profile_enabled and (not strict_scale_required):
            trustworthy_for_auto_profile = (
                inference_mode == 'model'
                and bool(measurements)
                and self._detections_trustworthy(yolo_result['detections'], main_img.shape[:2])
            )
            if trustworthy_for_auto_profile:
                self.calibrator.update_auto_scale(
                    mm_per_px=scale_mm_per_px,
                    camera_id=resolved_camera_id,
                    source_type=source_type,
                    crop=crop,
                )

        disease_analysis = self.disease_service.analyze(
            image_bgr=main_img,
            detections=yolo_result['detections'],
            measurements=measurements,
        )
        plantcv_analysis = self.plantcv_service.analyze(
            image_bgr=main_img,
            detections=yolo_result['detections'],
        )
        phi = self.phi_service.evaluate(
            measurements=measurements,
            crop=crop,
            disease_analysis=disease_analysis,
            absolute_scale_reliable=calibration_reliable,
        )
        explainability = self.xai_service.generate(
            image_bgr=main_img,
            detections=yolo_result['detections'],
            run_dir=run_dir,
            uncertainty_map=yolo_result.get('uncertainty_map'),
        )
        active_learning = self.active_learning_service.collect(
            run_id=run_id,
            image_bgr=main_img,
            overlay_bgr=yolo_result.get('overlay'),
            detections=yolo_result['detections'],
            crop=crop,
            tenant_id=tenant_id,
        )

        csv_path = run_dir / 'measurements.csv'
        self.storage.save_csv(csv_path, measurements)

        summary = self.reporter.build_summary(measurements)
        summary['min_report_confidence'] = float(min_reliable_conf)
        measured_classes = {m['class_name'] for m in measurements}
        detected_classes = {d.class_name for d in yolo_result['detections']}
        summary['recognized_structures'] = sorted(measured_classes or detected_classes)
        summary['image_quality'] = image_quality
        summary['adaptive_inference'] = adaptive_info
        summary['inference_params'] = {
            'conf': float(adaptive_conf) if adaptive_conf is not None else None,
            'iou': float(adaptive_iou) if adaptive_iou is not None else None,
            'max_det': int(adaptive_max_det) if adaptive_max_det is not None else None,
        }
        summary['confidence_by_class'] = {
            cls: round(self._mean_confidence(yolo_result['detections'], cls), 5)
            for cls in sorted(detected_classes)
        }
        summary['segmentation'] = {
            cls: {
                'detected': cls in detected_classes,
                'confidence': round(summary['confidence_by_class'].get(cls, 0.0), 5),
            }
            for cls in ['root', 'stem', 'leaves']
        }
        summary['keypoints'] = {
            'root_bottom': list(root_bottom_global) if root_bottom_global is not None else None,
            'root_stem_transition': list(root_to_stem_global) if root_to_stem_global is not None else None,
            'leaf_tip': list(leaf_tip_global) if leaf_tip_global is not None else None,
        }
        summary['plantcv'] = plantcv_analysis
        summary['inference_mode'] = inference_mode
        summary['camera_id'] = requested_camera_id
        summary['calibration_camera_id'] = resolved_camera_id
        summary['calibration_reliable'] = calibration_reliable
        summary['calibration_cache_validated'] = bool(cache_entry_validated) if cache_like_source else None
        summary['model_based'] = inference_mode.startswith('model')
        summary['measurements_reliable'] = bool(
            measurements
            and all(bool(m.get('reliable', False)) for m in measurements)
            and logic_checks.get('passed', True)
        )
        summary['logic_checks'] = logic_checks
        trust_payload = self._measurement_trust_score(
            measurements=measurements,
            calibration_reliable=calibration_reliable,
            logic_checks=logic_checks,
            inference_mode=inference_mode,
        )
        summary['measurement_trust_score'] = trust_payload['score']
        summary['measurement_trust_level'] = trust_payload['level']
        summary['measurement_trust_factors'] = trust_payload['factors']
        summary['mm_conversion_possible'] = bool(mm_conversion_possible)
        summary['mm_estimated'] = bool(mm_estimated)
        summary['mm_conversion_mode'] = 'reliable' if calibration_reliable else ('estimated' if mm_estimated else 'disabled')
        summary['mm_per_pixel'] = float(scale_mm_per_px) if mm_conversion_possible else None
        summary['calibration_source'] = str(scale_source) if mm_conversion_possible else None
        cal_error_map = {
            'chessboard': 3.0,
            'charuco': 3.5,
            'cache_scene': 4.5,
            'cache_scene_near': 5.0,
            'cache': 6.0,
            'cache+auto_profile': 6.0,
            'auto_profile': 8.0,
            'adaptive_prior': 14.0,
            'fallback': 18.0,
        }
        summary['calibration_error_pct'] = cal_error_map.get(str(scale_source).split('+')[0]) if mm_conversion_possible else None

        if inference_mode in {'model_unavailable', 'model_low_confidence', 'model_no_detections'}:
            summary['inference_note'] = (
                'Нет уверенных детекций модели. Численная морфометрия ограничена. '
                'Загрузите более четкое изображение или дообучите модель на более близком домене.'
            )
        elif inference_mode == 'model_assisted_no_detections':
            summary['inference_note'] = (
                'Модель не выделила маски уверенно, поэтому применена ассистирующая геометрическая сегментация. '
                'Метрики предварительные; для точности рекомендуется повторная съемка и дообучение.'
            )
        elif inference_mode == 'heuristic_fallback':
            summary['inference_note'] = (
                'Использован эвристический fallback, так как выход модели был недоступен или ненадежен. '
                'Считайте метрики предварительными.'
            )
        if not summary['measurements_reliable']:
            summary['calibration_note'] = (
                'Измерения считаются недостоверными: confidence ниже 70% и/или обнаружены логические противоречия.'
            )
        elif calibration_reliable:
            summary['calibration_note'] = (
                'Перевод в мм разрешен: есть валидная геометрическая калибровка.'
            )
        elif mm_conversion_possible:
            summary['calibration_note'] = (
                'Показаны оценочные мм по профилю камеры/кэшу. Для высокой точности добавьте эталон в кадр.'
            )
        else:
            summary['calibration_note'] = (
                'Перевод в мм невозможен без валидной калибровки камеры.'
            )
        recommendations = [
            r.model_dump()
            for r in self.recommender.generate(
                measurements,
                absolute_scale_reliable=calibration_reliable,
                scale_source=scale_source,
            )
        ]

        plot_path = run_dir / 'distribution.png'
        self.reporter.save_distribution_plot(measurements, plot_path)

        pdf_path = run_dir / 'report.pdf'
        self.reporter.save_pdf_report(pdf_path, run_id, summary, recommendations)

        payload = {
            'run_id': run_id,
            'scale_mm_per_px': scale_mm_per_px,
            'scale_source': scale_source,
            'measurements': measurements,
            'summary': summary,
            'recommendations': recommendations,
            'disease_analysis': disease_analysis,
            'plantcv_analysis': plantcv_analysis,
            'phi': phi.model_dump(),
            'explainability': explainability.model_dump(),
            'active_learning': active_learning.model_dump(),
            'files': {
                'input': str(input_path),
                'overlay': str(overlay_path),
                'csv': str(csv_path),
                'distribution_plot': str(plot_path),
                'pdf_report': str(pdf_path),
                'xai_confidence_map': explainability.confidence_map,
                'xai_attention': explainability.attention_heatmap,
                'xai_gradcam': explainability.gradcam,
                'xai_uncertainty': explainability.uncertainty_map,
            },
        }
        self.storage.save_json(run_dir / 'result.json', payload)

        logger.info('Inference run completed: %s', run_id)
        response = PredictResponse(**payload)
        if self.history_service is not None:
            self.history_service.register_result(
                response,
                tenant_id=tenant_id,
                source_type=source_type,
                camera_id=resolved_camera_id,
            )
        return response

    async def run_batch(
        self,
        files: list[tuple[bytes, str, str]],
        calibration_bytes: bytes | None = None,
        camera_id: str = 'default',
        tenant_id: str = 'default',
        source_type: str = 'unknown',
        conf: float | None = None,
        iou: float | None = None,
        max_det: int | None = None,
        use_ensemble: bool | None = None,
    ) -> list[PredictResponse]:
        results = []
        for image_bytes, image_name, crop in files:
            result = await self.run_single(
                image_bytes=image_bytes,
                image_name=image_name,
                crop=crop,
                calibration_bytes=calibration_bytes,
                camera_id=camera_id,
                tenant_id=tenant_id,
                source_type=source_type,
                conf=conf,
                iou=iou,
                max_det=max_det,
                use_ensemble=use_ensemble,
            )
            results.append(result)
        return results
