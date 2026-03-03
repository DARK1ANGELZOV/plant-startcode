from __future__ import annotations

import cv2
import numpy as np
import pytest

from inference.predictor import Detection
from services.inference_service import InferenceService
from services.recommendation_service import RecommendationService
from services.report_service import ReportService
from services.storage_service import StorageService


class DummyCalibrator:
    def get_scale(self, image, camera_id: str = 'default', use_cache: bool = True):
        return 0.12, 'fallback'


class DummyModelService:
    def __init__(self, detections: list[Detection], image: np.ndarray) -> None:
        self._detections = detections
        self._image = image

    async def predict(self, **kwargs):
        return {
            'image': self._image.copy(),
            'overlay': self._image.copy(),
            'detections': list(self._detections),
        }


def _config(output_root: str) -> dict:
    return {
        'inference': {
            'output_root': output_root,
            'overlay_alpha': 0.45,
            'save_masks': True,
            'min_trust_confidence': 0.12,
            'min_trust_area_ratio': 0.0015,
            'min_trust_major_axis_ratio': 0.06,
            'min_confidence_for_measurements': 0.03,
            'class_colors': {
                'root': [217, 83, 79],
                'stem': [92, 184, 92],
                'leaves': [66, 139, 202],
            },
        },
        'morphometry': {
            'default_mm_per_px': 0.12,
            'min_mask_area_px': 120,
            'min_mask_area_px_by_class': {'root': 24, 'stem': 30, 'leaves': 120},
            'adaptive_scale': {
                'enabled': True,
                'min_confidence': 0.22,
                'min_length_px': 30,
                'mm_per_px_min': 0.05,
                'mm_per_px_max': 0.35,
                'max_relative_delta': 1.8,
                'generic_target_length_mm': 80,
                'priors_mm': {
                    'default': {'root': 45, 'stem': 35, 'leaves': 80},
                    'Wheat': {'root': 55, 'stem': 40, 'leaves': 95},
                },
            },
            'recommendation_thresholds': {
                'Wheat': {
                    'min_root_length_mm': 25.0,
                    'min_stem_length_mm': 16.0,
                    'min_leaf_area_mm2': 120.0,
                    'min_leaf_root_ratio': 1.0,
                    'max_leaf_cv': 0.9,
                }
            },
        },
        'active_learning': {
            'root_dir': output_root,
            'low_conf_threshold': 0.12,
        },
    }


def _build_service(tmp_path, model_service) -> InferenceService:
    cfg = _config(str(tmp_path / 'outputs'))
    return InferenceService(
        model_service=model_service,
        calibrator=DummyCalibrator(),
        storage=StorageService(cfg['inference']['output_root']),
        reporter=ReportService(),
        recommender=RecommendationService(cfg['morphometry'].get('recommendation_thresholds', {})),
        config=cfg,
    )


def test_detections_trustworthy_rejects_low_conf(tmp_path) -> None:
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    mask = np.zeros((200, 400), dtype=np.uint8)
    cv2.rectangle(mask, (50, 80), (350, 120), 1, -1)
    det = Detection(
        instance_id=0,
        class_id=1,
        class_name='stem',
        confidence=0.04,
        bbox_xyxy=[50.0, 80.0, 350.0, 120.0],
        mask=mask,
    )
    service = _build_service(tmp_path, DummyModelService([det], image))
    assert service._detections_trustworthy([det], image.shape[:2]) is False


def test_estimate_adaptive_scale_from_priors_uses_long_axis(tmp_path) -> None:
    image = np.zeros((240, 640, 3), dtype=np.uint8)
    mask = np.zeros((240, 640), dtype=np.uint8)
    cv2.rectangle(mask, (40, 110), (540, 130), 1, -1)
    det = Detection(
        instance_id=0,
        class_id=2,
        class_name='leaves',
        confidence=0.84,
        bbox_xyxy=[40.0, 110.0, 540.0, 130.0],
        mask=mask,
    )
    service = _build_service(tmp_path, DummyModelService([det], image))
    scale = service._estimate_adaptive_scale_from_priors([det], crop='Wheat', current_scale=0.12)
    assert scale is not None
    assert 0.16 <= scale <= 0.22


@pytest.mark.asyncio
async def test_run_single_drops_untrustworthy_model_output_in_strict_mode(tmp_path) -> None:
    image = np.zeros((240, 640, 3), dtype=np.uint8)
    cv2.line(image, (30, 130), (560, 118), (0, 255, 0), 5)
    cv2.line(image, (560, 118), (600, 190), (220, 220, 220), 2)
    cv2.line(image, (560, 118), (575, 200), (220, 220, 220), 2)
    cv2.line(image, (560, 118), (545, 205), (220, 220, 220), 2)

    ok, enc = cv2.imencode('.png', image)
    assert ok

    weak_mask = np.zeros((240, 640), dtype=np.uint8)
    cv2.rectangle(weak_mask, (300, 115), (320, 132), 1, -1)
    weak_det = Detection(
        instance_id=0,
        class_id=1,
        class_name='stem',
        confidence=0.04,
        bbox_xyxy=[300.0, 115.0, 320.0, 132.0],
        mask=weak_mask,
    )

    service = _build_service(tmp_path, DummyModelService([weak_det], image))
    result = await service.run_single(
        image_bytes=enc.tobytes(),
        image_name='long_seedling.png',
        crop='Wheat',
    )

    assert (result.summary or {}).get('inference_mode') == 'model_low_confidence'
    assert not result.measurements
    assert 'Нет уверенных детекций модели' in str((result.summary or {}).get('inference_note', ''))

@pytest.mark.asyncio
async def test_run_single_blocks_mm_without_valid_calibration(tmp_path) -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(image, (120, 60), (180, 210), (0, 180, 0), -1)
    ok, enc = cv2.imencode('.png', image)
    assert ok

    mask = np.zeros((240, 320), dtype=np.uint8)
    cv2.rectangle(mask, (120, 60), (180, 210), 1, -1)
    det = Detection(
        instance_id=0,
        class_id=1,
        class_name='stem',
        confidence=0.92,
        bbox_xyxy=[120.0, 60.0, 180.0, 210.0],
        mask=mask,
    )

    service = _build_service(tmp_path, DummyModelService([det], image))
    result = await service.run_single(
        image_bytes=enc.tobytes(),
        image_name='strict_metric.png',
        crop='Wheat',
        camera_id='default',
        source_type='unknown',
    )

    assert result.measurements
    measurement = result.measurements[0]
    assert measurement.length_px > 0
    assert measurement.length_mm is None
    assert measurement.area_mm2 is None
    assert (result.summary or {}).get('mm_conversion_possible') is False
