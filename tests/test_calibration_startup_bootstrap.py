from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from api.main import _bootstrap_calibration_profiles
from calibration.chessboard import ScaleCalibrator


def _write_checkerboard(path: Path) -> None:
    square_px = 40
    rows_squares = 5
    cols_squares = 8
    margin = 20
    h = rows_squares * square_px + 2 * margin
    w = cols_squares * square_px + 2 * margin
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    for r in range(rows_squares):
        for c in range(cols_squares):
            color = 0 if (r + c) % 2 == 0 else 255
            y1 = margin + r * square_px
            x1 = margin + c * square_px
            cv2.rectangle(image, (x1, y1), (x1 + square_px, y1 + square_px), (color, color, color), thickness=-1)
    ok, enc = cv2.imencode('.png', image)
    assert ok
    path.write_bytes(enc.tobytes())


def test_startup_bootstrap_calibrates_profile(tmp_path: Path) -> None:
    cache_file = tmp_path / 'scale_cache.json'
    checker = tmp_path / 'checker.png'
    _write_checkerboard(checker)

    calibrator = ScaleCalibrator(
        cache_path=str(cache_file),
        default_mm_per_px=0.2,
        board_size=(7, 4),
        board_size_candidates=[(7, 4), (4, 7)],
        square_size_mm=10.0,
    )
    config = {
        'calibration': {
            'default_camera_profiles': {'lab_camera': 'lab_camera', 'default': 'default'},
            'startup_bootstrap': {
                'enabled': True,
                'only_if_missing': True,
                'max_images_per_profile': 3,
                'profiles': [
                    {
                        'camera_id': 'lab_camera',
                        'source_type': 'lab_camera',
                        'image_paths': [str(checker)],
                    }
                ],
            },
        }
    }

    _bootstrap_calibration_profiles(config=config, calibrator=calibrator)
    assert calibrator.is_cache_scale_validated('lab_camera') is True
    profile = calibrator.get_profile('lab_camera')
    assert profile is not None
    assert 0.2 <= float(profile['mm_per_px']) <= 0.3


def test_startup_bootstrap_skips_existing_validated_profile(tmp_path: Path) -> None:
    cache_file = tmp_path / 'scale_cache.json'
    calibrator = ScaleCalibrator(
        cache_path=str(cache_file),
        default_mm_per_px=0.2,
        board_size=(7, 4),
        board_size_candidates=[(7, 4), (4, 7)],
        square_size_mm=10.0,
    )
    calibrator.upsert_scale('lab_camera', 0.1234, fingerprint='manual_bootstrap')

    config = {
        'calibration': {
            'default_camera_profiles': {'lab_camera': 'lab_camera', 'default': 'default'},
            'startup_bootstrap': {
                'enabled': True,
                'only_if_missing': True,
                'max_images_per_profile': 3,
                'profiles': [
                    {
                        'camera_id': 'lab_camera',
                        'source_type': 'lab_camera',
                        'image_paths': [str(tmp_path / 'missing.png')],
                    }
                ],
            },
        }
    }

    _bootstrap_calibration_profiles(config=config, calibrator=calibrator)
    profile = calibrator.get_profile('lab_camera')
    assert profile is not None
    assert abs(float(profile['mm_per_px']) - 0.1234) < 1e-9
