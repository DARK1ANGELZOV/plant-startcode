from pathlib import Path

import cv2
import numpy as np

from calibration.chessboard import ScaleCalibrator


def test_fallback_without_chessboard(tmp_path: Path) -> None:
    cache_file = tmp_path / 'scale_cache.json'
    calibrator = ScaleCalibrator(
        cache_path=str(cache_file),
        default_mm_per_px=0.2,
        board_size=(7, 7),
        square_size_mm=5.0,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    scale, source = calibrator.get_scale(image, camera_id='cam_a')
    assert abs(scale - 0.2) < 1e-8
    assert source == 'fallback'


def test_cache_used_when_no_new_chessboard(tmp_path: Path) -> None:
    cache_file = tmp_path / 'scale_cache.json'
    cache_file.write_text('{"cam_b": {"mm_per_px": 0.155, "fingerprint": "abc"}}', encoding='utf-8')

    calibrator = ScaleCalibrator(
        cache_path=str(cache_file),
        default_mm_per_px=0.2,
        board_size=(7, 7),
        square_size_mm=5.0,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    scale, source = calibrator.get_scale(image, camera_id='cam_b')
    assert abs(scale - 0.155) < 1e-8
    assert source == 'cache'


def test_cache_can_be_disabled(tmp_path: Path) -> None:
    cache_file = tmp_path / 'scale_cache.json'
    cache_file.write_text('{"cam_b": {"mm_per_px": 0.155, "fingerprint": "abc"}}', encoding='utf-8')

    calibrator = ScaleCalibrator(
        cache_path=str(cache_file),
        default_mm_per_px=0.2,
        board_size=(7, 7),
        square_size_mm=5.0,
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    scale, source = calibrator.get_scale(image, camera_id='cam_b', use_cache=False)
    assert abs(scale - 0.2) < 1e-8
    assert source == 'fallback'


def test_upsert_scale_updates_cache(tmp_path: Path) -> None:
    cache_file = tmp_path / 'scale_cache.json'
    calibrator = ScaleCalibrator(
        cache_path=str(cache_file),
        default_mm_per_px=0.2,
        board_size=(7, 7),
        square_size_mm=5.0,
    )
    calibrator.upsert_scale('cam_custom', 0.1234, fingerprint='fit')
    scale, source = calibrator.get_scale(None, camera_id='cam_custom', use_cache=True)
    assert abs(scale - 0.1234) < 1e-8
    assert source == 'cache'


def test_estimate_scale_none_image() -> None:
    calibrator = ScaleCalibrator(
        cache_path='calibration/scale_cache.json',
        default_mm_per_px=0.2,
        board_size=(7, 7),
        square_size_mm=5.0,
    )
    scale, source = calibrator.estimate_scale(None)
    assert scale is None
    assert source is None


def test_detects_5x8_checkerboard_with_10mm_cells(tmp_path: Path) -> None:
    cache_file = tmp_path / 'scale_cache.json'
    calibrator = ScaleCalibrator(
        cache_path=str(cache_file),
        default_mm_per_px=0.2,
        board_size=(7, 4),
        board_size_candidates=[(7, 4), (4, 7)],
        square_size_mm=10.0,
    )

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

    scale, source = calibrator.get_scale(image, camera_id='cam_checker', use_cache=False)
    assert source == 'chessboard'
    assert 0.22 <= scale <= 0.28
