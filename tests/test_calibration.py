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
        scene_aware_cache_enabled=False,
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
        scene_aware_cache_enabled=False,
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
        allow_legacy_cache_without_scene=True,
    )
    calibrator.upsert_scale('cam_custom', 0.1234, fingerprint='fit')
    scale, source = calibrator.get_scale(None, camera_id='cam_custom', use_cache=True)
    assert abs(scale - 0.1234) < 1e-8
    assert source == 'cache'


def test_scene_aware_cache_rejects_other_scene(tmp_path: Path) -> None:
    cache_file = tmp_path / 'scale_cache.json'
    calibrator = ScaleCalibrator(
        cache_path=str(cache_file),
        default_mm_per_px=0.2,
        board_size=(7, 7),
        square_size_mm=5.0,
        scene_aware_cache_enabled=True,
        allow_legacy_cache_without_scene=False,
    )
    img_a = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.rectangle(img_a, (20, 20), (100, 100), (255, 255, 255), -1)
    img_b = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(img_b, (60, 60), 35, (255, 255, 255), -1)

    sig_a = calibrator._scene_signature(img_a)
    assert sig_a is not None
    calibrator.upsert_scale('cam_scene', 0.1234, fingerprint='manual_api', scene_signature=sig_a)

    scale_ok, source_ok = calibrator.get_scale(img_a, camera_id='cam_scene', use_cache=True)
    assert abs(scale_ok - 0.1234) < 1e-8
    assert source_ok in {'cache_scene', 'cache_scene_near'}

    scale_miss, source_miss = calibrator.get_scale(img_b, camera_id='cam_scene', use_cache=True)
    assert abs(scale_miss - 0.2) < 1e-8
    assert source_miss == 'fallback'


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


def test_calibrate_and_store_marks_cache_as_validated(tmp_path: Path) -> None:
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

    scale, source = calibrator.calibrate_and_store(image=image, camera_id='lab_camera')
    assert source == 'chessboard'
    assert scale is not None
    assert calibrator.is_cache_scale_validated('lab_camera') is True

    cached_scale, cached_source = calibrator.get_scale(image, camera_id='lab_camera', use_cache=True)
    assert cached_source in {'chessboard', 'cache_scene', 'cache_scene_near'}
    assert abs(float(cached_scale) - float(scale)) < 1e-6

    no_board = np.zeros((120, 120, 3), dtype=np.uint8)
    fallback_scale, fallback_source = calibrator.get_scale(no_board, camera_id='lab_camera', use_cache=True)
    assert fallback_source == 'fallback'
    assert abs(float(fallback_scale) - 0.2) < 1e-8


def test_list_profiles_skips_auto_profile_keys(tmp_path: Path) -> None:
    cache_file = tmp_path / 'scale_cache.json'
    calibrator = ScaleCalibrator(
        cache_path=str(cache_file),
        default_mm_per_px=0.2,
        board_size=(7, 4),
        square_size_mm=10.0,
    )
    calibrator.upsert_scale('lab_camera', 0.12, fingerprint='manual_api')
    calibrator.update_auto_scale(0.12, camera_id='default', source_type='lab_camera', crop='Wheat')
    calibrator.update_auto_scale(0.121, camera_id='default', source_type='lab_camera', crop='Wheat')
    calibrator.update_auto_scale(0.119, camera_id='default', source_type='lab_camera', crop='Wheat')

    profiles = calibrator.list_profiles(validated_only=True)
    ids = {p['camera_id'] for p in profiles}
    assert 'lab_camera' in ids
    assert all(not str(x).startswith(ScaleCalibrator.AUTO_PREFIX) for x in ids)
