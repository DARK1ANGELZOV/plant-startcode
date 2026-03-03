from __future__ import annotations

from calibration.chessboard import ScaleCalibrator


def test_auto_profile_learns_and_stabilizes(tmp_path) -> None:
    cache_path = tmp_path / 'scale_cache.json'
    calibrator = ScaleCalibrator(
        cache_path=str(cache_path),
        default_mm_per_px=0.12,
        auto_profile_enabled=True,
        auto_min_samples=3,
        auto_stable_samples=6,
        auto_max_cv=0.2,
        charuco_enabled=False,
    )

    samples = [0.108, 0.107, 0.109, 0.106, 0.108, 0.107, 0.108]
    for value in samples:
        calibrator.update_auto_scale(
            mm_per_px=value,
            camera_id='default',
            source_type='lab_camera',
            crop='Wheat',
        )

    scale, source = calibrator.get_auto_scale(
        camera_id='default',
        source_type='lab_camera',
        crop='Wheat',
        min_samples=3,
    )
    assert scale is not None
    assert source is not None
    assert '__auto_profile__' in source
    assert abs(scale - 0.1076) < 0.01
    assert calibrator.is_auto_profile_stable(
        camera_id='default',
        source_type='lab_camera',
        crop='Wheat',
    )


def test_auto_profile_is_persistent(tmp_path) -> None:
    cache_path = tmp_path / 'scale_cache.json'
    calibrator = ScaleCalibrator(
        cache_path=str(cache_path),
        default_mm_per_px=0.12,
        auto_profile_enabled=True,
        auto_min_samples=2,
        auto_stable_samples=4,
        auto_max_cv=0.3,
        charuco_enabled=False,
    )

    for value in [0.101, 0.103, 0.102, 0.102]:
        calibrator.update_auto_scale(
            mm_per_px=value,
            camera_id='mobile_1',
            source_type='phone_camera',
            crop='Arugula',
        )

    reloaded = ScaleCalibrator(
        cache_path=str(cache_path),
        default_mm_per_px=0.12,
        auto_profile_enabled=True,
        auto_min_samples=2,
        auto_stable_samples=4,
        auto_max_cv=0.3,
        charuco_enabled=False,
    )

    scale, source = reloaded.get_auto_scale(
        camera_id='mobile_1',
        source_type='phone_camera',
        crop='Arugula',
        min_samples=2,
    )
    assert source is not None
    assert scale is not None
    assert abs(scale - 0.102) < 0.01
