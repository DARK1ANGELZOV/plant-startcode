from __future__ import annotations

from typing import Iterable

import pytest

from services.recommendation_service import RecommendationService


THRESHOLDS = {
    'Arugula': {
        'min_root_length_mm': 18.0,
        'min_stem_length_mm': 12.0,
        'min_leaf_area_mm2': 95.0,
        'min_leaf_root_ratio': 1.2,
        'max_leaf_cv': 0.8,
    },
    'Wheat': {
        'min_root_length_mm': 25.0,
        'min_stem_length_mm': 16.0,
        'min_leaf_area_mm2': 120.0,
        'min_leaf_root_ratio': 1.0,
        'max_leaf_cv': 0.9,
    },
}


def _service() -> RecommendationService:
    return RecommendationService(THRESHOLDS)


def _mk_measurements(
    crop: str,
    root_lengths: Iterable[float],
    stem_lengths: Iterable[float],
    leaf_areas: Iterable[float],
) -> list[dict]:
    rows: list[dict] = []
    idx = 0

    for length in root_lengths:
        rows.append(
            {
                'instance_id': idx,
                'crop': crop,
                'class_name': 'root',
                'confidence': 0.95,
                'area_px': int(max(1, length * 3)),
                'area_mm2': float(max(1.0, length * 3.0)),
                'length_px': float(length / 0.1),
                'length_mm': float(length),
            }
        )
        idx += 1

    for length in stem_lengths:
        rows.append(
            {
                'instance_id': idx,
                'crop': crop,
                'class_name': 'stem',
                'confidence': 0.95,
                'area_px': int(max(1, length * 2)),
                'area_mm2': float(max(1.0, length * 2.0)),
                'length_px': float(length / 0.1),
                'length_mm': float(length),
            }
        )
        idx += 1

    for area in leaf_areas:
        rows.append(
            {
                'instance_id': idx,
                'crop': crop,
                'class_name': 'leaves',
                'confidence': 0.95,
                'area_px': int(max(1, area)),
                'area_mm2': float(max(1.0, area)),
                'length_px': float((area ** 0.5) / 0.1),
                'length_mm': float(area ** 0.5),
            }
        )
        idx += 1

    return rows


def _has_message(recs, token: str) -> bool:
    return any(token in r.message for r in recs)


def test_generate_empty_measurements_returns_warning() -> None:
    recs = _service().generate([])
    assert len(recs) == 1
    assert recs[0].severity == 'warning'
    assert 'не найдено сегментированных частей' in recs[0].message


ROOT_CASES = [
    (crop, root_len, root_len < THRESHOLDS[crop]['min_root_length_mm'])
    for crop in ('Arugula', 'Wheat')
    for root_len in (8.0, 10.0, 12.0, 15.0, 17.0, 18.0, 20.0, 22.0, 24.0, 26.0, 30.0, 35.0)
]


@pytest.mark.parametrize(('crop', 'root_len', 'expect_critical'), ROOT_CASES)
def test_root_threshold_triggers_critical(crop: str, root_len: float, expect_critical: bool) -> None:
    recs = _service().generate(
        _mk_measurements(crop, root_lengths=(root_len,), stem_lengths=(25.0,), leaf_areas=(500.0, 500.0))
    )
    assert _has_message(recs, 'корневая система недоразвита') is expect_critical
    if expect_critical:
        assert any(r.severity == 'critical' for r in recs)


STEM_CASES = [
    (crop, stem_len, stem_len < THRESHOLDS[crop]['min_stem_length_mm'])
    for crop in ('Arugula', 'Wheat')
    for stem_len in (8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0)
]


@pytest.mark.parametrize(('crop', 'stem_len', 'expect_warning'), STEM_CASES)
def test_stem_threshold_triggers_warning(crop: str, stem_len: float, expect_warning: bool) -> None:
    recs = _service().generate(
        _mk_measurements(crop, root_lengths=(35.0,), stem_lengths=(stem_len,), leaf_areas=(500.0, 500.0))
    )
    assert _has_message(recs, 'слабое развитие стебля') is expect_warning


LEAF_CASES = [
    (crop, leaf_area, leaf_area < THRESHOLDS[crop]['min_leaf_area_mm2'])
    for crop in ('Arugula', 'Wheat')
    for leaf_area in (40.0, 60.0, 80.0, 95.0, 100.0, 110.0, 120.0, 140.0, 180.0, 220.0)
]


@pytest.mark.parametrize(('crop', 'leaf_area', 'expect_warning'), LEAF_CASES)
def test_leaf_area_threshold_triggers_warning(crop: str, leaf_area: float, expect_warning: bool) -> None:
    recs = _service().generate(
        _mk_measurements(crop, root_lengths=(30.0,), stem_lengths=(20.0,), leaf_areas=(leaf_area, leaf_area))
    )
    assert _has_message(recs, 'недостаточная площадь листьев') is expect_warning


RATIO_CASES = [
    (crop, ratio, ratio < THRESHOLDS[crop]['min_leaf_root_ratio'])
    for crop in ('Arugula', 'Wheat')
    for ratio in (0.6, 0.8, 0.95, 0.99, 1.0, 1.05, 1.2, 1.5)
]


@pytest.mark.parametrize(('crop', 'ratio', 'expect_warning'), RATIO_CASES)
def test_leaf_root_ratio_warning(crop: str, ratio: float, expect_warning: bool) -> None:
    root_len = 200.0
    leaf_area = root_len * ratio
    recs = _service().generate(
        _mk_measurements(crop, root_lengths=(root_len,), stem_lengths=(30.0,), leaf_areas=(leaf_area, leaf_area))
    )
    assert _has_message(recs, 'дисбаланс лист/корень') is expect_warning


CV_CASES = [
    (crop, cv, cv > THRESHOLDS[crop]['max_leaf_cv'])
    for crop in ('Arugula', 'Wheat')
    for cv in (0.10, 0.20, 0.40, 0.60, 0.80, 0.85, 0.90, 0.95)
]


@pytest.mark.parametrize(('crop', 'cv', 'expect_warning'), CV_CASES)
def test_leaf_area_cv_warning(crop: str, cv: float, expect_warning: bool) -> None:
    mean_area = 200.0
    delta = mean_area * cv
    leaf_areas = (mean_area - delta, mean_area + delta)
    recs = _service().generate(
        _mk_measurements(crop, root_lengths=(30.0,), stem_lengths=(20.0,), leaf_areas=leaf_areas)
    )
    assert _has_message(recs, 'высокая неоднородность листовой площади') is expect_warning


NOISE_CASES = [(count, count > 35) for count in (34, 35, 36, 40, 50, 80)]


@pytest.mark.parametrize(('count', 'expect_warning'), NOISE_CASES)
def test_noise_detection_by_instance_count(count: int, expect_warning: bool) -> None:
    rows = _mk_measurements('Wheat', root_lengths=(30.0,) * count, stem_lengths=(), leaf_areas=())
    recs = _service().generate(rows)
    assert _has_message(recs, 'вероятен шум сегментации') is expect_warning


@pytest.mark.parametrize('crop', ['Arugula', 'Wheat'])
def test_ok_recommendation_for_healthy_profile(crop: str) -> None:
    recs = _service().generate(
        _mk_measurements(crop, root_lengths=(40.0, 42.0), stem_lengths=(24.0, 26.0), leaf_areas=(180.0, 190.0, 200.0))
    )
    assert any(r.severity == 'ok' for r in recs)
    assert any('в целевом диапазоне' in r.message for r in recs)


def test_actions_are_actionable_and_non_empty() -> None:
    recs = _service().generate(
        _mk_measurements('Arugula', root_lengths=(10.0,), stem_lengths=(8.0,), leaf_areas=(50.0, 400.0))
    )
    assert recs
    for rec in recs:
        assert rec.action.strip()
        assert any(verb in rec.action for verb in ('Проверить', 'Увеличить', 'Снизить', 'Сохранять', 'Повысить'))


def test_multi_crop_input_returns_crop_specific_messages() -> None:
    rows = []
    rows.extend(_mk_measurements('Arugula', root_lengths=(12.0,), stem_lengths=(20.0,), leaf_areas=(200.0, 200.0)))
    rows.extend(_mk_measurements('Wheat', root_lengths=(40.0,), stem_lengths=(20.0,), leaf_areas=(200.0, 200.0)))

    recs = _service().generate(rows)
    assert any('Arugula:' in r.message for r in recs)
    assert any('Wheat:' in r.message for r in recs)
