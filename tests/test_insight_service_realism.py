from __future__ import annotations

import pytest

from services.insight_service import InsightService
from utils.schemas import PlantMeasurement, PredictResponse, Recommendation


def _measurement(
    *,
    instance_id: int,
    cls: str,
    confidence: float,
    length_px: float,
    area_px: float,
    length_mm: float | None = None,
    area_mm2: float | None = None,
    crop: str = 'Wheat',
) -> PlantMeasurement:
    return PlantMeasurement(
        instance_id=instance_id,
        crop=crop,
        class_name=cls,
        confidence=confidence,
        area_px=max(1, int(area_px)),
        area_mm2=area_mm2,
        length_px=length_px,
        length_mm=length_mm,
    )


def _response(
    measurements: list[PlantMeasurement],
    recommendations: list[Recommendation] | None = None,
    *,
    scale_mm_per_px: float = 0.123456,
    scale_source: str = 'chessboard',
    summary: dict | None = None,
) -> PredictResponse:
    return PredictResponse(
        run_id='run_quality',
        scale_mm_per_px=scale_mm_per_px,
        scale_source=scale_source,
        measurements=measurements,
        summary=summary or {},
        recommendations=recommendations or [],
        files={},
    )


def test_reply_contains_compact_sections() -> None:
    service = InsightService()
    result = _response(
        measurements=[
            _measurement(
                instance_id=1,
                cls='root',
                confidence=0.9,
                length_px=200.0,
                area_px=1200.0,
                length_mm=24.0,
                area_mm2=150.0,
            )
        ],
        recommendations=[Recommendation(severity='ok', message='Все стабильно', action='Продолжать мониторинг.')],
        summary={'calibration_reliable': True},
    )
    text = service.compose_reply(result, user_message='Сделай анализ')
    assert 'Результаты анализа изображения:' in text
    assert '1. Сегментация:' in text
    assert '2. Измерения (мм):' in text
    assert '3. Перевод в мм:' in text
    assert '4. Вывод:' in text
    assert '5. Рекомендации:' in text


@pytest.mark.parametrize('user_message', ['  Проверить лист  ', 'root?', 'стебель', '  ', '\n\t'])
def test_reply_user_message_trimming(user_message: str) -> None:
    service = InsightService()
    result = _response(measurements=[_measurement(instance_id=1, cls='root', confidence=0.9, length_px=100.0, area_px=500.0)])
    text = service.compose_reply(result, user_message=user_message)
    assert 'Результаты анализа изображения:' in text
    assert '5. Рекомендации:' in text


def test_unreliable_scale_blocks_mm_conversion() -> None:
    service = InsightService()
    result = _response(
        measurements=[_measurement(instance_id=1, cls='stem', confidence=0.8, length_px=120.0, area_px=1000.0)],
        scale_mm_per_px=0.12,
        scale_source='fallback',
        summary={'calibration_reliable': False},
    )
    text = service.compose_reply(result)
    assert 'Перевод в мм невозможен' in text
    assert 'нет валидной геометрической калибровки' in text


def test_reliable_scale_shows_metric_mm() -> None:
    service = InsightService()
    result = _response(
        measurements=[
            _measurement(instance_id=1, cls='root', confidence=0.91, length_px=120.0, area_px=900.0, length_mm=12.0, area_mm2=90.0),
            _measurement(instance_id=2, cls='leaves', confidence=0.92, length_px=200.0, area_px=2400.0, length_mm=20.0, area_mm2=240.0),
        ],
        scale_mm_per_px=0.1,
        scale_source='chessboard',
        summary={'calibration_reliable': True},
    )
    text = service.compose_reply(result)
    assert 'Источник: chessboard' in text
    assert 'Длина корня' in text
    assert 'Площадь листьев' in text


def test_reply_uses_prior_context_feedback() -> None:
    service = InsightService()
    result = _response(
        measurements=[_measurement(instance_id=1, cls='stem', confidence=0.8, length_px=90.0, area_px=300.0)],
        recommendations=[],
    )
    text = service.compose_reply(
        result,
        prior_context=['В прошлый раз это не помогло', 'Стало хуже после полива'],
    )
    assert 'прошлую неудачу' in text


def test_recommendations_fallback_when_empty() -> None:
    service = InsightService()
    result = _response(measurements=[], recommendations=[])
    text = service.compose_reply(result)
    assert 'Переснимите фото' in text
    assert 'проблемную зону' in text


def test_reply_uses_px_when_mm_not_reliable() -> None:
    service = InsightService()
    result = _response(
        measurements=[
            _measurement(instance_id=1, cls='root', confidence=0.92, length_px=140.0, area_px=1200.0),
        ],
        scale_mm_per_px=0.12,
        scale_source='fallback',
        summary={'calibration_reliable': False},
    )
    text = service.compose_reply(result)
    assert '2. Измерения (px):' in text
    assert 'Длина корня: 140.0 px' in text
    assert 'Перевод в мм невозможен' in text


def test_reply_hides_low_confidence_numbers() -> None:
    service = InsightService({'min_confidence_for_numeric': 0.8})
    result = _response(
        measurements=[
            _measurement(
                instance_id=1,
                cls='stem',
                confidence=0.5,
                length_px=99.0,
                area_px=500.0,
                length_mm=9.9,
                area_mm2=50.0,
            )
        ],
        summary={'calibration_reliable': True},
    )
    text = service.compose_reply(result)
    assert '2. Измерения (мм):' in text
    assert 'Длина стебля: н/д' in text
