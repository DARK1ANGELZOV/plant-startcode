from services.insight_service import InsightService
from utils.schemas import PlantMeasurement, PredictResponse, Recommendation


def test_compose_reply_with_measurements() -> None:
    service = InsightService()
    result = PredictResponse(
        run_id='run_test',
        scale_mm_per_px=0.1,
        scale_source='chessboard',
        measurements=[
            PlantMeasurement(
                instance_id=0,
                crop='Wheat',
                class_name='root',
                confidence=0.9,
                area_px=200,
                area_mm2=2.0,
                length_px=40.0,
                length_mm=4.0,
            )
        ],
        summary={},
        recommendations=[
            Recommendation(
                severity='warning',
                message='test warning',
                action='test action',
            )
        ],
        files={},
    )
    text = service.compose_reply(result, user_message='Проверь корень')
    assert 'Результаты анализа изображения:' in text
    assert '1. Сегментация:' in text
    assert '5. Рекомендации:' in text


def test_compose_reply_without_measurements() -> None:
    service = InsightService()
    result = PredictResponse(
        run_id='run_empty',
        scale_mm_per_px=0.12,
        scale_source='fallback',
        measurements=[],
        summary={},
        recommendations=[],
        files={},
    )
    text = service.compose_reply(result, user_message=None)
    assert 'Результаты анализа изображения:' in text
    assert '3. Перевод в мм:' in text
    assert 'Могу продолжить диалог' in text
