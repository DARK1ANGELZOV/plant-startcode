from __future__ import annotations

from pathlib import Path

from services.history_service import RunHistoryService
from utils.schemas import PlantMeasurement, PredictResponse


def _predict(run_id: str, root_len: float, crop: str = 'Wheat') -> PredictResponse:
    measurement = PlantMeasurement(
        instance_id=1,
        crop=crop,
        class_name='root',
        confidence=0.9,
        area_px=120,
        area_mm2=12.0,
        length_px=root_len / 0.1,
        length_mm=root_len,
    )
    return PredictResponse(
        run_id=run_id,
        scale_mm_per_px=0.1,
        scale_source='fallback',
        measurements=[measurement],
        summary={'count': 1},
        recommendations=[],
        files={'overlay': f'outputs/{run_id}/overlay.png'},
    )


def test_register_and_list_runs(tmp_path: Path) -> None:
    svc = RunHistoryService(tmp_path / 'run_history.json')

    svc.register_result(_predict('run_1', 20.0), tenant_id='tenant_a', source_type='smartphone', camera_id='cam_1')
    svc.register_result(_predict('run_2', 22.0), tenant_id='tenant_a', source_type='drone', camera_id='cam_2')

    items = svc.list_runs(tenant_id='tenant_a', crop='Wheat', limit=10)
    assert len(items) == 2
    assert items[0].tenant_id == 'tenant_a'
    assert all('Wheat' in x.crops for x in items)


def test_trend_and_compare(tmp_path: Path) -> None:
    svc = RunHistoryService(tmp_path / 'run_history.json')

    svc.register_result(_predict('run_a', 10.0), tenant_id='tenant_t')
    svc.register_result(_predict('run_b', 20.0), tenant_id='tenant_t')

    trend = svc.trend(
        crop='Wheat',
        class_name='root',
        metric='avg_length_mm',
        tenant_id='tenant_t',
        limit=10,
    )
    assert len(trend.points) == 2
    assert trend.latest_value == 20.0
    assert trend.slope_per_step > 0

    cmp = svc.compare_runs(
        run_a='run_a',
        run_b='run_b',
        crop='Wheat',
        class_name='root',
        metric='avg_length_mm',
        tenant_id='tenant_t',
    )
    assert cmp.delta == 10.0
    assert cmp.delta_pct == 100.0
