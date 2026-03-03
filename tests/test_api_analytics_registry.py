from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from api.main import app
from utils.schemas import PlantMeasurement, PredictResponse


def _predict(run_id: str, length_mm: float) -> PredictResponse:
    return PredictResponse(
        run_id=run_id,
        scale_mm_per_px=0.1,
        scale_source='fallback',
        measurements=[
            PlantMeasurement(
                instance_id=1,
                crop='Wheat',
                class_name='root',
                confidence=0.9,
                area_px=200,
                area_mm2=20.0,
                length_px=length_mm / 0.1,
                length_mm=length_mm,
            )
        ],
        summary={'count': 1},
        recommendations=[],
        files={},
    )


def test_analytics_and_registry_endpoints() -> None:
    with TestClient(app) as client:
        tenant = f'tenant_{uuid4().hex[:6]}'
        run_a = f'run_{uuid4().hex[:8]}'
        run_b = f'run_{uuid4().hex[:8]}'

        app.state.history_service.register_result(_predict(run_a, 10.0), tenant_id=tenant)
        app.state.history_service.register_result(_predict(run_b, 15.0), tenant_id=tenant)

        runs = client.get('/analytics/runs', headers={'X-Tenant-ID': tenant})
        assert runs.status_code == 200
        assert isinstance(runs.json(), list)
        assert any(x['run_id'] == run_a for x in runs.json())

        trend = client.get(
            '/analytics/trends',
            params={'crop': 'Wheat', 'class_name': 'root', 'metric': 'avg_length_mm'},
            headers={'X-Tenant-ID': tenant},
        )
        assert trend.status_code == 200
        assert trend.json()['metric'] == 'avg_length_mm'

        compare = client.get(
            '/analytics/compare',
            params={
                'run_a': run_a,
                'run_b': run_b,
                'crop': 'Wheat',
                'class_name': 'root',
                'metric': 'avg_length_mm',
            },
            headers={'X-Tenant-ID': tenant},
        )
        assert compare.status_code == 200
        assert compare.json()['delta'] == 5.0

        model_reg = client.post(
            '/models/register',
            json={
                'path': f'models/{uuid4().hex}.pt',
                'metrics': {'map50': 0.77},
                'dataset_version': 'v-test',
                'tags': ['test'],
                'source': 'pytest',
            },
        )
        assert model_reg.status_code == 200
        assert 'version_id' in model_reg.json()

        best = client.get('/models/best', params={'metric': 'map50'})
        assert best.status_code == 200

        ds_reg = client.post(
            '/datasets/register',
            json={
                'dataset_version': f'pytest-{uuid4().hex[:6]}',
                'source': 'roboflow',
                'task_type': 'instance_segmentation',
                'classes': ['root', 'stem', 'leaves'],
                'augmentation': {'flipud': 0.1},
                'notes': 'test dataset',
            },
        )
        assert ds_reg.status_code == 200
        assert ds_reg.json()['source'] == 'roboflow'

        ds_versions = client.get('/datasets/versions')
        assert ds_versions.status_code == 200
        assert isinstance(ds_versions.json(), list)
