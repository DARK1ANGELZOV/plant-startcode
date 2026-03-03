from __future__ import annotations

from pathlib import Path

from services.dataset_registry_service import DatasetRegistryService
from services.model_registry_service import ModelRegistryService
from utils.schemas import RegisterDatasetRequest, RegisterModelRequest


def test_model_registry_register_and_best(tmp_path: Path) -> None:
    svc = ModelRegistryService(tmp_path / 'models_registry.json')

    first = svc.register(
        RegisterModelRequest(
            path='models/m1.pt',
            metrics={'map50': 0.44, 'precision': 0.52},
            dataset_version='v1',
            tags=['baseline'],
        )
    )
    second = svc.register(
        RegisterModelRequest(
            path='models/m2.pt',
            metrics={'map50': 0.62, 'precision': 0.66},
            dataset_version='v2',
            tags=['improved'],
        )
    )

    assert first.version_id != second.version_id
    versions = svc.list_versions(limit=10)
    assert len(versions) == 2

    best = svc.best_by_metric('map50')
    assert best is not None
    assert best.path == 'models/m2.pt'


def test_dataset_registry_upsert_and_list(tmp_path: Path) -> None:
    svc = DatasetRegistryService(tmp_path / 'datasets_registry.json')

    req = RegisterDatasetRequest(
        dataset_version='roboflow-v12',
        source='roboflow',
        classes=['root', 'stem', 'leaves'],
        augmentation={'mosaic': 0.2, 'hsv_s': 0.5},
        notes='production candidate',
    )
    first = svc.register(req)
    second = svc.register(req)

    versions = svc.list_versions(limit=10)
    assert len(versions) == 1
    assert versions[0].dataset_version == 'roboflow-v12'
    assert first.dataset_version == second.dataset_version
