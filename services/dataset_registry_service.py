from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from utils.schemas import DatasetVersionEntry, RegisterDatasetRequest


class DatasetRegistryService:
    def __init__(self, registry_path: str | Path = 'data/datasets_registry.json') -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._save([])

    def _load(self) -> list[dict]:
        payload = json.loads(self.registry_path.read_text(encoding='utf-8'))
        if not isinstance(payload, list):
            return []
        return payload

    def _save(self, rows: list[dict]) -> None:
        self.registry_path.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding='utf-8')

    def list_versions(self, limit: int = 100) -> list[DatasetVersionEntry]:
        rows = self._load()
        rows.sort(key=lambda x: str(x.get('created_at', '')), reverse=True)
        return [DatasetVersionEntry(**r) for r in rows[:limit]]

    def register(self, request: RegisterDatasetRequest) -> DatasetVersionEntry:
        rows = self._load()

        rows = [r for r in rows if str(r.get('dataset_version')) != request.dataset_version]
        entry = DatasetVersionEntry(
            dataset_version=request.dataset_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            source=request.source,
            task_type=request.task_type,
            classes=request.classes,
            augmentation=request.augmentation,
            notes=request.notes,
        )

        rows.append(entry.model_dump())
        rows.sort(key=lambda x: str(x.get('created_at', '')), reverse=True)
        self._save(rows)
        return entry
