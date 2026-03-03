from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from utils.schemas import ModelVersionEntry, RegisterModelRequest


class ModelRegistryService:
    def __init__(self, registry_path: str | Path = 'models/registry.json') -> None:
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

    def list_versions(self, limit: int = 100) -> list[ModelVersionEntry]:
        rows = self._load()
        rows.sort(key=lambda x: str(x.get('created_at', '')), reverse=True)
        return [ModelVersionEntry(**r) for r in rows[:limit]]

    def register(self, request: RegisterModelRequest) -> ModelVersionEntry:
        rows = self._load()
        now = datetime.now(timezone.utc).isoformat()
        version_id = f"m_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

        entry = ModelVersionEntry(
            version_id=version_id,
            created_at=now,
            path=request.path,
            metrics={k: float(v) for k, v in request.metrics.items()},
            dataset_version=request.dataset_version,
            tags=request.tags,
            source=request.source,
        )

        rows.append(entry.model_dump())
        rows.sort(key=lambda x: str(x.get('created_at', '')), reverse=True)
        self._save(rows)
        return entry

    def ensure_registered(self, path: str, source: str = 'bootstrap') -> ModelVersionEntry | None:
        rows = self._load()
        for row in rows:
            if str(row.get('path')) == path:
                return ModelVersionEntry(**row)

        if not Path(path).exists():
            return None

        return self.register(
            RegisterModelRequest(
                path=path,
                source=source,
                tags=['auto-registered'],
            )
        )

    def best_by_metric(self, metric: str = 'map50') -> ModelVersionEntry | None:
        candidates = [x for x in self.list_versions(limit=1000) if metric in x.metrics]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x.metrics.get(metric, float('-inf')))
