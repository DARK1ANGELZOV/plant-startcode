from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.schemas import CompareRunsResponse, PredictResponse, RunRecord, TrendPoint, TrendResponse

logger = logging.getLogger(__name__)


class RunHistoryService:
    def __init__(self, history_path: str | Path = 'outputs/run_history.json') -> None:
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_path.exists():
            self._bootstrap_from_outputs()

    def _bootstrap_from_outputs(self) -> None:
        records: list[dict[str, Any]] = []
        outputs_dir = self.history_path.parent
        for result_path in sorted(outputs_dir.glob('run_*/result.json')):
            try:
                payload = json.loads(result_path.read_text(encoding='utf-8'))
                measurements = payload.get('measurements', [])
                record = {
                    'run_id': payload['run_id'],
                    'created_at': datetime.fromtimestamp(result_path.stat().st_mtime, tz=timezone.utc).isoformat(),
                    'tenant_id': 'default',
                    'source_type': 'unknown',
                    'camera_id': 'default',
                    'scale_mm_per_px': float(payload.get('scale_mm_per_px', 0.0)),
                    'scale_source': str(payload.get('scale_source', 'unknown')),
                    'count': int(len(measurements)),
                    'crops': sorted({str(m.get('crop', 'Unknown')) for m in measurements}),
                    'traits': self._compute_traits(measurements),
                    'summary': payload.get('summary', {}),
                    'files': payload.get('files', {}),
                }
                records.append(record)
            except Exception:
                continue

        self._save_records(records)

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        try:
            payload = json.loads(self.history_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            # Recover from partial/empty writes without crashing request handlers.
            logger.warning('History file is corrupted, resetting: %s', self.history_path)
            self._save_records([])
            return []
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        return payload

    def _save_records(self, records: list[dict[str, Any]]) -> None:
        self.history_path.write_text(json.dumps(records, ensure_ascii=True, indent=2), encoding='utf-8')

    @staticmethod
    def _compute_traits(measurements: list[dict[str, Any]]) -> dict[str, Any]:
        def _safe_float(value: Any) -> float | None:
            if value is None:
                return None
            try:
                return float(value)
            except Exception:
                return None

        grouped: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
        for row in measurements:
            crop = str(row.get('crop', 'Unknown'))
            cls = str(row.get('class_name', 'unknown'))
            length_mm = _safe_float(row.get('length_mm'))
            area_mm2 = _safe_float(row.get('area_mm2'))
            grouped[crop][cls].append(
                {
                    'length_px': float(row.get('length_px', 0.0)),
                    'area_px': float(row.get('area_px', 0.0)),
                    'length_mm': length_mm,
                    'area_mm2': area_mm2,
                    'confidence': float(row.get('confidence', 0.0)),
                }
            )

        traits: dict[str, Any] = {}
        for crop, by_class in grouped.items():
            traits[crop] = {}
            for class_name, rows in by_class.items():
                n = len(rows)
                if n == 0:
                    continue
                avg_length_px = sum(r['length_px'] for r in rows) / n
                avg_area_px = sum(r['area_px'] for r in rows) / n
                mm_rows = [r for r in rows if r['length_mm'] is not None and r['area_mm2'] is not None]
                avg_conf = sum(r['confidence'] for r in rows) / n
                traits[crop][class_name] = {
                    'count': n,
                    'avg_length_px': round(avg_length_px, 6),
                    'avg_area_px': round(avg_area_px, 6),
                    'avg_confidence': round(avg_conf, 6),
                }
                if mm_rows:
                    avg_length_mm = sum(float(r['length_mm']) for r in mm_rows) / len(mm_rows)
                    avg_area_mm2 = sum(float(r['area_mm2']) for r in mm_rows) / len(mm_rows)
                    traits[crop][class_name]['avg_length_mm'] = round(avg_length_mm, 6)
                    traits[crop][class_name]['avg_area_mm2'] = round(avg_area_mm2, 6)
        return traits

    def register_result(
        self,
        result: PredictResponse,
        tenant_id: str = 'default',
        source_type: str = 'unknown',
        camera_id: str = 'default',
    ) -> RunRecord:
        records = self._load_records()
        measurements = [m.model_dump() for m in result.measurements]
        record = RunRecord(
            run_id=result.run_id,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            tenant_id=tenant_id,
            source_type=source_type,
            camera_id=camera_id,
            scale_mm_per_px=float(result.scale_mm_per_px),
            scale_source=str(result.scale_source),
            count=len(measurements),
            crops=sorted({str(m.get('crop', 'Unknown')) for m in measurements}),
            traits=self._compute_traits(measurements),
            summary=result.summary,
            files=result.files,
        )

        records = [r for r in records if str(r.get('run_id')) != result.run_id]
        records.append(record.model_dump())
        records.sort(key=lambda r: str(r.get('created_at', '')), reverse=True)
        self._save_records(records)
        return record

    def list_runs(
        self,
        tenant_id: str = 'default',
        crop: str | None = None,
        limit: int = 50,
    ) -> list[RunRecord]:
        records = self._load_records()
        items: list[RunRecord] = []
        for row in records:
            if tenant_id and row.get('tenant_id') != tenant_id:
                continue
            if crop and crop not in set(row.get('crops', [])):
                continue
            items.append(RunRecord(**row))
            if len(items) >= limit:
                break
        return items

    def get_run(self, run_id: str, tenant_id: str | None = None) -> RunRecord | None:
        records = self._load_records()
        for row in records:
            if str(row.get('run_id')) != run_id:
                continue
            if tenant_id and row.get('tenant_id') != tenant_id:
                continue
            return RunRecord(**row)
        return None

    @staticmethod
    def _extract_metric(record: RunRecord, crop: str, class_name: str, metric: str) -> float:
        crop_blob = record.traits.get(crop, {})
        class_blob = crop_blob.get(class_name, {})
        value = class_blob.get(metric)
        if value is None:
            raise ValueError(
                f'Metric {metric} for crop={crop} class={class_name} is not available in run {record.run_id}.'
            )
        return float(value)

    def compare_runs(
        self,
        run_a: str,
        run_b: str,
        crop: str,
        class_name: str,
        metric: str = 'avg_length_mm',
        tenant_id: str | None = None,
    ) -> CompareRunsResponse:
        a = self.get_run(run_a, tenant_id=tenant_id)
        b = self.get_run(run_b, tenant_id=tenant_id)
        if a is None or b is None:
            raise ValueError('One or both runs not found for comparison.')

        value_a = self._extract_metric(a, crop, class_name, metric)
        value_b = self._extract_metric(b, crop, class_name, metric)

        delta = value_b - value_a
        delta_pct = (delta / value_a * 100.0) if value_a != 0 else 0.0

        return CompareRunsResponse(
            run_a=run_a,
            run_b=run_b,
            crop=crop,
            class_name=class_name,
            metric=metric,
            value_a=value_a,
            value_b=value_b,
            delta=delta,
            delta_pct=delta_pct,
        )

    def trend(
        self,
        crop: str,
        class_name: str,
        metric: str = 'avg_length_mm',
        tenant_id: str = 'default',
        limit: int = 30,
    ) -> TrendResponse:
        runs = list(reversed(self.list_runs(tenant_id=tenant_id, crop=crop, limit=max(1, limit))))

        points: list[TrendPoint] = []
        for run in runs:
            try:
                value = self._extract_metric(run, crop, class_name, metric)
            except ValueError:
                continue
            points.append(
                TrendPoint(
                    run_id=run.run_id,
                    created_at=run.created_at,
                    value=value,
                )
            )

        if len(points) >= 2:
            slope = (points[-1].value - points[0].value) / (len(points) - 1)
        else:
            slope = 0.0

        latest = points[-1].value if points else None
        return TrendResponse(
            crop=crop,
            class_name=class_name,
            metric=metric,
            points=points,
            slope_per_step=float(slope),
            latest_value=latest,
        )
