from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from errno import ENOSPC
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import pandas as pd


class StorageService:
    def __init__(
        self,
        output_root: str,
        *,
        retention_cfg: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self._logger = logger or logging.getLogger(__name__)
        cfg = retention_cfg or {}
        self._retention_enabled = bool(cfg.get('enabled', True))
        self._keep_latest_runs = max(0, int(cfg.get('keep_latest_runs', 300)))
        self._min_free_gb = max(0.1, float(cfg.get('min_free_gb', 3.0)))
        self._target_free_gb = max(self._min_free_gb, float(cfg.get('target_free_gb', 6.0)))
        self._delete_batch = max(1, int(cfg.get('delete_batch', 50)))
        self._warn_only = bool(cfg.get('warn_only', False))

    def _run_dirs(self) -> list[Path]:
        runs = [p for p in self.output_root.glob('run_*') if p.is_dir()]
        runs.sort(key=lambda p: p.stat().st_mtime)
        return runs

    def _disk_free_gb(self) -> float:
        usage = shutil.disk_usage(self.output_root)
        return float(usage.free) / float(1024**3)

    def _disk_total_gb(self) -> float:
        usage = shutil.disk_usage(self.output_root)
        return float(usage.total) / float(1024**3)

    def _delete_run_dirs(self, dirs: list[Path]) -> int:
        removed = 0
        for p in dirs:
            try:
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
            except Exception as exc:
                self._logger.warning('Failed to remove run dir %s: %s', p, exc)
        return removed

    def _enforce_retention(self) -> None:
        if not self._retention_enabled:
            return

        removed_total = 0
        runs = self._run_dirs()

        if len(runs) > self._keep_latest_runs:
            to_remove = runs[: max(0, len(runs) - self._keep_latest_runs)]
            removed_total += self._delete_run_dirs(to_remove)
            runs = self._run_dirs()

        free_gb = self._disk_free_gb()
        if free_gb < self._min_free_gb:
            self._logger.warning(
                'Low disk space detected (free=%.2f GB, min=%.2f GB). Running aggressive cleanup.',
                free_gb,
                self._min_free_gb,
            )
            while free_gb < self._target_free_gb and runs:
                chunk = runs[: self._delete_batch]
                removed_total += self._delete_run_dirs(chunk)
                runs = self._run_dirs()
                free_gb = self._disk_free_gb()

        free_gb = self._disk_free_gb()
        if removed_total > 0:
            self._logger.info(
                'Storage retention removed %s run directories. Free space: %.2f GB.',
                removed_total,
                free_gb,
            )

        if free_gb < self._min_free_gb:
            msg = (
                f'Insufficient disk space in {self.output_root}: '
                f'free={free_gb:.2f} GB, required={self._min_free_gb:.2f} GB.'
            )
            if self._warn_only:
                self._logger.warning(msg)
            else:
                raise OSError(ENOSPC, msg)

    def health_status(self) -> dict[str, Any]:
        free_gb = self._disk_free_gb()
        total_gb = self._disk_total_gb()
        run_count = len(self._run_dirs())
        low = free_gb < self._min_free_gb
        return {
            'free_gb': round(free_gb, 2),
            'total_gb': round(total_gb, 2),
            'run_dirs_count': int(run_count),
            'low_space': bool(low),
            'min_free_gb': round(self._min_free_gb, 2),
            'retention_enabled': bool(self._retention_enabled),
        }

    def create_run_dir(self) -> tuple[str, Path]:
        self._enforce_retention()
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    @staticmethod
    def save_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    @staticmethod
    def save_image(path: Path, image) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)

    @staticmethod
    def save_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')

    @staticmethod
    def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
