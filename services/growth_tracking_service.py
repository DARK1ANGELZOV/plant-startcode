from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy.orm import Session

from inference.predictor import Detection
from morphometry.analysis import analyze_mask
from services.phi_service import PHIService
from services.storage_service import StorageService
from utils.schemas import (
    GrowthFrameSummary,
    GrowthTrack,
    GrowthTrackPoint,
    GrowthTrackingResponse,
)


@dataclass
class _TrackedDetection:
    track_id: int
    class_name: str
    detection: Detection
    length_mm: float
    area_mm2: float


class _SimpleIoUTracker:
    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = float(iou_threshold)
        self.next_track_id = 1
        self.prev: list[_TrackedDetection] = []

    @staticmethod
    def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        inter = float(np.logical_and(mask_a > 0, mask_b > 0).sum())
        union = float(np.logical_or(mask_a > 0, mask_b > 0).sum())
        return inter / union if union > 0 else 0.0

    def assign(self, detections: list[Detection]) -> list[tuple[int, Detection]]:
        assigned: list[tuple[int, Detection]] = []
        used_prev: set[int] = set()

        for det in detections:
            best_idx = -1
            best_iou = 0.0
            for idx, prev in enumerate(self.prev):
                if idx in used_prev:
                    continue
                if prev.class_name != det.class_name:
                    continue
                cur_iou = self._iou(prev.detection.mask, det.mask)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_idx = idx

            if best_idx >= 0 and best_iou >= self.iou_threshold:
                track_id = self.prev[best_idx].track_id
                used_prev.add(best_idx)
            else:
                track_id = self.next_track_id
                self.next_track_id += 1

            assigned.append((track_id, det))
        return assigned

    def update(self, tracked: list[_TrackedDetection]) -> None:
        self.prev = tracked


class GrowthTrackingService:
    def __init__(
        self,
        model_service,
        calibrator,
        storage: StorageService,
        phi_service: PHIService,
        config: dict[str, Any],
    ) -> None:
        self.model_service = model_service
        self.calibrator = calibrator
        self.storage = storage
        self.phi_service = phi_service
        self.config = config

    @staticmethod
    def _ema(values: list[float], alpha: float = 0.35) -> list[float]:
        if not values:
            return []
        out = [float(values[0])]
        for v in values[1:]:
            out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
        return out

    @staticmethod
    def _growth_rate_mm_day(points: list[GrowthTrackPoint], frame_interval_hours: float) -> float:
        if len(points) < 2:
            return 0.0
        dt_days = max(1e-6, (len(points) - 1) * (frame_interval_hours / 24.0))
        return (points[-1].length_mm - points[0].length_mm) / dt_days

    def _save_growth_plot(self, tracks: list[GrowthTrack], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
        for tr in tracks:
            xs = [p.frame_index for p in tr.points]
            ys = [p.length_mm for p in tr.points]
            if not xs:
                continue
            ax.plot(xs, ys, marker='o', label=f'{tr.class_name}#{tr.track_id}')
        ax.set_title('Growth Dynamics (Length mm)')
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Length (mm)')
        ax.grid(True, alpha=0.3)
        if tracks:
            ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)

    def _store_series_db(
        self,
        db: Session,
        user_id: int | None,
        tenant_id: str,
        crop: str,
        name: str,
        series_id: str,
        frames: list[GrowthFrameSummary],
    ) -> None:
        from db.models import PlantObservation, PlantSeries

        row = PlantSeries(user_id=user_id, tenant_id=tenant_id, crop=crop, name=name)
        db.add(row)
        db.flush()

        for fr in frames:
            obs = PlantObservation(
                series_id=row.id,
                frame_index=fr.frame_index,
                run_id=f'{series_id}_f{fr.frame_index}',
                phi_score=float(fr.phi.score),
                phi_status=str(fr.phi.status),
                metrics_json=json.dumps(fr.model_dump(), ensure_ascii=True),
            )
            db.add(obs)
        db.commit()

    async def run_series(
        self,
        files: list[tuple[bytes, str]],
        crop: str = 'Unknown',
        camera_id: str = 'default',
        frame_interval_hours: float = 24.0,
        calibration_bytes: bytes | None = None,
        tenant_id: str = 'default',
        user_id: int | None = None,
        db: Session | None = None,
    ) -> GrowthTrackingResponse:
        if not files:
            raise ValueError('No frames were provided.')
        if not self.model_service.is_loaded():
            raise RuntimeError('Model is not loaded.')

        series_id = f'series_{datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")}'
        series_dir = Path(self.config['inference'].get('output_root', 'outputs')) / series_id
        series_dir.mkdir(parents=True, exist_ok=True)

        calib_img = None
        if calibration_bytes:
            calib_img = cv2.imdecode(np.frombuffer(calibration_bytes, np.uint8), cv2.IMREAD_COLOR)

        tracker = _SimpleIoUTracker(iou_threshold=float(self.config.get('tracking', {}).get('iou_threshold', 0.3)))
        tracks_buffer: dict[tuple[int, str], list[GrowthTrackPoint]] = {}
        frame_summaries: list[GrowthFrameSummary] = []

        for frame_idx, (payload, filename) in enumerate(files):
            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            frame_path = series_dir / f'frame_{frame_idx:04d}_{Path(filename).name}'
            cv2.imwrite(str(frame_path), img)

            scale_mm_per_px, _ = self.calibrator.get_scale(
                calib_img if calib_img is not None else img,
                camera_id=f'{camera_id}_series',
            )

            class_colors = {k: tuple(v) for k, v in self.config['inference'].get('class_colors', {}).items()}
            yolo_result = await self.model_service.predict(
                image_path=str(frame_path),
                class_colors=class_colors,
                overlay_alpha=float(self.config['inference'].get('overlay_alpha', 0.45)),
            )
            overlay_path = series_dir / f'overlay_{frame_idx:04d}.png'
            cv2.imwrite(str(overlay_path), yolo_result['overlay'])

            assigned = tracker.assign(yolo_result['detections'])
            tracked_detections: list[_TrackedDetection] = []
            measurements: list[dict[str, Any]] = []
            now_ts = (datetime.now(tz=timezone.utc) + timedelta(hours=frame_idx * frame_interval_hours)).isoformat()

            for track_id, det in assigned:
                morph = analyze_mask(det.mask, scale_mm_per_px)
                tracked_detections.append(
                    _TrackedDetection(
                        track_id=track_id,
                        class_name=det.class_name,
                        detection=det,
                        length_mm=float(morph.length_mm),
                        area_mm2=float(morph.area_mm2),
                    )
                )
                measurements.append(
                    {
                        'instance_id': det.instance_id,
                        'crop': crop,
                        'class_name': det.class_name,
                        'confidence': float(det.confidence),
                        'area_mm2': float(morph.area_mm2),
                        'length_mm': float(morph.length_mm),
                    }
                )

                key = (track_id, det.class_name)
                tracks_buffer.setdefault(key, []).append(
                    GrowthTrackPoint(
                        frame_index=frame_idx,
                        timestamp=now_ts,
                        length_mm=float(morph.length_mm),
                        area_mm2=float(morph.area_mm2),
                        confidence=float(det.confidence),
                    )
                )

            tracker.update(tracked_detections)
            phi = self.phi_service.evaluate(measurements=measurements, crop=crop)
            frame_summaries.append(
                GrowthFrameSummary(
                    frame_index=frame_idx,
                    timestamp=now_ts,
                    count=len(measurements),
                    phi=phi,
                )
            )

        tracks: list[GrowthTrack] = []
        for (track_id, class_name), points in sorted(tracks_buffer.items(), key=lambda x: x[0][0]):
            smoothed = self._ema([p.length_mm for p in points], alpha=float(self.config.get('tracking', {}).get('ema_alpha', 0.35)))
            tracks.append(
                GrowthTrack(
                    track_id=track_id,
                    class_name=class_name,
                    points=points,
                    growth_rate_mm_per_day=round(self._growth_rate_mm_day(points, frame_interval_hours), 6),
                    smoothed_length_mm=[round(v, 6) for v in smoothed],
                )
            )

        plot_path = series_dir / 'growth_dynamics.png'
        self._save_growth_plot(tracks, plot_path)

        response = GrowthTrackingResponse(
            series_id=series_id,
            crop=crop,
            tracks=tracks,
            frames=frame_summaries,
            files={
                'series_dir': str(series_dir.resolve()),
                'growth_plot': str(plot_path.resolve()),
            },
        )

        (series_dir / 'series_result.json').write_text(
            response.model_dump_json(indent=2),
            encoding='utf-8',
        )

        if db is not None:
            self._store_series_db(
                db=db,
                user_id=user_id,
                tenant_id=tenant_id,
                crop=crop,
                name='Growth Tracking',
                series_id=series_id,
                frames=frame_summaries,
            )

        return response
