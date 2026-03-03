from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from inference.predictor import Detection
from utils.schemas import ActiveLearningItem, ActiveLearningQueueResponse, ActiveLearningSummary


class ActiveLearningService:
    def __init__(
        self,
        root_dir: str = 'data/active_learning',
        low_conf_threshold: float = 0.12,
    ) -> None:
        self.root = Path(root_dir)
        self.low_conf_threshold = float(low_conf_threshold)
        for status in ('pending', 'approved', 'rejected'):
            (self.root / status).mkdir(parents=True, exist_ok=True)

    def collect(
        self,
        run_id: str,
        image_bgr: np.ndarray,
        overlay_bgr: np.ndarray | None,
        detections: list[Detection],
        crop: str,
        tenant_id: str = 'default',
    ) -> ActiveLearningSummary:
        queue_items: list[str] = []

        for det in detections:
            if float(det.confidence) >= self.low_conf_threshold:
                continue

            item_id = f'al_{datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")}_{uuid4().hex[:8]}'
            item_dir = self.root / 'pending' / item_id
            item_dir.mkdir(parents=True, exist_ok=True)

            img_path = item_dir / 'image.png'
            cv2.imwrite(str(img_path), image_bgr)

            overlay_path = None
            if overlay_bgr is not None:
                overlay_path = item_dir / 'overlay.png'
                cv2.imwrite(str(overlay_path), overlay_bgr)

            mask_u8 = (det.mask > 0).astype(np.uint8) * 255
            cv2.imwrite(str(item_dir / 'mask.png'), mask_u8)

            payload = {
                'item_id': item_id,
                'status': 'pending',
                'run_id': run_id,
                'crop': crop,
                'tenant_id': tenant_id,
                'class_name': det.class_name,
                'confidence': float(det.confidence),
                'bbox_xyxy': [float(v) for v in det.bbox_xyxy],
                'image_path': str(img_path.resolve()),
                'overlay_path': str(overlay_path.resolve()) if overlay_path is not None else '',
                'mask_path': str((item_dir / 'mask.png').resolve()),
                'created_at': datetime.now(tz=timezone.utc).isoformat(),
            }
            (item_dir / 'metadata.json').write_text(
                json.dumps(payload, indent=2, ensure_ascii=True),
                encoding='utf-8',
            )
            queue_items.append(item_id)

        return ActiveLearningSummary(
            collected=len(queue_items),
            threshold=self.low_conf_threshold,
            queue_items=queue_items,
        )

    def list_items(self, status: str = 'pending', limit: int = 200) -> ActiveLearningQueueResponse:
        base = self.root / status
        if not base.exists():
            return ActiveLearningQueueResponse(status=status, count=0, items=[])

        rows: list[ActiveLearningItem] = []
        for meta in sorted(base.glob('*/metadata.json'), reverse=True):
            try:
                payload = json.loads(meta.read_text(encoding='utf-8'))
                rows.append(
                    ActiveLearningItem(
                        item_id=str(payload.get('item_id', meta.parent.name)),
                        status=str(payload.get('status', status)),
                        run_id=str(payload.get('run_id', '')),
                        class_name=str(payload.get('class_name', 'unknown')),
                        confidence=float(payload.get('confidence', 0.0)),
                        metadata_path=str(meta.resolve()),
                        created_at=str(payload.get('created_at', '')),
                    )
                )
            except Exception:
                continue
            if len(rows) >= limit:
                break

        return ActiveLearningQueueResponse(status=status, count=len(rows), items=rows)

    def export_manifest(self, out_path: str, status: str = 'pending') -> dict[str, Any]:
        queue = self.list_items(status=status, limit=100000)
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w', encoding='utf-8') as fh:
            for item in queue.items:
                fh.write(
                    json.dumps(
                        {
                            'item_id': item.item_id,
                            'status': item.status,
                            'run_id': item.run_id,
                            'class_name': item.class_name,
                            'confidence': item.confidence,
                            'metadata_path': item.metadata_path,
                            'created_at': item.created_at,
                        },
                        ensure_ascii=True,
                    )
                    + '\n'
                )
        return {
            'status': status,
            'count': queue.count,
            'manifest_path': str(target.resolve()),
        }

    def set_status(self, item_id: str, new_status: str) -> dict[str, Any]:
        if new_status not in {'pending', 'approved', 'rejected'}:
            raise ValueError('Unsupported status value.')
        src_meta = None
        src_dir = None
        src_status = None
        for status in ('pending', 'approved', 'rejected'):
            candidate = self.root / status / item_id / 'metadata.json'
            if candidate.exists():
                src_meta = candidate
                src_dir = candidate.parent
                src_status = status
                break
        if src_meta is None or src_dir is None:
            raise FileNotFoundError(f'Active learning item not found: {item_id}')

        dst_dir = self.root / new_status / item_id
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        if src_dir.resolve() != dst_dir.resolve():
            src_dir.rename(dst_dir)
        meta = dst_dir / 'metadata.json'
        payload = json.loads(meta.read_text(encoding='utf-8'))
        payload['status'] = new_status
        payload['updated_at'] = datetime.now(tz=timezone.utc).isoformat()
        meta.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')
        return {
            'item_id': item_id,
            'old_status': src_status,
            'new_status': new_status,
            'metadata_path': str(meta.resolve()),
        }
