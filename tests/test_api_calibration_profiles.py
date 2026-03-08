from __future__ import annotations

from uuid import uuid4

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.main import app


def _checkerboard_png_bytes() -> bytes:
    square_px = 44
    rows_squares = 5
    cols_squares = 8
    margin = 30
    h = rows_squares * square_px + 2 * margin
    w = cols_squares * square_px + 2 * margin
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    for r in range(rows_squares):
        for c in range(cols_squares):
            color = 0 if (r + c) % 2 == 0 else 255
            y1 = margin + r * square_px
            x1 = margin + c * square_px
            cv2.rectangle(image, (x1, y1), (x1 + square_px, y1 + square_px), (color, color, color), thickness=-1)
    ok, enc = cv2.imencode('.png', image)
    assert ok
    return enc.tobytes()


def test_calibration_profile_manual_roundtrip() -> None:
    with TestClient(app) as client:
        camera_id = f'api_cam_{uuid4().hex[:10]}'
        create = client.post(
            '/calibration/profile/manual',
            data={'camera_id': camera_id, 'source_type': 'lab_camera', 'mm_per_px': '0.1234'},
        )
        assert create.status_code == 200
        payload = create.json()
        assert payload['camera_id'] == camera_id
        assert payload['validated'] is True
        assert abs(float(payload['mm_per_px']) - 0.1234) < 1e-9

        profile = client.get(
            '/calibration/profile',
            params={'camera_id': camera_id, 'source_type': 'lab_camera'},
        )
        assert profile.status_code == 200
        p = profile.json()
        assert p['camera_id'] == camera_id
        assert p['validated'] is True
        assert abs(float(p['mm_per_px']) - 0.1234) < 1e-9

        rows = client.get('/calibration/profiles', params={'validated_only': 'true'})
        assert rows.status_code == 200
        assert any(str(x.get('camera_id')) == camera_id for x in (rows.json().get('items') or []))


def test_calibration_profile_auto_detects_checkerboard() -> None:
    with TestClient(app) as client:
        camera_id = f'api_cam_{uuid4().hex[:10]}'
        image_bytes = _checkerboard_png_bytes()
        create = client.post(
            '/calibration/profile/auto',
            data={'camera_id': camera_id, 'source_type': 'lab_camera'},
            files={'calibration_image': ('checker.png', image_bytes, 'image/png')},
        )
        assert create.status_code == 200, create.text
        payload = create.json()
        assert payload['camera_id'] == camera_id
        assert payload['validated'] is True
        assert payload['calibration_source'] in {'chessboard', 'charuco'}
        assert 0.15 <= float(payload['mm_per_px']) <= 0.35
