from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calibration.chessboard import ScaleCalibrator
from utils.config import load_app_config


def _render_checkerboard(
    inner_cols: int,
    inner_rows: int,
    square_px: int,
    margin_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    squares_x = inner_cols + 1
    squares_y = inner_rows + 1
    width = squares_x * square_px + 2 * margin_px
    height = squares_y * square_px + 2 * margin_px

    canvas = np.full((height, width), 255, dtype=np.uint8)
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 0:
                x1 = margin_px + x * square_px
                y1 = margin_px + y * square_px
                x2 = x1 + square_px
                y2 = y1 + square_px
                cv2.rectangle(canvas, (x1, y1), (x2, y2), 0, thickness=-1)

    # Inner-corner grid (OpenCV order: row-major, left->right, top->bottom).
    corners: list[list[float]] = []
    for r in range(inner_rows):
        for c in range(inner_cols):
            corners.append(
                [
                    float(margin_px + (c + 1) * square_px),
                    float(margin_px + (r + 1) * square_px),
                ]
            )
    return canvas, np.asarray(corners, dtype=np.float32)


def _random_projective_image(
    rng: np.random.Generator,
    inner_cols: int,
    inner_rows: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    square_px = int(rng.integers(24, 72))
    margin_px = int(rng.integers(24, 80))

    board, ideal_corners = _render_checkerboard(
        inner_cols=inner_cols,
        inner_rows=inner_rows,
        square_px=square_px,
        margin_px=margin_px,
    )
    h, w = board.shape

    src_quad = np.asarray(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
        dtype=np.float32,
    )

    jitter_x = float(w) * float(rng.uniform(0.03, 0.16))
    jitter_y = float(h) * float(rng.uniform(0.03, 0.16))
    dst_quad = src_quad.copy()
    dst_quad += np.asarray(
        [
            [rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)],
            [rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)],
            [rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)],
            [rng.uniform(-jitter_x, jitter_x), rng.uniform(-jitter_y, jitter_y)],
        ],
        dtype=np.float32,
    )

    scale_out = float(rng.uniform(1.0, 1.8))
    out_w = int(max(320, round(w * scale_out)))
    out_h = int(max(240, round(h * scale_out)))

    # Keep destination corners inside image frame.
    dst_quad[:, 0] = np.clip(dst_quad[:, 0] + (out_w - w) * 0.5, 8, out_w - 9)
    dst_quad[:, 1] = np.clip(dst_quad[:, 1] + (out_h - h) * 0.5, 8, out_h - 9)

    H = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.warpPerspective(board, H, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=255)

    corners_h = cv2.perspectiveTransform(ideal_corners.reshape(-1, 1, 2), H).reshape(-1, 2)

    # Photometric degradation to mimic field/lab variance.
    img = warped.astype(np.float32)
    alpha = float(rng.uniform(0.75, 1.25))  # contrast
    beta = float(rng.uniform(-28, 28))  # brightness shift
    img = np.clip(img * alpha + beta, 0, 255)

    if rng.random() < 0.55:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), float(rng.uniform(0.4, 1.8)))

    if rng.random() < 0.35:
        noise = rng.normal(0, float(rng.uniform(2, 11)), size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 255)

    # JPEG artifacts.
    if rng.random() < 0.35:
        q = int(rng.integers(45, 92))
        _, enc = cv2.imencode('.jpg', img.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), q])
        img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Soft vignette.
    if rng.random() < 0.4:
        yy, xx = np.indices(img.shape)
        cx = float(img.shape[1]) * float(rng.uniform(0.35, 0.65))
        cy = float(img.shape[0]) * float(rng.uniform(0.35, 0.65))
        rx = float(img.shape[1]) * float(rng.uniform(0.65, 1.2))
        ry = float(img.shape[0]) * float(rng.uniform(0.65, 1.2))
        vignette = ((xx - cx) ** 2 / (rx**2) + (yy - cy) ** 2 / (ry**2)).astype(np.float32)
        vignette = np.clip(1.12 - 0.35 * vignette, 0.62, 1.06)
        img = np.clip(img * vignette, 0, 255)

    # Convert to 3-channel BGR.
    img_bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return img_bgr, corners_h.astype(np.float32), float(square_px)


def _gt_mm_per_px_from_corners(corners: np.ndarray, board_size: tuple[int, int], square_size_mm: float) -> float | None:
    cols, rows = int(board_size[0]), int(board_size[1])
    if corners.shape[0] != cols * rows:
        return None

    horizontal: list[float] = []
    vertical: list[float] = []
    for r in range(rows):
        for c in range(cols - 1):
            i1 = r * cols + c
            i2 = i1 + 1
            horizontal.append(float(np.linalg.norm(corners[i1] - corners[i2])))
    for c in range(cols):
        for r in range(rows - 1):
            i1 = r * cols + c
            i2 = (r + 1) * cols + c
            vertical.append(float(np.linalg.norm(corners[i1] - corners[i2])))

    distances = np.asarray(horizontal + vertical, dtype=np.float32)
    distances = distances[np.isfinite(distances)]
    distances = distances[distances > 0.0]
    if distances.size == 0:
        return None
    px_per_square = float(np.median(distances))
    if px_per_square <= 0.0:
        return None
    return float(square_size_mm / px_per_square)


def _collect_image_paths(root: Path, max_images: int) -> list[Path]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    if not root.exists():
        return []
    images = [p for p in sorted(root.rglob('*')) if p.is_file() and p.suffix.lower() in exts]
    return images[: max(0, int(max_images))]


def _robust_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0.0]
    if arr.size == 0:
        return None
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    q10, q90 = np.percentile(arr, [10, 90]).tolist()
    return {
        'median': med,
        'mad': mad,
        'p10': float(q10),
        'p90': float(q90),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'count': int(arr.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Strict mm-per-pixel benchmark for chessboard calibration.')
    parser.add_argument('--app-config', default='configs/app.yaml')
    parser.add_argument('--n', type=int, default=40, help='Synthetic benchmark cases count')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='reports/calibrated_mm_benchmark.json')
    parser.add_argument(
        '--real-root',
        default='data/calibration/predictor_cloud_calib/raw/source/calib',
        help='Optional real calibration folder for detection-rate and scale dispersion check',
    )
    parser.add_argument('--real-max-images', type=int, default=40)
    args = parser.parse_args()

    cfg = load_app_config(args.app_config)
    cal_cfg = cfg.get('calibration', {})
    morph_cfg = cfg.get('morphometry', {})

    board_size = tuple(int(x) for x in cal_cfg.get('board_size', [7, 4]))
    board_candidates = [tuple(int(y) for y in x) for x in cal_cfg.get('board_size_candidates', [])]
    square_size_mm = float(cal_cfg.get('square_size_mm', 10.0))

    calibrator = ScaleCalibrator(
        cache_path=str(cal_cfg.get('cache_path', 'calibration/scale_cache.json')),
        default_mm_per_px=float(morph_cfg.get('default_mm_per_px', 0.12)),
        board_size=board_size,
        board_size_candidates=board_candidates,
        square_size_mm=square_size_mm,
        charuco_enabled=bool(cal_cfg.get('charuco', {}).get('enabled', False)),
        charuco_squares_x=int(cal_cfg.get('charuco', {}).get('squares_x', 5)),
        charuco_squares_y=int(cal_cfg.get('charuco', {}).get('squares_y', 7)),
        charuco_square_size_mm=float(cal_cfg.get('charuco', {}).get('square_size_mm', 8.0)),
        charuco_marker_size_mm=float(cal_cfg.get('charuco', {}).get('marker_size_mm', 6.0)),
        charuco_dictionary=str(cal_cfg.get('charuco', {}).get('dictionary', 'DICT_4X4_50')),
    )

    rng = np.random.default_rng(int(args.seed))
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    detections = 0
    misses = 0
    worst_cases: list[dict[str, float | int]] = []

    n_cases = max(1, int(args.n))
    for case_idx in range(n_cases):
        image, gt_corners, _ = _random_projective_image(rng, inner_cols=board_size[0], inner_rows=board_size[1])
        gt = _gt_mm_per_px_from_corners(gt_corners, board_size=board_size, square_size_mm=square_size_mm)
        if gt is None:
            misses += 1
            continue
        pred, source = calibrator.estimate_scale(image)
        if pred is None or source != 'chessboard':
            misses += 1
            continue
        detections += 1
        abs_err = float(abs(pred - gt))
        rel_err = float(abs_err / max(1e-9, gt))
        abs_errors.append(abs_err)
        rel_errors.append(rel_err)
        worst_cases.append(
            {
                'case': int(case_idx),
                'gt_mm_per_px': float(gt),
                'pred_mm_per_px': float(pred),
                'abs_err': abs_err,
                'rel_err_pct': float(rel_err * 100.0),
            }
        )

    worst_cases.sort(key=lambda x: float(x['abs_err']), reverse=True)
    worst_cases = worst_cases[:5]

    synthetic_report: dict[str, Any] = {
        'cases': int(n_cases),
        'detected': int(detections),
        'missed': int(misses),
        'detection_rate': float(detections / max(1, n_cases)),
        'mae_mm_per_px': float(np.mean(abs_errors)) if abs_errors else None,
        'rmse_mm_per_px': float(np.sqrt(np.mean(np.square(abs_errors)))) if abs_errors else None,
        'mape_pct': float(np.mean(rel_errors) * 100.0) if rel_errors else None,
        'p90_abs_err_mm_per_px': float(np.percentile(abs_errors, 90)) if abs_errors else None,
        'worst_cases': worst_cases,
    }

    # Real dataset check: no GT mm_per_px, so we report detection rate + dispersion.
    real_root = Path(args.real_root)
    real_paths = _collect_image_paths(real_root, max_images=int(args.real_max_images))
    real_scales: list[float] = []
    real_detected = 0
    for p in real_paths:
        image = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if image is None:
            continue
        pred, source = calibrator.estimate_scale(image)
        if pred is None:
            continue
        real_detected += 1
        real_scales.append(float(pred))

    real_report: dict[str, Any] = {
        'root': str(real_root),
        'images': int(len(real_paths)),
        'detected': int(real_detected),
        'detection_rate': float(real_detected / max(1, len(real_paths))),
        'scale_summary': _robust_summary(real_scales),
    }

    report = {
        'generated_at_utc': datetime.now(tz=timezone.utc).isoformat(),
        'seed': int(args.seed),
        'board_size_inner_corners': [int(board_size[0]), int(board_size[1])],
        'square_size_mm': float(square_size_mm),
        'board_candidates': [[int(a), int(b)] for a, b in calibrator.board_size_candidates],
        'synthetic_benchmark': synthetic_report,
        'real_dataset_check': real_report,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
