from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from calibration.chessboard import ScaleCalibrator
from utils.config import load_app_config, load_yaml


def _collect_images(root: Path) -> list[Path]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    if not root.exists():
        return []
    out: list[Path] = []
    if root.is_file() and root.suffix.lower() in exts:
        return [root]
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def _robust_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0.0]
    if arr.size == 0:
        raise ValueError('No valid mm/px values to aggregate.')

    q1, q3 = np.percentile(arr, [25, 75])
    iqr = max(1e-6, q3 - q1)
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    arr2 = arr[(arr >= lo) & (arr <= hi)]
    if arr2.size == 0:
        arr2 = arr

    med = float(np.median(arr2))
    mad = float(np.median(np.abs(arr2 - med))) if arr2.size > 0 else 0.0
    rel_mad = float(mad / med) if med > 0 else 0.0
    return {
        'median': med,
        'mad': mad,
        'rel_mad': rel_mad,
        'count': int(arr2.size),
    }


def _apply_board_settings(calibrator: ScaleCalibrator, board_cfg: dict | None) -> None:
    if not board_cfg:
        return
    kind = str(board_cfg.get('kind', '')).lower()
    if kind == 'chessboard':
        bs = board_cfg.get('board_size')
        if isinstance(bs, (list, tuple)) and len(bs) == 2:
            calibrator.board_size = (int(bs[0]), int(bs[1]))
        if 'square_size_mm' in board_cfg:
            calibrator.square_size_mm = float(board_cfg['square_size_mm'])
    if kind == 'charuco':
        calibrator.charuco_enabled = True
        calibrator.charuco_squares_x = int(board_cfg.get('squares_x', calibrator.charuco_squares_x))
        calibrator.charuco_squares_y = int(board_cfg.get('squares_y', calibrator.charuco_squares_y))
        calibrator.charuco_square_size_mm = float(board_cfg.get('square_size_mm', calibrator.charuco_square_size_mm))
        calibrator.charuco_marker_size_mm = float(board_cfg.get('marker_size_mm', calibrator.charuco_marker_size_mm))
        calibrator.charuco_dictionary = str(board_cfg.get('dictionary', calibrator.charuco_dictionary))


def main() -> None:
    parser = argparse.ArgumentParser(description='Fit calibration scale cache from calibration datasets.')
    parser.add_argument('--app-config', default='configs/app.yaml')
    parser.add_argument('--sources-config', default='configs/calibration_datasets.yaml')
    parser.add_argument('--camera-id', default='lab_camera')
    parser.add_argument('--source-ids', nargs='*', default=[])
    parser.add_argument('--extra-roots', nargs='*', default=[])
    parser.add_argument('--min-detections', type=int, default=20)
    parser.add_argument('--min-mm-per-px', type=float, default=0.001)
    parser.add_argument('--max-mm-per-px', type=float, default=2.0)
    parser.add_argument('--max-rel-mad', type=float, default=0.2)
    parser.add_argument('--allow-mixed-sources', action='store_true')
    parser.add_argument('--commit', action='store_true', help='Persist fitted scale into scale_cache.json')
    parser.add_argument('--report', default='reports/calibration_fit_report.json')
    args = parser.parse_args()

    app_cfg = load_app_config(args.app_config)
    cal_cfg = app_cfg.get('calibration', {})
    morph_cfg = app_cfg.get('morphometry', {})

    calibrator = ScaleCalibrator(
        cache_path=str(cal_cfg.get('cache_path', 'calibration/scale_cache.json')),
        default_mm_per_px=float(morph_cfg.get('default_mm_per_px', 0.12)),
        board_size=tuple(cal_cfg.get('board_size', [7, 7])),
        square_size_mm=float(cal_cfg.get('square_size_mm', 5.0)),
        charuco_enabled=bool(cal_cfg.get('charuco', {}).get('enabled', False)),
        charuco_squares_x=int(cal_cfg.get('charuco', {}).get('squares_x', 5)),
        charuco_squares_y=int(cal_cfg.get('charuco', {}).get('squares_y', 7)),
        charuco_square_size_mm=float(cal_cfg.get('charuco', {}).get('square_size_mm', 8.0)),
        charuco_marker_size_mm=float(cal_cfg.get('charuco', {}).get('marker_size_mm', 6.0)),
        charuco_dictionary=str(cal_cfg.get('charuco', {}).get('dictionary', 'DICT_4X4_50')),
    )

    src_cfg = load_yaml(args.sources_config)
    sources: list[dict] = src_cfg.get('calibration_datasets', [])
    selected_ids = {x.strip() for x in args.source_ids if x.strip()}

    roots: list[tuple[str, Path, dict | None]] = []
    for item in sources:
        sid = str(item.get('id', '')).strip()
        if selected_ids and sid not in selected_ids:
            continue
        local_dir = Path(str(item.get('local_dir', '')))
        if not local_dir:
            continue
        roots.append((sid or local_dir.name, local_dir, item.get('board')))

    for idx, root in enumerate(args.extra_roots):
        p = Path(root)
        roots.append((f'extra_{idx}', p, None))

    if not roots:
        raise RuntimeError('No calibration roots were selected.')

    scanned = 0
    per_root: list[dict] = []
    source_to_scales: dict[str, list[float]] = {}
    detection_by_method: dict[str, int] = {'chessboard': 0, 'charuco': 0}

    for root_id, root_path, board_cfg in roots:
        _apply_board_settings(calibrator, board_cfg)
        images = _collect_images(root_path)
        root_scales: list[float] = []
        root_detected = 0

        for img_path in images:
            scanned += 1
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            mm_per_px, method = calibrator.estimate_scale(img)
            if mm_per_px is None or method is None:
                continue
            if not (args.min_mm_per_px <= mm_per_px <= args.max_mm_per_px):
                continue
            root_scales.append(float(mm_per_px))
            root_detected += 1
            detection_by_method[method] = detection_by_method.get(method, 0) + 1

        source_to_scales[root_id] = root_scales
        stats = _robust_stats(root_scales) if root_scales else None
        per_root.append(
            {
                'id': root_id,
                'path': str(root_path),
                'images_total': len(images),
                'detected': root_detected,
                'stats': stats,
            }
        )

    valid_roots = [r for r in per_root if r['detected'] > 0]
    total_detections = sum(r['detected'] for r in valid_roots)

    fitted_mm_per_px = None
    fitted_from = None
    status = 'insufficient_detections'
    commit_applied = False

    if total_detections >= int(args.min_detections):
        if len(valid_roots) == 1:
            fitted_from = valid_roots[0]['id']
            fitted_mm_per_px = float(valid_roots[0]['stats']['median'])
            status = 'ok_single_source'
        elif args.allow_mixed_sources:
            mixed_values: list[float] = []
            for r in valid_roots:
                mixed_values.extend(source_to_scales.get(r['id'], []))
            mstats = _robust_stats(mixed_values)
            fitted_mm_per_px = float(mstats['median'])
            fitted_from = 'mixed_sources'
            status = 'ok_mixed_sources'
        else:
            status = 'blocked_mixed_sources'

    if fitted_mm_per_px is not None and args.commit and status.startswith('ok_'):
        # Stability gate: require bounded relative spread for selected source.
        if fitted_from == 'mixed_sources':
            chosen_rel_mad = _robust_stats(
                [v for vals in source_to_scales.values() for v in vals]
            )['rel_mad']
        else:
            chosen_rel_mad = float(next(r['stats']['rel_mad'] for r in valid_roots if r['id'] == fitted_from))

        if chosen_rel_mad <= float(args.max_rel_mad):
            fingerprint = f"fit_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            calibrator.upsert_scale(args.camera_id, fitted_mm_per_px, fingerprint=fingerprint)
            commit_applied = True
        else:
            status = 'blocked_high_variance'

    report = {
        'status': status,
        'camera_id': args.camera_id,
        'scanned_images': scanned,
        'detections': total_detections,
        'detection_by_method': detection_by_method,
        'fitted_mm_per_px': fitted_mm_per_px,
        'fitted_from': fitted_from,
        'min_detections': int(args.min_detections),
        'allow_mixed_sources': bool(args.allow_mixed_sources),
        'commit_requested': bool(args.commit),
        'commit_applied': bool(commit_applied),
        'max_rel_mad': float(args.max_rel_mad),
        'report_time_utc': datetime.now(tz=timezone.utc).isoformat(),
        'roots': per_root,
    }

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
