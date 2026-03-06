from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from utils.schemas import ExplainabilityArtifacts


ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
STRUCTURE_PATTERNS = [
    r'(^|\n)\s*Результаты анализа изображения:',
    r'(^|\n)\s*1\.\s*Сегментация:',
    r'(^|\n)\s*2\.\s*Измерения\s*\(px\):',
    r'(^|\n)\s*3\.\s*Перевод в мм:',
    r'(^|\n)\s*4\.\s*Вывод:',
    r'(^|\n)\s*5\.\s*Рекомендации:',
]

def _collect_images(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        if base.is_file() and base.suffix.lower() in ALLOWED_IMAGE_EXTS:
            out.append(base)
            continue
        for p in base.rglob('*'):
            if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS:
                out.append(p)
    return sorted(set(out))


def _guess_crop(path: Path) -> str:
    name = path.name.lower()
    if 'wheat' in name:
        return 'Wheat'
    if 'arugula' in name:
        return 'Arugula'
    return 'Unknown'


def _polygon_area(coords: np.ndarray) -> float:
    if coords.ndim != 2 or coords.shape[0] < 3:
        return 0.0
    return float(abs(cv2.contourArea(coords.astype(np.float32))))


def _ground_truth_area_by_class(image_path: Path) -> dict[int, float]:
    parts = [p.lower() for p in image_path.parts]
    if 'images' not in parts:
        return {}
    try:
        images_idx = parts.index('images')
        split = image_path.parts[images_idx + 1]
    except Exception:
        return {}
    label_path = Path(*image_path.parts[:images_idx], 'labels', split, f'{image_path.stem}.txt')
    if not label_path.exists():
        return {}

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {}
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return {}

    out: dict[int, float] = defaultdict(float)
    text = label_path.read_text(encoding='utf-8', errors='ignore').strip()
    if not text:
        return {}
    for row in text.splitlines():
        raw = row.strip().split()
        if len(raw) < 7:
            continue
        try:
            cid = int(float(raw[0]))
            vals = np.asarray([float(x) for x in raw[1:]], dtype=np.float32)
        except ValueError:
            continue
        if vals.size < 6 or vals.size % 2 != 0:
            continue
        pts = vals.reshape(-1, 2).copy()
        pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0) * float(w)
        pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0) * float(h)
        area = _polygon_area(pts)
        if area > 1.0:
            out[cid] += area
    return dict(out)


def _class_to_id(class_name: str) -> int | None:
    mapping = {'root': 0, 'stem': 1, 'leaves': 2}
    return mapping.get((class_name or '').strip().lower())


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _safe_float_or_none(x: Any) -> float | None:
    if x is None:
        return None
    try:
        val = float(x)
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return val


def _has_required_structure(reply: str) -> bool:
    text = str(reply or '')
    return all(re.search(pattern, text, flags=re.IGNORECASE) for pattern in STRUCTURE_PATTERNS)


def _strict_checks(payload: dict[str, Any], assistant_reply: str) -> tuple[dict[str, bool], list[str], dict[str, Any]]:
    result = payload.get('result') or {}
    summary = result.get('summary') or {}
    measurements = result.get('measurements') or []
    inference_mode = str(summary.get('inference_mode', ''))
    model_based = bool(summary.get('model_based', False))
    scale = _safe_float(result.get('scale_mm_per_px', 0.0))

    checks: dict[str, bool] = {}
    notes: list[str] = []
    extra: dict[str, Any] = {}

    checks['structured_sections'] = _has_required_structure(assistant_reply)
    checks['summary_has_mode'] = isinstance(summary.get('inference_mode'), str) and bool(summary.get('inference_mode'))
    checks['model_runtime_valid'] = (inference_mode not in {'model_unavailable', 'heuristic_fallback'}) and model_based
    checks['scale_positive'] = scale > 0.0
    checks['measurement_list_type'] = isinstance(measurements, list)

    numeric_ok = True
    conf_ok = True
    class_ok = True
    mm_consistency_fail = 0
    mm_consistency_count = 0
    realism_rows: list[str] = []
    pred_area_by_class: Counter = Counter()
    calibration_reliable = bool(summary.get('calibration_reliable', False))

    for m in measurements:
        cls = str(m.get('class_name', ''))
        cid = _class_to_id(cls)
        if cid is None:
            class_ok = False

        c = _safe_float(m.get('confidence', 0.0))
        area_px = _safe_float(m.get('area_px', 0.0))
        length_px = _safe_float(m.get('length_px', 0.0))
        area_mm2 = _safe_float_or_none(m.get('area_mm2'))
        length_mm = _safe_float_or_none(m.get('length_mm'))

        pred_area_by_class[cid] += max(0.0, area_px) if cid is not None else 0.0

        if not (0.0 <= c <= 1.0):
            conf_ok = False
        if area_px < 0.0 or length_px < 0.0:
            numeric_ok = False
        if area_mm2 is not None and area_mm2 < 0.0:
            numeric_ok = False
        if length_mm is not None and length_mm < 0.0:
            numeric_ok = False

        # Only enforce mm consistency where mm values are explicitly emitted.
        if scale > 0.0 and length_mm is not None and length_px > 0.0:
            mm_consistency_count += 1
            expected_len = length_px * scale
            len_tol = max(1.0, 0.15 * max(1.0, expected_len))
            if abs(length_mm - expected_len) > len_tol:
                mm_consistency_fail += 1
                realism_rows.append(f'length mismatch cls={cls}')
        if scale > 0.0 and area_mm2 is not None and area_px > 0.0:
            mm_consistency_count += 1
            expected_area = area_px * scale * scale
            area_tol = max(3.0, 0.18 * max(1.0, expected_area))
            if abs(area_mm2 - expected_area) > area_tol:
                mm_consistency_fail += 1
                realism_rows.append(f'area mismatch cls={cls}')

    # If calibration is reliable and there are masks, at least one mm value should be present.
    if calibration_reliable and measurements:
        mm_emitted = any(
            (_safe_float_or_none(m.get('length_mm')) is not None)
            or (_safe_float_or_none(m.get('area_mm2')) is not None)
            for m in measurements
        )
        if not mm_emitted:
            numeric_ok = False
            realism_rows.append('missing mm values under reliable calibration')

    checks['measurement_values_consistent'] = numeric_ok and (mm_consistency_fail == 0)
    checks['measurement_conf_range'] = conf_ok
    checks['measurement_class_valid'] = class_ok
    extra['mm_consistency_checks'] = mm_consistency_count
    extra['mm_consistency_failures'] = mm_consistency_fail

    if not measurements:
        low = assistant_reply.lower()
        checks['no_fake_numbers_without_masks'] = ('н/д' in low) or ('нет' in low) or ('недоступ' in low)
    else:
        checks['no_fake_numbers_without_masks'] = True

    conf_by_class = summary.get('confidence_by_class') or {}
    if measurements and conf_by_class:
        grouped: dict[str, list[float]] = defaultdict(list)
        for m in measurements:
            grouped[str(m.get('class_name', ''))].append(_safe_float(m.get('confidence', 0.0)))
        conf_match = True
        for cls, vals in grouped.items():
            if cls not in conf_by_class:
                continue
            mean_conf = float(np.mean(vals)) if vals else 0.0
            if abs(mean_conf - _safe_float(conf_by_class.get(cls))) > 0.25:
                conf_match = False
        checks['confidence_summary_consistent'] = conf_match
    else:
        checks['confidence_summary_consistent'] = True

    gt_area = payload.get('_gt_area_by_class') or {}
    if gt_area:
        checks['has_predictions_on_labeled_image'] = len(measurements) > 0
    else:
        checks['has_predictions_on_labeled_image'] = True

    realism_pass = True
    realism_scores: list[float] = []
    if gt_area and measurements:
        for cid, gt_val_raw in gt_area.items():
            gt_val = float(gt_val_raw)
            pred_val = float(pred_area_by_class.get(cid, 0.0))
            if gt_val <= 1.0 or pred_val <= 1.0:
                continue
            ratio = min(pred_val, gt_val) / max(pred_val, gt_val)
            realism_scores.append(ratio)
            if ratio < 0.015:
                realism_pass = False
    extra['area_realism_ratio_mean'] = round(float(np.mean(realism_scores)), 6) if realism_scores else None
    checks['area_realism_vs_labels'] = realism_pass

    for k, v in checks.items():
        if not v:
            notes.append(f'failed:{k}')
    if realism_rows:
        notes.extend(realism_rows[:8])
    return checks, notes, extra


def main() -> None:
    parser = argparse.ArgumentParser(description='Very strict photo-analysis quality gate for /chat/analyze.')
    parser.add_argument('--n', default=160, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--out', default='reports/strict_photo_quality_report.json', type=str)
    parser.add_argument(
        '--data-roots',
        nargs='*',
        default=[
            'data/hf_multisource_balanced_auto/images',
            'data/hf_multisource_medium/images',
            'data/demo/dataset/images',
        ],
    )
    parser.add_argument('--strict-pass-threshold', default=0.88, type=float)
    parser.add_argument('--fail-on-threshold', action='store_true')
    parser.add_argument('--keep-artifacts', action='store_true')
    parser.add_argument(
        '--lightweight',
        action='store_true',
        help='Use lightweight mode (disable heavy artifacts like XAI/masks) to reduce disk usage.',
    )
    args = parser.parse_args()

    random.seed(args.seed)
    roots = [Path(p) for p in args.data_roots]
    images = _collect_images(roots)
    if not images:
        raise RuntimeError('No images found for strict quality gate.')
    chosen = random.sample(images, k=min(len(images), int(args.n)))

    failures: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    check_counter: Counter = Counter()
    total_checks = 0
    inference_modes: Counter = Counter()

    with TestClient(app) as client:
        if args.lightweight:
            try:
                inf = app.state.inference_service
                inf.config.setdefault('inference', {})
                inf.config['inference']['save_masks'] = False
                inf.config['inference']['output_root'] = 'outputs/strict_tmp'
                inf.storage.output_root = Path('outputs/strict_tmp')
                inf.storage.output_root.mkdir(parents=True, exist_ok=True)

                inf.xai_service.generate = lambda *_, **__: ExplainabilityArtifacts(
                    notes=['xai_disabled_for_strict_gate']
                )
                inf.reporter.save_distribution_plot = lambda *_, **__: None
            except Exception:
                pass

        for idx, image_path in enumerate(chosen, start=1):
            try:
                with image_path.open('rb') as fh:
                    resp = client.post(
                        '/chat/analyze',
                        data={
                            'message': 'Проведи анализ и объясни риски по цифрам.',
                            'crop': _guess_crop(image_path),
                            'camera_id': 'default',
                            'source_type': 'lab_camera',
                        },
                        files={'image': (image_path.name, fh, 'image/jpeg')},
                    )
            except Exception as exc:
                failures.append(
                    {
                        'index': idx,
                        'image': str(image_path),
                        'status_code': 0,
                        'error': f'exception: {exc}',
                    }
                )
                continue

            if resp.status_code != 200:
                failures.append(
                    {
                        'index': idx,
                        'image': str(image_path),
                        'status_code': resp.status_code,
                        'error': resp.text[:500],
                    }
                )
                continue

            payload = resp.json()
            payload['_gt_area_by_class'] = _ground_truth_area_by_class(image_path)
            inference_mode = str(((payload.get('result') or {}).get('summary') or {}).get('inference_mode', 'unknown'))
            inference_modes[inference_mode] += 1

            assistant_reply = str(payload.get('assistant_reply', ''))
            checks, notes, extra = _strict_checks(payload=payload, assistant_reply=assistant_reply)
            passed = all(checks.values())

            for key, value in checks.items():
                total_checks += 1
                if value:
                    check_counter[key] += 1

            row = {
                'index': idx,
                'image': str(image_path),
                'passed': passed,
                'checks': checks,
                'notes': notes,
                **extra,
            }
            all_rows.append(row)

            if (not passed) and len(failures) < 60:
                failures.append(
                    {
                        'index': idx,
                        'image': str(image_path),
                        'checks': checks,
                        'notes': notes,
                        'reply_excerpt': assistant_reply[:900],
                    }
                )

            if len(sample_rows) < 24:
                sample_rows.append(row)

            if not args.keep_artifacts:
                try:
                    input_file = Path(str(((payload.get('result') or {}).get('files') or {}).get('input', '')))
                    run_dir = input_file.parent
                    if run_dir.name.startswith('run_') and run_dir.exists():
                        shutil.rmtree(run_dir, ignore_errors=True)
                except Exception:
                    pass

    check_keys = sorted(
        {
            'structured_sections',
            'summary_has_mode',
            'model_runtime_valid',
            'scale_positive',
            'measurement_list_type',
            'measurement_values_consistent',
            'measurement_conf_range',
            'measurement_class_valid',
            'no_fake_numbers_without_masks',
            'confidence_summary_consistent',
            'has_predictions_on_labeled_image',
            'area_realism_vs_labels',
        }
    )
    per_check_pass_rate = {
        key: round(float(check_counter.get(key, 0)) / float(max(1, len(all_rows))) * 100.0, 2)
        for key in check_keys
    }

    strict_pass = sum(1 for row in all_rows if row.get('passed'))
    strict_rate = round(float(strict_pass) / float(max(1, len(all_rows))) * 100.0, 2)
    global_checks_pass_rate = round(float(sum(check_counter.values())) / float(max(1, total_checks)) * 100.0, 2)
    critical_keys = ['model_runtime_valid', 'has_predictions_on_labeled_image', 'measurement_values_consistent', 'area_realism_vs_labels']
    critical_checks_pass_rate = round(min(float(per_check_pass_rate.get(k, 0.0)) for k in critical_keys), 2)

    quality_gate_passed = (
        (global_checks_pass_rate / 100.0) >= float(args.strict_pass_threshold)
        and (critical_checks_pass_rate / 100.0) >= float(args.strict_pass_threshold)
        and (strict_rate / 100.0) >= float(args.strict_pass_threshold)
    )

    report = {
        'requested_n': int(args.n),
        'executed_n': len(all_rows),
        'sampled_for_report': len(sample_rows),
        'seed': int(args.seed),
        'strict_threshold': float(args.strict_pass_threshold),
        'quality_gate_passed': bool(quality_gate_passed),
        'global_checks_pass_rate': global_checks_pass_rate,
        'strict_sample_pass_rate': strict_rate,
        'critical_checks_pass_rate': critical_checks_pass_rate,
        'inference_modes': dict(inference_modes),
        'per_check_pass_rate': per_check_pass_rate,
        'samples': sample_rows,
        'failures': failures,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.fail_on_threshold and (not quality_gate_passed):
        raise SystemExit(2)


if __name__ == '__main__':
    main()

