from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ArchitectureSummary:
    name: str
    report_path: str
    miou: float
    robustness_score: float
    boundary_iou: float
    precision: float
    recall: float
    mean_drop: float
    composite_score: float
    checkpoint: str
    onnx_path: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))


def _extract_common(report: dict[str, Any]) -> dict[str, float]:
    clean = report.get('clean', {})
    robust = report.get('robustness', {})
    return {
        'miou': _safe_float(clean.get('miou', 0.0)),
        'precision': _safe_float(clean.get('precision', 0.0)),
        'recall': _safe_float(clean.get('recall', 0.0)),
        'boundary_iou': _safe_float(clean.get('boundary_iou', 0.0)),
        'robustness_score': _safe_float(robust.get('robustness_score', 0.0)),
        'mean_drop': _safe_float(robust.get('mean_drop', 0.0)),
    }


def _extract_deeplab(report: dict[str, Any]) -> dict[str, Any]:
    stats = _extract_common(report)
    return {
        **stats,
        'checkpoint': str(report.get('checkpoint', '')),
        'onnx_path': str(report.get('onnx_path', '')),
    }


def _extract_yolo(report: dict[str, Any]) -> dict[str, Any]:
    stats = _extract_common(report)
    return {
        **stats,
        'checkpoint': str(report.get('best_checkpoint', '')),
        'onnx_path': str(report.get('onnx_path', '')),
    }


def _score(stats: dict[str, Any], weights: dict[str, float]) -> float:
    return (
        _safe_float(stats.get('miou')) * weights['miou']
        + _safe_float(stats.get('robustness_score')) * weights['robustness']
        + _safe_float(stats.get('boundary_iou')) * weights['boundary_iou']
        + _safe_float(stats.get('precision')) * weights['precision']
        + _safe_float(stats.get('recall')) * weights['recall']
    )


def compare_architectures(
    yolo_report_path: str,
    deeplab_report_path: str,
    output_path: str,
    score_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    weights = score_weights or {
        'miou': 0.45,
        'robustness': 0.30,
        'boundary_iou': 0.10,
        'precision': 0.075,
        'recall': 0.075,
    }

    yolo_raw = _load_json(yolo_report_path)
    deeplab_raw = _load_json(deeplab_report_path)
    if not yolo_raw and not deeplab_raw:
        raise FileNotFoundError('Both reports are missing or empty.')

    candidates: list[ArchitectureSummary] = []

    if yolo_raw:
        stats = _extract_yolo(yolo_raw)
        candidates.append(
            ArchitectureSummary(
                name='yolo_seg',
                report_path=str(Path(yolo_report_path).resolve()),
                miou=_safe_float(stats['miou']),
                robustness_score=_safe_float(stats['robustness_score']),
                boundary_iou=_safe_float(stats['boundary_iou']),
                precision=_safe_float(stats['precision']),
                recall=_safe_float(stats['recall']),
                mean_drop=_safe_float(stats['mean_drop']),
                composite_score=_score(stats, weights),
                checkpoint=str(stats.get('checkpoint', '')),
                onnx_path=str(stats.get('onnx_path', '')),
            )
        )

    if deeplab_raw:
        stats = _extract_deeplab(deeplab_raw)
        candidates.append(
            ArchitectureSummary(
                name='deeplabv3',
                report_path=str(Path(deeplab_report_path).resolve()),
                miou=_safe_float(stats['miou']),
                robustness_score=_safe_float(stats['robustness_score']),
                boundary_iou=_safe_float(stats['boundary_iou']),
                precision=_safe_float(stats['precision']),
                recall=_safe_float(stats['recall']),
                mean_drop=_safe_float(stats['mean_drop']),
                composite_score=_score(stats, weights),
                checkpoint=str(stats.get('checkpoint', '')),
                onnx_path=str(stats.get('onnx_path', '')),
            )
        )

    candidates.sort(key=lambda x: x.composite_score, reverse=True)
    best = candidates[0]

    result = {
        'best_architecture': best.name,
        'selection_date_utc': dt.datetime.now(dt.UTC).isoformat(),
        'score_weights': weights,
        'candidates': [c.__dict__ for c in candidates],
        'decision_rationale': (
            'Best composite score using weighted mIoU, robustness score, '
            'boundary IoU, precision, and recall.'
        ),
        'recommended_checkpoint': best.checkpoint,
        'recommended_onnx': best.onnx_path,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare robust segmentation architectures and pick the best.')
    parser.add_argument('--yolo-report', default='models/robust/yolo_metrics.json')
    parser.add_argument('--deeplab-report', default='models/robust/deeplab_metrics.json')
    parser.add_argument('--output', default='models/robust/best_architecture.json')
    args = parser.parse_args()

    result = compare_architectures(
        yolo_report_path=args.yolo_report,
        deeplab_report_path=args.deeplab_report,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
