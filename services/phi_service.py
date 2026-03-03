from __future__ import annotations

from collections import defaultdict
from statistics import mean, pstdev
from typing import Any

from utils.schemas import PHIResult


class PHIService:
    def __init__(self, thresholds: dict[str, Any] | None = None) -> None:
        self.thresholds = thresholds or {}

    @staticmethod
    def _risk_penalty(disease_analysis: dict[str, Any] | None) -> tuple[float, str | None]:
        if not disease_analysis:
            return 0.0, None
        risk = str(disease_analysis.get('risk_level', '')).lower()
        if risk == 'critical':
            return 25.0, 'Disease module reported critical risk.'
        if risk in {'high', 'risk'}:
            return 12.0, 'Disease module reported elevated risk.'
        if risk in {'medium', 'warning'}:
            return 6.0, 'Disease module reported moderate risk.'
        return 0.0, None

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def evaluate(
        self,
        measurements: list[dict[str, Any]],
        crop: str,
        growth_context: dict[str, Any] | None = None,
        disease_analysis: dict[str, Any] | None = None,
        absolute_scale_reliable: bool = True,
    ) -> PHIResult:
        if not measurements:
            return PHIResult(
                score=20.0,
                status='Critical',
                components={'signal': 0.0},
                reasons=['No measurable plant organs were detected.'],
            )

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in measurements:
            grouped[str(row.get('class_name', 'unknown'))].append(row)

        if not absolute_scale_reliable:
            classes_present = {k for k, rows in grouped.items() if rows}
            conf_values = [float(x.get('confidence', 0.0)) for x in measurements]
            mean_conf = mean(conf_values) if conf_values else 0.0
            organ_score = 45.0 * self._clamp(len(classes_present) / 3.0, 0.0, 1.0)
            confidence_score = 35.0 * self._clamp(mean_conf, 0.0, 1.0)
            disease_penalty, disease_reason = self._risk_penalty(disease_analysis)
            score = self._clamp(20.0 + organ_score + confidence_score - disease_penalty, 0.0, 100.0)
            if score >= 75.0:
                status = 'Healthy'
            elif score >= 45.0:
                status = 'Risk'
            else:
                status = 'Critical'

            reasons = [
                'Перевод в мм отключен: нет валидной геометрической калибровки.',
                f'Detected classes: {", ".join(sorted(classes_present)) if classes_present else "none"}.',
            ]
            if disease_reason:
                reasons.append(disease_reason)
            return PHIResult(
                score=round(score, 4),
                status=status,
                components={
                    'organ_presence': round(organ_score, 4),
                    'confidence': round(confidence_score, 4),
                    'disease_penalty': round(disease_penalty, 4),
                },
                reasons=reasons[:6],
            )

        thr = self.thresholds.get(crop, {})
        min_root = float(thr.get('min_root_length_mm', 18.0))
        min_stem = float(thr.get('min_stem_length_mm', 12.0))
        min_leaf_area = float(thr.get('min_leaf_area_mm2', 90.0))

        root_vals = [float(x.get('length_mm')) for x in grouped.get('root', []) if x.get('length_mm') is not None]
        stem_vals = [float(x.get('length_mm')) for x in grouped.get('stem', []) if x.get('length_mm') is not None]
        leaf_vals = [float(x.get('area_mm2')) for x in grouped.get('leaves', []) if x.get('area_mm2') is not None]

        root_len = mean(root_vals) if root_vals else 0.0
        stem_len = mean(stem_vals) if stem_vals else 0.0
        leaf_area = mean(leaf_vals) if leaf_vals else 0.0

        root_score = 30.0 * self._clamp(root_len / max(1e-6, min_root), 0.0, 1.2)
        stem_score = 20.0 * self._clamp(stem_len / max(1e-6, min_stem), 0.0, 1.2)
        leaf_score = 25.0 * self._clamp(leaf_area / max(1e-6, min_leaf_area), 0.0, 1.2)

        leaf_root_ratio = leaf_area / max(1e-6, root_len)
        ratio_score = 10.0 * self._clamp(1.0 - abs(leaf_root_ratio - 1.2) / 1.2, 0.0, 1.0)

        leaf_areas = [float(x.get('area_mm2')) for x in grouped.get('leaves', []) if x.get('area_mm2') is not None]
        cv = (pstdev(leaf_areas) / max(1e-6, mean(leaf_areas))) if len(leaf_areas) > 1 else 0.0
        uniformity_score = 10.0 * self._clamp(1.0 - cv, 0.0, 1.0)

        growth_rate = 0.0
        if growth_context:
            growth_rate = float(growth_context.get('growth_rate_mm_per_day', 0.0))
        growth_score = 5.0 * self._clamp((growth_rate + 2.0) / 4.0, 0.0, 1.0)

        disease_penalty, disease_reason = self._risk_penalty(disease_analysis)

        raw = root_score + stem_score + leaf_score + ratio_score + uniformity_score + growth_score - disease_penalty
        score = self._clamp(raw, 0.0, 100.0)

        if score >= 75.0:
            status = 'Healthy'
        elif score >= 45.0:
            status = 'Risk'
        else:
            status = 'Critical'

        reasons = []
        if root_len < min_root:
            reasons.append(f'Root length below target ({root_len:.2f} < {min_root:.2f} mm).')
        if stem_len < min_stem:
            reasons.append(f'Stem length below target ({stem_len:.2f} < {min_stem:.2f} mm).')
        if leaf_area < min_leaf_area:
            reasons.append(f'Leaf area below target ({leaf_area:.2f} < {min_leaf_area:.2f} mm2).')
        if disease_reason:
            reasons.append(disease_reason)
        if not reasons:
            reasons.append('Morphometric indicators are in acceptable range.')

        return PHIResult(
            score=round(score, 4),
            status=status,
            components={
                'root': round(root_score, 4),
                'stem': round(stem_score, 4),
                'leaves': round(leaf_score, 4),
                'balance': round(ratio_score, 4),
                'uniformity': round(uniformity_score, 4),
                'growth': round(growth_score, 4),
                'disease_penalty': round(disease_penalty, 4),
            },
            reasons=reasons[:6],
        )
