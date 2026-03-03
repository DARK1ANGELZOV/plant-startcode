from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class DiseaseService:
    """Heuristic disease diagnostics from leaf pixels and morphometry signals."""

    def analyze(
        self,
        image_bgr: np.ndarray,
        detections: list[Any],
        measurements: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if image_bgr is None or image_bgr.size == 0:
            return {
                'risk_level': 'unknown',
                'confidence': 0.0,
                'findings': ['Некорректное изображение для диагностики.'],
                'actions': ['Проверить качество входного изображения.'],
            }

        leaf_masks = [d.mask for d in detections if str(d.class_name) == 'leaves']
        if not leaf_masks:
            return {
                'risk_level': 'unknown',
                'confidence': 0.25,
                'findings': ['Листья не сегментированы, оценка болезней ограничена.'],
                'actions': [
                    'Переснять изображение с лучшим освещением и ракурсом.',
                    'Если листья пропущены, временно снизить порог confidence до 0.01-0.05 и повторить анализ.',
                ],
            }

        leaf_union = np.zeros(leaf_masks[0].shape, dtype=np.uint8)
        for m in leaf_masks:
            leaf_union = np.maximum(leaf_union, (m > 0).astype(np.uint8))

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        leaf_pixels = leaf_union > 0
        leaf_count = int(leaf_pixels.sum())
        if leaf_count == 0:
            return {
                'risk_level': 'unknown',
                'confidence': 0.25,
                'findings': ['Листовая область пуста после объединения масок.'],
                'actions': ['Проверить корректность модели и масок.'],
            }

        # Chlorosis proxy: yellow-ish pixels
        yellow = ((h >= 18) & (h <= 38) & (s >= 35) & (v >= 60) & leaf_pixels)
        chlorosis_ratio = float(yellow.sum() / leaf_count)

        # Necrosis proxy: dark/brown spots
        dark = ((v <= 60) & (s >= 30) & leaf_pixels)
        necrosis_ratio = float(dark.sum() / leaf_count)

        # Low greenness proxy: non-green dominance
        green = ((h >= 40) & (h <= 95) & (s >= 30) & (v >= 40) & leaf_pixels)
        greenness_ratio = float(green.sum() / leaf_count)

        findings: list[str] = []
        actions: list[str] = []
        risk_score = 0.0

        if chlorosis_ratio > 0.22:
            risk_score += 0.35
            findings.append(
                f'Обнаружены признаки хлороза: желтые зоны на листьях {chlorosis_ratio * 100:.1f}%.'
            )
            actions.append('Проверить азот/железо в питании и pH раствора, скорректировать питание.')

        if necrosis_ratio > 0.08:
            risk_score += 0.35
            findings.append(
                f'Обнаружены темные/некротические участки: {necrosis_ratio * 100:.1f}% листовой площади.'
            )
            actions.append('Проверить грибковые/бактериальные поражения, снизить влажность и осмотреть очаги.')

        if greenness_ratio < 0.45:
            risk_score += 0.2
            findings.append(
                f'Низкая доля зеленого пигмента: {greenness_ratio * 100:.1f}%.'
            )
            actions.append('Проверить стресс от света/температуры и водный режим.')

        # Morphometry stress signal
        low_conf = [x for x in measurements if float(x.get('confidence', 0.0)) < 0.1]
        if len(low_conf) > max(5, int(len(measurements) * 0.4)):
            risk_score += 0.1
            findings.append('Высокая доля неуверенных инстансов может указывать на сильный шум/дефект съемки.')
            actions.append('Повторить съемку с калибровкой и стабильным освещением.')

        if not findings:
            risk_level = 'low'
            confidence = 0.6
            findings.append('Выраженных визуальных признаков болезней не обнаружено.')
            actions.append('Продолжать мониторинг раз в 24 часа.')
        else:
            if risk_score >= 0.65:
                risk_level = 'high'
            elif risk_score >= 0.35:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            confidence = min(0.95, 0.55 + risk_score)

        return {
            'risk_level': risk_level,
            'confidence': round(float(confidence), 4),
            'scores': {
                'chlorosis_ratio': round(chlorosis_ratio, 5),
                'necrosis_ratio': round(necrosis_ratio, 5),
                'greenness_ratio': round(greenness_ratio, 5),
            },
            'findings': findings,
            'actions': actions,
        }
