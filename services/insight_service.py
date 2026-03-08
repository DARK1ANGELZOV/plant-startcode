from __future__ import annotations

from statistics import mean
from typing import Any

from utils.schemas import PredictResponse


class InsightService:
    POSITIVE_FEEDBACK_HINTS = ['помогло', 'лучше', 'стало лучше', 'сработало']
    NEGATIVE_FEEDBACK_HINTS = ['не помог', 'не помогло', 'хуже', 'стало хуже', 'без изменений', 'не сработало']

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.compact_mode = bool(cfg.get('compact_mode', True))
        self.min_confidence_for_numeric = float(cfg.get('min_confidence_for_numeric', 0.35))
        self.max_recommendations = max(1, int(cfg.get('max_recommendations', 3)))
        self.max_quality_notes = max(1, int(cfg.get('max_quality_notes', 2)))
        self.include_px_fallback = bool(cfg.get('include_px_fallback', True))

    @staticmethod
    def _class_ru_name(class_name: str) -> str:
        mapping = {
            'root': 'Корень',
            'stem': 'Стебель',
            'leaves': 'Листья',
        }
        return mapping.get(class_name, class_name)

    @staticmethod
    def _class_measurements(
        result: PredictResponse,
        class_name: str,
        min_confidence: float = 0.0,
    ) -> list:
        return [
            m
            for m in result.measurements
            if m.class_name == class_name and float(m.confidence) >= float(min_confidence)
        ]

    @staticmethod
    def _avg_measure(
        result: PredictResponse,
        class_name: str,
        key: str,
        min_confidence: float = 0.0,
    ) -> float:
        rows = InsightService._class_measurements(
            result=result,
            class_name=class_name,
            min_confidence=min_confidence,
        )
        values = []
        for row in rows:
            value = getattr(row, key, None)
            if value is None:
                continue
            values.append(float(value))
        if not values:
            return 0.0
        return float(mean(values))

    @staticmethod
    def _max_conf(result: PredictResponse, class_name: str, summary_conf: dict) -> float:
        if class_name in summary_conf:
            try:
                return float(summary_conf[class_name])
            except Exception:
                pass
        vals = [float(m.confidence) for m in result.measurements if m.class_name == class_name]
        return max(vals) if vals else 0.0

    @staticmethod
    def _detected(result: PredictResponse, class_name: str, segmentation: dict) -> bool:
        info = segmentation.get(class_name, {}) if isinstance(segmentation, dict) else {}
        if isinstance(info, dict) and 'detected' in info:
            return bool(info.get('detected'))
        return any(m.class_name == class_name for m in result.measurements)

    def _feedback_state(self, prior_context: list[str] | None) -> str:
        if not prior_context:
            return 'none'
        joined = ' '.join(str(x).lower() for x in prior_context[-12:])
        if any(token in joined for token in self.NEGATIVE_FEEDBACK_HINTS):
            return 'negative'
        if any(token in joined for token in self.POSITIVE_FEEDBACK_HINTS):
            return 'positive'
        return 'none'

    def _real_mm_block(
        self,
        result: PredictResponse,
        summary: dict,
    ) -> tuple[list[str], bool]:
        calibration_reliable = bool(summary.get('calibration_reliable', result.scale_source == 'chessboard'))
        mm_lines: list[str] = []
        if not calibration_reliable:
            mm_lines.append('- Невозможен: нет валидной геометрической калибровки камеры.')
            mm_lines.append('- Для реальных мм используйте калиброванный профиль камеры или кадр с эталоном.')
            return mm_lines, False

        source = str(summary.get('calibration_source', result.scale_source or 'unknown'))
        camera_profile = str(summary.get('calibration_camera_id', summary.get('camera_id', 'default')))
        mm_lines.append(f'- Возможен: {float(result.scale_mm_per_px):.5f} мм/пикс')
        mm_lines.append(f'- Источник: {source} (camera_id: {camera_profile})')
        cal_err = summary.get('calibration_error_pct')
        if cal_err is not None:
            try:
                mm_lines.append(f'- Оценка погрешности масштаба: ±{float(cal_err):.1f}%')
            except Exception:
                pass
        return mm_lines, True

    @staticmethod
    def _short_quality_notes(summary: dict, max_notes: int = 2) -> list[str]:
        quality = summary.get('image_quality', {}) if isinstance(summary, dict) else {}
        notes = quality.get('notes') or []
        if not notes:
            return ['Качество изображения оценено автоматически.']
        return [str(x) for x in notes[:max(1, int(max_notes))]]

    def _short_recommendations(
        self,
        result: PredictResponse,
        feedback_state: str,
        trust_level: str,
        limit: int,
    ) -> list[str]:
        tips: list[str] = []
        for rec in result.recommendations[:2]:
            action = rec.action.strip()
            tips.append(action if action else rec.message.strip())

        if not tips:
            if not result.measurements:
                tips.append('Переснимите фото при ровном свете и четком фокусе.')
                tips.append('Снимите растение целиком и проблемную зону крупным планом.')
            else:
                tips.append('Повторите съемку через 24 часа и сравните динамику.')
                tips.append('При ухудшении тренда скорректируйте полив и освещение постепенно.')

        if trust_level == 'low':
            tips.insert(0, 'Доверие к измерениям низкое: используйте вывод как скрининг.')
        elif trust_level == 'medium':
            tips.append('Доверие среднее: подтвердите вывод повторным снимком.')

        if feedback_state == 'negative':
            tips.insert(0, 'Учитываю прошлую неудачу: меняйте только один фактор за шаг.')
        elif feedback_state == 'positive':
            tips.insert(0, 'Учитываю позитивный результат: зафиксируйте текущий режим ухода.')

        return tips[:max(1, int(limit))]

    def compose_reply(
        self,
        result: PredictResponse,
        user_message: str | None = None,
        prior_context: list[str] | None = None,
    ) -> str:
        _ = user_message
        summary = result.summary or {}
        segmentation = summary.get('segmentation') or {}
        conf_by_class = summary.get('confidence_by_class') or {}
        trust_score = float(summary.get('measurement_trust_score', 0.0))
        trust_level = str(summary.get('measurement_trust_level', 'low'))
        feedback_state = self._feedback_state(prior_context)
        min_conf = max(0.0, min(1.0, float(self.min_confidence_for_numeric)))

        mm_lines, has_real_mm = self._real_mm_block(result=result, summary=summary)
        units = 'мм' if has_real_mm else 'px'

        lines: list[str] = ['Результаты анализа изображения:']

        lines.append('')
        lines.append('1. Сегментация:')
        for cls in ['root', 'stem', 'leaves']:
            detected = self._detected(result, cls, segmentation)
            conf = self._max_conf(result, cls, conf_by_class)
            if detected and conf >= min_conf:
                lines.append(f"- {self._class_ru_name(cls)}: да ({int(round(conf * 100))}%)")
            elif detected:
                lines.append(f"- {self._class_ru_name(cls)}: обнаружено с низкой уверенностью ({int(round(conf * 100))}%)")
            else:
                lines.append(f"- {self._class_ru_name(cls)}: нет")

        lines.append('')
        lines.append(f'2. Измерения ({units}):')
        if has_real_mm:
            root_mm = self._avg_measure(result, 'root', 'length_mm', min_confidence=min_conf)
            stem_mm = self._avg_measure(result, 'stem', 'length_mm', min_confidence=min_conf)
            leaves_mm2 = self._avg_measure(result, 'leaves', 'area_mm2', min_confidence=min_conf)
            root_mm2 = self._avg_measure(result, 'root', 'area_mm2', min_confidence=min_conf)
            stem_mm2 = self._avg_measure(result, 'stem', 'area_mm2', min_confidence=min_conf)
            lines.append(f'- Длина корня: {root_mm:.1f} мм' if root_mm > 0 else '- Длина корня: н/д')
            lines.append(f'- Длина стебля: {stem_mm:.1f} мм' if stem_mm > 0 else '- Длина стебля: н/д')
            lines.append(f'- Площадь листьев: {leaves_mm2 / 100.0:.2f} см2' if leaves_mm2 > 0 else '- Площадь листьев: н/д')
            lines.append(f'- Площадь корней: {root_mm2 / 100.0:.2f} см2' if root_mm2 > 0 else '- Площадь корней: н/д')
            lines.append(f'- Площадь стебля: {stem_mm2 / 100.0:.2f} см2' if stem_mm2 > 0 else '- Площадь стебля: н/д')
        else:
            root_px = self._avg_measure(result, 'root', 'length_px', min_confidence=min_conf)
            stem_px = self._avg_measure(result, 'stem', 'length_px', min_confidence=min_conf)
            leaves_px2 = self._avg_measure(result, 'leaves', 'area_px', min_confidence=min_conf)
            root_px2 = self._avg_measure(result, 'root', 'area_px', min_confidence=min_conf)
            stem_px2 = self._avg_measure(result, 'stem', 'area_px', min_confidence=min_conf)
            lines.append(f'- Длина корня: {root_px:.1f} px' if root_px > 0 else '- Длина корня: н/д')
            lines.append(f'- Длина стебля: {stem_px:.1f} px' if stem_px > 0 else '- Длина стебля: н/д')
            lines.append(f'- Площадь листьев: {leaves_px2:.0f} px2' if leaves_px2 > 0 else '- Площадь листьев: н/д')
            lines.append(f'- Площадь корней: {root_px2:.0f} px2' if root_px2 > 0 else '- Площадь корней: н/д')
            lines.append(f'- Площадь стебля: {stem_px2:.0f} px2' if stem_px2 > 0 else '- Площадь стебля: н/д')
            if self.include_px_fallback:
                lines.append(f'- Порог надежности чисел: confidence >= {min_conf:.2f}')

        lines.append('')
        lines.append('3. Перевод в мм:')
        lines.extend(mm_lines)

        lines.append('')
        lines.append('4. Вывод:')
        phi = getattr(result, 'phi', None)
        if phi is not None:
            title = f"Plant Health Index: {phi.score:.1f}/100 ({phi.status})"
            if not has_real_mm:
                title += ' [без валидной калибровки]'
            lines.append(f'- {title}')
        else:
            lines.append('- Индекс PHI недоступен.')
        lines.append(f'- Доверие к измерениям: {trust_score:.1f}/100 ({trust_level})')

        inference_note = str(summary.get('inference_note', '')).strip()
        if inference_note:
            lines.append(f'- {inference_note}')
        for note in self._short_quality_notes(summary, max_notes=self.max_quality_notes):
            lines.append(f'- {note}')

        lines.append('')
        lines.append('5. Рекомендации:')
        for tip in self._short_recommendations(
            result=result,
            feedback_state=feedback_state,
            trust_level=trust_level,
            limit=self.max_recommendations,
        ):
            lines.append(f'- {tip}')

        lines.append('')
        lines.append('Могу продолжить диалог в этом же чате: напишите, что изменилось после следующего шага.')
        return '\n'.join(lines)
