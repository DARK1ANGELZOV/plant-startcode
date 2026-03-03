from __future__ import annotations

from statistics import mean

from utils.schemas import PredictResponse


class InsightService:
    POSITIVE_FEEDBACK_HINTS = ['помогло', 'лучше', 'стало лучше', 'сработало']
    NEGATIVE_FEEDBACK_HINTS = ['не помог', 'не помогло', 'хуже', 'стало хуже', 'без изменений', 'не сработало']

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
    ):
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
    def _max_conf(
        result: PredictResponse,
        class_name: str,
        summary_conf: dict,
    ) -> float:
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

    def _mm_block(
        self,
        result: PredictResponse,
        summary: dict,
        confidence_gate: float,
    ) -> tuple[list[str], bool]:
        calibration_reliable = bool(summary.get('calibration_reliable', result.scale_source == 'chessboard'))
        mm_lines: list[str] = []
        approx_mode = False

        root_px = self._avg_measure(result, 'root', 'length_px', min_confidence=0.0)
        stem_px = self._avg_measure(result, 'stem', 'length_px', min_confidence=0.0)
        leaves_px2 = self._avg_measure(result, 'leaves', 'area_px', min_confidence=0.0)
        root_px2 = self._avg_measure(result, 'root', 'area_px', min_confidence=0.0)
        stem_px2 = self._avg_measure(result, 'stem', 'area_px', min_confidence=0.0)

        if calibration_reliable:
            mm_lines.append(f"- Масштаб: {result.scale_mm_per_px:.5f} мм/пикс (источник: {result.scale_source})")
            root_mm = self._avg_measure(result, 'root', 'length_mm', min_confidence=confidence_gate) or (
                root_px * result.scale_mm_per_px
            )
            stem_mm = self._avg_measure(result, 'stem', 'length_mm', min_confidence=confidence_gate) or (
                stem_px * result.scale_mm_per_px
            )
            leaves_mm2 = self._avg_measure(result, 'leaves', 'area_mm2', min_confidence=confidence_gate) or (
                leaves_px2 * result.scale_mm_per_px * result.scale_mm_per_px
            )
            root_mm2 = self._avg_measure(result, 'root', 'area_mm2', min_confidence=confidence_gate) or (
                root_px2 * result.scale_mm_per_px * result.scale_mm_per_px
            )
            stem_mm2 = self._avg_measure(result, 'stem', 'area_mm2', min_confidence=confidence_gate) or (
                stem_px2 * result.scale_mm_per_px * result.scale_mm_per_px
            )
            mm_lines.append(f"- Длина корня: {root_mm:.1f} мм" if root_mm > 0 else '- Длина корня: н/д')
            mm_lines.append(f"- Длина стебля: {stem_mm:.1f} мм" if stem_mm > 0 else '- Длина стебля: н/д')
            mm_lines.append(f"- Площадь листьев: {leaves_mm2 / 100.0:.2f} см²" if leaves_mm2 > 0 else '- Площадь листьев: н/д')
            mm_lines.append(f"- Площадь корней: {root_mm2 / 100.0:.2f} см²" if root_mm2 > 0 else '- Площадь корней: н/д')
            mm_lines.append(f"- Площадь стебля: {stem_mm2 / 100.0:.2f} см²" if stem_mm2 > 0 else '- Площадь стебля: н/д')
            return mm_lines, approx_mode

        # Fallback/approximate mm path requested by user.
        if float(result.scale_mm_per_px) > 0.0:
            approx_mode = True
            scale = float(result.scale_mm_per_px)
            mm_lines.append(f"- Примерный масштаб: {scale:.5f} мм/пикс (источник: {result.scale_source}, низкая точность)")
            if root_px > 0:
                mm_lines.append(f"- Примерная длина корня: ~{root_px * scale:.1f} мм")
            if stem_px > 0:
                mm_lines.append(f"- Примерная длина стебля: ~{stem_px * scale:.1f} мм")
            if leaves_px2 > 0:
                mm_lines.append(f"- Примерная площадь листьев: ~{(leaves_px2 * scale * scale) / 100.0:.2f} см²")
            mm_lines.append('- Для точных мм добавьте шахматку/линейку в этот же кадр.')
            return mm_lines, approx_mode

        mm_lines.append('- Перевод в мм сейчас невозможен: нет масштаба.')
        return mm_lines, approx_mode

    @staticmethod
    def _short_quality_notes(summary: dict) -> list[str]:
        quality = summary.get('image_quality', {}) if isinstance(summary, dict) else {}
        notes = quality.get('notes') or []
        if not notes:
            return ['Качество изображения оценено автоматически.']
        return [str(x) for x in notes[:2]]

    def _short_recommendations(
        self,
        result: PredictResponse,
        feedback_state: str,
        trust_level: str,
    ) -> list[str]:
        tips: list[str] = []
        for rec in result.recommendations[:2]:
            action = rec.action.strip()
            if action:
                tips.append(action)
            else:
                tips.append(rec.message.strip())

        if not tips:
            if not result.measurements:
                tips.append('Переснимите фото при ровном свете и четком фокусе.')
                tips.append('Снимите растение целиком и отдельно проблемную зону крупным планом.')
            else:
                tips.append('Сравните этот замер с предыдущим через 24 часа.')
                tips.append('Если тренд ухудшается, скорректируйте полив и освещение постепенно.')

        if trust_level == 'low':
            tips.insert(0, 'Доверие к измерениям низкое: используйте вывод как скрининг, не как точный диагноз.')
            tips.append('Для повышения точности: добавьте шахматку 10 мм в кадр и переснимите без бликов.')
        elif trust_level == 'medium':
            tips.append('Доверие к измерениям среднее: подтверждайте тренд повторным снимком через 24 часа.')

        if feedback_state == 'negative':
            tips.insert(0, 'Учитываю, что прошлые действия не помогли: на следующем шаге меняйте только 1 фактор за раз.')
        elif feedback_state == 'positive':
            tips.insert(0, 'Учитываю, что прошлый шаг помог: закрепите режим и сделайте контрольный снимок через сутки.')

        return tips[:3]

    def compose_reply(
        self,
        result: PredictResponse,
        user_message: str | None = None,
        prior_context: list[str] | None = None,
    ) -> str:
        summary = result.summary or {}
        segmentation = summary.get('segmentation') or {}
        confidence_gate = float(summary.get('min_report_confidence', 0.7))
        conf_by_class = summary.get('confidence_by_class') or {}
        trust_score = float(summary.get('measurement_trust_score', 0.0))
        trust_level = str(summary.get('measurement_trust_level', 'low'))
        feedback_state = self._feedback_state(prior_context)

        lines: list[str] = ['Результаты анализа изображения:']

        lines.append('')
        lines.append('1. Сегментация:')
        for cls in ['root', 'stem', 'leaves']:
            detected = self._detected(result, cls, segmentation)
            conf = self._max_conf(result, cls, conf_by_class)
            if detected:
                lines.append(f"- {self._class_ru_name(cls)}: да ({int(round(conf * 100))}%)")
            else:
                lines.append(f"- {self._class_ru_name(cls)}: нет")

        lines.append('')
        lines.append('2. Измерения (px):')
        root_px = self._avg_measure(result, 'root', 'length_px', min_confidence=0.0)
        stem_px = self._avg_measure(result, 'stem', 'length_px', min_confidence=0.0)
        leaves_px2 = self._avg_measure(result, 'leaves', 'area_px', min_confidence=0.0)
        root_px2 = self._avg_measure(result, 'root', 'area_px', min_confidence=0.0)
        stem_px2 = self._avg_measure(result, 'stem', 'area_px', min_confidence=0.0)
        lines.append(f"- Длина корня: {root_px:.1f} px" if root_px > 0 else '- Длина корня: н/д')
        lines.append(f"- Длина стебля: {stem_px:.1f} px" if stem_px > 0 else '- Длина стебля: н/д')
        lines.append(f"- Площадь листьев: {leaves_px2:.1f} px²" if leaves_px2 > 0 else '- Площадь листьев: н/д')
        lines.append(f"- Площадь корней: {root_px2:.1f} px²" if root_px2 > 0 else '- Площадь корней: н/д')
        lines.append(f"- Площадь стебля: {stem_px2:.1f} px²" if stem_px2 > 0 else '- Площадь стебля: н/д')

        lines.append('')
        lines.append('3. Перевод в мм:')
        mm_lines, approx_mode = self._mm_block(result=result, summary=summary, confidence_gate=confidence_gate)
        lines.extend(mm_lines)

        lines.append('')
        lines.append('4. Вывод:')
        phi = getattr(result, 'phi', None)
        if phi is not None:
            title = f"Plant Health Index: {phi.score:.1f}/100 ({phi.status})"
            if approx_mode:
                title += ' [оценочно]'
            lines.append(f'- {title}')
        else:
            lines.append('- Индекс PHI недоступен.')
        lines.append(f"- Доверие к измерениям: {trust_score:.1f}/100 ({trust_level})")

        inference_note = str(summary.get('inference_note', '')).strip()
        if inference_note:
            lines.append(f'- {inference_note}')
        for note in self._short_quality_notes(summary):
            lines.append(f'- {note}')

        lines.append('')
        lines.append('5. Рекомендации:')
        for tip in self._short_recommendations(
            result=result,
            feedback_state=feedback_state,
            trust_level=trust_level,
        ):
            lines.append(f'- {tip}')

        lines.append('')
        lines.append('Могу продолжить диалог по этому чату: уточните, что изменилось после последнего шага.')
        return '\n'.join(lines)
