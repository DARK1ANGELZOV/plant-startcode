from __future__ import annotations


class ConsultationService:
    """Text-only consultant with short actionable replies and chat-memory hints."""

    POSITIVE_FEEDBACK_HINTS = ['помогло', 'лучше', 'стало лучше', 'сработало']
    NEGATIVE_FEEDBACK_HINTS = ['не помог', 'не помогло', 'хуже', 'стало хуже', 'без изменений', 'не сработало']

    @staticmethod
    def _detect_flags(text: str) -> dict[str, bool]:
        t = text.lower()
        return {
            'yellow': any(k in t for k in ['желт', 'yellow', 'хлороз']),
            'dry': any(k in t for k in ['сух', 'dry', 'вян', 'увяд']),
            'spots': any(k in t for k in ['пятн', 'spot', 'некроз']),
            'rot': any(k in t for k in ['гни', 'rot', 'плес', 'mold']),
            'slow_growth': any(k in t for k in ['медленно', 'не растет', 'slow', 'small']),
            'pest': any(k in t for k in ['вредител', 'клещ', 'тля', 'thrips', 'mite', 'aphid']),
            'watering': any(k in t for k in ['полив', 'water', 'вода']),
            'light': any(k in t for k in ['свет', 'ламп', 'lighting']),
        }

    def _feedback_state(self, prior_context: list[str] | None) -> str:
        if not prior_context:
            return 'none'
        joined = ' '.join(str(x).lower() for x in prior_context[-12:])
        if any(token in joined for token in self.NEGATIVE_FEEDBACK_HINTS):
            return 'negative'
        if any(token in joined for token in self.POSITIVE_FEEDBACK_HINTS):
            return 'positive'
        return 'none'

    def compose(self, message: str, prior_context: list[str] | None = None) -> str:
        text = (message or '').strip()
        flags = self._detect_flags(text)
        feedback_state = self._feedback_state(prior_context)

        possible_causes: list[str] = []
        actions_today: list[str] = []
        risk_signs: list[str] = []

        if flags['yellow']:
            possible_causes.append('Пожелтение чаще связано с переливом или дефицитом питания.')
            actions_today.append('Проверьте влажность субстрата на глубине 2-3 см перед следующим поливом.')
        if flags['dry']:
            possible_causes.append('Сухие края и вялость указывают на водный стресс или перегрев.')
            actions_today.append('Стабилизируйте режим полива и исключите резкие пересушки.')
        if flags['spots']:
            possible_causes.append('Пятна могут быть грибковой/бактериальной природы или ожогом.')
            actions_today.append('Изолируйте растение и не оставляйте капли воды на листьях при ярком свете.')
            risk_signs.append('Если пятна быстро растут 24-48 часов, нужен ускоренный фитосанитарный протокол.')
        if flags['rot']:
            possible_causes.append('Признаки гнили часто связаны с застоем влаги и слабой аэрацией корней.')
            actions_today.append('Сократите полив и удалите явно пораженные ткани стерильным инструментом.')
            risk_signs.append('Если есть запах гнили и размягчение тканей, риск критический.')
        if flags['slow_growth']:
            possible_causes.append('Замедленный рост связан с дефицитом света, питания или слабой корневой системой.')
            actions_today.append('Проверьте световой режим и питание (pH/EC), корректируйте постепенно.')
        if flags['pest']:
            possible_causes.append('Есть риск поражения вредителями (клещ/тля/трипс).')
            actions_today.append('Осмотрите нижнюю сторону листа и при подтверждении начните мягкую обработку.')
            risk_signs.append('При паутинке и мозаике лист быстро теряет фотосинтез.')

        if not possible_causes:
            possible_causes.append('Без фото диагноз предварительный: причин может быть несколько одновременно.')
        if not actions_today:
            actions_today.extend(
                [
                    'Сравните состояние верхнего и нижнего яруса листьев.',
                    'Сделайте контрольный замер через 24 часа (цвет, тургор, новые пятна, прирост).',
                ]
            )
        if not risk_signs:
            risk_signs.append('Срочно: резкое ухудшение за сутки, массовое увядание, потемнение у основания стебля.')

        if feedback_state == 'negative':
            actions_today.append('Учту, что прошлый шаг не помог: меняйте только 1 фактор за раз и фиксируйте эффект за 24 часа.')
        elif feedback_state == 'positive':
            actions_today.append('Учту, что прошлый шаг помог: сохраните режим и сделайте повторный контроль завтра.')

        lines = [
            'Консультация без фото (предварительно):',
            '',
            '1. Вероятные причины:',
        ]
        lines.extend([f'- {x}' for x in possible_causes[:3]])
        lines.append('')
        lines.append('2. Что сделать сейчас:')
        lines.extend([f'- {x}' for x in actions_today[:3]])
        lines.append('')
        lines.append('3. Когда срочно реагировать:')
        lines.extend([f'- {x}' for x in risk_signs[:2]])
        lines.append('')
        lines.append('Для точной оценки отправьте фото растения целиком и крупный план проблемной зоны.')
        return '\n'.join(lines)
