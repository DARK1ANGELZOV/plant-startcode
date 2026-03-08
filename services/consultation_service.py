from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentSpec:
    name: str
    keywords: tuple[str, ...]
    phrase_weight: float = 1.3
    token_weight: float = 1.0
    question_boost: float = 0.0


class ConsultationService:
    """Conversational assistant with intent scoring and plant diagnostics mode."""

    BOT_NAME = "PlantVision AI"

    _TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]+", re.UNICODE)

    POSITIVE_FEEDBACK_HINTS = (
        "помогло",
        "лучше",
        "сработало",
        "восстановилось",
    )
    NEGATIVE_FEEDBACK_HINTS = (
        "не помогло",
        "хуже",
        "без изменений",
        "не сработало",
        "стало хуже",
    )

    FOLLOW_UP_HINTS = (
        "что делать",
        "и дальше",
        "а дальше",
        "а теперь",
        "почему",
        "как исправить",
        "что еще",
        "как лучше",
    )

    INTENTS: tuple[IntentSpec, ...] = (
        IntentSpec(
            "plant",
            (
                "растен",
                "лист",
                "корен",
                "стеб",
                "почв",
                "грунт",
                "полив",
                "удобр",
                "вредител",
                "болезн",
                "пятн",
                "желт",
                "гни",
                "рост",
                "урож",
                "wheat",
                "arugula",
                "leaf",
                "root",
                "stem",
                "plant",
            ),
            question_boost=0.15,
        ),
        IntentSpec(
            "greeting",
            (
                "привет",
                "здравствуй",
                "здравствуйте",
                "добрый день",
                "добрый вечер",
                "доброе утро",
                "hello",
                "hi",
                "hey",
            ),
            question_boost=0.05,
        ),
        IntentSpec(
            "how_are_you",
            (
                "как ты",
                "как дела",
                "как сам",
                "как настроение",
                "how are you",
            ),
            question_boost=0.2,
        ),
        IntentSpec(
            "name",
            (
                "как тебя зовут",
                "твое имя",
                "твое имя",
                "кто ты",
                "what is your name",
            ),
            question_boost=0.2,
        ),
        IntentSpec(
            "capabilities",
            (
                "что умеешь",
                "что можешь",
                "твои возможности",
                "что ты умеешь",
                "what can you do",
            ),
            question_boost=0.2,
        ),
        IntentSpec(
            "thanks",
            ("спасибо", "благодарю", "thanks", "thank you"),
        ),
        IntentSpec(
            "bye",
            ("пока", "до встречи", "увидимся", "goodbye", "bye"),
        ),
    )

    @classmethod
    def _normalize(cls, text: str) -> str:
        return " ".join((text or "").strip().lower().split())

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return cls._TOKEN_RE.findall(text.lower())

    @staticmethod
    def _is_question(text: str) -> bool:
        return "?" in text or text.startswith(("как ", "что ", "кто ", "почему ", "зачем ", "когда "))

    @staticmethod
    def _stable_choice(options: list[str], seed_text: str) -> str:
        if not options:
            return ""
        digest = hashlib.sha1(seed_text.encode("utf-8", errors="ignore")).hexdigest()
        idx = int(digest[:8], 16) % len(options)
        return options[idx]

    def _feedback_state(self, prior_context: list[str] | None) -> str:
        if not prior_context:
            return "none"
        joined = " ".join(self._normalize(x) for x in prior_context[-12:])
        if any(token in joined for token in self.NEGATIVE_FEEDBACK_HINTS):
            return "negative"
        if any(token in joined for token in self.POSITIVE_FEEDBACK_HINTS):
            return "positive"
        return "none"

    def _context_plant_bias(self, prior_context: list[str] | None, current_text: str) -> float:
        if not prior_context:
            return 0.0
        joined = " ".join(self._normalize(x) for x in prior_context[-10:])
        plant_mentions = sum(1 for kw in self.INTENTS[0].keywords if kw in joined)
        follow_up = any(h in current_text for h in self.FOLLOW_UP_HINTS)
        if plant_mentions == 0:
            return 0.0
        bias = min(1.1, plant_mentions * 0.12)
        if follow_up:
            bias += 0.2
        return min(1.3, bias)

    def _has_follow_up_hint(self, text: str) -> bool:
        return any(h in text for h in self.FOLLOW_UP_HINTS)

    def _intent_scores(self, text: str, tokens: list[str], prior_context: list[str] | None) -> dict[str, float]:
        question = self._is_question(text)
        scores: dict[str, float] = {}
        token_set = set(tokens)
        for spec in self.INTENTS:
            score = 0.0
            for kw in spec.keywords:
                if " " in kw:
                    if kw in text:
                        score += spec.phrase_weight
                else:
                    if kw in token_set:
                        score += spec.token_weight
                    elif any(t.startswith(kw) for t in token_set):
                        score += spec.token_weight * 0.8
            if score > 0 and question:
                score += spec.question_boost
            if score > 0:
                scores[spec.name] = score

        scores["plant"] = scores.get("plant", 0.0) + self._context_plant_bias(prior_context, text)
        return scores

    @staticmethod
    def _detect_plant_flags(text: str) -> dict[str, bool]:
        t = text.lower()
        return {
            "yellow": any(k in t for k in ("желт", "yellow", "хлороз")),
            "dry": any(k in t for k in ("сух", "dry", "вян", "увяд")),
            "spots": any(k in t for k in ("пятн", "spot", "некроз")),
            "rot": any(k in t for k in ("гни", "rot", "плес", "mold")),
            "slow_growth": any(k in t for k in ("медлен", "не растет", "slow", "small", "замедл")),
            "pest": any(k in t for k in ("вредител", "клещ", "тля", "thrips", "mite", "aphid")),
        }

    def _compose_social_reply(self, text: str, scores: dict[str, float]) -> str:
        best = max(scores.items(), key=lambda x: x[1])[0] if scores else "generic"
        seed = f"{best}:{text}"

        responses: dict[str, list[str]] = {
            "greeting": [
                "Привет! Я на связи. Можем просто пообщаться или разобрать вопрос по растению.",
                f"Привет! Я {self.BOT_NAME}. Готов и к обычному диалогу, и к анализу растений.",
            ],
            "how_are_you": [
                "Все отлично, спасибо. Как у вас дела?",
                "Хорошо, работаю в боевом режиме. Чем помочь?",
            ],
            "name": [
                f"Меня зовут {self.BOT_NAME}.",
                f"Я {self.BOT_NAME}, ваш AI-помощник по растениям и диалогу.",
            ],
            "capabilities": [
                "Могу поддержать диалог, дать рекомендации по уходу и разобрать фото растения.",
                "Умею: общение, консультации по симптомам, анализ фото и краткий план действий.",
            ],
            "thanks": [
                "Пожалуйста. Если нужно, продолжим.",
                "Всегда рад помочь.",
            ],
            "bye": [
                "Хорошо, до связи.",
                "До встречи. Если что, пишите.",
            ],
        }

        if best in responses:
            return self._stable_choice(responses[best], seed)

        if self._is_question(text):
            return (
                "Хороший вопрос. Могу ответить в обычном формате разговора, "
                "а если нужно — отдельно перейти к диагностике растений."
            )

        return self._stable_choice(
            [
                "Понял вас. Продолжим диалог?",
                "Я на связи. Можем говорить свободно, без фото тоже.",
            ],
            seed,
        )

    def _compose_plant_consultation(self, text: str, feedback_state: str) -> str:
        flags = self._detect_plant_flags(text)

        causes: list[str] = []
        actions: list[str] = []
        urgent: list[str] = []

        if flags["yellow"]:
            causes.append("Пожелтение часто связано с переливом или дефицитом питания.")
            actions.append("Проверьте влажность грунта и режим подкормки.")
        if flags["dry"]:
            causes.append("Сухие края и вялость обычно указывают на водный стресс.")
            actions.append("Стабилизируйте полив и избегайте пересушивания.")
        if flags["spots"]:
            causes.append("Пятна могут быть инфекцией, ожогом или реакцией на стресс.")
            actions.append("Изолируйте растение и уберите прямые капли воды с листьев.")
            urgent.append("Если пятна быстро растут за 24-48 часов — нужна быстрая обработка.")
        if flags["rot"]:
            causes.append("Признаки гнили связаны с переувлажнением и слабой аэрацией.")
            actions.append("Снизьте полив и удалите пораженные участки стерильным инструментом.")
            urgent.append("Размягчение тканей и запах гнили — высокий риск.")
        if flags["slow_growth"]:
            causes.append("Замедление роста бывает при нехватке света или питания.")
            actions.append("Проверьте световой режим и корректируйте питание постепенно.")
        if flags["pest"]:
            causes.append("Возможны вредители (тля, клещ, трипс).")
            actions.append("Осмотрите нижнюю сторону листьев и точки роста.")
            urgent.append("При паутине или активной колонии действуйте сразу.")

        if not causes:
            causes.append("Без фото диагноз предварительный: причин может быть несколько одновременно.")
        if not actions:
            actions.extend(
                [
                    "Сравните верхние и нижние листья (цвет, тургор, пятна).",
                    "Сделайте контрольный снимок через 24 часа при том же освещении.",
                ]
            )
        if not urgent:
            urgent.append("Срочно реагируйте при резком ухудшении за сутки и массовом увядании.")

        if feedback_state == "negative":
            actions.append("Раз меняли схему ранее без эффекта, меняйте только один фактор за шаг.")
        elif feedback_state == "positive":
            actions.append("Раз стало лучше, закрепите режим еще на 2-3 дня и проверьте динамику.")

        lines = [
            "Краткая консультация по симптомам:",
            "",
            "1. Возможные причины:",
            f"- {causes[0]}",
            f"- {causes[1]}" if len(causes) > 1 else "",
            "",
            "2. Что сделать сейчас:",
            f"- {actions[0]}",
            f"- {actions[1]}" if len(actions) > 1 else "",
            "",
            "3. Когда срочно реагировать:",
            f"- {urgent[0]}",
            "",
            "Если хотите точнее, отправьте фото целиком и крупный план проблемной зоны.",
        ]
        return "\n".join([x for x in lines if x])

    def compose(self, message: str, prior_context: list[str] | None = None) -> str:
        text = self._normalize(message)
        if not text:
            return "Напишите сообщение. Могу и просто пообщаться, и помочь с растением."

        tokens = self._tokenize(text)
        scores = self._intent_scores(text=text, tokens=tokens, prior_context=prior_context)
        feedback_state = self._feedback_state(prior_context)

        plant_score = float(scores.get("plant", 0.0))
        social_score = max((v for k, v in scores.items() if k != "plant"), default=0.0)

        # Plant mode is enabled only when plant intent clearly dominates.
        follow_up = self._has_follow_up_hint(text)
        if (plant_score >= 0.9 and plant_score >= social_score + 0.2) or (
            follow_up and plant_score >= 0.65 and plant_score >= social_score
        ):
            return self._compose_plant_consultation(text=text, feedback_state=feedback_state)

        return self._compose_social_reply(text=text, scores=scores)
