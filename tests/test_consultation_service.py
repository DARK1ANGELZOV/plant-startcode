from services.consultation_service import ConsultationService


def test_greeting_returns_conversational_reply() -> None:
    service = ConsultationService()
    text = service.compose("Привет")
    assert "Привет" in text or "на связи" in text
    assert "Краткая консультация по симптомам" not in text


def test_how_are_you_stays_in_social_mode() -> None:
    service = ConsultationService()
    text = service.compose("Как ты?")
    assert "дела" in text.lower() or "чем помочь" in text.lower() or "отлично" in text.lower()
    assert "Краткая консультация по симптомам" not in text


def test_name_question_has_identity_reply() -> None:
    service = ConsultationService()
    text = service.compose("Как тебя зовут?")
    assert "PlantVision AI" in text


def test_plant_message_returns_consultation_block() -> None:
    service = ConsultationService()
    text = service.compose("У растения желтеют листья и есть пятна")
    assert "Краткая консультация по симптомам" in text
    assert "Возможные причины" in text
    assert "Что сделать сейчас" in text


def test_follow_up_uses_plant_context() -> None:
    service = ConsultationService()
    text = service.compose(
        "Что делать дальше?",
        prior_context=[
            "У растения желтеют листья и есть пятна.",
            "Проверьте полив и освещение.",
        ],
    )
    assert "Краткая консультация по симптомам" in text

