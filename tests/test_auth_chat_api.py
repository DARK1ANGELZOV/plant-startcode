from uuid import uuid4

from fastapi.testclient import TestClient

from api.main import app
from utils.schemas import PHIResult, PlantMeasurement, PredictResponse, Recommendation


def test_register_login_and_chat_sessions() -> None:
    with TestClient(app) as client:
        email = f'user_auth_chat_{uuid4().hex[:8]}@example.com'
        password = 'StrongPass123'

        reg = client.post('/auth/register', json={'email': email, 'password': password})
        assert reg.status_code == 200
        token = reg.json()['access_token']

        me = client.get('/auth/me', headers={'Authorization': f'Bearer {token}'})
        assert me.status_code == 200
        assert me.json()['email'] == email

        create = client.post(
            '/chat/sessions',
            json={'title': 'My Session'},
            headers={'Authorization': f'Bearer {token}'},
        )
        assert create.status_code == 200
        session_id = create.json()['id']

        sessions = client.get('/chat/sessions', headers={'Authorization': f'Bearer {token}'})
        assert sessions.status_code == 200
        assert any(s['id'] == session_id for s in sessions.json())

        messages = client.get(f'/chat/sessions/{session_id}/messages', headers={'Authorization': f'Bearer {token}'})
        assert messages.status_code == 200
        assert isinstance(messages.json(), list)


def test_login_existing_user() -> None:
    with TestClient(app) as client:
        email = f'user_login_flow_{uuid4().hex[:8]}@example.com'
        password = 'StrongPass123'
        client.post('/auth/register', json={'email': email, 'password': password})

        login = client.post('/auth/login', json={'email': email, 'password': password})
        assert login.status_code == 200
        assert login.json()['token_type'] == 'bearer'


def test_chat_text_and_search() -> None:
    with TestClient(app) as client:
        email = f'user_text_chat_{uuid4().hex[:8]}@example.com'
        password = 'StrongPass123'
        reg = client.post('/auth/register', json={'email': email, 'password': password})
        assert reg.status_code == 200
        token = reg.json()['access_token']
        headers = {'Authorization': f'Bearer {token}'}

        ask = client.post(
            '/chat/message',
            json={'message': 'Листья желтеют и рост стал медленным'},
            headers=headers,
        )
        assert ask.status_code == 200
        payload = ask.json()
        assert payload.get('assistant_reply')
        assert payload.get('session_id') is not None
        session_id = int(payload['session_id'])

        rows = client.get(f'/chat/sessions/{session_id}/messages', headers=headers)
        assert rows.status_code == 200
        messages = rows.json()
        assert len(messages) >= 2
        assert messages[0]['role'] == 'user'
        assert messages[1]['role'] == 'assistant'

        search = client.get('/chat/search', params={'query': 'желтеют'}, headers=headers)
        assert search.status_code == 200
        assert any(int(hit['session_id']) == session_id for hit in search.json())


def test_delete_chat_session_removes_it_permanently() -> None:
    with TestClient(app) as client:
        email = f'user_delete_chat_{uuid4().hex[:8]}@example.com'
        password = 'StrongPass123'
        reg = client.post('/auth/register', json={'email': email, 'password': password})
        assert reg.status_code == 200
        token = reg.json()['access_token']
        headers = {'Authorization': f'Bearer {token}'}

        created = client.post('/chat/sessions', json={'title': 'Delete me'}, headers=headers)
        assert created.status_code == 200
        session_id = int(created.json()['id'])

        deleted = client.delete(f'/chat/sessions/{session_id}', headers=headers)
        assert deleted.status_code == 200
        assert deleted.json().get('status') == 'deleted'

        sessions = client.get('/chat/sessions', headers=headers)
        assert sessions.status_code == 200
        assert not any(int(s['id']) == session_id for s in sessions.json())

        rows = client.get(f'/chat/sessions/{session_id}/messages', headers=headers)
        assert rows.status_code == 404


def test_chat_analyze_reply_template_same_for_guest_and_auth() -> None:
    with TestClient(app) as client:
        async def fake_run_single(**kwargs):
            _ = kwargs
            return PredictResponse(
                run_id='run_test_template',
                scale_mm_per_px=0.3122,
                scale_source='cache',
                measurements=[
                    PlantMeasurement(
                        instance_id=1,
                        crop='Wheat',
                        class_name='leaves',
                        confidence=0.92,
                        area_px=12000,
                        area_mm2=41180.0,
                        length_px=400.0,
                        length_mm=125.0,
                        reliable=True,
                    ),
                    PlantMeasurement(
                        instance_id=2,
                        crop='Wheat',
                        class_name='stem',
                        confidence=0.15,
                        area_px=2200,
                        area_mm2=6667.0,
                        length_px=295.8,
                        length_mm=92.4,
                        reliable=False,
                    ),
                ],
                summary={
                    'segmentation': {
                        'root': {'detected': False, 'confidence': 0.0},
                        'stem': {'detected': True, 'confidence': 0.15},
                        'leaves': {'detected': True, 'confidence': 0.92},
                    },
                    'confidence_by_class': {'stem': 0.15, 'leaves': 0.92},
                    'measurement_trust_score': 63.2,
                    'measurement_trust_level': 'medium',
                    'image_quality': {'notes': ['Изображение умеренно четкое']},
                    'calibration_reliable': True,
                    'calibration_source': 'cache',
                    'calibration_camera_id': 'lab_camera',
                    'calibration_error_pct': 6.0,
                },
                recommendations=[
                    Recommendation(
                        severity='warning',
                        message='Нужно усилить контроль полива.',
                        action='Увеличить глубину полива, проверить pH и EC субстрата.',
                    ),
                ],
                disease_analysis={},
                phi=PHIResult(score=54.5, status='Risk', reasons=['Test']),
                explainability={},
                active_learning={},
                files={'overlay': 'outputs/run_test_template/overlay.png'},
            )

        client.app.state.inference_service.run_single = fake_run_single

        payload = {
            'crop': 'Wheat',
            'message': 'Проверь растение по фото',
            'source_type': 'lab_camera',
            'camera_id': 'default',
        }

        guest = client.post(
            '/chat/analyze',
            data=payload,
            files={'image': ('leaf.jpg', b'img-bytes', 'image/jpeg')},
        )
        assert guest.status_code == 200
        guest_reply = guest.json()['assistant_reply']

        email = f'user_template_{uuid4().hex[:8]}@example.com'
        password = 'StrongPass123'
        reg = client.post('/auth/register', json={'email': email, 'password': password})
        assert reg.status_code == 200
        token = reg.json()['access_token']
        headers = {'Authorization': f'Bearer {token}'}

        auth = client.post(
            '/chat/analyze',
            data=payload,
            files={'image': ('leaf.jpg', b'img-bytes', 'image/jpeg')},
            headers=headers,
        )
        assert auth.status_code == 200
        auth_reply = auth.json()['assistant_reply']

        required_blocks = [
            'Результаты анализа изображения:',
            '1. Сегментация:',
            '2. Измерения',
            '3. Перевод в мм:',
            '4. Вывод:',
            '5. Рекомендации:',
        ]
        for block in required_blocks:
            assert block in guest_reply
            assert block in auth_reply

        assert guest_reply == auth_reply
