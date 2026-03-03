from uuid import uuid4

from fastapi.testclient import TestClient

from api.main import app


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
