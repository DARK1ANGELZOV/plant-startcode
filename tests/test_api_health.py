from fastapi.testclient import TestClient

from api.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get('/health')
        assert response.status_code == 200
        payload = response.json()
        assert payload['status'] == 'ok'
        assert payload['model_loaded'] is False
