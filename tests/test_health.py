from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_http_200() -> None:
    response = client.get("/health")

    assert response.status_code == 200


def test_health_returns_expected_body() -> None:
    response = client.get("/health")

    assert response.json() == {"status": "ok"}
