"""
Tests for the health endpoint.

Uses FastAPI's TestClient (backed by httpx) — no real network I/O is performed.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    """GET /health must respond with HTTP 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_expected_json() -> None:
    """GET /health must return exactly {'status': 'healthy'}."""
    response = client.get("/health")
    assert response.json() == {"status": "healthy"}
