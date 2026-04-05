from io import BytesIO

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_analyze_text_file():
    content = (
        "John Smith from Acme Corp invested $5000 on 15 January 2024. "
        "The meeting was cancelled due to budget cuts but the team will reschedule."
    )
    files = {"file": ("sample.txt", BytesIO(content.encode("utf-8")), "text/plain")}
    response = client.post("/api/v1/documents/analyze", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "entities" in data
    assert "sentiment" in data
    assert "confidence_scores" in data
    assert 0.0 <= data["confidence_scores"]["overall"] <= 1.0


def test_api_v1_health_and_metrics():
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    metrics = client.get("/api/v1/metrics")
    assert metrics.status_code == 200
    payload = metrics.json()
    assert "total_processed" in payload


def test_api_key_enforcement_and_rate_limit():
    prev_require = settings.require_api_key
    prev_key = settings.api_key
    prev_limit = settings.rate_limit_per_minute
    try:
        settings.require_api_key = True
        settings.api_key = "unit-test-key"
        settings.rate_limit_per_minute = 1

        content = "A short note about Acme revenue growth."
        def build_files():
            return {"file": ("sample.txt", BytesIO(content.encode("utf-8")), "text/plain")}

        missing = client.post("/api/v1/documents/analyze", files=build_files())
        assert missing.status_code == 401

        ok = client.post(
            "/api/v1/documents/analyze",
            files=build_files(),
            headers={"Authorization": "Bearer unit-test-key"},
        )
        assert ok.status_code == 200

        limited = client.post(
            "/api/v1/documents/analyze",
            files=build_files(),
            headers={"Authorization": "Bearer unit-test-key"},
        )
        assert limited.status_code == 429
    finally:
        settings.require_api_key = prev_require
        settings.api_key = prev_key
        settings.rate_limit_per_minute = prev_limit
