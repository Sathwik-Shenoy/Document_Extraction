from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_dependency_health_endpoint_shape():
    response = client.get("/health/dependencies")
    assert response.status_code == 200
    data = response.json()
    assert "extractors" in data
    assert "pdf" in data["extractors"]
    assert "docx" in data["extractors"]
    assert "image" in data["extractors"]
    assert "all_extractors_ready" in data["extractors"]
