"""API tests that need neither a database nor an API key."""

from fastapi.testclient import TestClient

from app.main import app

# Note: not used as a context manager, so the lifespan (and its schema check)
# does not run -- these routes touch nothing.
client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_rejects_empty_question():
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_ingest_rejects_missing_filename():
    assert client.post("/ingest", json={"content": "text"}).status_code == 422


def test_query_rejects_out_of_range_top_k():
    response = client.post("/query", json={"question": "hi", "top_k": 999})
    assert response.status_code == 422
