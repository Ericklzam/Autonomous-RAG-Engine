"""End-to-end ingest -> search against a real pgvector database.

Skipped automatically when Postgres is not running. The embedding client is
still the deterministic fake, so these cost nothing and need no API key --
they verify the SQL, the index path, and the threshold, not the model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal, get_db
from app.main import app
from app.models import DocumentChunk
from app.services.embeddings import get_embedding_client
from app.services.ingestion import ingest_document
from app.services.retrieval import search
from tests.conftest import FakeEmbeddingClient, requires_db

pytestmark = requires_db

DOC = "test_handbook.txt"
OTHER_DOC = "test_other.txt"

CHUNK_A = "Employees accrue twenty days of paid time off each year."
CHUNK_B = "The office kitchen is restocked every Tuesday morning."
CONTENT = f"{CHUNK_A}\n\n{CHUNK_B}"


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.query(DocumentChunk).filter(
            DocumentChunk.document_name.in_([DOC, OTHER_DOC])
        ).delete(synchronize_session=False)
        session.commit()
        session.close()


def test_ingest_stores_chunks(db, fake_client):
    count = ingest_document(db, fake_client, DOC, CONTENT)
    db.commit()

    assert count > 0
    stored = db.query(DocumentChunk).filter_by(document_name=DOC).count()
    assert stored == count


def test_empty_content_stores_nothing(db, fake_client):
    assert ingest_document(db, fake_client, DOC, "   \n  ") == 0
    db.commit()
    assert db.query(DocumentChunk).filter_by(document_name=DOC).count() == 0


def test_reingest_replaces_rather_than_duplicates(db, fake_client):
    first = ingest_document(db, fake_client, DOC, CONTENT)
    db.commit()

    second = ingest_document(db, fake_client, DOC, CONTENT)
    db.commit()

    assert first == second
    total = db.query(DocumentChunk).filter_by(document_name=DOC).count()
    assert total == first, "re-ingesting must refresh, not append"


def test_failed_embedding_leaves_existing_chunks_intact(db, fake_client):
    ingest_document(db, fake_client, DOC, CONTENT)
    db.commit()
    before = db.query(DocumentChunk).filter_by(document_name=DOC).count()

    class Exploding(FakeEmbeddingClient):
        def embed_texts(self, texts):
            raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        ingest_document(db, Exploding(), DOC, "replacement content")
    db.rollback()

    assert db.query(DocumentChunk).filter_by(document_name=DOC).count() == before


def test_search_returns_the_semantically_nearest_chunk(db, fake_client):
    ingest_document(db, fake_client, DOC, CONTENT)
    db.commit()

    hits = search(db, fake_client, question=CHUNK_A, top_k=2)

    assert hits, "expected at least one hit"
    assert hits[0].chunk_text == CHUNK_A
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)
    assert hits[0].document_name == DOC


def test_search_drops_irrelevant_results(db, fake_client):
    ingest_document(db, fake_client, DOC, CONTENT)
    db.commit()

    hits = search(db, fake_client, question="completely unrelated subject", top_k=5)

    assert hits == [], "distance threshold should filter out noise"


def test_search_can_be_scoped_to_one_document(db, fake_client):
    ingest_document(db, fake_client, DOC, CHUNK_A)
    ingest_document(db, fake_client, OTHER_DOC, CHUNK_A)
    db.commit()

    hits = search(db, fake_client, question=CHUNK_A, top_k=10, document_name=DOC)

    assert hits
    assert {hit.document_name for hit in hits} == {DOC}


def test_api_round_trip(db, fake_client):
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_embedding_client] = lambda: fake_client
    try:
        client = TestClient(app)

        ingested = client.post("/ingest", json={"filename": DOC, "content": CONTENT})
        assert ingested.status_code == 200
        assert ingested.json()["chunks_created"] > 0

        queried = client.post("/query", json={"question": CHUNK_A, "top_k": 3})
        assert queried.status_code == 200

        matches = queried.json()["matches"]
        assert matches
        assert matches[0]["chunk_text"] == CHUNK_A
        assert matches[0]["distance"] == pytest.approx(0.0, abs=1e-6)
    finally:
        app.dependency_overrides.clear()
