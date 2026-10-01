"""Shared fixtures.

Unit tests need neither a database nor an API key. Integration tests need a
running Postgres+pgvector and are skipped automatically when it is absent.
"""

from __future__ import annotations

import hashlib
import math
import random

import pytest

from app.config import settings


class FakeEmbeddingClient:
    """Deterministic stand-in for OpenAI.

    Maps text -> a stable pseudo-random unit vector. Identical text yields an
    identical vector (cosine distance 0), and unrelated text yields a
    near-orthogonal one (distance ~1.0), which is exactly the property the
    retrieval threshold is meant to exploit.
    """

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embedding_dim
        self.calls: list[list[str]] = []

    def _vector(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = random.Random(seed)
        raw = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


@pytest.fixture
def fake_client() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


def _database_available() -> bool:
    try:
        from sqlalchemy import text

        from app.database import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _database_available(),
    reason="Postgres+pgvector not reachable; start it with `docker compose up -d`.",
)
