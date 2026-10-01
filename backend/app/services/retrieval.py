"""Semantic search over stored document chunks."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import DocumentChunk
from app.services.embeddings import EmbeddingClient


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_name: str
    chunk_text: str
    distance: float


def search(
    db: Session,
    client: EmbeddingClient,
    question: str,
    top_k: int | None = None,
    max_distance: float | None = None,
    document_name: str | None = None,
) -> list[SearchHit]:
    """Return the chunks nearest to ``question`` by cosine distance.

    ``max_distance`` is applied *after* the top-k cut: take the k best, then
    drop any that are not actually relevant. A question with no good match
    returns an empty list rather than the least-bad noise.
    """
    top_k = top_k or settings.default_top_k
    threshold = settings.max_distance if max_distance is None else max_distance

    query_vector = client.embed_query(question)

    # Must match the index's operator class (vector_cosine_ops) or Postgres
    # will ignore the index and sequential-scan the table.
    distance = DocumentChunk.embedding.cosine_distance(query_vector).label(
        "distance"
    )

    stmt = select(DocumentChunk, distance)
    if document_name:
        stmt = stmt.where(DocumentChunk.document_name == document_name)
    stmt = stmt.order_by(distance).limit(top_k)

    return [
        SearchHit(
            chunk_id=str(chunk.id),
            document_name=chunk.document_name,
            chunk_text=chunk.chunk_text,
            distance=float(dist),
        )
        for chunk, dist in db.execute(stmt).all()
        if dist is not None and float(dist) <= threshold
    ]
