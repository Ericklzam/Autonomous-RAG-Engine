"""Document ingestion: split raw text into chunks and embed them.

Sprint 2 replaced the zero-vector placeholder with real embeddings, and made
re-ingesting a document idempotent instead of duplicating every chunk.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from app.config import settings
from app.models import DocumentChunk
from app.services.embeddings import EmbeddingClient

# A single splitter instance is reusable and stateless across calls.
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    length_function=len,
)


def chunk_text(content: str) -> list[str]:
    """Split ``content`` into overlapping chunks.

    Pure and API-free, so it is testable without network or credentials.
    Whitespace-only or empty input yields an empty list.
    """
    return [chunk for chunk in _splitter.split_text(content) if chunk.strip()]


def ingest_document(
    db: Session,
    client: EmbeddingClient,
    document_name: str,
    content: str,
) -> int:
    """Chunk, embed, and stage ``content`` for persistence.

    Re-ingesting the same ``document_name`` replaces its existing chunks, so
    ingestion is a "refresh" rather than an append.

    Stages the work in ``db`` but does not commit -- the caller owns the
    transaction. Returns the number of chunks staged.
    """
    texts = chunk_text(content)
    if not texts:
        return 0

    # Embed BEFORE touching the database. If the embedding call fails, the
    # document's existing chunks are still intact.
    vectors = client.embed_texts(texts)

    if len(vectors) != len(texts):
        raise ValueError(
            f"Embedding count ({len(vectors)}) does not match chunk count "
            f"({len(texts)})."
        )

    # Replace, don't append.
    db.query(DocumentChunk).filter(
        DocumentChunk.document_name == document_name
    ).delete(synchronize_session=False)

    db.add_all(
        [
            DocumentChunk(
                document_name=document_name,
                chunk_text=text,
                embedding=vector,
            )
            for text, vector in zip(texts, vectors)
        ]
    )

    return len(texts)
