"""SQLAlchemy ORM models."""

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.database import Base

# OpenAI text-embedding-3-small / text-embedding-ada-002 produce 1536-dim
# vectors. Real embeddings arrive in Sprint 2; chunks are stored with zeros
# for now (see services/ingestion.py).
EMBEDDING_DIM = 1536


class DocumentChunk(Base):
    """A single chunk of an ingested document plus its embedding vector."""

    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_name = Column(String, nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk id={self.id} "
            f"document_name={self.document_name!r}>"
        )
