"""SQLAlchemy ORM models."""

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.config import settings
from app.database import Base

# Single source of truth is `settings.embedding_dim`; re-exported here because
# the embedding and ingestion services validate against it. Changing it
# requires a migration -- the Vector column width is fixed at DDL time.
EMBEDDING_DIM = settings.embedding_dim


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
