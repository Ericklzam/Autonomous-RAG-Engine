"""FastAPI application entrypoint for the Autonomous RAG Engine."""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.services.ingestion import chunk_document


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify the pgvector extension + tables exist before any request is served.
    init_db()
    yield


app = FastAPI(
    title="Autonomous RAG Engine",
    version="0.1.0",
    lifespan=lifespan,
)


# Schemas

class IngestRequest(BaseModel):
    filename: str = Field(..., min_length=1, examples=["handbook.txt"])
    content: str = Field(..., description="Raw document text to ingest.")


class IngestResponse(BaseModel):
    document_name: str
    chunks_created: int


# Endpoints

@app.get("/")
def health_check():
    """Liveness probe."""
    return {"status": "ok", "service": "rag-ingestion"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest, db: Session = Depends(get_db)):
    """Chunk raw text and persist each chunk (with a mock embedding)."""
    chunks = chunk_document(payload.filename, payload.content)

    if not chunks:
        # Empty / whitespace-only content produces no chunks; nothing to store.
        return IngestResponse(document_name=payload.filename, chunks_created=0)

    try:
        for chunk in chunks:
            db.add(chunk)
        db.commit()
    except Exception as exc:  # noqa: BLE001 - roll back, then surface as 500
        db.rollback()
        raise HTTPException(
            status_code=500, detail="Failed to persist document chunks."
        ) from exc

    return IngestResponse(
        document_name=payload.filename,
        chunks_created=len(chunks),
    )
