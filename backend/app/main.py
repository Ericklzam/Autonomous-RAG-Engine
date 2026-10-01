"""FastAPI application entrypoint for the Autonomous RAG Engine."""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, verify_schema
from app.services.embeddings import (
    EmbeddingClient,
    EmbeddingError,
    get_embedding_client,
)
from app.services.ingestion import ingest_document
from app.services.retrieval import search


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail at boot with instructions rather than mid-request with a traceback.
    verify_schema()
    yield


app = FastAPI(
    title="Autonomous RAG Engine",
    version="0.2.0",
    lifespan=lifespan,
)


# Schemas

class IngestRequest(BaseModel):
    filename: str = Field(..., min_length=1, examples=["handbook.txt"])
    content: str = Field(..., description="Raw document text to ingest.")


class IngestResponse(BaseModel):
    document_name: str
    chunks_created: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is the PTO policy?"])
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_name: str | None = Field(
        default=None, description="Restrict the search to one document."
    )


class Match(BaseModel):
    chunk_id: str
    document_name: str
    chunk_text: str
    # Cosine distance: 0.0 is identical, 1.0 is unrelated. Returned so a bad
    # answer can be diagnosed as retrieval failure vs. generation failure.
    distance: float


class QueryResponse(BaseModel):
    question: str
    matches: list[Match]


# Endpoints

@app.get("/")
def health_check():
    """Liveness probe."""
    return {"status": "ok", "service": "rag-engine", "version": app.version}


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    payload: IngestRequest,
    db: Session = Depends(get_db),
    client: EmbeddingClient = Depends(get_embedding_client),
):
    """Chunk, embed, and store a document. Re-ingesting replaces its chunks."""
    try:
        chunks_created = ingest_document(
            db, client, payload.filename, payload.content
        )
        db.commit()
    except EmbeddingError as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - roll back, then surface as 500
        db.rollback()
        raise HTTPException(
            status_code=500, detail="Failed to persist document chunks."
        ) from exc

    return IngestResponse(
        document_name=payload.filename,
        chunks_created=chunks_created,
    )


@app.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    db: Session = Depends(get_db),
    client: EmbeddingClient = Depends(get_embedding_client),
):
    """Return the stored chunks most semantically similar to the question."""
    try:
        hits = search(
            db,
            client,
            question=payload.question,
            top_k=payload.top_k or settings.default_top_k,
            document_name=payload.document_name,
        )
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return QueryResponse(
        question=payload.question,
        matches=[
            Match(
                chunk_id=hit.chunk_id,
                document_name=hit.document_name,
                chunk_text=hit.chunk_text,
                distance=hit.distance,
            )
            for hit in hits
        ],
    )
