# Autonomous RAG Engine

Semantic search over ingested documents, backed by Postgres + pgvector.

## Status

| Sprint | Scope | State |
|---|---|---|
| 1 | Ingestion skeleton: chunking, schema, placeholder vectors | Done |
| 2 | Real embeddings, migrations + HNSW index, `/query` retrieval | Done |
| 3 | Answer generation (LLM synthesis over retrieved chunks) | Not started |

Sprint 2 makes retrieval genuinely work. There is deliberately **no LLM answer
generation yet** — `/query` returns the source chunks and their distances.

## Prerequisites

- Python 3.11+
- Docker (for Postgres + pgvector)
- An OpenAI API key with billing credits

## Setup

```bash
cd backend

# 1. Environment
cp .env.example .env          # then edit: set POSTGRES_PASSWORD and OPENAI_API_KEY

# 2. Dependencies
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -r requirements-dev.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt   # macOS/Linux

# 3. Database
docker compose up -d
.venv/Scripts/python -m alembic upgrade head

# 4. Run
.venv/Scripts/python -m uvicorn app.main:app --reload
```

Interactive API docs: <http://127.0.0.1:8000/docs>

## API

### `POST /ingest`

Chunks, embeds, and stores a document. **Re-ingesting the same `filename`
replaces its chunks** rather than appending duplicates.

```json
{ "filename": "handbook.txt", "content": "Employees accrue twenty days..." }
```
→ `{ "document_name": "handbook.txt", "chunks_created": 12 }`

### `POST /query`

```json
{ "question": "How much PTO do I get?", "top_k": 5, "document_name": null }
```
→
```json
{
  "question": "How much PTO do I get?",
  "matches": [
    { "chunk_id": "...", "document_name": "handbook.txt",
      "chunk_text": "Employees accrue twenty days...", "distance": 0.18 }
  ]
}
```

`distance` is cosine distance: `0.0` identical, `1.0` unrelated. It is returned
on purpose — once Sprint 3 adds generation, it is how you tell a retrieval
failure apart from a generation failure. Matches worse than `MAX_DISTANCE`
(default `0.6`) are dropped, so an unanswerable question returns `[]` rather
than the least-bad noise.

Embedding-provider failures (missing key, quota, upstream outage) return
**502**, not 500 — the request was fine, the dependency was not.

## Resetting the database

**Any rows written during Sprint 1 must be discarded.** They hold all-zero
embeddings, and cosine distance against a zero vector divides by zero — those
rows produce `NaN` and sort unpredictably into every result set. They cannot be
back-filled in place; re-ingest the source documents instead.

```bash
cd backend
docker compose down -v          # drops the volume, including Sprint 1 rows
docker compose up -d
.venv/Scripts/python -m alembic upgrade head
```

## Tests

```bash
cd backend
.venv/Scripts/python -m pytest
```

Two tiers, split by what they need:

- **Unit** — no database, no network, no API key. Chunking, batching, response
  realignment, dimension validation, request validation.
- **Integration** — needs Postgres running; **skipped automatically** when it
  is not. Uses a deterministic fake embedder, so the full ingest→search round
  trip is verified without spending a cent or needing a key.

## Configuration

Everything is centralized in [`app/config.py`](backend/app/config.py) and
overridable via `.env`. Notable values:

| Setting | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 1536 dims |
| `EMBEDDING_DIM` | `1536` | **Changing this requires a migration** — the `Vector` column width is fixed at DDL time. `text-embedding-3-large` is 3072. |
| `EMBED_BATCH_SIZE` | `128` | Chunks per API request |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Characters |
| `DEFAULT_TOP_K` | `5` | Results per query |
| `MAX_DISTANCE` | `0.6` | Relevance ceiling |

## Design notes

- **Alembic owns the schema.** `create_all()` was removed in Sprint 2 — running
  both races, and only a migration can add the HNSW index. Startup fails fast
  with instructions if migrations have not been applied.
- **HNSW, not IVFFlat.** HNSW builds on an empty table and needs no training
  pass. The index uses `vector_cosine_ops`, which *must* match the query-time
  operator (`<=>`); a mismatch is silent — Postgres just ignores the index.
- **Embed before writing.** Ingestion generates vectors before deleting the old
  chunks, so a provider failure leaves the existing document intact.
- **Batched embeddings.** A 200-chunk document is two HTTP round trips, not 200.
- **Responses are realigned by `index`,** never trusted positionally — a
  shuffled response would otherwise pair chunks with the wrong vectors.
- **The embedding client is constructed lazily.** FastAPI resolves dependencies
  before validating request bodies, so an eager constructor that raised would
  turn every malformed request into a 500 and mask ordinary 422s.
