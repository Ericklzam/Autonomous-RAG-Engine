"""Document ingestion: split raw text into chunks and attach embeddings.

Real embeddings arrive in Sprint 2. For now each chunk gets a mock vector of
1536 zeros so it satisfies the non-nullable `embedding` column on
`DocumentChunk`.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import EMBEDDING_DIM, DocumentChunk

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# A single splitter instance is reusable and stateless across calls.
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)


def _mock_embedding() -> list[float]:
    """Placeholder embedding (1536 zero floats) until Sprint 2 wires a model."""
    return [0.0] * EMBEDDING_DIM


def chunk_document(document_name: str, content: str) -> list[DocumentChunk]:
    """Split ``content`` into overlapping chunks ready for DB insertion.

    Returns unpersisted ``DocumentChunk`` ORM instances. Whitespace-only or
    empty input yields an empty list (the splitter returns no chunks).
    """
    texts = _splitter.split_text(content)
    return [
        DocumentChunk(
            document_name=document_name,
            chunk_text=text,
            embedding=_mock_embedding(),
        )
        for text in texts
    ]
