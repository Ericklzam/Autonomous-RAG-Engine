"""Chunking is pure -- no database, no network, no API key."""

from app.config import settings
from app.services.ingestion import chunk_text


def test_empty_content_yields_no_chunks():
    assert chunk_text("") == []


def test_whitespace_only_content_yields_no_chunks():
    assert chunk_text("   \n\t  \n ") == []


def test_short_content_is_a_single_chunk():
    chunks = chunk_text("A short policy document.")
    assert chunks == ["A short policy document."]


def test_long_content_is_split_into_multiple_chunks():
    content = "word " * 1000  # ~5000 chars, well over chunk_size
    chunks = chunk_text(content)
    assert len(chunks) > 1
    assert all(len(chunk) <= settings.chunk_size for chunk in chunks)


def test_chunks_never_contain_blank_entries():
    content = "\n\n\n".join(["paragraph one", "", "paragraph two", "   "])
    assert all(chunk.strip() for chunk in chunk_text(content))
