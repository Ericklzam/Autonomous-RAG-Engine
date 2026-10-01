"""Embedding generation.

Replaces Sprint 1's zero-vector placeholder with real OpenAI embeddings.

The public surface is the ``EmbeddingClient`` protocol rather than a concrete
class so tests can inject a deterministic fake and so a future provider swap
touches only this module.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, runtime_checkable

from app.config import settings


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced (auth, quota, bad response)."""


@runtime_checkable
class EmbeddingClient(Protocol):
    """Anything that can turn text into vectors."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Output order must match input order."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...


def _batched(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


class OpenAIEmbeddingClient:
    """OpenAI-backed embedding client.

    Batches inputs into as few HTTP round trips as the API allows: a
    200-chunk document is two requests, not two hundred.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dim
        self.batch_size = batch_size or settings.embed_batch_size

        self._api_key = (
            api_key if api_key is not None else settings.openai_api_key
        )
        self._client = None

    def _sdk(self):
        """Build the SDK client on first use.

        Construction is deliberately lazy. FastAPI resolves dependencies
        *before* validating the request body, so a constructor that raised
        would escape the endpoint's error handling (an unhandled 500 instead
        of a 502) and would mask ordinary 422s on malformed requests.
        """
        if self._client is None:
            if not self._api_key:
                raise EmbeddingError(
                    "OPENAI_API_KEY is not set. Add it to backend/.env before "
                    "ingesting or querying documents."
                )

            # Imported here so merely importing this module does not pay for
            # the SDK import.
            from openai import OpenAI

            # The SDK retries 429s, 5xx, and connection errors with
            # exponential backoff internally -- no hand-rolled retry loop.
            self._client = OpenAI(
                api_key=self._api_key,
                max_retries=settings.openai_max_retries,
                timeout=settings.openai_timeout_seconds,
            )
        return self._client

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        kwargs = {"model": self.model, "input": batch}

        # Only the v3 models accept an explicit output width. Sending it makes
        # a misconfigured model fail loudly here instead of silently writing
        # wrong-width vectors.
        if self.model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self.dimensions

        try:
            response = self._sdk().embeddings.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize SDK errors
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc

        # The API documents `index` on each item; sort rather than trusting
        # positional order, because a mismatch would silently pair chunks with
        # the wrong vectors.
        items = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in items]

        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"Expected {len(batch)} embeddings, received {len(vectors)}."
            )

        for vector in vectors:
            if len(vector) != self.dimensions:
                raise EmbeddingError(
                    f"Model {self.model!r} returned {len(vector)}-dim vectors, "
                    f"but the database column is {self.dimensions}-dim."
                )

        return vectors

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise EmbeddingError("Cannot embed empty or whitespace-only text.")

        vectors: list[list[float]] = []
        for batch in _batched(texts, self.batch_size):
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


@lru_cache
def _cached_client() -> OpenAIEmbeddingClient:
    return OpenAIEmbeddingClient()


def get_embedding_client() -> EmbeddingClient:
    """FastAPI dependency. Override in tests via ``dependency_overrides``."""
    return _cached_client()
