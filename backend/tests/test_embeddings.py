"""Embedding client behavior, with the OpenAI SDK mocked out.

These tests cost nothing and need no API key, but they cover the failure modes
that would otherwise corrupt the index silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import openai
import pytest

from app.services.embeddings import EmbeddingError, OpenAIEmbeddingClient

DIM = 8


@dataclass
class _Item:
    index: int
    embedding: list[float]


@dataclass
class _Response:
    data: list[_Item]


class _FakeEmbeddings:
    def __init__(self, owner: "_FakeOpenAI") -> None:
        self._owner = owner

    def create(self, **kwargs):
        self._owner.calls.append(kwargs)
        return self._owner.responder(kwargs)


class _FakeOpenAI:
    def __init__(self, responder, **_ignored):
        self.calls: list[dict] = []
        self.responder = responder
        self.embeddings = _FakeEmbeddings(self)


@pytest.fixture
def patch_openai(monkeypatch):
    """Install a fake OpenAI class; returns the instance it produced."""
    created: list[_FakeOpenAI] = []

    def install(responder):
        def factory(**kwargs):
            client = _FakeOpenAI(responder, **kwargs)
            created.append(client)
            return client

        monkeypatch.setattr(openai, "OpenAI", factory)
        return created

    return install


def _ok_responder(width: int = DIM):
    def responder(kwargs):
        texts = kwargs["input"]
        return _Response(
            data=[_Item(index=i, embedding=[0.1] * width) for i in range(len(texts))]
        )

    return responder


def _client(**overrides) -> OpenAIEmbeddingClient:
    params = {
        "api_key": "sk-test",
        "model": "text-embedding-3-small",
        "dimensions": DIM,
        "batch_size": 2,
    }
    params.update(overrides)
    return OpenAIEmbeddingClient(**params)


def test_missing_api_key_fails_with_a_clear_message():
    """Constructing is safe; the error surfaces at call time.

    This matters because FastAPI builds dependencies before validating the
    request body -- a raising constructor would turn every malformed request
    into a 500.
    """
    client = OpenAIEmbeddingClient(api_key="")
    with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
        client.embed_texts(["anything"])


def test_empty_input_makes_no_api_call(patch_openai):
    created = patch_openai(_ok_responder())
    client = _client()
    assert client.embed_texts([]) == []
    assert created == [], "no SDK client should even be constructed"


def test_inputs_are_batched_into_few_requests(patch_openai):
    created = patch_openai(_ok_responder())
    client = _client(batch_size=2)

    vectors = client.embed_texts(["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    # 5 texts at batch_size 2 -> 3 round trips, not 5.
    assert len(created[0].calls) == 3
    assert [len(call["input"]) for call in created[0].calls] == [2, 2, 1]


def test_out_of_order_response_is_realigned_to_input_order(patch_openai):
    """A shuffled response must not pair chunks with the wrong vectors."""

    def responder(kwargs):
        texts = kwargs["input"]
        items = [
            _Item(index=i, embedding=[float(i)] * DIM) for i in range(len(texts))
        ]
        return _Response(data=list(reversed(items)))

    patch_openai(responder)
    client = _client(batch_size=10)

    vectors = client.embed_texts(["first", "second", "third"])

    assert [vector[0] for vector in vectors] == [0.0, 1.0, 2.0]


def test_wrong_dimension_is_rejected(patch_openai):
    """Guards against someone switching to text-embedding-3-large in config."""
    patch_openai(_ok_responder(width=DIM + 1))
    client = _client()

    with pytest.raises(EmbeddingError, match="dim vectors"):
        client.embed_texts(["a"])


def test_short_response_is_rejected(patch_openai):
    def responder(kwargs):
        return _Response(data=[_Item(index=0, embedding=[0.1] * DIM)])

    patch_openai(responder)
    client = _client(batch_size=10)

    with pytest.raises(EmbeddingError, match="Expected 2 embeddings"):
        client.embed_texts(["a", "b"])


def test_blank_text_is_rejected_before_the_api_call(patch_openai):
    created = patch_openai(_ok_responder())
    client = _client()

    with pytest.raises(EmbeddingError, match="empty or whitespace"):
        client.embed_texts(["fine", "   "])
    assert created == [], "validation must happen before any network call"


def test_dimensions_param_sent_for_v3_models(patch_openai):
    created = patch_openai(_ok_responder())
    _client(model="text-embedding-3-small").embed_texts(["a"])
    assert created[0].calls[0]["dimensions"] == DIM


def test_dimensions_param_omitted_for_ada_002(patch_openai):
    """ada-002 has a fixed width and rejects the `dimensions` argument."""
    created = patch_openai(_ok_responder())
    _client(model="text-embedding-ada-002").embed_texts(["a"])
    assert "dimensions" not in created[0].calls[0]


def test_sdk_errors_are_normalized_to_embedding_error(patch_openai):
    def responder(kwargs):
        raise RuntimeError("429 rate limit exceeded")

    patch_openai(responder)

    with pytest.raises(EmbeddingError, match="Embedding request failed"):
        _client().embed_texts(["a"])
