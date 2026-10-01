"""Centralized application settings.

Every tunable (credentials, model names, chunking, retrieval) lives here so
modules never reach for ``os.getenv`` directly. Values come from the process
environment first, then ``backend/.env``.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/.env, regardless of the working directory
# the app happens to be launched from.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    database_url: str = Field(
        ...,
        description="SQLAlchemy connection string for the pgvector database.",
    )

    # --- OpenAI ---
    # Intentionally optional: the app must import and its non-embedding tests
    # must run before an API key exists. The embedding client raises a clear
    # error at call time if it is still empty.
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"

    # Must match the Vector() column width in models.py. text-embedding-3-small
    # and ada-002 are both 1536; text-embedding-3-large is 3072 and would
    # require a schema migration + full re-ingest.
    embedding_dim: int = 1536

    # The embeddings endpoint accepts arrays of up to 2048 inputs. 128 keeps
    # individual requests small enough to retry cheaply.
    embed_batch_size: int = 128
    openai_max_retries: int = 5
    openai_timeout_seconds: float = 30.0

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Retrieval ---
    default_top_k: int = 5

    # Cosine distance ceiling: 0.0 is identical, 1.0 is orthogonal. Hits above
    # this are dropped so an unanswerable question returns nothing rather than
    # the least-bad noise.
    max_distance: float = 0.6


@lru_cache
def get_settings() -> Settings:
    """Settings are read once per process and cached."""
    return Settings()


settings = get_settings()
