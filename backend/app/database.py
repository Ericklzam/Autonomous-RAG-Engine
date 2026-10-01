"""SQLAlchemy engine, session factory, and startup schema verification.

V1 uses synchronous SQLAlchemy for simplicity (per sprint brief). The session
factory and `get_db` dependency are structured so swapping to async later only
touches this module.

Schema ownership moved to Alembic in Sprint 2. This module no longer calls
``create_all`` -- running both would race, and only Alembic can add the HNSW
index the retrieval path depends on.
"""

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

# pool_pre_ping avoids handing out dead connections after the DB restarts.
engine = create_engine(settings.database_url, pool_pre_ping=True, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base shared by all ORM models.
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_schema() -> None:
    """Fail fast at startup if migrations have not been applied.

    Without this the first ``/ingest`` call dies deep inside SQLAlchemy with a
    ProgrammingError; here it dies immediately with instructions.
    """
    # Imported for its side effect of registering models on Base.metadata.
    from app import models  # noqa: F401

    inspector = inspect(engine)
    if not inspector.has_table(models.DocumentChunk.__tablename__):
        raise RuntimeError(
            "Table 'document_chunks' is missing. Start the database with "
            "`docker compose up -d`, then apply migrations with "
            "`alembic upgrade head` from the backend/ directory."
        )
