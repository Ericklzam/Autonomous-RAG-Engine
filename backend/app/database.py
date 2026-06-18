"""SQLAlchemy engine, session factory, and DB initialization.

V1 uses synchronous SQLAlchemy for simplicity (per sprint brief). The session
factory and `get_db` dependency are structured so swapping to async later only
touches this module.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

# Load variables from backend/.env into the process environment.
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Copy backend/.env.example to backend/.env "
        "and fill it in."
    )

# pool_pre_ping avoids handing out dead connections after the DB restarts.
engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)

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


def init_db() -> None:
    """Ensure the pgvector extension exists, then create all tables.

    The `vector` extension must exist before any table with a Vector column is
    created, so we run it first in its own transaction.
    """
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # Imported here (not at module top) to avoid a circular import:
    # models.py imports Base from this module.
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
