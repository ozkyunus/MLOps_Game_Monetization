"""Postgres engine + FastAPI session dependency.

The engine is a lazy module-level proxy: the first attribute access materialises
it against `SQLALCHEMY_DATABASE_URL`. This keeps `import src.database` cheap and
CI-safe when the env var is unset (unit tests that don't touch the DB won't
trigger a bogus connect attempt at import time).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine

load_dotenv()


class _LazyEngine:
    """Lazily materialised SQLAlchemy engine. Behaves like a real Engine on
    first attribute access; before that it's a cheap no-op object that Python
    can import without a live database."""

    _instance = None

    def _materialise(self):
        if self._instance is None:
            url = os.getenv("SQLALCHEMY_DATABASE_URL")
            if not url:
                raise RuntimeError(
                    "SQLALCHEMY_DATABASE_URL is not set. Create a `.env` or set the "
                    "env var to reach Postgres."
                )
            # echo=False for production sanity; flip in local .env if you want SQL logs.
            self._instance = create_engine(url, echo=False)
        return self._instance

    def __getattr__(self, name):
        return getattr(self._materialise(), name)


engine = _LazyEngine()


def get_engine():
    """Public accessor for the real (materialised) SQLAlchemy engine.
    Use inside request handlers / scripts; never at module import time."""
    return engine._materialise()


def create_db_and_tables():
    SQLModel.metadata.create_all(engine._materialise())


def get_db():
    db = Session(engine._materialise())
    try:
        yield db
    finally:
        db.close()
