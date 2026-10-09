"""Engine and session lifecycle.

Nothing is created at import time. Create one engine per process with
`create_db_engine()` and share it; create sessions from the factory returned by
`create_session_factory()`.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.config import get_database_url


def create_db_engine(database_url: str | None = None) -> Engine:
    """Create an engine; defaults to `DATABASE_URL`. Does not connect."""
    return create_engine(database_url or get_database_url())


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Provide a session that commits on success and rolls back on error."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
