"""Database engine, session factory, and connection pool management.

Pool configuration (from core/constants.py):
    pool_size=10, max_overflow=5, pool_timeout=3s, pool_recycle=1800s
    pool_pre_ping=True for stale connection detection.

Lifecycle:
    - init_engine() called once during FastAPI lifespan startup.
    - get_session() used as FastAPI Depends() for per-request sessions.
    - dispose_engine() called during graceful shutdown.
"""
from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from mlsentry.core.constants import (
    DB_MAX_OVERFLOW,
    DB_POOL_RECYCLE,
    DB_POOL_SIZE,
    DB_POOL_TIMEOUT,
)

# Module-level engine and session factory.
# Engine is None until init_engine() is called during lifespan startup.
engine: Engine | None = None
SessionLocal: sessionmaker[Session] = sessionmaker(
    autocommit=False, autoflush=False,
)


def init_engine(database_url: str) -> Engine:
    """Create and configure the SQLAlchemy engine.

    Called once during application lifespan startup.
    Binds the global SessionLocal factory to the created engine.

    Args:
        database_url: PostgreSQL connection string
            (e.g. postgresql://user:pass@host:5432/dbname).

    Returns:
        Configured SQLAlchemy Engine instance.

    Raises:
        sqlalchemy.exc.ArgumentError: If database_url is malformed.
    """
    global engine
    engine = create_engine(
        database_url,
        pool_size=DB_POOL_SIZE,
        max_overflow=DB_MAX_OVERFLOW,
        pool_timeout=DB_POOL_TIMEOUT,
        pool_recycle=DB_POOL_RECYCLE,
        pool_pre_ping=True,
    )
    SessionLocal.configure(bind=engine)
    return engine


def get_session() -> Generator[Session, None, None]:
    """Yield a database session and ensure cleanup.

    Usage as a FastAPI dependency:
        session: Session = Depends(get_session)

    Yields:
        Active SQLAlchemy Session bound to the configured engine.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def dispose_engine() -> None:
    """Dispose of the engine connection pool.

    Called during graceful shutdown to release all DB connections.
    Safe to call if engine is None (no-op).
    """
    global engine
    if engine is not None:
        engine.dispose()
        engine = None
