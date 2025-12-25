"""
Database Infrastructure
=======================

Manages database connections, session lifecycle, and engine configuration.

Uses SQLAlchemy 2.0 with asyncpg for async PostgreSQL operations.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.config import settings


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy models.

    SQLAlchemy 2.0 style using DeclarativeBase.
    All models inherit from this class.
    """
    pass


# Global engine and session maker
_engine: AsyncEngine | None = None
_session_maker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """
    Get or create the database engine.

    Returns:
        AsyncEngine: SQLAlchemy async engine

    Raises:
        RuntimeError: If engine has not been initialized
    """
    global _engine
    if _engine is None:
        raise RuntimeError("Database engine not initialized. Call init_database() first.")
    return _engine


def init_database() -> AsyncEngine:
    """
    Initialize the database engine and session maker.

    Should be called during application startup.

    Returns:
        AsyncEngine: The initialized engine
    """
    global _engine, _session_maker

    # Fix asyncpg SSL: replace sslmode with ssl for asyncpg compatibility
    database_url = settings.database_url.replace("sslmode=", "ssl=")

    _engine = create_async_engine(
        database_url,
        echo=settings.debug,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,  # Verify connections before using
    )

    _session_maker = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,  # Prevent lazy loading after commit
        autocommit=False,
        autoflush=False,
    )

    return _engine


async def close_database() -> None:
    """
    Close the database engine and dispose of connections.

    Should be called during application shutdown.
    """
    global _engine, _session_maker

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_maker = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async generator for database sessions.

    For use with FastAPI's Depends() - FastAPI handles the lifecycle.

    Usage in FastAPI:
        @app.get("/users")
        async def get_users(session: AsyncSession = Depends(get_session)):
            result = await session.execute(select(User))
            return result.scalars().all()

    Manual usage:
        async for session in get_session():
            result = await session.execute(select(User))
            users = result.scalars().all()
            break  # Session is cleaned up after exit

    Yields:
        AsyncSession: SQLAlchemy async session
    """
    global _session_maker

    if _session_maker is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with _session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.

    For use in background tasks, scripts, and manual async operations.
    Use this when you need to manually manage the session lifecycle.

    Usage:
        async with get_session_context() as session:
            result = await session.execute(select(User))
            users = result.scalars().all()

    Yields:
        AsyncSession: SQLAlchemy async session
    """
    global _session_maker

    if _session_maker is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with _session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """
    Create all database tables.

    This should only be used for development/testing.
    Production should use migrations (Alembic).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
