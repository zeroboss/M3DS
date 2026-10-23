"""Async engine/session setup and the database call recorder."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from m3ds.llm.tracing import LLMCallRecord
from m3ds.storage.models import Base, LLMCall


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=not url.startswith("sqlite"))


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


class DatabaseRecorder:
    """Persists every LLM call (tokens, cost, latency) for auditing and budgets."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]):
        self.sessionmaker = sessionmaker

    async def record(self, call: LLMCallRecord) -> None:
        u = call.usage
        async with self.sessionmaker() as session:
            session.add(
                LLMCall(
                    conversation_id=call.conversation_id,
                    task=call.task,
                    provider=call.provider,
                    model=call.model,
                    alias=call.alias,
                    ok=call.ok,
                    error=call.error,
                    stop_reason=call.stop_reason,
                    input_tokens=u.input_tokens,
                    output_tokens=u.output_tokens,
                    cache_read_tokens=u.cache_read_tokens,
                    cache_write_tokens=u.cache_write_tokens,
                    reasoning_tokens=u.reasoning_tokens,
                    cost_usd=call.cost_usd,
                    latency_ms=call.latency_ms,
                    request_id=call.request_id,
                    created_at=call.created_at,
                )
            )
            await session.commit()
