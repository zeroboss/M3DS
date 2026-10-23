from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest

from m3ds.llm.gateway import LLMGateway
from m3ds.llm.providers.fake import FakeProvider
from m3ds.llm.registry import ModelRegistry
from m3ds.storage.db import create_schema, make_engine, make_sessionmaker
from m3ds.storage.models import Base

# Set to run storage/API tests against PostgreSQL (as CI does); defaults to SQLite.
TEST_DATABASE_URL = os.environ.get("M3DS_TEST_DATABASE_URL")

_MODEL = {
    "context_window": 1000,
    "max_output_tokens": 512,
    "default_max_output_tokens": 256,
    "price": {"input": 1.0, "output": 2.0},
}

REGISTRY_DATA = {
    "routes": {"chat": ["primary", "secondary"], "extract": ["primary"]},
    "models": {
        "primary": {"provider": "p1", "model_id": "p1-model", **_MODEL},
        "secondary": {"provider": "p2", "model_id": "p2-model", **_MODEL},
    },
}


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.from_dict(REGISTRY_DATA)


@pytest.fixture
def p1() -> FakeProvider:
    return FakeProvider("p1")


@pytest.fixture
def p2() -> FakeProvider:
    return FakeProvider("p2")


@pytest.fixture
def gateway(registry: ModelRegistry, p1: FakeProvider, p2: FakeProvider) -> LLMGateway:
    return LLMGateway(registry, {"p1": p1, "p2": p2})


@pytest.fixture
def database_url(tmp_path) -> str:
    return TEST_DATABASE_URL or f"sqlite+aiosqlite:///{tmp_path}/test.db"


@pytest.fixture
async def sessionmaker(database_url: str) -> AsyncIterator:
    engine = make_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await create_schema(engine)
    yield make_sessionmaker(engine)
    await engine.dispose()
