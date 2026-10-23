"""Model registry: maps aliases to provider/model ids, capabilities and prices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from m3ds.llm.types import Usage


class Capabilities(BaseModel):
    tools: bool = True
    vision: bool = False
    structured_output: bool = True
    effort: bool = False
    prompt_caching: bool = False
    temperature: bool = True


class Price(BaseModel):
    """USD per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float | None = None
    cache_write: float | None = None


class ModelInfo(BaseModel):
    alias: str
    provider: str
    model_id: str
    context_window: int = 128_000
    max_output_tokens: int = 8_192
    default_max_output_tokens: int = 4_096
    supports: Capabilities = Field(default_factory=Capabilities)
    price: Price | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    def cost(self, usage: Usage) -> float | None:
        if self.price is None:
            return None
        p = self.price
        cache_read = p.cache_read if p.cache_read is not None else p.input
        cache_write = p.cache_write if p.cache_write is not None else p.input
        return (
            usage.input_tokens * p.input
            + usage.cache_read_tokens * cache_read
            + usage.cache_write_tokens * cache_write
            + usage.output_tokens * p.output
        ) / 1_000_000


class UnknownModelError(KeyError):
    pass


class ModelRegistry:
    def __init__(self, models: dict[str, ModelInfo], routes: dict[str, list[str]]):
        self.models = models
        self.routes = routes

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelRegistry:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelRegistry:
        models = {
            alias: ModelInfo(alias=alias, **spec)
            for alias, spec in (data.get("models") or {}).items()
        }
        routes = {task: list(chain) for task, chain in (data.get("routes") or {}).items()}
        for task, chain in routes.items():
            missing = [a for a in chain if a not in models]
            if missing:
                raise ValueError(f"route {task!r} references unknown models: {missing}")
        return cls(models, routes)

    def resolve(self, ref: str) -> ModelInfo:
        """Resolve an alias, or an ad-hoc ``provider:model-id`` reference."""
        if ref in self.models:
            return self.models[ref]
        if ":" in ref:
            provider, model_id = ref.split(":", 1)
            for info in self.models.values():
                if info.provider == provider and info.model_id == model_id:
                    return info
            return ModelInfo(alias=ref, provider=provider, model_id=model_id)
        raise UnknownModelError(ref)

    def route(self, task: str) -> list[ModelInfo]:
        if task not in self.routes:
            raise UnknownModelError(f"no route for task {task!r}")
        return [self.models[a] for a in self.routes[task]]
