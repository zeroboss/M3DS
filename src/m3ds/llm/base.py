"""Provider interface and shared errors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from m3ds.llm.registry import ModelInfo
from m3ds.llm.types import LLMRequest, LLMResponse, StreamEvent


class ProviderError(Exception):
    """A provider call failed.

    ``retryable`` errors (rate limits, overload, timeouts, 5xx, network) let the
    gateway fall back to the next model in the route; others are surfaced directly.
    """

    def __init__(self, provider: str, message: str, *, retryable: bool, status: int | None = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable
        self.status = status


class ProviderNotConfiguredError(ProviderError):
    def __init__(self, provider: str):
        super().__init__(provider, "provider is not configured (missing API key?)", retryable=True)


class LLMProvider(ABC):
    name: str

    @abstractmethod
    async def generate(self, req: LLMRequest, model: ModelInfo) -> LLMResponse: ...

    @abstractmethod
    def stream(self, req: LLMRequest, model: ModelInfo) -> AsyncIterator[StreamEvent]:
        """Yield ``TextDelta`` events, then exactly one ``StreamDone``."""
        ...

    async def list_models(self) -> list[str]:
        return []

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        pass


def effective_max_tokens(req: LLMRequest, model: ModelInfo) -> int:
    return min(req.max_output_tokens or model.default_max_output_tokens, model.max_output_tokens)


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make a JSON schema acceptable to strict structured-output modes.

    OpenAI and Anthropic strict modes require every object to declare
    ``additionalProperties: false`` and list all properties as required. Pydantic's
    ``$defs``/``$ref`` layout is kept; ``title`` noise is dropped.
    """

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(v) for v in node]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "title" and isinstance(value, str):
                continue
            if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
                # Keys here are property/definition names, not schema keywords.
                out[key] = {name: walk(sub) for name, sub in value.items()}
            else:
                out[key] = walk(value)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        return out

    return walk(schema)
