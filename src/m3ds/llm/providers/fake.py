"""Deterministic offline provider for development and tests."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable

from m3ds.llm.base import LLMProvider, ProviderError
from m3ds.llm.registry import ModelInfo
from m3ds.llm.types import (
    LLMRequest,
    LLMResponse,
    Message,
    StopReason,
    StreamDone,
    StreamEvent,
    TextDelta,
    Usage,
)


class FakeProvider(LLMProvider):
    """Returns scripted replies (or errors) in order, then echoes the last user message.

    ``requests`` records every request received so tests can assert on what the
    gateway sent.
    """

    def __init__(self, name: str = "fake", script: Iterable[str | Exception] = ()):
        self.name = name
        self.script: deque[str | Exception] = deque(script)
        self.requests: list[LLMRequest] = []

    def _reply(self, req: LLMRequest) -> str:
        self.requests.append(req)
        if self.script:
            item = self.script.popleft()
            if isinstance(item, Exception):
                raise item
            return item
        last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
        return f"echo: {last_user.text if last_user else ''}"

    def _response(self, text: str, req: LLMRequest, model: ModelInfo) -> LLMResponse:
        words = sum(len(m.text.split()) for m in req.messages)
        return LLMResponse(
            provider=self.name,
            model=model.model_id,
            message=Message.assistant(text),
            stop_reason=StopReason.END,
            raw_stop_reason="end",
            usage=Usage(input_tokens=words, output_tokens=len(text.split())),
        )

    async def generate(self, req: LLMRequest, model: ModelInfo) -> LLMResponse:
        return self._response(self._reply(req), req, model)

    async def stream(self, req: LLMRequest, model: ModelInfo) -> AsyncIterator[StreamEvent]:
        text = self._reply(req)
        for i, word in enumerate(text.split(" ")):
            yield TextDelta(text=word if i == 0 else " " + word)
        yield StreamDone(response=self._response(text, req, model))

    async def list_models(self) -> list[str]:
        return ["fake-echo"]


def retryable_error(provider: str = "fake") -> ProviderError:
    return ProviderError(provider, "simulated overload", retryable=True, status=529)
