"""Anthropic (Claude) adapter over the official ``anthropic`` SDK (Messages API)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from m3ds.llm.base import LLMProvider, ProviderError, effective_max_tokens, strict_json_schema
from m3ds.llm.registry import ModelInfo
from m3ds.llm.types import (
    ImagePart,
    LLMRequest,
    LLMResponse,
    Message,
    NativeContent,
    Part,
    StopReason,
    StreamDone,
    StreamEvent,
    SystemBlock,
    TextDelta,
    TextPart,
    ToolCallPart,
    Usage,
)

PROVIDER = "anthropic"
SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_STOP_REASONS = {
    "end_turn": StopReason.END,
    "stop_sequence": StopReason.END,
    "max_tokens": StopReason.MAX_TOKENS,
    "model_context_window_exceeded": StopReason.MAX_TOKENS,
    "tool_use": StopReason.TOOL_USE,
    "refusal": StopReason.REFUSAL,
}


# ---------------------------------------------------------------- request mapping


def _part_to_block(part: Part) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ImagePart):
        if part.url:
            return {"type": "image", "source": {"type": "url", "url": part.url}}
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": part.media_type, "data": part.data_base64},
        }
    if isinstance(part, ToolCallPart):
        return {"type": "tool_use", "id": part.id, "name": part.name, "input": part.arguments}
    return {
        "type": "tool_result",
        "tool_use_id": part.tool_call_id,
        "content": part.content,
        "is_error": part.is_error,
    }


def sanitize_fallback_echo(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare content containing a server-side ``fallback`` block for replay.

    Before the last ``fallback`` boundary only text blocks and paired server-tool
    blocks may be echoed back; model-internal blocks from the declined attempt
    (thinking, tool_use, unknown types) must be dropped.
    """
    boundary = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=-1)
    if boundary < 0:
        return blocks
    result_ids = {
        b.get("tool_use_id") for b in blocks if b.get("type", "").endswith("_tool_result")
    }
    kept: list[dict[str, Any]] = []
    for i, block in enumerate(blocks):
        kind = block.get("type", "")
        if (
            i >= boundary
            or kind == "text"
            or kind.endswith("_tool_result")
            or (kind == "server_tool_use" and block.get("id") in result_ids)
        ):
            kept.append(block)
    return kept


def to_anthropic_messages(messages: list[Message], model: ModelInfo) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        native = msg.native
        if msg.role == "assistant" and native is not None and native.provider == PROVIDER:
            # Replay unchanged (thinking blocks included); the API drops blocks the
            # target model cannot read, and stripping them can cause 400s.
            content = sanitize_fallback_echo(list(native.payload))
        else:
            content = [_part_to_block(p) for p in msg.parts]
        out.append({"role": msg.role, "content": content})
    return out


def to_anthropic_system(blocks: list[SystemBlock]) -> list[dict[str, Any]]:
    system: list[dict[str, Any]] = [{"type": "text", "text": b.text} for b in blocks]
    last_cacheable = max((i for i, b in enumerate(blocks) if b.cacheable), default=-1)
    if last_cacheable >= 0:
        system[last_cacheable]["cache_control"] = {"type": "ephemeral"}
    return system


def build_params(req: LLMRequest, model: ModelInfo, *, max_tokens: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model.model_id,
        "max_tokens": max_tokens,
        "messages": to_anthropic_messages(req.messages, model),
    }
    if req.system:
        params["system"] = to_anthropic_system(req.system)
    if model.supports.prompt_caching:
        # Automatic caching of the growing conversation prefix.
        params["cache_control"] = {"type": "ephemeral"}
    if req.tools:
        params["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in req.tools
        ]
    output_config: dict[str, Any] = {}
    if req.effort and model.supports.effort:
        output_config["effort"] = req.effort
    if req.response_schema is not None:
        output_config["format"] = {
            "type": "json_schema",
            "schema": strict_json_schema(req.response_schema),
        }
    if output_config:
        params["output_config"] = output_config
    # Sampling parameters (temperature/top_p/top_k) are rejected by the current
    # Messages API version, so req.temperature is intentionally not forwarded.
    if model.options.get("server_fallback"):
        params["betas"] = [SERVER_FALLBACK_BETA]
        params["fallbacks"] = "default"
    return params


# --------------------------------------------------------------- response mapping


def from_anthropic_message(msg: Any, model: ModelInfo, latency_ms: float) -> LLMResponse:
    parts: list[Part] = []
    for block in msg.content:
        if block.type == "text":
            parts.append(TextPart(text=block.text))
        elif block.type == "tool_use":
            parts.append(ToolCallPart(id=block.id, name=block.name, arguments=dict(block.input)))
    native = [b.model_dump(mode="json", exclude_none=True) for b in msg.content]
    u = msg.usage
    usage = Usage(
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", None) or 0,
    )
    stop = msg.stop_reason or ""
    return LLMResponse(
        provider=PROVIDER,
        # With server-side fallback, msg.model names the model that actually answered.
        model=msg.model or model.model_id,
        message=Message(
            role="assistant",
            parts=parts,
            native=NativeContent(
                provider=PROVIDER, model=msg.model or model.model_id, payload=native
            ),
        ),
        stop_reason=_STOP_REASONS.get(stop, StopReason.OTHER),
        raw_stop_reason=stop,
        usage=usage,
        request_id=getattr(msg, "_request_id", None),
        latency_ms=latency_ms,
    )


def _map_error(exc: Exception) -> ProviderError:
    if isinstance(exc, anthropic.APIConnectionError):  # includes timeouts
        return ProviderError(PROVIDER, str(exc), retryable=True)
    if isinstance(exc, anthropic.APIStatusError):
        status = exc.status_code
        retryable = status in (408, 409, 429) or status >= 500
        return ProviderError(PROVIDER, exc.message, retryable=retryable, status=status)
    return ProviderError(PROVIDER, repr(exc), retryable=False)


class AnthropicProvider(LLMProvider):
    name = PROVIDER

    def __init__(self, api_key: str | None = None, *, timeout: float = 600.0, max_retries: int = 2):
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    def _endpoint(self, params: dict[str, Any]) -> Any:
        return self.client.beta.messages if "betas" in params else self.client.messages

    async def generate(self, req: LLMRequest, model: ModelInfo) -> LLMResponse:
        params = build_params(req, model, max_tokens=effective_max_tokens(req, model))
        start = time.perf_counter()
        try:
            msg = await self._endpoint(params).create(**params)
        except anthropic.AnthropicError as exc:
            raise _map_error(exc) from exc
        return from_anthropic_message(msg, model, (time.perf_counter() - start) * 1000)

    async def stream(self, req: LLMRequest, model: ModelInfo) -> AsyncIterator[StreamEvent]:
        params = build_params(req, model, max_tokens=effective_max_tokens(req, model))
        start = time.perf_counter()
        try:
            async with self._endpoint(params).stream(**params) as stream:
                async for text in stream.text_stream:
                    yield TextDelta(text=text)
                msg = await stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise _map_error(exc) from exc
        yield StreamDone(
            response=from_anthropic_message(msg, model, (time.perf_counter() - start) * 1000)
        )

    async def list_models(self) -> list[str]:
        try:
            return [m.id async for m in self.client.models.list()]
        except anthropic.AnthropicError as exc:
            raise _map_error(exc) from exc

    async def aclose(self) -> None:
        await self.client.close()
