"""OpenAI (GPT) adapter over the official ``openai`` SDK (Responses API, stateless)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import openai
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseTextDeltaEvent,
)

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
    TextDelta,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)

PROVIDER = "openai"


# ---------------------------------------------------------------- request mapping


def _user_content(parts: list[Part]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for p in parts:
        if isinstance(p, TextPart):
            content.append({"type": "input_text", "text": p.text})
        elif isinstance(p, ImagePart):
            url = p.url or f"data:{p.media_type};base64,{p.data_base64}"
            content.append({"type": "input_image", "image_url": url})
    return content


def to_openai_input(messages: list[Message]) -> list[dict[str, Any]]:
    """Map canonical messages to Responses API input items.

    Assistant turns produced by OpenAI are replayed from their native output items
    (including encrypted reasoning) so reasoning carries over between turns while
    ``store=False`` keeps the API stateless.
    """
    items: list[dict[str, Any]] = []
    for msg in messages:
        native = msg.native
        if msg.role == "assistant" and native is not None and native.provider == PROVIDER:
            items.extend(native.payload)
            continue
        if msg.role == "assistant":
            text = msg.text
            if text:
                items.append({"role": "assistant", "content": text})
            for call in msg.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    }
                )
            continue
        for p in msg.parts:
            if isinstance(p, ToolResultPart):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": p.tool_call_id,
                        "output": p.content,
                    }
                )
        content = _user_content(msg.parts)
        if content:
            items.append({"role": "user", "content": content})
    return items


def build_params(req: LLMRequest, model: ModelInfo) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model.model_id,
        "input": to_openai_input(req.messages),
        "max_output_tokens": effective_max_tokens(req, model),
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if req.system:
        params["instructions"] = "\n\n".join(b.text for b in req.system)
    if req.tools:
        params["tools"] = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "strict": False,
            }
            for t in req.tools
        ]
    if req.effort and model.supports.effort:
        params["reasoning"] = {"effort": req.effort}
    if req.temperature is not None and model.supports.temperature:
        params["temperature"] = req.temperature
    if req.response_schema is not None:
        params["text"] = {
            "format": {
                "type": "json_schema",
                "name": req.response_schema_name,
                "schema": strict_json_schema(req.response_schema),
                "strict": True,
            }
        }
    if cache_key := req.metadata.get("cache_key"):
        params["prompt_cache_key"] = cache_key
    return params


# --------------------------------------------------------------- response mapping


def from_openai_response(resp: Any, model: ModelInfo, latency_ms: float) -> LLMResponse:
    parts: list[Part] = []
    refused = False
    for item in resp.output or []:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    parts.append(TextPart(text=c.text))
                elif c.type == "refusal":
                    refused = True
                    parts.append(TextPart(text=c.refusal))
        elif item.type == "function_call":
            try:
                args = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": item.arguments}
            parts.append(ToolCallPart(id=item.call_id, name=item.name, arguments=args))

    incomplete = (
        getattr(resp.incomplete_details, "reason", None) if resp.incomplete_details else None
    )
    if refused or incomplete == "content_filter":
        stop = StopReason.REFUSAL
    elif incomplete == "max_output_tokens":
        stop = StopReason.MAX_TOKENS
    elif any(isinstance(p, ToolCallPart) for p in parts):
        stop = StopReason.TOOL_USE
    elif resp.status == "completed":
        stop = StopReason.END
    else:
        stop = StopReason.OTHER

    usage = Usage()
    if resp.usage is not None:
        u = resp.usage
        cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
        cache_write = getattr(u.input_tokens_details, "cache_write_tokens", 0) or 0
        usage = Usage(
            # input_tokens includes cache reads and writes; report them separately.
            input_tokens=(u.input_tokens or 0) - cached - cache_write,
            cache_read_tokens=cached,
            cache_write_tokens=cache_write,
            output_tokens=u.output_tokens or 0,
            reasoning_tokens=getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0,
        )
    native = [item.model_dump(mode="json", exclude_none=True) for item in resp.output or []]
    return LLMResponse(
        provider=PROVIDER,
        model=resp.model or model.model_id,
        message=Message(
            role="assistant",
            parts=parts,
            native=NativeContent(provider=PROVIDER, model=model.model_id, payload=native),
        ),
        stop_reason=stop,
        raw_stop_reason=incomplete or resp.status,
        usage=usage,
        request_id=getattr(resp, "_request_id", None) or resp.id,
        latency_ms=latency_ms,
    )


def _map_error(exc: Exception) -> ProviderError:
    if isinstance(exc, openai.APIConnectionError):  # includes timeouts
        return ProviderError(PROVIDER, str(exc), retryable=True)
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        retryable = status in (408, 409, 429) or status >= 500
        return ProviderError(PROVIDER, exc.message, retryable=retryable, status=status)
    return ProviderError(PROVIDER, repr(exc), retryable=False)


class OpenAIProvider(LLMProvider):
    name = PROVIDER

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
    ):
        self.client = openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
        )

    async def generate(self, req: LLMRequest, model: ModelInfo) -> LLMResponse:
        params = build_params(req, model)
        start = time.perf_counter()
        try:
            resp = await self.client.responses.create(**params)
        except openai.OpenAIError as exc:
            raise _map_error(exc) from exc
        return from_openai_response(resp, model, (time.perf_counter() - start) * 1000)

    async def stream(self, req: LLMRequest, model: ModelInfo) -> AsyncIterator[StreamEvent]:
        params = build_params(req, model)
        start = time.perf_counter()
        final: Any = None
        try:
            events = await self.client.responses.create(**params, stream=True)
            async for event in events:
                if isinstance(event, ResponseTextDeltaEvent):
                    yield TextDelta(text=event.delta)
                elif isinstance(event, ResponseCompletedEvent | ResponseIncompleteEvent):
                    final = event.response
                elif isinstance(event, ResponseFailedEvent):
                    raise ProviderError(PROVIDER, str(event.response.error), retryable=False)
                elif isinstance(event, ResponseErrorEvent):
                    raise ProviderError(PROVIDER, event.message, retryable=False)
        except openai.OpenAIError as exc:
            raise _map_error(exc) from exc
        if final is None:
            raise ProviderError(PROVIDER, "stream ended without a final response", retryable=True)
        yield StreamDone(
            response=from_openai_response(final, model, (time.perf_counter() - start) * 1000)
        )

    async def list_models(self) -> list[str]:
        try:
            return sorted([m.id async for m in self.client.models.list()])
        except openai.OpenAIError as exc:
            raise _map_error(exc) from exc

    async def aclose(self) -> None:
        await self.client.close()
