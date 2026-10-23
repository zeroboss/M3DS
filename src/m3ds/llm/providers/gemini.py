"""Google Gemini adapter over the official ``google-genai`` SDK."""

from __future__ import annotations

import base64
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

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

PROVIDER = "gemini"
# Prefix for tool-call ids we synthesise when Gemini returns calls without ids.
_SYNTHETIC_ID = "gemini-call-"
# Serialised by the SDK (URL-safe base64) as "skip_thought_signature_validator".
_SKIP_SIGNATURE = base64.urlsafe_b64decode("skip_thought_signature_validator")

_REFUSAL_REASONS = {
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
}


# ---------------------------------------------------------------- request mapping


def _part_to_gemini(part: Part) -> types.Part:
    if isinstance(part, TextPart):
        return types.Part(text=part.text)
    if isinstance(part, ImagePart):
        if part.url:
            return types.Part(
                file_data=types.FileData(file_uri=part.url, mime_type=part.media_type)
            )
        return types.Part(
            inline_data=types.Blob(
                data=base64.b64decode(part.data_base64 or ""), mime_type=part.media_type
            )
        )
    if isinstance(part, ToolCallPart):
        call_id = None if part.id.startswith(_SYNTHETIC_ID) else part.id
        return types.Part(
            function_call=types.FunctionCall(id=call_id, name=part.name, args=part.arguments)
        )
    assert isinstance(part, ToolResultPart)
    call_id = None if part.tool_call_id.startswith(_SYNTHETIC_ID) else part.tool_call_id
    key = "error" if part.is_error else "result"
    return types.Part(
        function_response=types.FunctionResponse(
            id=call_id, name=part.name, response={key: part.content}
        )
    )


def to_gemini_contents(messages: list[Message]) -> list[types.Content]:
    contents: list[types.Content] = []
    for msg in messages:
        native = msg.native
        if msg.role == "assistant" and native is not None and native.provider == PROVIDER:
            # Replay with thought signatures intact.
            contents.append(types.Content.model_validate(native.payload))
            continue
        role = "model" if msg.role == "assistant" else "user"
        gparts = [_part_to_gemini(p) for p in msg.parts]
        if role == "model":
            # Function calls produced by another provider have no thought signature,
            # which Gemini 3 rejects; the documented dummy value skips validation.
            # Only the first call of a (parallel) batch carries a signature.
            first_call = next((gp for gp in gparts if gp.function_call is not None), None)
            if first_call is not None:
                first_call.thought_signature = _SKIP_SIGNATURE
        contents.append(types.Content(role=role, parts=gparts))
    return contents


def build_config(req: LLMRequest, model: ModelInfo) -> types.GenerateContentConfig:
    cfg: dict[str, Any] = {"max_output_tokens": effective_max_tokens(req, model)}
    if req.system:
        cfg["system_instruction"] = "\n\n".join(b.text for b in req.system)
    if req.tools:
        cfg["tools"] = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=t.name, description=t.description, parameters_json_schema=t.parameters
                    )
                    for t in req.tools
                ]
            )
        ]
        cfg["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    if req.temperature is not None and model.supports.temperature:
        cfg["temperature"] = req.temperature
    if req.effort and model.supports.effort:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=req.effort.upper())  # type: ignore[arg-type]
    if req.response_schema is not None:
        cfg["response_mime_type"] = "application/json"
        cfg["response_json_schema"] = strict_json_schema(req.response_schema)
    return types.GenerateContentConfig(**cfg)


# --------------------------------------------------------------- response mapping


def _parts_from_gemini(gparts: list[types.Part]) -> list[Part]:
    parts: list[Part] = []
    for gp in gparts:
        if gp.function_call is not None:
            fc = gp.function_call
            parts.append(
                ToolCallPart(
                    id=fc.id or f"{_SYNTHETIC_ID}{uuid.uuid4().hex[:12]}",
                    name=fc.name or "",
                    arguments=dict(fc.args or {}),
                )
            )
        elif gp.text and not gp.thought:
            if parts and isinstance(parts[-1], TextPart):
                parts[-1] = TextPart(text=parts[-1].text + gp.text)
            else:
                parts.append(TextPart(text=gp.text))
    return parts


def _usage(meta: types.GenerateContentResponseUsageMetadata | None) -> Usage:
    if meta is None:
        return Usage()
    cached = meta.cached_content_token_count or 0
    thoughts = meta.thoughts_token_count or 0
    return Usage(
        input_tokens=(meta.prompt_token_count or 0) - cached,
        cache_read_tokens=cached,
        # Thinking tokens are billed as output, matching how the other providers count.
        output_tokens=(meta.candidates_token_count or 0) + thoughts,
        reasoning_tokens=thoughts,
    )


def build_response(
    model: ModelInfo,
    gparts: list[types.Part],
    finish_reason: str | None,
    blocked: bool,
    usage: Usage,
    latency_ms: float,
    response_id: str | None = None,
) -> LLMResponse:
    parts = _parts_from_gemini(gparts)
    if blocked or finish_reason in _REFUSAL_REASONS:
        stop = StopReason.REFUSAL
    elif finish_reason == "MAX_TOKENS":
        stop = StopReason.MAX_TOKENS
    elif any(isinstance(p, ToolCallPart) for p in parts):
        stop = StopReason.TOOL_USE
    elif finish_reason == "STOP":
        stop = StopReason.END
    else:
        stop = StopReason.OTHER
    native = types.Content(role="model", parts=gparts).model_dump(mode="json", exclude_none=True)
    return LLMResponse(
        provider=PROVIDER,
        model=model.model_id,
        message=Message(
            role="assistant",
            parts=parts,
            native=NativeContent(provider=PROVIDER, model=model.model_id, payload=native),
        ),
        stop_reason=stop,
        raw_stop_reason=finish_reason or ("BLOCKED" if blocked else None),
        usage=usage,
        request_id=response_id,
        latency_ms=latency_ms,
    )


def _finish(resp: types.GenerateContentResponse) -> tuple[list[types.Part], str | None, bool]:
    blocked = bool(resp.prompt_feedback and resp.prompt_feedback.block_reason)
    if not resp.candidates:
        return [], None, blocked
    cand = resp.candidates[0]
    gparts = list(cand.content.parts or []) if cand.content else []
    reason = cand.finish_reason.name if cand.finish_reason else None
    return gparts, reason, blocked


def _map_error(exc: Exception) -> ProviderError:
    if isinstance(exc, genai_errors.APIError):
        status = exc.code
        retryable = status in (408, 409, 429) or status >= 500
        return ProviderError(PROVIDER, str(exc.message or exc), retryable=retryable, status=status)
    return ProviderError(PROVIDER, repr(exc), retryable=True)


class GeminiProvider(LLMProvider):
    name = PROVIDER

    def __init__(self, api_key: str | None = None):
        self.client = genai.Client(api_key=api_key)

    async def generate(self, req: LLMRequest, model: ModelInfo) -> LLMResponse:
        start = time.perf_counter()
        try:
            resp = await self.client.aio.models.generate_content(
                model=model.model_id,
                contents=to_gemini_contents(req.messages),  # type: ignore[arg-type]
                config=build_config(req, model),
            )
        except (genai_errors.APIError, OSError) as exc:
            raise _map_error(exc) from exc
        gparts, reason, blocked = _finish(resp)
        return build_response(
            model,
            gparts,
            reason,
            blocked,
            _usage(resp.usage_metadata),
            (time.perf_counter() - start) * 1000,
            resp.response_id,
        )

    async def stream(self, req: LLMRequest, model: ModelInfo) -> AsyncIterator[StreamEvent]:
        start = time.perf_counter()
        gparts: list[types.Part] = []
        reason: str | None = None
        blocked = False
        usage = Usage()
        response_id: str | None = None
        try:
            chunks = await self.client.aio.models.generate_content_stream(
                model=model.model_id,
                contents=to_gemini_contents(req.messages),  # type: ignore[arg-type]
                config=build_config(req, model),
            )
            async for chunk in chunks:
                cparts, creason, cblocked = _finish(chunk)
                for gp in cparts:
                    if gp.text and not gp.thought:
                        yield TextDelta(text=gp.text)
                gparts.extend(cparts)
                reason = creason or reason
                blocked = blocked or cblocked
                if chunk.usage_metadata is not None:
                    usage = _usage(chunk.usage_metadata)
                response_id = chunk.response_id or response_id
        except (genai_errors.APIError, OSError) as exc:
            raise _map_error(exc) from exc
        yield StreamDone(
            response=build_response(
                model,
                gparts,
                reason,
                blocked,
                usage,
                (time.perf_counter() - start) * 1000,
                response_id,
            )
        )

    async def list_models(self) -> list[str]:
        try:
            pager = await self.client.aio.models.list()
            return sorted([m.name or "" async for m in pager])
        except genai_errors.APIError as exc:
            raise _map_error(exc) from exc

    async def aclose(self) -> None:
        await self.client.aio.aclose()
