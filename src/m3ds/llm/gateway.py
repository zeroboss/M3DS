"""LLM gateway: model routing, cross-provider fallback, structured output and tracing."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence

from opentelemetry.trace import Span
from pydantic import BaseModel, ValidationError

from m3ds.llm.base import LLMProvider, ProviderError, ProviderNotConfiguredError
from m3ds.llm.registry import ModelInfo, ModelRegistry
from m3ds.llm.tracing import CallRecorder, LLMCallRecord, tracer
from m3ds.llm.types import LLMRequest, LLMResponse, Message, StopReason, StreamDone, StreamEvent

log = logging.getLogger(__name__)


class StructuredOutputError(Exception):
    pass


class LLMGateway:
    def __init__(
        self,
        registry: ModelRegistry,
        providers: dict[str, LLMProvider],
        recorders: Sequence[CallRecorder] = (),
    ):
        self.registry = registry
        self.providers = providers
        self.recorders = list(recorders)

    # ------------------------------------------------------------------ routing

    def is_available(self, info: ModelInfo) -> bool:
        return info.provider in self.providers

    def candidates(self, model: str | None, task: str) -> list[ModelInfo]:
        """Models to try, in order.

        An explicit ``model`` is used alone (no silent switch to another model the
        caller did not ask for); otherwise the task route is filtered to providers
        that are configured.
        """
        if model:
            return [self.registry.resolve(model)]
        chain = [m for m in self.registry.route(task) if self.is_available(m)]
        if not chain:
            raise ProviderNotConfiguredError(f"any provider for task {task!r}")
        return chain

    def _prepare(self, req: LLMRequest, info: ModelInfo) -> LLMRequest:
        updates: dict[str, object] = {"model": info.alias}
        if not info.supports.temperature:
            updates["temperature"] = None
        if not info.supports.effort:
            updates["effort"] = None
        return req.model_copy(update=updates)

    def _provider(self, info: ModelInfo) -> LLMProvider:
        provider = self.providers.get(info.provider)
        if provider is None:
            raise ProviderNotConfiguredError(info.provider)
        return provider

    # ------------------------------------------------------------------ tracing

    async def _record(
        self,
        task: str,
        info: ModelInfo,
        req: LLMRequest,
        *,
        resp: LLMResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        call = LLMCallRecord(
            task=task,
            provider=info.provider,
            model=resp.model if resp else info.model_id,
            alias=info.alias,
            ok=error is None,
            error=str(error) if error else None,
            conversation_id=req.metadata.get("conversation_id"),
        )
        if resp is not None:
            call.usage = resp.usage
            call.cost_usd = resp.cost_usd
            call.latency_ms = resp.latency_ms
            call.request_id = resp.request_id
            call.stop_reason = resp.stop_reason.value
        for recorder in self.recorders:
            try:
                await recorder.record(call)
            except Exception:  # tracing must never break a turn
                log.exception("call recorder failed")

    @staticmethod
    def _annotate(span: Span, info: ModelInfo, resp: LLMResponse) -> None:
        span.set_attribute("gen_ai.response.model", resp.model)
        span.set_attribute("gen_ai.usage.input_tokens", resp.usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", resp.usage.output_tokens)
        span.set_attribute("m3ds.cache_read_tokens", resp.usage.cache_read_tokens)
        span.set_attribute("m3ds.stop_reason", resp.stop_reason.value)
        if resp.cost_usd is not None:
            span.set_attribute("m3ds.cost_usd", resp.cost_usd)

    # --------------------------------------------------------------- generation

    async def generate(self, req: LLMRequest, task: str = "chat") -> LLMResponse:
        last_error: ProviderError | None = None
        chain = self.candidates(req.model, task)
        for i, info in enumerate(chain):
            prepared = self._prepare(req, info)
            with tracer.start_as_current_span(f"llm.generate {info.alias}") as span:
                span.set_attribute("gen_ai.system", info.provider)
                span.set_attribute("gen_ai.request.model", info.model_id)
                span.set_attribute("m3ds.task", task)
                try:
                    resp = await self._provider(info).generate(prepared, info)
                except ProviderError as exc:
                    span.record_exception(exc)
                    await self._record(task, info, prepared, error=exc)
                    if not exc.retryable:
                        raise
                    if i + 1 < len(chain):
                        log.warning(
                            "%s failed (%s); trying %s", info.alias, exc, chain[i + 1].alias
                        )
                    last_error = exc
                    continue
                resp.alias = info.alias
                resp.cost_usd = info.cost(resp.usage)
                self._annotate(span, info, resp)
            await self._record(task, info, prepared, resp=resp)
            return resp
        assert last_error is not None
        raise last_error

    async def stream(self, req: LLMRequest, task: str = "chat") -> AsyncIterator[StreamEvent]:
        """Stream a response. Falls back to the next model only if a model fails
        before emitting any text, so the client never sees mixed output."""
        last_error: ProviderError | None = None
        chain = self.candidates(req.model, task)
        for i, info in enumerate(chain):
            prepared = self._prepare(req, info)
            emitted = False
            # Not start_as_current_span: an async generator must not hold an
            # attached OTel context across yields.
            span = tracer.start_span(f"llm.stream {info.alias}")
            span.set_attribute("gen_ai.system", info.provider)
            span.set_attribute("gen_ai.request.model", info.model_id)
            span.set_attribute("m3ds.task", task)
            try:
                async for event in self._provider(info).stream(prepared, info):
                    if isinstance(event, StreamDone):
                        event.response.alias = info.alias
                        event.response.cost_usd = info.cost(event.response.usage)
                        self._annotate(span, info, event.response)
                        await self._record(task, info, prepared, resp=event.response)
                    else:
                        emitted = True
                    yield event
                return
            except ProviderError as exc:
                span.record_exception(exc)
                await self._record(task, info, prepared, error=exc)
                if emitted or not exc.retryable:
                    raise
                if i + 1 < len(chain):
                    log.warning("%s failed (%s); trying %s", info.alias, exc, chain[i + 1].alias)
                last_error = exc
            finally:
                span.end()
        assert last_error is not None
        raise last_error

    async def structured[T: BaseModel](
        self, req: LLMRequest, schema: type[T], task: str = "extract", max_attempts: int = 2
    ) -> T:
        """Generate a response validated against ``schema``.

        Uses each provider's native JSON-schema mode, validates with Pydantic, and on
        failure retries once with the validation error fed back to the model.
        """
        req = req.model_copy(
            update={
                "response_schema": schema.model_json_schema(),
                "response_schema_name": schema.__name__,
            }
        )
        messages = list(req.messages)
        error: Exception | None = None
        for _ in range(max_attempts):
            resp = await self.generate(req.model_copy(update={"messages": messages}), task=task)
            if resp.stop_reason == StopReason.REFUSAL:
                raise StructuredOutputError(f"model refused: {resp.text[:200]}")
            try:
                return schema.model_validate_json(_strip_fences(resp.text))
            except (ValidationError, json.JSONDecodeError) as exc:
                error = exc
                messages = [
                    *messages,
                    Message.assistant(resp.text),
                    Message.user(
                        "That output did not match the required JSON schema:\n"
                        f"{exc}\nReply again with only valid JSON."
                    ),
                ]
        raise StructuredOutputError(str(error))

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()
