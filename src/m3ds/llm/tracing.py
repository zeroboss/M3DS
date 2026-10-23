"""Call tracing: OpenTelemetry spans plus pluggable call recorders (e.g. the database)."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Protocol

from opentelemetry import trace
from pydantic import BaseModel, Field

from m3ds.llm.types import Usage

log = logging.getLogger(__name__)
tracer = trace.get_tracer("m3ds.llm")


class LLMCallRecord(BaseModel):
    task: str
    provider: str
    model: str
    alias: str
    ok: bool
    error: str | None = None
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float | None = None
    latency_ms: float = 0.0
    request_id: str | None = None
    stop_reason: str | None = None
    conversation_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CallRecorder(Protocol):
    async def record(self, call: LLMCallRecord) -> None: ...


class LoggingRecorder:
    async def record(self, call: LLMCallRecord) -> None:
        log.info(
            "llm_call task=%s model=%s ok=%s in=%d out=%d cache_read=%d cost=%s latency_ms=%.0f",
            call.task,
            call.model,
            call.ok,
            call.usage.input_tokens,
            call.usage.output_tokens,
            call.usage.cache_read_tokens,
            call.cost_usd,
            call.latency_ms,
        )


def setup_tracing(service_name: str = "m3ds") -> None:
    """Export spans over OTLP when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set.

    Langfuse, Jaeger, Honeycomb, Grafana Tempo etc. all accept OTLP.
    """
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # pyright: ignore[reportMissingImports]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning("OTEL endpoint set but exporter missing; install m3ds[otlp]")
        return
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
