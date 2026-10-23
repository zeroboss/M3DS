"""HTTP API (REST + Server-Sent Events streaming)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from m3ds import __version__
from m3ds.chat.service import ChatService, ConversationNotFoundError
from m3ds.config import Settings, get_settings
from m3ds.llm.base import ProviderError
from m3ds.llm.factory import build_gateway
from m3ds.llm.gateway import LLMGateway
from m3ds.llm.registry import UnknownModelError
from m3ds.llm.tracing import setup_tracing
from m3ds.llm.types import LLMResponse, Message, StreamDone, TextDelta, Usage
from m3ds.storage.db import DatabaseRecorder, create_schema, make_engine, make_sessionmaker

# ------------------------------------------------------------------ schemas


class ConversationCreate(BaseModel):
    robot_id: str = "default"
    user_id: str = "default"
    title: str | None = None


class ConversationOut(BaseModel):
    id: str
    robot_id: str
    user_id: str
    title: str | None
    created_at: datetime


class SendMessage(BaseModel):
    text: str
    model: str | None = None
    stream: bool = False


class TurnOut(BaseModel):
    message: Message
    provider: str
    model: str
    alias: str | None
    stop_reason: str
    usage: Usage
    cost_usd: float | None
    latency_ms: float


class ModelOut(BaseModel):
    alias: str
    provider: str
    model_id: str
    available: bool
    context_window: int


def turn_out(resp: LLMResponse) -> TurnOut:
    return TurnOut(
        message=resp.message.model_copy(update={"native": None}),
        provider=resp.provider,
        model=resp.model,
        alias=resp.alias,
        stop_reason=resp.stop_reason.value,
        usage=resp.usage,
        cost_usd=resp.cost_usd,
        latency_ms=resp.latency_ms,
    )


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------------ app


def _chat_service(request: Request) -> ChatService:
    return request.app.state.chat


# Module level so FastAPI can resolve it from the postponed annotations.
ChatDep = Annotated[ChatService, Depends(_chat_service)]


def create_app(
    settings: Settings | None = None,
    gateway: LLMGateway | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_tracing()
        engine = make_engine(settings.database_url)
        if settings.auto_create_schema:
            await create_schema(engine)
        sessionmaker = make_sessionmaker(engine)
        gw = gateway or build_gateway(settings)
        gw.recorders.append(DatabaseRecorder(sessionmaker))
        app.state.gateway = gw
        app.state.chat = ChatService(
            gw, sessionmaker, settings.system_prompt, settings.history_max_messages
        )
        try:
            yield
        finally:
            await gw.aclose()
            await engine.dispose()

    app = FastAPI(title="M3DS Knowledge Core", version=__version__, lifespan=lifespan)

    async def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    auth = [Depends(require_api_key)]

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/models", dependencies=auth)
    async def list_models(request: Request) -> list[ModelOut]:
        gw: LLMGateway = request.app.state.gateway
        return [
            ModelOut(
                alias=m.alias,
                provider=m.provider,
                model_id=m.model_id,
                available=gw.is_available(m),
                context_window=m.context_window,
            )
            for m in gw.registry.models.values()
        ]

    @app.post("/v1/conversations", dependencies=auth, status_code=201)
    async def create_conversation(body: ConversationCreate, svc: ChatDep) -> ConversationOut:
        conv = await svc.create_conversation(body.robot_id, body.user_id, body.title)
        return ConversationOut.model_validate(conv, from_attributes=True)

    @app.get("/v1/conversations/{conversation_id}/messages", dependencies=auth)
    async def get_messages(conversation_id: str, svc: ChatDep) -> list[Message]:
        try:
            history = await svc.history(conversation_id)
        except ConversationNotFoundError as exc:
            raise HTTPException(404, "conversation not found") from exc
        return [m.model_copy(update={"native": None}) for m in history]

    @app.post(
        "/v1/conversations/{conversation_id}/messages", dependencies=auth, response_model=None
    )
    async def send_message(
        conversation_id: str, body: SendMessage, svc: ChatDep
    ) -> TurnOut | StreamingResponse:
        try:
            await svc.get_conversation(conversation_id)
            if body.model:
                svc.gateway.registry.resolve(body.model)
        except ConversationNotFoundError as exc:
            raise HTTPException(404, "conversation not found") from exc
        except UnknownModelError as exc:
            raise HTTPException(400, f"unknown model {body.model!r}") from exc

        if not body.stream:
            try:
                resp = await svc.send(conversation_id, body.text, body.model)
            except ProviderError as exc:
                raise HTTPException(502, str(exc)) from exc
            return turn_out(resp)

        async def events() -> AsyncIterator[str]:
            try:
                async for event in svc.send_stream(conversation_id, body.text, body.model):
                    if isinstance(event, TextDelta):
                        yield _sse("delta", {"text": event.text})
                    elif isinstance(event, StreamDone):
                        yield _sse("done", turn_out(event.response).model_dump(mode="json"))
            except ProviderError as exc:
                yield _sse("error", {"detail": str(exc)})

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
