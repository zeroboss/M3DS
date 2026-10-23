from __future__ import annotations

import json

import httpx
import pytest
from sqlalchemy import select

from m3ds.api.app import create_app
from m3ds.chat.service import ChatService, trim_history
from m3ds.config import Settings
from m3ds.llm.gateway import LLMGateway
from m3ds.llm.providers.fake import FakeProvider
from m3ds.llm.types import Message, StreamDone, TextPart, ToolResultPart
from m3ds.storage.models import LLMCall


async def test_conversation_history_persists_and_is_sent(
    gateway: LLMGateway, p1: FakeProvider, sessionmaker
):
    svc = ChatService(gateway, sessionmaker, "be nice")
    conv = await svc.create_conversation(robot_id="r1")
    await svc.send(conv.id, "first")
    await svc.send(conv.id, "second")
    history = await svc.history(conv.id)
    assert [m.role for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[3].text == "echo: second"
    last = p1.requests[-1]
    assert [m.text for m in last.messages] == ["first", "echo: first", "second"]
    assert last.system[0].text == "be nice" and last.system[0].cacheable
    assert last.metadata["conversation_id"] == conv.id


async def test_switching_model_mid_conversation(
    gateway: LLMGateway, p2: FakeProvider, sessionmaker
):
    svc = ChatService(gateway, sessionmaker, "sys")
    conv = await svc.create_conversation()
    await svc.send(conv.id, "on p1")
    resp = await svc.send(conv.id, "on p2", model="secondary")
    assert resp.provider == "p2"
    assert [m.text for m in p2.requests[0].messages] == ["on p1", "echo: on p1", "on p2"]


async def test_stream_persists_on_done(gateway: LLMGateway, sessionmaker):
    svc = ChatService(gateway, sessionmaker, "sys")
    conv = await svc.create_conversation()
    events = [e async for e in svc.send_stream(conv.id, "hey")]
    assert isinstance(events[-1], StreamDone)
    assert len(await svc.history(conv.id)) == 2


def test_trim_history_never_starts_with_orphan_turns():
    msgs = [
        Message.user("u1"),
        Message.assistant("a1"),
        Message(role="user", parts=[ToolResultPart(tool_call_id="x", name="t", content="r")]),
        Message.assistant("a2"),
        Message(role="user", parts=[TextPart(text="u3")]),
        Message.assistant("a3"),
    ]
    trimmed = trim_history(msgs, 4)
    assert trimmed[0].text == "u3"
    assert trim_history(msgs, 10) == msgs


@pytest.fixture
async def client(database_url: str, sessionmaker, gateway: LLMGateway):
    # Depends on `sessionmaker` so the schema is reset before the app starts.
    settings = Settings(database_url=database_url, api_key="secret")
    app = create_app(settings, gateway=gateway)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", headers={"X-API-Key": "secret"}
        ) as c:
            c.app = app  # type: ignore[attr-defined]
            yield c


async def test_api_requires_key(client: httpx.AsyncClient):
    r = await client.get("/v1/models", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
    assert (await client.get("/healthz")).status_code == 200


async def test_api_conversation_flow(client: httpx.AsyncClient):
    models = (await client.get("/v1/models")).json()
    assert {m["alias"]: m["available"] for m in models} == {"primary": True, "secondary": True}

    conv = (await client.post("/v1/conversations", json={"robot_id": "r1"})).json()
    r = await client.post(f"/v1/conversations/{conv['id']}/messages", json={"text": "hello"})
    assert r.status_code == 200
    body = r.json()
    assert body["message"]["parts"][0]["text"] == "echo: hello"
    assert body["alias"] == "primary" and body["cost_usd"] > 0

    history = (await client.get(f"/v1/conversations/{conv['id']}/messages")).json()
    assert [m["role"] for m in history] == ["user", "assistant"]

    sm = client.app.state.chat.sessionmaker  # type: ignore[attr-defined]
    async with sm() as s:
        calls = (await s.scalars(select(LLMCall))).all()
    assert len(calls) == 1 and calls[0].conversation_id == conv["id"] and calls[0].ok


async def test_api_streaming_sse(client: httpx.AsyncClient):
    conv = (await client.post("/v1/conversations", json={})).json()
    async with client.stream(
        "POST",
        f"/v1/conversations/{conv['id']}/messages",
        json={"text": "stream me", "stream": True},
    ) as r:
        raw = (await r.aread()).decode()
    events = [
        (
            block.split("\n")[0].removeprefix("event: "),
            json.loads(block.split("\n")[1].removeprefix("data: ")),
        )
        for block in raw.strip().split("\n\n")
    ]
    text = "".join(d["text"] for e, d in events if e == "delta")
    assert text == "echo: stream me"
    assert events[-1][0] == "done"


async def test_api_errors(client: httpx.AsyncClient):
    assert (
        await client.post("/v1/conversations/nope/messages", json={"text": "x"})
    ).status_code == 404
    conv = (await client.post("/v1/conversations", json={})).json()
    r = await client.post(
        f"/v1/conversations/{conv['id']}/messages", json={"text": "x", "model": "zzz"}
    )
    assert r.status_code == 400
