from __future__ import annotations

import pytest
from pydantic import BaseModel

from m3ds.llm.base import ProviderError
from m3ds.llm.gateway import LLMGateway, StructuredOutputError
from m3ds.llm.providers.fake import FakeProvider, retryable_error
from m3ds.llm.tracing import LLMCallRecord
from m3ds.llm.types import LLMRequest, Message, StreamDone, TextDelta


def req(text: str = "hi", **kw) -> LLMRequest:
    return LLMRequest(messages=[Message.user(text)], **kw)


async def test_routes_to_first_available_model(gateway: LLMGateway, p1: FakeProvider):
    resp = await gateway.generate(req("hello"))
    assert resp.provider == "p1"
    assert resp.alias == "primary"
    assert resp.text == "echo: hello"
    assert p1.requests[0].model == "primary"


async def test_cost_is_computed_from_registry_price(gateway: LLMGateway, p1: FakeProvider):
    p1.script.append("one two three")
    resp = await gateway.generate(req("a b"))
    # 2 input tokens * $1/M + 3 output tokens * $2/M
    assert resp.cost_usd == pytest.approx((2 * 1.0 + 3 * 2.0) / 1_000_000)


async def test_falls_back_on_retryable_error(
    gateway: LLMGateway, p1: FakeProvider, p2: FakeProvider
):
    p1.script.append(retryable_error("p1"))
    resp = await gateway.generate(req())
    assert resp.provider == "p2"
    assert len(p1.requests) == 1 and len(p2.requests) == 1


async def test_non_retryable_error_is_raised(
    gateway: LLMGateway, p1: FakeProvider, p2: FakeProvider
):
    p1.script.append(ProviderError("p1", "bad request", retryable=False, status=400))
    with pytest.raises(ProviderError, match="bad request"):
        await gateway.generate(req())
    assert p2.requests == []


async def test_explicit_model_does_not_fall_back(
    gateway: LLMGateway, p1: FakeProvider, p2: FakeProvider
):
    p1.script.append(retryable_error("p1"))
    with pytest.raises(ProviderError):
        await gateway.generate(req(model="primary"))
    assert p2.requests == []


async def test_unconfigured_providers_are_skipped(registry):
    p2 = FakeProvider("p2")
    gw = LLMGateway(registry, {"p2": p2})
    resp = await gw.generate(req())
    assert resp.provider == "p2"


async def test_unsupported_params_are_dropped(gateway: LLMGateway, p1: FakeProvider):
    await gateway.generate(req(temperature=0.2, effort="high"))
    # The default Capabilities support temperature but not effort.
    assert p1.requests[0].temperature == 0.2
    assert p1.requests[0].effort is None


async def test_recorders_receive_success_and_failure(gateway: LLMGateway, p1: FakeProvider):
    calls: list[LLMCallRecord] = []

    class Rec:
        async def record(self, call: LLMCallRecord) -> None:
            calls.append(call)

    gateway.recorders.append(Rec())
    p1.script.append(retryable_error("p1"))
    await gateway.generate(req(metadata={"conversation_id": "c1"}))
    assert [(c.alias, c.ok) for c in calls] == [("primary", False), ("secondary", True)]
    assert calls[1].conversation_id == "c1"


async def test_stream_yields_deltas_then_done(gateway: LLMGateway):
    events = [e async for e in gateway.stream(req("hello there"))]
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "echo: hello there"
    assert isinstance(events[-1], StreamDone)
    assert events[-1].response.alias == "primary"


async def test_stream_falls_back_before_first_token(gateway: LLMGateway, p1: FakeProvider):
    p1.script.append(retryable_error("p1"))
    events = [e async for e in gateway.stream(req())]
    done = events[-1]
    assert isinstance(done, StreamDone) and done.response.provider == "p2"


class Person(BaseModel):
    name: str
    age: int


async def test_structured_output_validates(gateway: LLMGateway, p1: FakeProvider):
    p1.script.append('```json\n{"name": "Ada", "age": 36}\n```')
    person = await gateway.structured(req("extract"), Person)
    assert person == Person(name="Ada", age=36)
    sent = p1.requests[0]
    assert sent.response_schema is not None
    assert sent.response_schema_name == "Person"


async def test_structured_output_retries_with_error_feedback(gateway: LLMGateway, p1: FakeProvider):
    p1.script.extend(['{"name": "Ada"}', '{"name": "Ada", "age": 36}'])
    person = await gateway.structured(req("extract"), Person)
    assert person.age == 36
    retry = p1.requests[1]
    assert "did not match" in retry.messages[-1].text


async def test_structured_output_gives_up(gateway: LLMGateway, p1: FakeProvider):
    p1.script.extend(["nope", "still nope"])
    with pytest.raises(StructuredOutputError):
        await gateway.structured(req("extract"), Person)
