"""Live contract tests against the real provider APIs.

Run with: ``uv run pytest -m live`` (each provider is skipped when its key is unset).
These spend a small amount of money.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from m3ds.config import Settings
from m3ds.llm.factory import build_gateway
from m3ds.llm.gateway import LLMGateway
from m3ds.llm.types import (
    LLMRequest,
    Message,
    StopReason,
    StreamDone,
    SystemBlock,
    TextDelta,
    ToolResultPart,
    ToolSpec,
)

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[1]
ALIASES = ["claude", "gpt", "gemini"]


@pytest.fixture
async def gw():
    gateway = build_gateway(Settings(models_config=str(ROOT / "config" / "models.yaml")))
    yield gateway
    await gateway.aclose()


def require(gw: LLMGateway, alias: str) -> None:
    if not gw.is_available(gw.registry.resolve(alias)):
        pytest.skip(f"{alias}: provider not configured")


@pytest.mark.parametrize("alias", ALIASES)
async def test_generate_and_stream(gw: LLMGateway, alias: str):
    require(gw, alias)
    req = LLMRequest(
        model=alias,
        system=[SystemBlock(text="Reply with exactly one word.", cacheable=True)],
        messages=[Message.user("What colour is a clear daytime sky?")],
        max_output_tokens=2000,
        effort="low",
    )
    resp = await gw.generate(req)
    assert resp.stop_reason == StopReason.END
    assert "blue" in resp.text.lower()
    assert resp.usage.output_tokens > 0

    events = [e async for e in gw.stream(req)]
    assert isinstance(events[-1], StreamDone)
    streamed = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert streamed == events[-1].response.text


class City(BaseModel):
    name: str
    country: str
    population_millions: float


@pytest.mark.parametrize("alias", ALIASES)
async def test_structured_output(gw: LLMGateway, alias: str):
    require(gw, alias)
    city = await gw.structured(
        LLMRequest(model=alias, messages=[Message.user("Describe Tokyo.")], effort="low"), City
    )
    assert city.country.lower() == "japan"


@pytest.mark.parametrize("alias", ALIASES)
async def test_tool_round_trip_and_provider_switch(gw: LLMGateway, alias: str):
    """Tool call on one provider, then continue the same history on every other one."""
    require(gw, alias)
    tool = ToolSpec(
        name="get_robot_battery",
        description="Returns the robot's current battery level in percent.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )
    history = [Message.user("What's your battery level? Use the tool.")]
    first = await gw.generate(LLMRequest(model=alias, messages=history, tools=[tool], effort="low"))
    assert first.stop_reason == StopReason.TOOL_USE
    call = first.message.tool_calls[0]
    history += [
        first.message,
        Message(
            role="user",
            parts=[ToolResultPart(tool_call_id=call.id, name=call.name, content="42")],
        ),
    ]
    for other in ALIASES:
        if not gw.is_available(gw.registry.resolve(other)):
            continue
        final = await gw.generate(
            LLMRequest(model=other, messages=history, tools=[tool], effort="low")
        )
        assert "42" in final.text, f"{alias} -> {other}: {final.text!r}"
