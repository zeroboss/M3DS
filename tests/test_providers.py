"""Request/response mapping tests for each adapter. No network access."""

from __future__ import annotations

from types import SimpleNamespace

from anthropic.types import Message as AnthropicMessage
from google.genai import types as gtypes
from openai.types.responses import Response as OpenAIResponse

from m3ds.llm.providers import anthropic as anth
from m3ds.llm.providers import gemini as gem
from m3ds.llm.providers import openai as oai
from m3ds.llm.registry import Capabilities, ModelInfo
from m3ds.llm.types import (
    LLMRequest,
    Message,
    NativeContent,
    StopReason,
    SystemBlock,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
)

CAPS = Capabilities(effort=True, prompt_caching=True, temperature=False)


def model(provider: str, **options) -> ModelInfo:
    return ModelInfo(
        alias=provider,
        provider=provider,
        model_id=f"{provider}-m",
        max_output_tokens=1000,
        default_max_output_tokens=500,
        supports=CAPS,
        options=options,
    )


TOOL = ToolSpec(
    name="get_time", description="Current time", parameters={"type": "object", "properties": {}}
)

TOOL_CONVERSATION = [
    Message.user("what time is it?"),
    Message(
        role="assistant", parts=[TextPart(text="Checking."), ToolCallPart(id="c1", name="get_time")]
    ),
    Message(
        role="user", parts=[ToolResultPart(tool_call_id="c1", name="get_time", content="12:00")]
    ),
]


def full_request(**kw) -> LLMRequest:
    return LLMRequest(
        system=[SystemBlock(text="persona", cacheable=True), SystemBlock(text="volatile")],
        messages=TOOL_CONVERSATION,
        tools=[TOOL],
        effort="low",
        temperature=0.3,
        max_output_tokens=5000,
        **kw,
    )


# ------------------------------------------------------------------ anthropic


def test_anthropic_params():
    p = anth.build_params(full_request(), model("anthropic", server_fallback=True), max_tokens=1000)
    assert p["max_tokens"] == 1000
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in p["system"][1]
    assert p["cache_control"] == {"type": "ephemeral"}
    assert p["output_config"] == {"effort": "low"}
    assert "temperature" not in p
    assert p["fallbacks"] == "default" and p["betas"] == [anth.SERVER_FALLBACK_BETA]
    msgs = p["messages"]
    assert msgs[1]["content"][1] == {
        "type": "tool_use",
        "id": "c1",
        "name": "get_time",
        "input": {},
    }
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert p["tools"][0]["input_schema"] == TOOL.parameters


def test_anthropic_structured_output_format():
    r = LLMRequest(
        messages=[Message.user("x")],
        response_schema={"type": "object", "properties": {"a": {"type": "string"}}},
    )
    p = anth.build_params(r, model("anthropic"), max_tokens=100)
    fmt = p["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["additionalProperties"] is False


def test_anthropic_response_roundtrip_replays_native():
    msg = AnthropicMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "anthropic-m",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "tu_1", "name": "get_time", "input": {"tz": "UTC"}},
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 7,
                "cache_creation_input_tokens": 3,
            },
        }
    )
    resp = anth.from_anthropic_message(msg, model("anthropic"), 12.0)
    assert resp.stop_reason == StopReason.TOOL_USE
    assert resp.text == "Let me check."
    assert resp.message.tool_calls[0].arguments == {"tz": "UTC"}
    assert (resp.usage.cache_read_tokens, resp.usage.cache_write_tokens) == (7, 3)
    replay = anth.to_anthropic_messages([Message.user("hi"), resp.message], model("anthropic"))
    assert replay[1]["content"][0]["type"] == "thinking"  # thinking block preserved
    assert replay[1]["content"][0]["signature"] == "sig"


def test_anthropic_ignores_other_providers_native():
    m = Message(
        role="assistant",
        parts=[TextPart(text="hi")],
        native=NativeContent(provider="openai", model="x", payload=[{"type": "reasoning"}]),
    )
    assert anth.to_anthropic_messages([m], model("anthropic"))[0]["content"] == [
        {"type": "text", "text": "hi"}
    ]


def test_anthropic_fallback_echo_sanitized():
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "s"},
        {"type": "text", "text": "partial"},
        {"type": "tool_use", "id": "t", "name": "x", "input": {}},
        {"type": "fallback"},
        {"type": "thinking", "thinking": "", "signature": "s2"},
        {"type": "text", "text": "rest"},
    ]
    kinds = [b["type"] for b in anth.sanitize_fallback_echo(blocks)]
    assert kinds == ["text", "fallback", "thinking", "text"]


# ------------------------------------------------------------------ openai


def test_openai_params():
    p = oai.build_params(full_request(metadata={"cache_key": "conv-1"}), model("openai"))
    assert p["instructions"] == "persona\n\nvolatile"
    assert p["store"] is False
    assert p["reasoning"] == {"effort": "low"}
    assert p["max_output_tokens"] == 1000  # clamped to the model maximum
    assert "temperature" not in p
    assert p["prompt_cache_key"] == "conv-1"
    items = p["input"]
    assert items[0] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "what time is it?"}],
    }
    assert items[1] == {"role": "assistant", "content": "Checking."}
    assert items[2]["type"] == "function_call" and items[2]["call_id"] == "c1"
    assert items[3] == {"type": "function_call_output", "call_id": "c1", "output": "12:00"}


def test_openai_response_mapping():
    resp = OpenAIResponse.model_validate(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "openai-m",
            "status": "completed",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"},
                {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                },
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 10},
                "output_tokens_details": {"reasoning_tokens": 8},
            },
        }
    )
    out = oai.from_openai_response(resp, model("openai"), 5.0)
    assert out.text == "hello"
    assert out.stop_reason == StopReason.END
    assert (out.usage.input_tokens, out.usage.cache_read_tokens, out.usage.cache_write_tokens) == (
        30,
        60,
        10,
    )
    assert out.usage.reasoning_tokens == 8
    replay = oai.to_openai_input([Message.user("hi"), out.message])
    assert replay[1]["type"] == "reasoning" and replay[1]["encrypted_content"] == "enc"


def test_openai_incomplete_maps_to_max_tokens():
    resp = SimpleNamespace(
        output=[],
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        status="incomplete",
        usage=None,
        model="openai-m",
        id="r",
    )
    assert oai.from_openai_response(resp, model("openai"), 1.0).stop_reason == StopReason.MAX_TOKENS


# ------------------------------------------------------------------ gemini


def test_gemini_contents_and_config():
    contents = gem.to_gemini_contents(TOOL_CONVERSATION)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts is not None and contents[1].parts[1].function_call is not None
    # Calls from another provider get the dummy signature Gemini 3 requires.
    dumped = contents[1].parts[1].model_dump(mode="json", exclude_none=True)
    assert dumped["thought_signature"] == "skip_thought_signature_validator"
    fr = contents[2].parts[0].function_response if contents[2].parts else None
    assert fr is not None and fr.response == {"result": "12:00"} and fr.id == "c1"
    cfg = gem.build_config(full_request(), model("gemini"))
    assert cfg.system_instruction == "persona\n\nvolatile"
    assert cfg.max_output_tokens == 1000
    assert (
        cfg.thinking_config is not None
        and cfg.thinking_config.thinking_level == gtypes.ThinkingLevel.LOW
    )
    assert cfg.temperature is None  # model declares no temperature support


def test_gemini_response_preserves_thought_signature():
    gparts = [
        gtypes.Part(text="thinking...", thought=True),
        gtypes.Part(text="Hi "),
        gtypes.Part(text="there", thought_signature=b"\x01\x02sig"),
        gtypes.Part(function_call=gtypes.FunctionCall(name="get_time", args={})),
    ]
    resp = gem.build_response(model("gemini"), gparts, "STOP", False, gem._usage(None), 1.0)
    assert resp.text == "Hi there"
    assert resp.stop_reason == StopReason.TOOL_USE
    call = resp.message.tool_calls[0]
    assert call.id.startswith("gemini-call-")
    # Native payload survives JSON (database) round-trip with signature bytes intact.
    stored = Message.model_validate_json(resp.message.model_dump_json())
    replay = gem.to_gemini_contents([stored])[0]
    assert replay.parts is not None and replay.parts[2].thought_signature == b"\x01\x02sig"
    # Synthetic ids are not sent back to Gemini.
    synthetic = gem.to_gemini_contents([Message(role="assistant", parts=[call])])[0]
    assert synthetic.parts is not None and synthetic.parts[0].function_call is not None
    assert synthetic.parts[0].function_call.id is None


def test_gemini_safety_maps_to_refusal():
    resp = gem.build_response(model("gemini"), [], "SAFETY", False, gem._usage(None), 1.0)
    assert resp.stop_reason == StopReason.REFUSAL
