"""Canonical, provider-neutral request/response types.

Everything above the provider adapters speaks only these types. Adapters translate
them to and from each vendor's wire format.

Provider-native content (reasoning/thinking blocks, thought signatures, encrypted
reasoning items) cannot be represented portably, but some providers require it to be
sent back unchanged on the next turn. Each assistant message therefore carries an
optional ``native`` payload: when the next request goes to the same provider, the
adapter replays it verbatim; any other provider uses the portable ``parts``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    media_type: str = "image/png"
    data_base64: str | None = None
    url: str | None = None


class ToolCallPart(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultPart(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


Part = Annotated[TextPart | ImagePart | ToolCallPart | ToolResultPart, Field(discriminator="type")]


class NativeContent(BaseModel):
    """Opaque provider-specific content for lossless replay to the same provider."""

    provider: str
    model: str
    payload: Any


class Message(BaseModel):
    role: Literal["user", "assistant"]
    parts: list[Part]
    native: NativeContent | None = None

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", parts=[TextPart(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role="assistant", parts=[TextPart(text=text)])

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.parts if isinstance(p, ToolCallPart)]


class SystemBlock(BaseModel):
    """A block of system instructions.

    ``cacheable`` marks the end of a stable prefix (persona, standing directives) so
    adapters can place prompt-cache breakpoints. Put stable blocks first.
    """

    text: str
    cacheable: bool = False


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


Effort = Literal["low", "medium", "high"]


class LLMRequest(BaseModel):
    """A provider-neutral generation request.

    ``model`` is a registry alias (e.g. ``"claude"``) or ``"provider:model-id"``; when
    omitted the gateway picks a model from the task's route.
    """

    messages: list[Message]
    model: str | None = None
    system: list[SystemBlock] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    max_output_tokens: int | None = None
    effort: Effort | None = None
    # Ignored by providers that no longer accept sampling parameters (Anthropic).
    temperature: float | None = None
    # JSON schema the final answer must conform to (structured output).
    response_schema: dict[str, Any] | None = None
    response_schema_name: str = "response"
    metadata: dict[str, str] = Field(default_factory=dict)


class StopReason(StrEnum):
    END = "end"
    MAX_TOKENS = "max_tokens"
    TOOL_USE = "tool_use"
    REFUSAL = "refusal"
    OTHER = "other"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0


class LLMResponse(BaseModel):
    provider: str
    model: str
    alias: str | None = None
    message: Message
    stop_reason: StopReason
    raw_stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    request_id: str | None = None
    latency_ms: float = 0.0
    cost_usd: float | None = None

    @property
    def text(self) -> str:
        return self.message.text


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class StreamDone(BaseModel):
    type: Literal["done"] = "done"
    response: LLMResponse


StreamEvent = TextDelta | StreamDone
