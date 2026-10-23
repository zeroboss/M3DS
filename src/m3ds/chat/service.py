"""Conversation service: persists sessions and runs turns through the gateway.

Phase 0 builds the prompt from a static system prompt plus recent history. Later
phases plug persona, directives, memory and knowledge retrieval into ``build_request``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import TypeAdapter
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from m3ds.llm.gateway import LLMGateway
from m3ds.llm.types import (
    LLMRequest,
    LLMResponse,
    Message,
    NativeContent,
    Part,
    StreamDone,
    StreamEvent,
    SystemBlock,
)
from m3ds.storage.models import Conversation, MessageRow

_parts_adapter = TypeAdapter(list[Part])


class ConversationNotFoundError(LookupError):
    pass


def row_to_message(row: MessageRow) -> Message:
    return Message(
        role=row.role,  # type: ignore[arg-type]
        parts=_parts_adapter.validate_python(row.parts),
        native=NativeContent.model_validate(row.native) if row.native else None,
    )


def trim_history(messages: list[Message], max_messages: int) -> list[Message]:
    """Keep the most recent messages, starting on a plain user turn.

    Never starts on an assistant turn or on a user turn that only carries tool
    results, which every provider would reject as an orphaned tool exchange.
    """
    if len(messages) <= max_messages:
        return messages
    start = len(messages) - max_messages
    while start < len(messages):
        m = messages[start]
        if m.role == "user" and any(p.type != "tool_result" for p in m.parts):
            break
        start += 1
    return messages[start:]


class ChatService:
    def __init__(
        self,
        gateway: LLMGateway,
        sessionmaker: async_sessionmaker[AsyncSession],
        system_prompt: str,
        history_max_messages: int = 40,
    ):
        self.gateway = gateway
        self.sessionmaker = sessionmaker
        self.system_prompt = system_prompt
        self.history_max_messages = history_max_messages

    async def create_conversation(
        self, robot_id: str = "default", user_id: str = "default", title: str | None = None
    ) -> Conversation:
        async with self.sessionmaker() as session:
            conv = Conversation(robot_id=robot_id, user_id=user_id, title=title)
            session.add(conv)
            await session.commit()
            return conv

    async def get_conversation(self, conversation_id: str) -> Conversation:
        async with self.sessionmaker() as session:
            conv = await session.get(Conversation, conversation_id)
            if conv is None:
                raise ConversationNotFoundError(conversation_id)
            return conv

    async def history(self, conversation_id: str) -> list[Message]:
        await self.get_conversation(conversation_id)
        async with self.sessionmaker() as session:
            rows = await session.scalars(
                select(MessageRow)
                .where(MessageRow.conversation_id == conversation_id)
                .order_by(MessageRow.seq)
            )
            return [row_to_message(r) for r in rows]

    def build_request(
        self, history: list[Message], user_message: Message, model: str | None, conversation_id: str
    ) -> LLMRequest:
        return LLMRequest(
            model=model,
            # Stable, cacheable prefix first; per-turn context will follow in later phases.
            system=[SystemBlock(text=self.system_prompt, cacheable=True)],
            messages=[*trim_history(history, self.history_max_messages), user_message],
            metadata={"conversation_id": conversation_id, "cache_key": conversation_id},
        )

    async def _append(
        self, conversation_id: str, messages: list[tuple[Message, LLMResponse | None]]
    ) -> None:
        async with self.sessionmaker() as session, session.begin():
            last = await session.scalar(
                select(func.max(MessageRow.seq)).where(
                    MessageRow.conversation_id == conversation_id
                )
            )
            seq = -1 if last is None else last
            for msg, resp in messages:
                seq += 1
                session.add(
                    MessageRow(
                        conversation_id=conversation_id,
                        seq=seq,
                        role=msg.role,
                        parts=[p.model_dump(mode="json") for p in msg.parts],
                        native=msg.native.model_dump(mode="json") if msg.native else None,
                        model_alias=resp.alias if resp else None,
                        provider_model=f"{resp.provider}:{resp.model}" if resp else None,
                    )
                )
            conv = await session.get(Conversation, conversation_id)
            if conv is not None:
                conv.updated_at = func.now()  # type: ignore[assignment]

    async def send(self, conversation_id: str, text: str, model: str | None = None) -> LLMResponse:
        history = await self.history(conversation_id)
        user_message = Message.user(text)
        resp = await self.gateway.generate(
            self.build_request(history, user_message, model, conversation_id)
        )
        await self._append(conversation_id, [(user_message, None), (resp.message, resp)])
        return resp

    async def send_stream(
        self, conversation_id: str, text: str, model: str | None = None
    ) -> AsyncIterator[StreamEvent]:
        history = await self.history(conversation_id)
        user_message = Message.user(text)
        req = self.build_request(history, user_message, model, conversation_id)
        async for event in self.gateway.stream(req):
            if isinstance(event, StreamDone):
                await self._append(
                    conversation_id,
                    [(user_message, None), (event.response.message, event.response)],
                )
            yield event
