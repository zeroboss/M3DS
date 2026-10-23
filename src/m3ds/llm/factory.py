"""Build an LLMGateway from settings."""

from __future__ import annotations

from collections.abc import Sequence

from m3ds.config import Settings
from m3ds.llm.base import LLMProvider
from m3ds.llm.gateway import LLMGateway
from m3ds.llm.registry import ModelRegistry
from m3ds.llm.tracing import CallRecorder, LoggingRecorder


def build_providers(settings: Settings) -> dict[str, LLMProvider]:
    providers: dict[str, LLMProvider] = {}
    # SDK imports are deferred so a missing optional SDK only affects its provider.
    if settings.provider_enabled("anthropic"):
        from m3ds.llm.providers.anthropic import AnthropicProvider

        providers["anthropic"] = AnthropicProvider(settings.anthropic_api_key)
    if settings.provider_enabled("openai"):
        from m3ds.llm.providers.openai import OpenAIProvider

        providers["openai"] = OpenAIProvider(
            settings.openai_api_key, base_url=settings.openai_base_url
        )
    if settings.provider_enabled("gemini"):
        from m3ds.llm.providers.gemini import GeminiProvider

        providers["gemini"] = GeminiProvider(settings.gemini_api_key)
    from m3ds.llm.providers.fake import FakeProvider

    providers["fake"] = FakeProvider()
    return providers


def build_gateway(settings: Settings, recorders: Sequence[CallRecorder] = ()) -> LLMGateway:
    registry = ModelRegistry.from_yaml(settings.models_config)
    return LLMGateway(registry, build_providers(settings), [LoggingRecorder(), *recorders])
