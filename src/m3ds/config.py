"""Application settings (environment / .env driven)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful robot assistant. Answer clearly and concisely, and say so when "
    "you are unsure."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="M3DS_", env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./m3ds.db"
    models_config: str = "config/models.yaml"
    # Create tables on startup (convenient for SQLite/dev; use Alembic in production).
    auto_create_schema: bool = True

    # Provider credentials use each vendor's conventional variable names.
    anthropic_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "M3DS_ANTHROPIC_API_KEY")
    )
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "M3DS_OPENAI_API_KEY")
    )
    openai_base_url: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_BASE_URL", "M3DS_OPENAI_BASE_URL")
    )
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY", "M3DS_GEMINI_API_KEY"),
    )
    # Force-enable providers whose SDK finds credentials another way (e.g. an
    # `ant auth login` profile, or Vertex/ADC), comma separated: "anthropic,gemini".
    enabled_providers: str = ""

    # Protects the HTTP API when set (sent as the X-API-Key header).
    api_key: str | None = None

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    history_max_messages: int = 40

    def provider_enabled(self, name: str) -> bool:
        forced = {p.strip() for p in self.enabled_providers.split(",") if p.strip()}
        if name in forced:
            return True
        return bool(
            {
                "anthropic": self.anthropic_api_key,
                "openai": self.openai_api_key,
                "gemini": self.gemini_api_key,
                "fake": True,
            }.get(name)
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
