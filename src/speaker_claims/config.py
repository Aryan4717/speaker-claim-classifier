from __future__ import annotations

from pydantic import SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_POSTGRES_PREFIXES = ("postgresql://", "postgresql+psycopg://")


class Settings(BaseSettings):
    """Process settings loaded from the environment.

    OPENAI_API_KEY and DATABASE_URL are required. SENTRY_DSN may be empty;
    an empty value means error monitoring stays off until a later branch
    selects a no-op monitor.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr
    database_url: str
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    langfuse_public_key: str = ""
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = ""
    sentry_dsn: str = ""
    sentry_environment: str = "local"

    @field_validator(
        "anthropic_api_key",
        "gemini_api_key",
        "langfuse_secret_key",
        mode="before",
    )
    @classmethod
    def blank_optional_secret(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("openai_api_key")
    @classmethod
    def openai_key_required(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("OPENAI_API_KEY is required")
        return value

    @field_validator("database_url")
    @classmethod
    def database_url_required(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned.startswith(_POSTGRES_PREFIXES):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL")
        return cleaned

    @field_validator("sentry_dsn")
    @classmethod
    def sentry_dsn_optional(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned and not cleaned.startswith(("http://", "https://")):
            raise ValueError("SENTRY_DSN must be empty or an HTTP(S) URL")
        return cleaned


def load_settings() -> Settings:
    """Load settings and exit the process when they are missing or malformed."""
    try:
        return Settings()
    except ValidationError as exc:
        raise SystemExit(f"Invalid configuration: {exc}") from exc
