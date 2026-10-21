import pytest
from pydantic import ValidationError

from speaker_claims.config import Settings, load_settings

_ENV_KEYS = (
    "OPENAI_API_KEY",
    "DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "SENTRY_DSN",
    "SENTRY_ENVIRONMENT",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_missing_openai_key_is_rejected(clean_env: None) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            database_url="postgresql://user:password@localhost:5432/speaker_claims",
        )


def test_missing_database_url_is_rejected(clean_env: None) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, openai_api_key="sk-test")


def test_non_postgres_database_url_is_rejected(clean_env: None) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            openai_api_key="sk-test",
            database_url="mysql://localhost/speaker_claims",
        )


def test_malformed_sentry_dsn_is_rejected(clean_env: None) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            openai_api_key="sk-test",
            database_url="postgresql://user:password@localhost:5432/speaker_claims",
            sentry_dsn="not-a-url",
        )


def test_load_settings_exits_when_required_values_are_missing(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        load_settings()


def test_empty_sentry_dsn_is_valid(clean_env: None) -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="sk-test",
        database_url="postgresql://user:password@localhost:5432/speaker_claims",
        sentry_dsn="",
    )
    assert settings.sentry_dsn == ""
    assert settings.sentry_environment == "local"
