from speaker_claims.runtime import engine_url
from speaker_claims.server import main


def test_plain_postgres_url_uses_psycopg() -> None:
    converted = engine_url("postgresql://user:password@localhost:5433/speaker_claims")
    assert converted == "postgresql+psycopg://user:password@localhost:5433/speaker_claims"
    assert engine_url(converted) == converted


def test_server_import_does_not_start() -> None:
    assert callable(main)
