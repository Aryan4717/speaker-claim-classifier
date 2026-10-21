from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine

from speaker_claims.adapters.csv_catalogue import CsvAnswerSource, CsvFactCatalogue
from speaker_claims.adapters.csv_gold import CsvGoldSource
from speaker_claims.adapters.openai_provider import DEFAULT_MODEL
from speaker_claims.adapters.run_store import SqlRunStore
from speaker_claims.application.evaluation import EvaluationService
from speaker_claims.composition import build_monitor, build_pipelines, build_tracer
from speaker_claims.config import load_settings

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def engine_url(database_url: str) -> str:
    """Use the psycopg v3 driver for a plain PostgreSQL URL."""
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_url.removeprefix("postgresql://")
    return database_url


def migrate(database_url: str) -> None:
    os.environ["DATABASE_URL"] = database_url
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    command.upgrade(config, "head")


def build_service() -> EvaluationService:
    """Wire the API dependencies: model, catalogue, Postgres, Langfuse, and Sentry."""
    settings = load_settings()
    database_url = engine_url(settings.database_url)
    migrate(database_url)
    catalogue = CsvFactCatalogue.from_path(DATA / "brand_facts.csv")
    return EvaluationService(
        build_pipelines(settings.openai_api_key.get_secret_value(), catalogue),
        catalogue,
        CsvAnswerSource(DATA / "answers.csv"),
        CsvGoldSource(DATA / "development_gold.csv"),
        SqlRunStore(create_engine(database_url)),
        provider="openai",
        model=DEFAULT_MODEL,
        settings={
            "timeout_seconds": 15,
            "max_model_turns": 3,
            "max_in_flight": 3,
            "max_attempts": 2,
        },
        source_revision=_revision(),
        file_hashes=_file_hashes(),
        tracer=build_tracer(settings),
        monitor=build_monitor(settings),
    )


def _file_hashes() -> dict[str, str]:
    names = ("answers.csv", "brand_facts.csv", "development_gold.csv")
    return {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in names}


def _revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = completed.stdout.strip()
    return revision or None
