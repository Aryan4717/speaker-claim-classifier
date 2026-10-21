from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from speaker_claims.adapters.csv_catalogue import CsvFactCatalogue
from speaker_claims.adapters.openai_provider import DEFAULT_MODEL, OpenAIProvider
from speaker_claims.application.claim_pipeline import V1ClaimPipeline, V2ClaimPipeline
from speaker_claims.application.fact_tool import FactToolExecutor
from speaker_claims.application.monitoring import NoOpMonitor
from speaker_claims.application.reliability import ProviderGate, RetryingProvider, RetryPolicy
from speaker_claims.application.tracing import NoOpTracer
from speaker_claims.config import Settings
from speaker_claims.ports.monitor import ErrorMonitorPort
from speaker_claims.ports.tracer import TracerPort


def build_monitor(settings: Settings) -> ErrorMonitorPort:
    """Select the Sentry adapter only when a DSN is configured."""
    dsn = settings.sentry_dsn.strip()
    if dsn == "":
        return NoOpMonitor()
    from speaker_claims.adapters.sentry_monitor import SentryMonitor

    return SentryMonitor(dsn=dsn, environment=settings.sentry_environment)


def build_tracer(settings: Settings) -> TracerPort:
    """Select the Langfuse adapter only when both keys are present."""
    secret = settings.langfuse_secret_key
    public_key = settings.langfuse_public_key.strip()
    if secret is None or public_key == "":
        return NoOpTracer()
    from speaker_claims.adapters.langfuse_tracer import LangfuseTracer

    return LangfuseTracer(
        public_key=public_key,
        secret_key=secret.get_secret_value(),
        host=settings.langfuse_host.strip(),
    )


def load_v1_prompt() -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "v1" / "system.txt"
    return path.read_text(encoding="utf-8")


def load_v2_prompt() -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "v2" / "system.txt"
    return path.read_text(encoding="utf-8")


def build_pipelines(
    api_key: str,
    catalogue: CsvFactCatalogue,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> dict[str, V1ClaimPipeline]:
    """Register v1 and v2 on one provider gate. v2 changes only its opening prompt."""
    provider = RetryingProvider(
        OpenAIProvider(api_key=api_key, model=model, client=client),
        policy=RetryPolicy(),
        gate=ProviderGate(),
    )
    executor = FactToolExecutor(catalogue)
    known_fact_ids = catalogue.fact_ids()
    return {
        "v1": V1ClaimPipeline(provider, executor, known_fact_ids, load_v1_prompt()),
        "v2": V2ClaimPipeline(provider, executor, known_fact_ids, load_v2_prompt()),
    }


def build_v1_pipeline(
    api_key: str,
    catalogue: CsvFactCatalogue,
    prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> V1ClaimPipeline:
    """Wire the v1 prompt, the OpenAI adapter, and the catalogue at the edge."""
    return V1ClaimPipeline(
        provider=RetryingProvider(
            OpenAIProvider(api_key=api_key, model=model, client=client),
            policy=RetryPolicy(),
            gate=ProviderGate(),
        ),
        executor=FactToolExecutor(catalogue),
        known_fact_ids=catalogue.fact_ids(),
        prompt=prompt if prompt is not None else load_v1_prompt(),
    )
