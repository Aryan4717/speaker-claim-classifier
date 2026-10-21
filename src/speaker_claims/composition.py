from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from speaker_claims.adapters.csv_catalogue import CsvFactCatalogue
from speaker_claims.adapters.openai_provider import DEFAULT_MODEL, OpenAIProvider
from speaker_claims.application.claim_pipeline import V1ClaimPipeline
from speaker_claims.application.fact_tool import FactToolExecutor


def load_v1_prompt() -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / "v1" / "system.txt"
    return path.read_text(encoding="utf-8")


def build_v1_pipeline(
    api_key: str,
    catalogue: CsvFactCatalogue,
    prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
) -> V1ClaimPipeline:
    """Wire the v1 prompt, the OpenAI adapter, and the catalogue at the edge."""
    return V1ClaimPipeline(
        provider=OpenAIProvider(api_key=api_key, model=model, client=client),
        executor=FactToolExecutor(catalogue),
        known_fact_ids=catalogue.fact_ids(),
        prompt=prompt if prompt is not None else load_v1_prompt(),
    )
