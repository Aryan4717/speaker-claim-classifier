from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class StoredClaim:
    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: str
    fact_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class StoredAnswer:
    answer_id: str
    status: str
    error_code: str | None
    error_message: str | None
    latency_ms: int | None
    trace_id: str | None
    claims: tuple[StoredClaim, ...]


@dataclass(frozen=True)
class StoredRun:
    run_id: UUID
    split: str
    agent_version: str
    prompt_version: str
    dataset_version: str
    evaluator_version: str
    fact_version: str
    provider: str
    model: str
    settings: dict[str, object]
    source_revision: str | None
    file_hashes: dict[str, str]
    created_at: datetime
    status: str
    answer_count: int
    answer_error_count: int
    accepted_prediction_count: int
    quality: dict[str, object] | None
    answers: tuple[StoredAnswer, ...]
