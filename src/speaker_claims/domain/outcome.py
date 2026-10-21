from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from speaker_claims.domain.claims import PredictedClaim


@dataclass(frozen=True)
class AnswerOutcome:
    answer_id: str
    agent_version: str
    status: Literal["ok", "error"]
    claims: tuple[PredictedClaim, ...]
    error_code: str | None = None
    error_message: str | None = None
    trace_id: str | None = None
