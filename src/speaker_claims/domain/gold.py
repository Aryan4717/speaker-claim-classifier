from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldClaim:
    answer_id: str
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    expected_label: str
    evidence_fact_ids: tuple[str, ...]
    explanation: str
    difficulty_tags: tuple[str, ...]
