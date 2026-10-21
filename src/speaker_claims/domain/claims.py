from __future__ import annotations

from dataclasses import dataclass

LABELS = frozenset({"supported", "contradicted", "unknown"})


@dataclass(frozen=True)
class AnswerRecord:
    answer_id: str
    split: str
    brand_id: str
    question: str
    ai_answer: str


@dataclass(frozen=True)
class PredictedClaim:
    answer_id: str
    start_char: int
    end_char: int
    quote: str
    label: str
    fact_ids: tuple[str, ...]
    reason: str
    claim_id: str

    @property
    def expected_claim_id(self) -> str:
        return f"{self.answer_id}:{self.start_char}:{self.end_char}"
