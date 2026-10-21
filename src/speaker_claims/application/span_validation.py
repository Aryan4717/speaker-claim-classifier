from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from speaker_claims.domain.claims import LABELS, PredictedClaim
from speaker_claims.domain.errors import SpanValidationError

_SENTENCE_SPLIT = re.compile(r"[.!?]+")


@dataclass(frozen=True)
class AnswerClaims:
    answer_id: str
    ai_answer: str
    claims: tuple[PredictedClaim, ...]
    known_fact_ids: frozenset[str]


class SpanCheck(ABC):
    def __init__(self) -> None:
        self._next: SpanCheck | None = None

    def set_next(self, handler: SpanCheck) -> SpanCheck:
        self._next = handler
        return handler

    def validate(self, context: AnswerClaims) -> None:
        self.check(context)
        if self._next is not None:
            self._next.validate(context)

    @abstractmethod
    def check(self, context: AnswerClaims) -> None:
        """Reject the answer when this one rule fails."""


class BoundsCheck(SpanCheck):
    def check(self, context: AnswerClaims) -> None:
        length = len(context.ai_answer)
        for claim in context.claims:
            out_of_range = (
                claim.start_char < 0
                or claim.end_char > length
                or claim.start_char >= claim.end_char
            )
            if out_of_range:
                raise SpanValidationError("out_of_bounds", claim.expected_claim_id)


class QuoteCheck(SpanCheck):
    def check(self, context: AnswerClaims) -> None:
        for claim in context.claims:
            if claim.quote != claim.quote.strip():
                raise SpanValidationError("quote_whitespace", claim.expected_claim_id)
            if context.ai_answer[claim.start_char : claim.end_char] != claim.quote:
                raise SpanValidationError("quote_mismatch", claim.expected_claim_id)


class ClaimIdCheck(SpanCheck):
    def check(self, context: AnswerClaims) -> None:
        for claim in context.claims:
            if claim.answer_id != context.answer_id or claim.claim_id != claim.expected_claim_id:
                raise SpanValidationError("claim_id_mismatch", claim.expected_claim_id)


class LabelCheck(SpanCheck):
    def check(self, context: AnswerClaims) -> None:
        for claim in context.claims:
            if claim.label not in LABELS:
                raise SpanValidationError("invalid_label", claim.label)


class FactIdCheck(SpanCheck):
    def check(self, context: AnswerClaims) -> None:
        for claim in context.claims:
            unknown = [
                fact_id for fact_id in claim.fact_ids if fact_id not in context.known_fact_ids
            ]
            if unknown:
                raise SpanValidationError("unknown_fact_id", unknown[0])


class DuplicateCheck(SpanCheck):
    def check(self, context: AnswerClaims) -> None:
        seen: set[tuple[int, int]] = set()
        for claim in context.claims:
            key = (claim.start_char, claim.end_char)
            if key in seen:
                raise SpanValidationError("duplicate_span", claim.claim_id)
            seen.add(key)


class OverlapCheck(SpanCheck):
    def check(self, context: AnswerClaims) -> None:
        ordered = sorted(context.claims, key=lambda claim: (claim.start_char, claim.end_char))
        for left, right in zip(ordered, ordered[1:], strict=False):
            if left.end_char > right.start_char:
                raise SpanValidationError("overlap", right.claim_id)


class OrderCheck(SpanCheck):
    def check(self, context: AnswerClaims) -> None:
        starts = [claim.start_char for claim in context.claims]
        if starts != sorted(starts) or len(starts) != len(set(starts)):
            raise SpanValidationError("unordered", context.answer_id)


class ReasonCheck(SpanCheck):
    def check(self, context: AnswerClaims) -> None:
        for claim in context.claims:
            parts = [part for part in _SENTENCE_SPLIT.split(claim.reason.strip()) if part.strip()]
            if len(parts) > 2:
                raise SpanValidationError("reason_too_long", claim.claim_id)


def build_span_chain() -> SpanCheck:
    checks: tuple[SpanCheck, ...] = (
        BoundsCheck(),
        QuoteCheck(),
        ClaimIdCheck(),
        LabelCheck(),
        FactIdCheck(),
        DuplicateCheck(),
        OverlapCheck(),
        OrderCheck(),
        ReasonCheck(),
    )
    for earlier, later in zip(checks, checks[1:], strict=False):
        earlier.set_next(later)
    return checks[0]
