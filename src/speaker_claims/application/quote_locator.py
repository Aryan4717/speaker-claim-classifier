from __future__ import annotations

from speaker_claims.domain.claims import AnswerRecord, PredictedClaim
from speaker_claims.domain.errors import SpanValidationError


def locate_quotes(
    answer: AnswerRecord,
    quotes: list[tuple[str, str, tuple[str, ...], str]],
) -> tuple[PredictedClaim, ...]:
    """Find each exact quote in ai_answer and assign offsets."""
    cursor = 0
    located: list[PredictedClaim] = []
    for quote, label, fact_ids, reason in quotes:
        if quote == "" or quote != quote.strip():
            raise SpanValidationError("quote_whitespace", answer.answer_id)
        start = answer.ai_answer.find(quote, cursor)
        if start < 0:
            raise SpanValidationError("quote_not_found", answer.answer_id)
        end = start + len(quote)
        cursor = end
        claim_id = f"{answer.answer_id}:{start}:{end}"
        located.append(
            PredictedClaim(
                answer_id=answer.answer_id,
                start_char=start,
                end_char=end,
                quote=quote,
                label=label,
                fact_ids=fact_ids,
                reason=reason,
                claim_id=claim_id,
            )
        )
    return tuple(located)
