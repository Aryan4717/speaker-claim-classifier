import csv
from pathlib import Path

import pytest

from speaker_claims.adapters.csv_catalogue import load_answers
from speaker_claims.application.span_validation import AnswerClaims, build_span_chain
from speaker_claims.domain.claims import PredictedClaim
from speaker_claims.domain.errors import SpanValidationError

DATA = Path(__file__).resolve().parents[1] / "data"
TEXT = "Alpha costs INR 1,299. Beta weighs 320 grams."


def _claim(
    text: str,
    quote: str,
    label: str = "supported",
    fact_ids: tuple[str, ...] = (),
    reason: str = "The catalogue matches this sentence.",
    answer_id: str = "A01",
    claim_id: str | None = None,
) -> PredictedClaim:
    start = text.index(quote)
    end = start + len(quote)
    return PredictedClaim(
        answer_id=answer_id,
        start_char=start,
        end_char=end,
        quote=quote,
        label=label,
        fact_ids=fact_ids,
        reason=reason,
        claim_id=claim_id or f"{answer_id}:{start}:{end}",
    )


def _validate(claims: tuple[PredictedClaim, ...], text: str = TEXT) -> None:
    build_span_chain().validate(
        AnswerClaims(
            answer_id="A01",
            ai_answer=text,
            claims=claims,
            known_fact_ids=frozenset({"cleartrail.mini.price_2026"}),
        )
    )


def test_quoted_comma_and_embedded_newline_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "answers.csv"
    question = "As of 1 October 2026, what does it cost?"
    answer = "The price is INR 1,299.\n\nThe warranty is separate."
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["answer_id", "split", "brand_id", "question", "ai_answer"])
        writer.writerow(["A00", "development", "cleartrail", question, answer])

    loaded = load_answers(path)
    assert loaded[0].question == question
    assert loaded[0].ai_answer == answer
    assert loaded[0].ai_answer[loaded[0].ai_answer.index("INR") :].startswith("INR 1,299.")


def test_issued_answers_keep_exact_gold_spans() -> None:
    answers = {row.answer_id: row for row in load_answers(DATA / "answers.csv")}
    assert len(answers) == 50
    assert answers["A01"].split == "development"
    assert answers["A50"].split == "check"
    assert "," in answers["A01"].question
    assert "\n" in answers["A01"].ai_answer
    raw = (DATA / "answers.csv").read_bytes()
    assert raw.count(b"\n") > len(answers)

    with (DATA / "development_gold.csv").open(encoding="utf-8", newline="") as handle:
        gold = list(csv.DictReader(handle))
    assert len(gold) == 376
    for row in gold:
        text = answers[row["answer_id"]].ai_answer
        start = int(row["start_char"])
        end = int(row["end_char"])
        assert text[start:end] == row["quote"]
        assert row["quote"] == row["quote"].strip()


def test_valid_span_passes() -> None:
    claim = _claim(TEXT, "Alpha costs INR 1,299.", fact_ids=("cleartrail.mini.price_2026",))
    _validate((claim,))


def test_bounds_quote_duplicate_and_overlap_are_rejected() -> None:
    valid = _claim(TEXT, "Alpha costs INR 1,299.")
    with pytest.raises(SpanValidationError) as bounds:
        _validate(
            (
                PredictedClaim(
                    answer_id="A01",
                    start_char=0,
                    end_char=len(TEXT) + 5,
                    quote="Alpha costs INR 1,299.",
                    label="supported",
                    fact_ids=(),
                    reason="Too wide.",
                    claim_id="A01:0:99",
                ),
            )
        )
    assert bounds.value.code == "out_of_bounds"

    mismatched = PredictedClaim(
        answer_id=valid.answer_id,
        start_char=valid.start_char,
        end_char=valid.end_char,
        quote="Alpha costs INR 1,199.",
        label="supported",
        fact_ids=(),
        reason="Wrong quote.",
        claim_id=valid.claim_id,
    )
    with pytest.raises(SpanValidationError) as mismatch:
        _validate((mismatched,))
    assert mismatch.value.code == "quote_mismatch"

    with pytest.raises(SpanValidationError) as duplicate:
        _validate((valid, valid))
    assert duplicate.value.code == "duplicate_span"

    first = _claim(TEXT, "Alpha costs INR 1,299.")
    second = _claim(TEXT, "INR 1,299. Beta weighs 320 grams.")
    with pytest.raises(SpanValidationError) as overlap:
        _validate((first, second))
    assert overlap.value.code == "overlap"


def test_unknown_fact_id_and_bad_label_are_rejected() -> None:
    with pytest.raises(SpanValidationError) as unknown:
        _validate((_claim(TEXT, "Alpha costs INR 1,299.", fact_ids=("missing.fact",)),))
    assert unknown.value.code == "unknown_fact_id"

    with pytest.raises(SpanValidationError) as label:
        _validate((_claim(TEXT, "Alpha costs INR 1,299.", label="maybe"),))
    assert label.value.code == "invalid_label"


def test_unordered_claims_are_rejected() -> None:
    later = _claim(TEXT, "Beta weighs 320 grams.")
    earlier = _claim(TEXT, "Alpha costs INR 1,299.")
    with pytest.raises(SpanValidationError) as unordered:
        _validate((later, earlier))
    assert unordered.value.code == "unordered"
