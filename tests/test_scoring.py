from pathlib import Path

import pytest

from speaker_claims.adapters.csv_gold import load_gold
from speaker_claims.application.scoring import score_run
from speaker_claims.domain.claims import PredictedClaim
from speaker_claims.domain.gold import GoldClaim
from speaker_claims.domain.outcome import AnswerOutcome

DATA = Path(__file__).resolve().parents[1] / "data"


def _gold(index: int, label: str = "supported", quote: str | None = None) -> GoldClaim:
    start = index * 10
    text = quote if quote is not None else f"claim{index}."
    return GoldClaim(
        answer_id="A01",
        claim_id=f"A01:{start}:{start + len(text)}",
        start_char=start,
        end_char=start + len(text),
        quote=text,
        expected_label=label,
        evidence_fact_ids=(),
        explanation="",
        difficulty_tags=(),
    )


def _prediction(
    gold: GoldClaim,
    label: str | None = None,
    quote: str | None = None,
) -> PredictedClaim:
    text = gold.quote if quote is None else quote
    return PredictedClaim(
        answer_id=gold.answer_id,
        start_char=gold.start_char,
        end_char=gold.end_char,
        quote=text,
        label=gold.expected_label if label is None else label,
        fact_ids=(),
        reason="Checked.",
        claim_id=f"{gold.answer_id}:{gold.start_char}:{gold.end_char}",
    )


def _extra(index: int, label: str = "unknown") -> PredictedClaim:
    start = 1000 + index
    return PredictedClaim(
        answer_id="A01",
        start_char=start,
        end_char=start + 4,
        quote="note",
        label=label,
        fact_ids=(),
        reason="Extra.",
        claim_id=f"A01:{start}:{start + 4}",
    )


def _ok(claims: tuple[PredictedClaim, ...], answer_id: str = "A01") -> AnswerOutcome:
    return AnswerOutcome(answer_id=answer_id, agent_version="v1", status="ok", claims=claims)


def _error(answer_id: str = "A01") -> AnswerOutcome:
    return AnswerOutcome(
        answer_id=answer_id,
        agent_version="v1",
        status="error",
        claims=(),
        error_code="provider_failure",
        error_message="The model request failed",
    )


def test_published_arithmetic_example() -> None:
    gold = tuple(_gold(index) for index in range(10))
    predictions = tuple(_prediction(claim) for claim in gold[:7])
    predictions += (_prediction(gold[7], label="contradicted"), _extra(0), _extra(1))
    quality = score_run("development", (_ok(predictions),), gold).quality
    assert quality is not None
    assert (quality.g, quality.p, quality.m, quality.c) == (10, 10, 8, 7)
    assert quality.extraction_precision == pytest.approx(0.80)
    assert quality.extraction_recall == pytest.approx(0.80)
    assert quality.extraction_f1 == pytest.approx(0.80)
    assert quality.joint_precision == pytest.approx(0.70)
    assert quality.joint_recall == pytest.approx(0.70)
    assert quality.joint_f1 == pytest.approx(0.70)
    assert quality.gold_coverage == pytest.approx(0.70)
    assert quality.confusion.at("no_gold", "missing_or_error") == 0
    assert quality.confusion.total == quality.g + len(quality.unmatched_prediction_ids)


def test_perfect_missed_extra_and_wrong_label() -> None:
    perfect_gold = (
        _gold(0, "supported"),
        _gold(1, "contradicted"),
        _gold(2, "unknown"),
    )
    perfect = score_run(
        "development",
        (_ok(tuple(_prediction(claim) for claim in perfect_gold)),),
        perfect_gold,
    ).quality
    assert perfect is not None
    assert perfect.joint_f1 == pytest.approx(1)
    assert perfect.macro_f1 == pytest.approx(1)
    assert perfect.answers[0].extraction_f1 == pytest.approx(1)

    missed_gold = (_gold(0), _gold(1))
    missed = score_run("development", (_ok((_prediction(missed_gold[0]),)),), missed_gold).quality
    assert missed is not None
    assert missed.missed_gold_ids == (missed_gold[1].claim_id,)
    assert missed.extraction_recall == pytest.approx(0.5)
    assert missed.extraction_f1 == pytest.approx(2 / 3)

    extra_gold = (_gold(0),)
    extra = score_run(
        "development",
        (_ok((_prediction(extra_gold[0]), _extra(0))),),
        extra_gold,
    ).quality
    assert extra is not None
    assert extra.extraction_precision == pytest.approx(0.5)
    assert extra.confusion.at("no_gold", "unknown") == 1

    wrong_gold = (_gold(0, "supported"),)
    wrong = score_run(
        "development",
        (_ok((_prediction(wrong_gold[0], label="contradicted"),)),),
        wrong_gold,
    ).quality
    assert wrong is not None
    assert wrong.m == 1
    assert wrong.c == 0
    assert wrong.extraction_f1 == pytest.approx(1)
    assert wrong.joint_f1 == 0
    assert wrong.matched_wrong_label_ids == (wrong_gold[0].claim_id,)
    assert wrong.confusion.at("supported", "contradicted") == 1


def test_error_answer_and_valid_empty_predictions() -> None:
    gold = (_gold(0),)
    leaked = (_prediction(gold[0]),)
    errored = score_run(
        "development",
        (AnswerOutcome("A01", "v1", "error", leaked, "timeout", "timed out"),),
        gold,
    )
    assert errored.answer_error_count == 1
    assert errored.accepted_prediction_count == 0
    assert errored.quality is not None
    assert errored.quality.p == 0
    assert errored.quality.joint_f1 == 0
    assert errored.quality.missed_gold_ids == (gold[0].claim_id,)

    empty = score_run("development", (_ok(()),), gold)
    assert empty.answer_error_count == 0
    assert empty.quality is not None
    assert empty.quality.p == 0
    assert empty.quality.missed_gold_ids == (gold[0].claim_id,)


def test_quote_must_match_before_a_span_counts() -> None:
    gold = (_gold(0, quote="Alpha."),)
    prediction = _prediction(gold[0], quote="Beta.")
    quality = score_run("development", (_ok((prediction,)),), gold).quality
    assert quality is not None
    assert quality.m == 0
    assert quality.missed_gold_ids == (gold[0].claim_id,)
    assert quality.unmatched_prediction_ids == (prediction.claim_id,)


def test_check_split_and_empty_schedule_have_null_quality() -> None:
    gold = (_gold(0),)
    outcome = _ok((_prediction(gold[0]),))
    check = score_run("check", (outcome,), gold)
    assert check.quality is None
    assert check.answer_count == 1
    assert check.accepted_prediction_count == 1
    empty = score_run("development", (), gold)
    assert empty.quality is None
    assert empty.answer_count == 0


def test_unscheduled_gold_is_not_scored() -> None:
    scheduled = _gold(0)
    other = GoldClaim(
        answer_id="A02",
        claim_id="A02:0:6",
        start_char=0,
        end_char=6,
        quote="Other.",
        expected_label="supported",
        evidence_fact_ids=(),
        explanation="",
        difficulty_tags=(),
    )
    quality = score_run(
        "development",
        (_ok((_prediction(scheduled),)),),
        (scheduled, other),
    ).quality
    assert quality is not None
    assert quality.g == 1
    assert quality.joint_f1 == pytest.approx(1)


def test_issued_gold_scores_perfect_when_predictions_copy_it() -> None:
    gold = load_gold(DATA / "development_gold.csv")
    by_answer: dict[str, list[PredictedClaim]] = {}
    for claim in gold:
        by_answer.setdefault(claim.answer_id, []).append(_prediction(claim))
    outcomes = tuple(
        _ok(tuple(claims), answer_id=answer_id) for answer_id, claims in sorted(by_answer.items())
    )
    quality = score_run("development", outcomes, gold).quality
    assert quality is not None
    assert quality.g == 376
    assert quality.m == 376
    assert quality.c == 376
    assert quality.joint_f1 == pytest.approx(1)
    assert quality.extraction_f1 == pytest.approx(1)
    assert quality.missed_gold_ids == ()


def test_duplicate_answer_outcomes_are_rejected() -> None:
    with pytest.raises(ValueError):
        score_run("development", (_ok(()), _ok(())), ())
