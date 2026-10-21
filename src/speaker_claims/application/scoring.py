from __future__ import annotations

from dataclasses import dataclass

from speaker_claims.domain.claims import LABELS, PredictedClaim
from speaker_claims.domain.gold import GoldClaim
from speaker_claims.domain.outcome import AnswerOutcome

_ROWS = ("supported", "contradicted", "unknown", "no_gold")
_COLUMNS = ("supported", "contradicted", "unknown", "missing_or_error")


@dataclass(frozen=True)
class LabelMetrics:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class AnswerQuality:
    answer_id: str
    g: int
    p: int
    m: int
    c: int
    extraction_f1: float
    joint_f1: float


@dataclass(frozen=True)
class ConfusionMatrix:
    counts: tuple[tuple[int, ...], ...]

    def at(self, row: str, column: str) -> int:
        return self.counts[_ROWS.index(row)][_COLUMNS.index(column)]

    @property
    def total(self) -> int:
        return sum(sum(row) for row in self.counts)


@dataclass(frozen=True)
class QualityMetrics:
    g: int
    p: int
    m: int
    c: int
    extraction_precision: float
    extraction_recall: float
    extraction_f1: float
    joint_precision: float
    joint_recall: float
    joint_f1: float
    gold_coverage: float
    macro_f1: float
    missed_gold_ids: tuple[str, ...]
    unmatched_prediction_ids: tuple[str, ...]
    matched_wrong_label_ids: tuple[str, ...]
    supported: LabelMetrics
    contradicted: LabelMetrics
    unknown: LabelMetrics
    confusion: ConfusionMatrix
    answers: tuple[AnswerQuality, ...]


@dataclass(frozen=True)
class ScoreReport:
    split: str
    answer_count: int
    answer_error_count: int
    accepted_prediction_count: int
    quality: QualityMetrics | None


def score_run(
    split: str,
    outcomes: tuple[AnswerOutcome, ...],
    gold: tuple[GoldClaim, ...] = (),
) -> ScoreReport:
    """Score exact spans. Check runs and empty schedules carry no quality metrics."""
    _require_unique_answers(outcomes)
    accepted = _accepted_predictions(outcomes)
    report = ScoreReport(
        split=split,
        answer_count=len(outcomes),
        answer_error_count=sum(1 for outcome in outcomes if outcome.status == "error"),
        accepted_prediction_count=len(accepted),
        quality=None,
    )
    if split == "check" or not outcomes:
        return report
    scheduled = {outcome.answer_id for outcome in outcomes}
    scheduled_gold = tuple(claim for claim in gold if claim.answer_id in scheduled)
    return ScoreReport(
        split=report.split,
        answer_count=report.answer_count,
        answer_error_count=report.answer_error_count,
        accepted_prediction_count=report.accepted_prediction_count,
        quality=_quality(outcomes, accepted, scheduled_gold),
    )


def _require_unique_answers(outcomes: tuple[AnswerOutcome, ...]) -> None:
    seen: set[str] = set()
    for outcome in outcomes:
        if outcome.answer_id in seen:
            raise ValueError(f"Duplicate answer outcome: {outcome.answer_id}")
        seen.add(outcome.answer_id)


def _accepted_predictions(outcomes: tuple[AnswerOutcome, ...]) -> tuple[PredictedClaim, ...]:
    claims: list[PredictedClaim] = []
    for outcome in outcomes:
        if outcome.status != "ok":
            continue
        for claim in outcome.claims:
            if claim.label not in LABELS:
                raise ValueError(f"Unsupported label on {claim.claim_id}")
            claims.append(claim)
    return tuple(claims)


def _quality(
    outcomes: tuple[AnswerOutcome, ...],
    predictions: tuple[PredictedClaim, ...],
    gold: tuple[GoldClaim, ...],
) -> QualityMetrics:
    gold_by_span = {(claim.answer_id, claim.start_char, claim.end_char): claim for claim in gold}
    matched_gold: set[tuple[str, int, int]] = set()
    matched_pairs: list[tuple[GoldClaim, PredictedClaim]] = []
    unmatched: list[PredictedClaim] = []
    for prediction in predictions:
        key = (prediction.answer_id, prediction.start_char, prediction.end_char)
        gold_claim = gold_by_span.get(key)
        quote_matches = gold_claim is not None and prediction.quote == gold_claim.quote
        if gold_claim is None or not quote_matches or key in matched_gold:
            unmatched.append(prediction)
            continue
        matched_gold.add(key)
        matched_pairs.append((gold_claim, prediction))
    missed = tuple(sorted(claim.claim_id for claim in gold if _span(claim) not in matched_gold))
    wrong = tuple(
        sorted(
            gold_claim.claim_id
            for gold_claim, prediction in matched_pairs
            if gold_claim.expected_label != prediction.label
        )
    )
    correct = sum(
        1
        for gold_claim, prediction in matched_pairs
        if gold_claim.expected_label == prediction.label
    )
    g_count = len(gold)
    p_count = len(predictions)
    m_count = len(matched_pairs)
    per_label = {
        label: _label_metrics(label, gold, matched_pairs, unmatched) for label in LABELS
    }
    return QualityMetrics(
        g=g_count,
        p=p_count,
        m=m_count,
        c=correct,
        extraction_precision=_ratio(m_count, p_count),
        extraction_recall=_ratio(m_count, g_count),
        extraction_f1=_ratio(2 * m_count, p_count + g_count),
        joint_precision=_ratio(correct, p_count),
        joint_recall=_ratio(correct, g_count),
        joint_f1=_ratio(2 * correct, p_count + g_count),
        gold_coverage=_ratio(correct, g_count),
        macro_f1=sum(metrics.f1 for metrics in per_label.values()) / 3,
        missed_gold_ids=missed,
        unmatched_prediction_ids=tuple(sorted(claim.claim_id for claim in unmatched)),
        matched_wrong_label_ids=wrong,
        supported=per_label["supported"],
        contradicted=per_label["contradicted"],
        unknown=per_label["unknown"],
        confusion=_confusion(gold, matched_pairs, unmatched, matched_gold),
        answers=tuple(_answer_quality(outcome, gold, predictions) for outcome in outcomes),
    )


def _label_metrics(
    label: str,
    gold: tuple[GoldClaim, ...],
    matched_pairs: list[tuple[GoldClaim, PredictedClaim]],
    unmatched: list[PredictedClaim],
) -> LabelMetrics:
    true_positive = sum(
        1
        for gold_claim, prediction in matched_pairs
        if gold_claim.expected_label == label and prediction.label == label
    )
    false_positive = sum(1 for prediction in unmatched if prediction.label == label)
    false_positive += sum(
        1
        for gold_claim, prediction in matched_pairs
        if prediction.label == label and gold_claim.expected_label != label
    )
    false_negative = sum(1 for claim in gold if claim.expected_label == label) - true_positive
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return LabelMetrics(
        precision=precision,
        recall=recall,
        f1=_ratio(2 * true_positive, (2 * true_positive) + false_positive + false_negative),
    )


def _confusion(
    gold: tuple[GoldClaim, ...],
    matched_pairs: list[tuple[GoldClaim, PredictedClaim]],
    unmatched: list[PredictedClaim],
    matched_gold: set[tuple[str, int, int]],
) -> ConfusionMatrix:
    counts = [[0, 0, 0, 0] for _ in _ROWS]
    for gold_claim, prediction in matched_pairs:
        counts[_ROWS.index(gold_claim.expected_label)][_COLUMNS.index(prediction.label)] += 1
    for claim in gold:
        if _span(claim) not in matched_gold:
            counts[_ROWS.index(claim.expected_label)][_COLUMNS.index("missing_or_error")] += 1
    for prediction in unmatched:
        counts[_ROWS.index("no_gold")][_COLUMNS.index(prediction.label)] += 1
    frozen = tuple(tuple(row) for row in counts)
    return ConfusionMatrix(counts=frozen)


def _answer_quality(
    outcome: AnswerOutcome,
    gold: tuple[GoldClaim, ...],
    predictions: tuple[PredictedClaim, ...],
) -> AnswerQuality:
    answer_gold = tuple(claim for claim in gold if claim.answer_id == outcome.answer_id)
    answer_predictions = tuple(
        claim for claim in predictions if claim.answer_id == outcome.answer_id
    )
    matched = 0
    correct = 0
    gold_by_span = {_span(claim): claim for claim in answer_gold}
    seen: set[tuple[str, int, int]] = set()
    for prediction in answer_predictions:
        key = (prediction.answer_id, prediction.start_char, prediction.end_char)
        gold_claim = gold_by_span.get(key)
        if gold_claim is None or prediction.quote != gold_claim.quote or key in seen:
            continue
        seen.add(key)
        matched += 1
        if gold_claim.expected_label == prediction.label:
            correct += 1
    g_count = len(answer_gold)
    p_count = len(answer_predictions)
    return AnswerQuality(
        answer_id=outcome.answer_id,
        g=g_count,
        p=p_count,
        m=matched,
        c=correct,
        extraction_f1=_ratio(2 * matched, p_count + g_count),
        joint_f1=_ratio(2 * correct, p_count + g_count),
    )


def _span(claim: GoldClaim) -> tuple[str, int, int]:
    return (claim.answer_id, claim.start_char, claim.end_char)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def quality_document(metrics: QualityMetrics) -> dict[str, object]:
    """JSON-ready score payload. Check runs do not call this."""
    return {
        "g": metrics.g,
        "p": metrics.p,
        "m": metrics.m,
        "c": metrics.c,
        "extraction_precision": metrics.extraction_precision,
        "extraction_recall": metrics.extraction_recall,
        "extraction_f1": metrics.extraction_f1,
        "joint_precision": metrics.joint_precision,
        "joint_recall": metrics.joint_recall,
        "joint_f1": metrics.joint_f1,
        "gold_coverage": metrics.gold_coverage,
        "macro_f1": metrics.macro_f1,
        "missed_gold_ids": list(metrics.missed_gold_ids),
        "unmatched_prediction_ids": list(metrics.unmatched_prediction_ids),
        "matched_wrong_label_ids": list(metrics.matched_wrong_label_ids),
        "supported": _label_document(metrics.supported),
        "contradicted": _label_document(metrics.contradicted),
        "unknown": _label_document(metrics.unknown),
        "confusion": {
            "rows": list(_ROWS),
            "columns": list(_COLUMNS),
            "counts": [list(row) for row in metrics.confusion.counts],
        },
        "answers": [
            {
                "answer_id": answer.answer_id,
                "g": answer.g,
                "p": answer.p,
                "m": answer.m,
                "c": answer.c,
                "extraction_f1": answer.extraction_f1,
                "joint_f1": answer.joint_f1,
            }
            for answer in metrics.answers
        ],
    }


def _label_document(metrics: LabelMetrics) -> dict[str, float]:
    return {"precision": metrics.precision, "recall": metrics.recall, "f1": metrics.f1}
