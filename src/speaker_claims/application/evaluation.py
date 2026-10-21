from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from speaker_claims.application.scoring import quality_document, score_run
from speaker_claims.domain.claims import AnswerRecord, PredictedClaim
from speaker_claims.domain.errors import UnavailableAgentVersion, UnknownBrandError
from speaker_claims.domain.gold import GoldClaim
from speaker_claims.domain.outcome import AnswerOutcome
from speaker_claims.domain.runs import StoredAnswer, StoredClaim, StoredRun
from speaker_claims.ports.catalogue import FactCataloguePort
from speaker_claims.ports.classifier import ClassifierPort
from speaker_claims.ports.run_store import RunStorePort

DATASET_VERSION = "long-brand-answers-50-v4"
EVALUATOR_VERSION = "sentence-spans-and-labels-v4"
FACT_VERSION = "speaker-catalogue-v4"


class AnswerSource(Protocol):
    def by_split(self, split: str) -> tuple[AnswerRecord, ...]:
        """Return the fixed answers for one split, in file order."""


class GoldSource(Protocol):
    def load(self) -> tuple[GoldClaim, ...]:
        """Return development gold. Check runs must not call this."""


class EvaluationService:
    """Classify one answer or a fixed split. Gold is used only after classification."""

    def __init__(
        self,
        classifiers: dict[str, ClassifierPort],
        catalogue: FactCataloguePort,
        answers: AnswerSource,
        gold: GoldSource,
        store: RunStorePort,
        *,
        provider: str,
        model: str,
        settings: dict[str, object],
        source_revision: str | None,
        file_hashes: dict[str, str],
    ) -> None:
        self._classifiers = classifiers
        self._catalogue = catalogue
        self._answers = answers
        self._gold = gold
        self._store = store
        self._provider = provider
        self._model = model
        self._settings = settings
        self._source_revision = source_revision
        self._file_hashes = file_hashes

    def classify(self, answer: AnswerRecord, agent_version: str) -> AnswerOutcome:
        classifier = self._require(agent_version)
        self._catalogue.get_brand_facts(answer.brand_id)
        return classifier.classify(answer)

    def start_run(self, agent_version: str, split: str) -> StoredRun:
        classifier = self._require(agent_version)
        scheduled = tuple(sorted(self._answers.by_split(split), key=lambda item: item.answer_id))
        outcomes: list[AnswerOutcome] = []
        stored_answers: list[StoredAnswer] = []
        for answer in scheduled:
            outcome, latency_ms = self._classify_scheduled(answer, classifier, agent_version)
            outcomes.append(outcome)
            stored_answers.append(_stored_answer(outcome, latency_ms))
        gold = self._gold.load() if split == "development" else ()
        report = score_run(split, tuple(outcomes), gold)
        quality = None if report.quality is None else quality_document(report.quality)
        run = StoredRun(
            run_id=uuid4(),
            split=split,
            agent_version=agent_version,
            prompt_version=agent_version,
            dataset_version=DATASET_VERSION,
            evaluator_version=EVALUATOR_VERSION,
            fact_version=FACT_VERSION,
            provider=self._provider,
            model=self._model,
            settings=self._settings,
            source_revision=self._source_revision,
            file_hashes=self._file_hashes,
            created_at=datetime.now(UTC),
            status="completed",
            answer_count=report.answer_count,
            answer_error_count=report.answer_error_count,
            accepted_prediction_count=report.accepted_prediction_count,
            quality=quality,
            answers=tuple(stored_answers),
        )
        self._store.save(run)
        return run

    def get_run(self, run_id: UUID) -> StoredRun | None:
        return self._store.get(run_id)

    def _require(self, agent_version: str) -> ClassifierPort:
        try:
            return self._classifiers[agent_version]
        except KeyError as exc:
            raise UnavailableAgentVersion(agent_version) from exc

    def _classify_scheduled(
        self,
        answer: AnswerRecord,
        classifier: ClassifierPort,
        agent_version: str,
    ) -> tuple[AnswerOutcome, int]:
        try:
            self._catalogue.get_brand_facts(answer.brand_id)
        except UnknownBrandError:
            outcome = AnswerOutcome(
                answer_id=answer.answer_id,
                agent_version=agent_version,
                status="error",
                claims=(),
                error_code="unknown_brand",
                error_message=f"Unknown brand: {answer.brand_id}",
            )
            return outcome, 0
        started = time.perf_counter()
        outcome = classifier.classify(answer)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        if outcome.status == "error":
            outcome = AnswerOutcome(
                answer_id=outcome.answer_id,
                agent_version=outcome.agent_version,
                status="error",
                claims=(),
                error_code=outcome.error_code,
                error_message=outcome.error_message,
            )
        return outcome, elapsed_ms


def _stored_answer(outcome: AnswerOutcome, latency_ms: int) -> StoredAnswer:
    claims = ()
    if outcome.status == "ok":
        ordered = sorted(outcome.claims, key=lambda claim: (claim.start_char, claim.end_char))
        claims = tuple(_stored_claim(claim) for claim in ordered)
    return StoredAnswer(
        answer_id=outcome.answer_id,
        status=outcome.status,
        error_code=outcome.error_code,
        error_message=outcome.error_message,
        latency_ms=latency_ms,
        trace_id=None,
        claims=claims,
    )


def _stored_claim(claim: PredictedClaim) -> StoredClaim:
    return StoredClaim(
        answer_id=claim.answer_id,
        claim_id=claim.claim_id,
        start_char=claim.start_char,
        end_char=claim.end_char,
        quote=claim.quote,
        label=claim.label,
        fact_ids=claim.fact_ids,
        reason=claim.reason,
    )
