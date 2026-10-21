from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from speaker_claims.application.monitoring import (
    NoOpMonitor,
    ReportedFailure,
    already_reported,
    capture_failure,
    monitor_scope,
)
from speaker_claims.application.scoring import ScoreReport, quality_document, score_run
from speaker_claims.application.tracing import NoOpTracer, open_answer
from speaker_claims.domain.claims import AnswerRecord, PredictedClaim
from speaker_claims.domain.errors import UnavailableAgentVersion, UnknownBrandError
from speaker_claims.domain.gold import GoldClaim
from speaker_claims.domain.outcome import AnswerOutcome
from speaker_claims.domain.runs import StoredAnswer, StoredClaim, StoredRun
from speaker_claims.ports.catalogue import FactCataloguePort
from speaker_claims.ports.classifier import ClassifierPort
from speaker_claims.ports.monitor import ErrorMonitorPort
from speaker_claims.ports.run_store import RunStorePort
from speaker_claims.ports.tracer import TracerPort

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
        tracer: TracerPort | None = None,
        monitor: ErrorMonitorPort | None = None,
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
        self._tracer = tracer if tracer is not None else NoOpTracer()
        self._monitor = monitor if monitor is not None else NoOpMonitor()

    @property
    def monitor(self) -> ErrorMonitorPort:
        return self._monitor

    def classify(self, answer: AnswerRecord, agent_version: str) -> AnswerOutcome:
        classifier = self._require(agent_version)
        with self._watch(answer, agent_version, None):
            with self._open(answer, agent_version, None) as trace_id:
                try:
                    self._catalogue.get_brand_facts(answer.brand_id)
                except UnknownBrandError:
                    self._tracer.finish_answer(
                        trace_id,
                        status="error",
                        error_code="unknown_brand",
                    )
                    self._tracer.flush()
                    raise
                try:
                    outcome = classifier.classify(answer)
                except Exception:
                    self._tracer.finish_answer(
                        trace_id,
                        status="error",
                        error_code="provider_failure",
                    )
                    self._tracer.flush()
                    raise
                tagged = _tagged(outcome, trace_id)
                self._report(tagged, answer, None)
                self._tracer.finish_answer(
                    trace_id,
                    status=tagged.status,
                    error_code=tagged.error_code,
                )
                self._tracer.flush()
                return tagged

    def start_run(self, agent_version: str, split: str) -> StoredRun:
        classifier = self._require(agent_version)
        run_id = uuid4()
        scheduled = tuple(sorted(self._answers.by_split(split), key=lambda item: item.answer_id))
        results = asyncio.run(
            self._classify_all(scheduled, classifier, agent_version, str(run_id))
        )
        outcomes = tuple(outcome for outcome, _latency in results)
        stored_answers = tuple(
            _stored_answer(outcome, latency_ms) for outcome, latency_ms in results
        )
        gold = self._gold.load() if split == "development" else ()
        report = score_run(split, tuple(outcomes), gold)
        self._attach_scores(split, report, outcomes)
        self._tracer.flush()
        quality = None if report.quality is None else quality_document(report.quality)
        run = StoredRun(
            run_id=run_id,
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
            answers=stored_answers,
        )
        self._store.save(run)
        return run

    def get_run(self, run_id: UUID) -> StoredRun | None:
        return self._store.get(run_id)

    async def _classify_all(
        self,
        answers: tuple[AnswerRecord, ...],
        classifier: ClassifierPort,
        agent_version: str,
        run_id: str,
    ) -> list[tuple[AnswerOutcome, int]]:
        tasks = [
            asyncio.to_thread(
                self._classify_scheduled,
                answer,
                classifier,
                agent_version,
                run_id,
            )
            for answer in answers
        ]
        return list(await asyncio.gather(*tasks))

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
        run_id: str,
    ) -> tuple[AnswerOutcome, int]:
        with self._watch(answer, agent_version, run_id):
            return self._classify_watched(answer, classifier, agent_version, run_id)

    def _classify_watched(
        self,
        answer: AnswerRecord,
        classifier: ClassifierPort,
        agent_version: str,
        run_id: str,
    ) -> tuple[AnswerOutcome, int]:
        with self._open(answer, agent_version, run_id) as trace_id:
            try:
                self._catalogue.get_brand_facts(answer.brand_id)
            except UnknownBrandError:
                self._tracer.finish_answer(trace_id, status="error", error_code="unknown_brand")
                outcome = AnswerOutcome(
                    answer_id=answer.answer_id,
                    agent_version=agent_version,
                    status="error",
                    claims=(),
                    error_code="unknown_brand",
                    error_message=f"Unknown brand: {answer.brand_id}",
                    trace_id=trace_id,
                )
                self._report(outcome, answer, run_id)
                return outcome, 0
            started = time.perf_counter()
            try:
                outcome = classifier.classify(answer)
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                capture_failure(exc)
                self._tracer.finish_answer(
                    trace_id,
                    status="error",
                    error_code="provider_failure",
                )
                outcome = AnswerOutcome(
                    answer_id=answer.answer_id,
                    agent_version=agent_version,
                    status="error",
                    claims=(),
                    error_code="provider_failure",
                    error_message="The answer failed",
                    trace_id=trace_id,
                )
                self._report(outcome, answer, run_id)
                return outcome, elapsed_ms
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            tagged = _tagged(outcome, trace_id)
            self._report(tagged, answer, run_id)
            self._tracer.finish_answer(
                trace_id,
                status=tagged.status,
                error_code=tagged.error_code,
            )
            return tagged, elapsed_ms

    def _open(self, answer: AnswerRecord, agent_version: str, run_id: str | None):
        return open_answer(
            self._tracer,
            answer_id=answer.answer_id,
            brand_id=answer.brand_id,
            split=answer.split or "request",
            agent_version=agent_version,
            model=self._model,
            run_id=run_id,
        )

    def _watch(self, answer: AnswerRecord, agent_version: str, run_id: str | None):
        return monitor_scope(
            self._monitor,
            run_id=run_id,
            answer_id=answer.answer_id,
            split=answer.split or "request",
            agent_version=agent_version,
        )

    def _report(self, outcome: AnswerOutcome, answer: AnswerRecord, run_id: str | None) -> None:
        if outcome.status != "error" or already_reported():
            return
        if outcome.error_code == "unknown_brand" and run_id is not None:
            self._monitor.breadcrumb(
                category="answer",
                message="unknown_brand",
                data={"answer_id": answer.answer_id, "brand_id": answer.brand_id},
            )
            return
        if outcome.error_code in {"timeout", "provider_failure", "authentication_failed"}:
            self._monitor.capture_exception(ReportedFailure(outcome.error_code))

    def _attach_scores(
        self,
        split: str,
        report: ScoreReport,
        outcomes: tuple[AnswerOutcome, ...],
    ) -> None:
        if split != "development" or report.quality is None:
            return
        by_answer = {item.answer_id: item for item in report.quality.answers}
        for outcome in outcomes:
            quality = by_answer.get(outcome.answer_id)
            if quality is None:
                continue
            self._tracer.score_answer(
                outcome.trace_id,
                extraction_f1=quality.extraction_f1,
                joint_f1=quality.joint_f1,
            )


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
        trace_id=outcome.trace_id,
        claims=claims,
    )


def _tagged(outcome: AnswerOutcome, trace_id: str | None) -> AnswerOutcome:
    return AnswerOutcome(
        answer_id=outcome.answer_id,
        agent_version=outcome.agent_version,
        status=outcome.status,
        claims=outcome.claims if outcome.status == "ok" else (),
        error_code=outcome.error_code,
        error_message=outcome.error_message,
        trace_id=trace_id,
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
