from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import Engine, insert, select
from sqlalchemy.engine import RowMapping

from speaker_claims.adapters.schema import answer_results, claim_results, runs
from speaker_claims.domain.runs import StoredAnswer, StoredClaim, StoredRun


class SqlRunStore:
    """Write runs with bound parameters. One transaction covers the whole run."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save(self, run: StoredRun) -> None:
        with self._engine.begin() as connection:
            connection.execute(insert(runs).values(_run_row(run)))
            if run.answers:
                connection.execute(
                    insert(answer_results),
                    [_answer_row(run.run_id, answer) for answer in run.answers],
                )
            claim_rows = [
                _claim_row(run.run_id, claim)
                for answer in run.answers
                for claim in answer.claims
            ]
            if claim_rows:
                connection.execute(insert(claim_results), claim_rows)

    def get(self, run_id: UUID) -> StoredRun | None:
        with self._engine.connect() as connection:
            run_row = connection.execute(select(runs).where(runs.c.id == run_id)).mappings().first()
            if run_row is None:
                return None
            answer_rows = (
                connection.execute(
                    select(answer_results)
                    .where(answer_results.c.run_id == run_id)
                    .order_by(answer_results.c.answer_id)
                )
                .mappings()
                .all()
            )
            claim_rows = (
                connection.execute(
                    select(claim_results)
                    .where(claim_results.c.run_id == run_id)
                    .order_by(
                        claim_results.c.answer_id,
                        claim_results.c.start_char,
                        claim_results.c.end_char,
                    )
                )
                .mappings()
                .all()
            )
        return _assemble(run_row, list(answer_rows), list(claim_rows))


def _run_row(run: StoredRun) -> dict[str, object]:
    return {
        "id": run.run_id,
        "split": run.split,
        "agent_version": run.agent_version,
        "prompt_version": run.prompt_version,
        "dataset_version": run.dataset_version,
        "evaluator_version": run.evaluator_version,
        "fact_version": run.fact_version,
        "provider": run.provider,
        "model": run.model,
        "settings": run.settings,
        "source_revision": run.source_revision,
        "file_hashes": run.file_hashes,
        "created_at": run.created_at,
        "status": run.status,
        "answer_count": run.answer_count,
        "answer_error_count": run.answer_error_count,
        "accepted_prediction_count": run.accepted_prediction_count,
        "quality": run.quality,
    }


def _answer_row(run_id: UUID, answer: StoredAnswer) -> dict[str, object]:
    return {
        "id": uuid4(),
        "run_id": run_id,
        "answer_id": answer.answer_id,
        "status": answer.status,
        "error_code": answer.error_code,
        "error_message": answer.error_message,
        "latency_ms": answer.latency_ms,
        "trace_id": answer.trace_id,
    }


def _claim_row(run_id: UUID, claim: StoredClaim) -> dict[str, object]:
    return {
        "id": uuid4(),
        "run_id": run_id,
        "answer_id": claim.answer_id,
        "claim_id": claim.claim_id,
        "start_char": claim.start_char,
        "end_char": claim.end_char,
        "quote": claim.quote,
        "label": claim.label,
        "fact_ids": list(claim.fact_ids),
        "reason": claim.reason,
    }


def _assemble(
    run_row: RowMapping,
    answer_rows: list[RowMapping],
    claim_rows: list[RowMapping],
) -> StoredRun:
    claims_by_answer: dict[str, list[StoredClaim]] = {}
    for row in claim_rows:
        claims_by_answer.setdefault(row["answer_id"], []).append(
            StoredClaim(
                answer_id=row["answer_id"],
                claim_id=row["claim_id"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                quote=row["quote"],
                label=row["label"],
                fact_ids=tuple(row["fact_ids"]),
                reason=row["reason"],
            )
        )
    answers = tuple(
        StoredAnswer(
            answer_id=row["answer_id"],
            status=row["status"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            latency_ms=row["latency_ms"],
            trace_id=row["trace_id"],
            claims=tuple(claims_by_answer.get(row["answer_id"], ())),
        )
        for row in answer_rows
    )
    return StoredRun(
        run_id=run_row["id"],
        split=run_row["split"],
        agent_version=run_row["agent_version"],
        prompt_version=run_row["prompt_version"],
        dataset_version=run_row["dataset_version"],
        evaluator_version=run_row["evaluator_version"],
        fact_version=run_row["fact_version"],
        provider=run_row["provider"],
        model=run_row["model"],
        settings=dict(run_row["settings"]),
        source_revision=run_row["source_revision"],
        file_hashes=dict(run_row["file_hashes"]),
        created_at=run_row["created_at"],
        status=run_row["status"],
        answer_count=run_row["answer_count"],
        answer_error_count=run_row["answer_error_count"],
        accepted_prediction_count=run_row["accepted_prediction_count"],
        quality=None if run_row["quality"] is None else dict(run_row["quality"]),
        answers=answers,
    )
