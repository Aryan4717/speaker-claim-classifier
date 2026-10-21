from __future__ import annotations

import json
from pathlib import Path

from speaker_claims.domain.runs import StoredAnswer, StoredClaim, StoredRun
from speaker_claims.runtime import build_service

REPORTS = Path(__file__).resolve().parents[2] / "reports"
RUN_ORDER = (
    ("v1", "development"),
    ("v2", "development"),
    ("v1", "check"),
    ("v2", "check"),
)


def artifact(run: StoredRun) -> dict[str, object]:
    """JSON evidence for one live run. Check runs keep quality null."""
    return {
        "run_id": str(run.run_id),
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
        "created_at": run.created_at.isoformat(),
        "status": run.status,
        "answer_count": run.answer_count,
        "answer_error_count": run.answer_error_count,
        "accepted_prediction_count": run.accepted_prediction_count,
        "quality": run.quality,
        "answers": [_answer(answer) for answer in run.answers],
    }


def write_artifact(run: StoredRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact(run), indent=2, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8")


def main() -> None:
    """Store both versions in Postgres and write the four live-run artifacts."""
    service = build_service()
    for version, split in RUN_ORDER:
        run = service.start_run(version, split)
        path = REPORTS / f"{version}-{split}.json"
        write_artifact(run, path)
        quality = run.quality or {}
        joint = quality.get("joint_f1")
        print(
            f"{path.name} answers={run.answer_count} "
            f"errors={run.answer_error_count} joint_f1={joint}",
            flush=True,
        )


def _answer(answer: StoredAnswer) -> dict[str, object]:
    return {
        "answer_id": answer.answer_id,
        "status": answer.status,
        "error_code": answer.error_code,
        "error_message": answer.error_message,
        "latency_ms": answer.latency_ms,
        "trace_id": answer.trace_id,
        "claims": [_claim(claim) for claim in answer.claims],
    }


def _claim(claim: StoredClaim) -> dict[str, object]:
    return {
        "claim_id": claim.claim_id,
        "start_char": claim.start_char,
        "end_char": claim.end_char,
        "quote": claim.quote,
        "label": claim.label,
        "fact_ids": list(claim.fact_ids),
        "reason": claim.reason,
    }


if __name__ == "__main__":
    main()
