from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from speaker_claims.adapters.csv_catalogue import CsvAnswerSource, CsvFactCatalogue
from speaker_claims.adapters.csv_gold import CsvGoldSource
from speaker_claims.adapters.openai_provider import DEFAULT_MODEL
from speaker_claims.application.evaluation import EvaluationService
from speaker_claims.composition import build_monitor, build_pipelines, build_tracer
from speaker_claims.config import load_settings
from speaker_claims.domain.runs import StoredAnswer, StoredClaim, StoredRun

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
RUN_ORDER = (
    ("v1", "development"),
    ("v2", "development"),
    ("v1", "check"),
    ("v2", "check"),
)


class MemoryStore:
    def save(self, run: StoredRun) -> None:
        return None

    def get(self, run_id: object) -> StoredRun | None:
        return None


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
    """Freeze both prompts, then write the four live-run artifacts."""
    settings = load_settings()
    catalogue = CsvFactCatalogue.from_path(DATA / "brand_facts.csv")
    service = EvaluationService(
        build_pipelines(settings.openai_api_key.get_secret_value(), catalogue),
        catalogue,
        CsvAnswerSource(DATA / "answers.csv"),
        CsvGoldSource(DATA / "development_gold.csv"),
        MemoryStore(),
        provider="openai",
        model=DEFAULT_MODEL,
        settings={
            "timeout_seconds": 15,
            "max_model_turns": 3,
            "max_in_flight": 3,
            "max_attempts": 2,
        },
        source_revision=_revision(),
        file_hashes=_file_hashes(),
        tracer=build_tracer(settings),
        monitor=build_monitor(settings),
    )
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


def _file_hashes() -> dict[str, str]:
    names = ("answers.csv", "brand_facts.csv", "development_gold.csv")
    return {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest() for name in names}


def _revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


if __name__ == "__main__":
    main()
