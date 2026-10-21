import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from speaker_claims.adapters.csv_catalogue import CsvFactCatalogue
from speaker_claims.api.app import create_app
from speaker_claims.application.claim_pipeline import V2ClaimPipeline
from speaker_claims.application.evaluation import EvaluationService
from speaker_claims.application.fact_tool import FactToolExecutor
from speaker_claims.composition import build_pipelines, load_v1_prompt, load_v2_prompt
from speaker_claims.domain.claims import AnswerRecord
from speaker_claims.domain.gold import GoldClaim
from speaker_claims.domain.outcome import AnswerOutcome
from speaker_claims.domain.runs import StoredAnswer, StoredClaim, StoredRun
from speaker_claims.ports.llm import ChatMessage, ModelTurn
from speaker_claims.reports import write_artifact

DATA = Path(__file__).resolve().parents[1] / "data"
ANCHOR = "Match brand, model, market, date, units, and stated conditions. "
TRANSFER_RULE = (
    "Do not transfer a fact across a different model, variant, feature, date, or condition; "
    "silence about that target is unknown unless a supplied fact explicitly refutes it."
)


class Scripted:
    def classify(self, answer: AnswerRecord) -> AnswerOutcome:
        return AnswerOutcome(answer.answer_id, "v2", "ok", ())


class MemoryGold:
    def load(self) -> tuple[GoldClaim, ...]:
        return ()


class MemoryAnswers:
    def by_split(self, split: str) -> tuple[AnswerRecord, ...]:
        return ()


class MemoryStore:
    def save(self, run: StoredRun) -> None:
        return None

    def get(self, run_id: object) -> StoredRun | None:
        return None


class IdleProvider:
    def complete(self, messages: tuple[ChatMessage, ...], timeout_seconds: float) -> ModelTurn:
        raise AssertionError("opening_messages must not call the model")


def test_v2_prompt_adds_only_the_transfer_rule() -> None:
    v1 = load_v1_prompt()
    v2 = load_v2_prompt()
    assert TRANSFER_RULE not in v1
    assert v2 == v1.replace(ANCHOR, f"{ANCHOR}{TRANSFER_RULE} ", 1)
    assert "development_gold" not in v2
    assert "expected_label" not in v2
    assert "condition_swap" not in v2


def test_v2_replaces_only_the_opening_prompt() -> None:
    catalogue = CsvFactCatalogue.from_path(DATA / "brand_facts.csv")
    pipeline = V2ClaimPipeline(
        IdleProvider(),
        FactToolExecutor(catalogue),
        catalogue.fact_ids(),
        load_v2_prompt(),
    )
    answer = AnswerRecord("A01", "development", "cleartrail", "What is the price?", "A claim.")
    messages = pipeline.opening_messages(answer)
    assert pipeline.version == "v2"
    assert messages[0].content == load_v2_prompt()
    assert json.loads(messages[1].content) == {
        "answer_id": "A01",
        "brand_id": "cleartrail",
        "question": "What is the price?",
        "ai_answer": "A claim.",
    }
    pipelines = build_pipelines("sk-test", catalogue, client=SimpleNamespace())
    assert pipelines["v1"].version == "v1"
    assert pipelines["v2"].version == "v2"


def test_registered_v2_is_accepted() -> None:
    catalogue = CsvFactCatalogue.from_path(DATA / "brand_facts.csv")
    scripted = Scripted()
    service = EvaluationService(
        {"v1": scripted, "v2": scripted},
        catalogue,
        MemoryAnswers(),
        MemoryGold(),
        MemoryStore(),
        provider="openai",
        model="gpt-4o-mini",
        settings={},
        source_revision=None,
        file_hashes={},
    )
    client = TestClient(create_app(service))
    response = client.post(
        "/classify",
        json={
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "What is the price?",
            "ai_answer": "A claim.",
            "agent_version": "v2",
        },
    )
    assert response.status_code == 200
    assert response.json()["agent_version"] == "v2"


def test_check_artifact_keeps_quality_null(tmp_path: Path) -> None:
    claim = StoredClaim("A41", "A41:0:7", 0, 7, "A claim.", "unknown", (), "No fact applies.")
    answer = StoredAnswer("A41", "ok", None, None, 12, "trace-1", (claim,))
    run = StoredRun(
        run_id=uuid4(),
        split="check",
        agent_version="v2",
        prompt_version="v2",
        dataset_version="long-brand-answers-50-v4",
        evaluator_version="sentence-spans-and-labels-v4",
        fact_version="speaker-catalogue-v4",
        provider="openai",
        model="gpt-4o-mini",
        settings={"timeout_seconds": 15},
        source_revision=None,
        file_hashes={"answers.csv": "hash"},
        created_at=datetime(2026, 10, 21, tzinfo=UTC),
        status="completed",
        answer_count=1,
        answer_error_count=0,
        accepted_prediction_count=1,
        quality=None,
        answers=(answer,),
    )
    path = tmp_path / "v2-check.json"
    write_artifact(run, path)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["quality"] is None
    assert document["answers"][0]["claims"][0]["quote"] == "A claim."
    assert "ai_answer" not in path.read_text(encoding="utf-8")
    assert "expected_label" not in path.read_text(encoding="utf-8")
