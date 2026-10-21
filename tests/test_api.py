import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from speaker_claims.adapters.csv_catalogue import CsvAnswerSource, CsvFactCatalogue
from speaker_claims.adapters.csv_gold import CsvGoldSource
from speaker_claims.adapters.run_store import SqlRunStore
from speaker_claims.api.app import create_app
from speaker_claims.application.evaluation import DATASET_VERSION, EvaluationService
from speaker_claims.domain.claims import AnswerRecord, PredictedClaim
from speaker_claims.domain.gold import GoldClaim
from speaker_claims.domain.outcome import AnswerOutcome
from speaker_claims.domain.runs import StoredAnswer, StoredRun

DATA = Path(__file__).resolve().parents[1] / "data"
DATABASE_CANDIDATES = (
    os.environ.get("TEST_DATABASE_URL", ""),
    "postgresql+psycopg://user:password@localhost:5432/speaker_claims",
    "postgresql+psycopg://user:password@localhost:5433/speaker_claims",
)


class ScriptedClassifier:
    def __init__(self, outcomes: dict[str, AnswerOutcome] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.seen: list[AnswerRecord] = []

    def classify(self, answer: AnswerRecord) -> AnswerOutcome:
        self.seen.append(answer)
        if answer.answer_id in self.outcomes:
            return self.outcomes[answer.answer_id]
        return AnswerOutcome(answer.answer_id, "v1", "ok", ())


class MemoryAnswers:
    def __init__(self, answers: tuple[AnswerRecord, ...]) -> None:
        self._answers = answers

    def by_split(self, split: str) -> tuple[AnswerRecord, ...]:
        return tuple(answer for answer in self._answers if answer.split == split)


class MemoryGold:
    def __init__(self, claims: tuple[GoldClaim, ...] = ()) -> None:
        self.claims = claims
        self.calls = 0

    def load(self) -> tuple[GoldClaim, ...]:
        self.calls += 1
        return self.claims


def _answer(
    answer_id: str,
    brand_id: str = "cleartrail",
    split: str = "development",
) -> AnswerRecord:
    return AnswerRecord(answer_id, split, brand_id, "What is the price?", "The price is INR 1,299.")


def _claim(answer_id: str, start: int) -> PredictedClaim:
    quote = "The price is INR 1,299."
    return PredictedClaim(
        answer_id=answer_id,
        start_char=start,
        end_char=start + len(quote),
        quote=quote,
        label="supported",
        fact_ids=("cleartrail.mini.price_2026",),
        reason="The catalogue matches.",
        claim_id=f"{answer_id}:{start}:{start + len(quote)}",
    )


def _body(**extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "answer_id": "A01",
        "brand_id": "cleartrail",
        "question": "What is the price?",
        "ai_answer": "The price is INR 1,299.",
        "agent_version": "v1",
    }
    payload.update(extra)
    return payload


@pytest.fixture(scope="session")
def engine():
    last_error: Exception | None = None
    chosen = ""
    candidate = None
    for url in DATABASE_CANDIDATES:
        if url == "":
            continue
        try:
            candidate = create_engine(url)
            with candidate.connect() as connection:
                connection.execute(text("SELECT 1"))
            chosen = url
            break
        except Exception as exc:
            last_error = exc
            if candidate is not None:
                candidate.dispose()
            candidate = None
    if candidate is None or chosen == "":
        raise RuntimeError("Postgres is not ready") from last_error
    os.environ["DATABASE_URL"] = chosen
    command.upgrade(Config("alembic.ini"), "head")
    yield candidate
    candidate.dispose()


@pytest.fixture
def store(engine) -> SqlRunStore:
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM claim_results"))
        connection.execute(text("DELETE FROM answer_results"))
        connection.execute(text("DELETE FROM runs"))
    return SqlRunStore(engine)


@pytest.fixture
def catalogue() -> CsvFactCatalogue:
    return CsvFactCatalogue.from_path(DATA / "brand_facts.csv")


def _service(
    store: SqlRunStore,
    catalogue: CsvFactCatalogue,
    **kwargs: object,
) -> EvaluationService:
    classifier = kwargs.get("classifier", ScriptedClassifier())
    answers = kwargs.get("answers", MemoryAnswers((_answer("A01"),)))
    gold = kwargs.get("gold", MemoryGold())
    return EvaluationService(
        {"v1": classifier},
        catalogue,
        answers,
        gold,
        store,
        provider="openai",
        model="gpt-4o-mini",
        settings={"timeout_seconds": 15, "max_model_turns": 3},
        source_revision="test",
        file_hashes={"answers.csv": "hash"},
    )


def _client(service: EvaluationService) -> TestClient:
    return TestClient(create_app(service))


def test_classify_rejects_gold_fields_and_unknown_brands(store, catalogue) -> None:
    classifier = ScriptedClassifier()
    client = _client(_service(store, catalogue, classifier=classifier))
    assert client.post("/classify", json=_body(expected_label="supported")).status_code == 422
    assert client.post("/classify", json=_body(claims=[])).status_code == 422
    missing = _body()
    del missing["ai_answer"]
    assert client.post("/classify", json=missing).status_code == 422
    unknown = client.post("/classify", json=_body(brand_id="northwind"))
    assert unknown.status_code == 404
    assert classifier.seen == []
    assert client.post("/classify", json=_body(agent_version="v2")).status_code == 422


def test_classify_maps_timeout_invalid_output_and_success(store, catalogue) -> None:
    success = _claim("A01", 0)
    classifier = ScriptedClassifier(
        {
            "A01": AnswerOutcome("A01", "v1", "ok", (success,)),
            "slow": AnswerOutcome(
                "slow",
                "v1",
                "error",
                (),
                "timeout",
                "The model request timed out",
            ),
            "bad": AnswerOutcome(
                "bad",
                "v1",
                "error",
                (),
                "invalid_output",
                "The model JSON was invalid",
            ),
        }
    )
    client = _client(_service(store, catalogue, classifier=classifier))
    timed_out = client.post("/classify", json=_body(answer_id="slow"))
    assert timed_out.status_code == 504
    assert timed_out.json()["detail"]["error_code"] == "timeout"
    invalid = client.post("/classify", json=_body(answer_id="bad"))
    assert invalid.status_code == 502
    assert invalid.json()["detail"]["error_code"] == "invalid_output"
    ok = client.post("/classify", json=_body())
    assert ok.status_code == 200
    assert ok.json()["claims"][0]["claim_id"] == success.claim_id
    assert "expected_label" not in ok.json()["claims"][0]


def test_evaluation_persists_success_empty_and_error_answers(store, catalogue) -> None:
    later = _claim("A01", 20)
    earlier = _claim("A01", 0)
    classifier = ScriptedClassifier(
        {
            "A01": AnswerOutcome("A01", "v1", "ok", (later, earlier)),
            "A02": AnswerOutcome("A02", "v1", "ok", ()),
            "A03": AnswerOutcome(
                "A03",
                "v1",
                "error",
                (),
                "provider_failure",
                "The model request failed",
            ),
        }
    )
    answers = MemoryAnswers((_answer("A02"), _answer("A01"), _answer("A03")))
    gold = MemoryGold()
    client = _client(_service(store, catalogue, classifier=classifier, answers=answers, gold=gold))
    created = client.post("/evaluation-runs", json={"agent_version": "v1", "split": "development"})
    assert created.status_code == 201
    body = created.json()
    assert body["answer_count"] == 3
    assert body["answer_error_count"] == 1
    assert body["accepted_prediction_count"] == 2
    assert body["quality"]["g"] == 0
    assert gold.calls == 1
    fetched = client.get(f"/evaluation-runs/{body['run_id']}")
    assert fetched.status_code == 200
    rows = fetched.json()["answers"]
    assert [row["answer_id"] for row in rows] == ["A01", "A02", "A03"]
    assert [claim["start_char"] for claim in rows[0]["claims"]] == [0, 20]
    assert rows[1]["claims"] == []
    assert rows[2]["status"] == "error"
    assert rows[2]["claims"] == []
    assert rows[2]["trace_id"] is None
    assert "expected_label" not in classifier.seen[0].__dict__


def test_check_run_keeps_quality_null_and_does_not_load_gold(store, catalogue) -> None:
    gold = MemoryGold((_gold_claim(),))
    answers = MemoryAnswers((_answer("A41", split="check"),))
    client = _client(_service(store, catalogue, answers=answers, gold=gold))
    created = client.post("/evaluation-runs", json={"agent_version": "v1", "split": "check"})
    assert created.status_code == 201
    assert created.json()["quality"] is None
    assert gold.calls == 0
    fetched = client.get(f"/evaluation-runs/{created.json()['run_id']}")
    assert fetched.json()["quality"] is None
    assert fetched.json()["dataset_version"] == DATASET_VERSION


def test_fixed_csv_splits_and_run_lookup_errors(store, catalogue) -> None:
    service = EvaluationService(
        {"v1": ScriptedClassifier()},
        catalogue,
        CsvAnswerSource(DATA / "answers.csv"),
        CsvGoldSource(DATA / "development_gold.csv"),
        store,
        provider="openai",
        model="gpt-4o-mini",
        settings={"timeout_seconds": 15},
        source_revision=None,
        file_hashes={"answers.csv": "hash"},
    )
    client = _client(service)
    development = client.post(
        "/evaluation-runs",
        json={"agent_version": "v1", "split": "development"},
    )
    assert development.status_code == 201
    assert development.json()["answer_count"] == 40
    assert development.json()["quality"]["g"] == 376
    check = client.post("/evaluation-runs", json={"agent_version": "v1", "split": "check"})
    assert check.json()["answer_count"] == 10
    assert check.json()["quality"] is None
    assert client.get("/evaluation-runs/not-a-uuid").status_code == 422
    assert client.get(f"/evaluation-runs/{uuid4()}").status_code == 404


def test_duplicate_answer_rows_roll_back(store) -> None:
    answer = StoredAnswer("A01", "ok", None, None, 0, None, ())
    run = StoredRun(
        run_id=uuid4(),
        split="development",
        agent_version="v1",
        prompt_version="v1",
        dataset_version=DATASET_VERSION,
        evaluator_version="sentence-spans-and-labels-v4",
        fact_version="speaker-catalogue-v4",
        provider="openai",
        model="gpt-4o-mini",
        settings={},
        source_revision=None,
        file_hashes={},
        created_at=datetime.now(UTC),
        status="completed",
        answer_count=2,
        answer_error_count=0,
        accepted_prediction_count=0,
        quality=None,
        answers=(answer, answer),
    )
    with pytest.raises(IntegrityError):
        store.save(run)
    assert store.get(run.run_id) is None


def _gold_claim() -> GoldClaim:
    return GoldClaim("A41", "A41:0:4", 0, 4, "Gold", "supported", (), "", ())
