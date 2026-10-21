import json
import threading
import time
from pathlib import Path

from speaker_claims.adapters.csv_catalogue import CsvFactCatalogue
from speaker_claims.application.claim_pipeline import V1ClaimPipeline
from speaker_claims.application.evaluation import EvaluationService
from speaker_claims.application.fact_tool import FactToolExecutor
from speaker_claims.application.reliability import ProviderGate, RetryingProvider, RetryPolicy
from speaker_claims.domain.claims import AnswerRecord
from speaker_claims.domain.errors import ProviderAuthError, ProviderRateLimit, ProviderTimeout
from speaker_claims.domain.outcome import AnswerOutcome
from speaker_claims.domain.runs import StoredRun
from speaker_claims.ports.llm import ChatMessage, ModelTurn

DATA = Path(__file__).resolve().parents[1] / "data"
PROMPT = "Return JSON only."


class ScriptProvider:
    def __init__(self, results: list[ModelTurn | Exception]) -> None:
        self._results = list(results)
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, messages: tuple[ChatMessage, ...], timeout_seconds: float) -> ModelTurn:
        with self._lock:
            self.calls += 1
            result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class SlowProvider:
    def __init__(self, delay: float) -> None:
        self.delay = delay

    def complete(self, messages: tuple[ChatMessage, ...], timeout_seconds: float) -> ModelTurn:
        time.sleep(self.delay)
        payload = json.loads(messages[-1].content)
        body = json.dumps({"answer_id": payload["answer_id"], "claims": []})
        return ModelTurn(tool_calls=(), content=body)


class MemoryStore:
    def __init__(self) -> None:
        self.saved: list[StoredRun] = []

    def save(self, run: StoredRun) -> None:
        self.saved.append(run)

    def get(self, run_id: object) -> StoredRun | None:
        for run in self.saved:
            if run.run_id == run_id:
                return run
        return None


class MemoryAnswers:
    def __init__(self, answers: tuple[AnswerRecord, ...]) -> None:
        self._answers = answers

    def by_split(self, split: str) -> tuple[AnswerRecord, ...]:
        return tuple(answer for answer in self._answers if answer.split == split)


class MemoryGold:
    def load(self) -> tuple:
        return ()


def _answer(answer_id: str) -> AnswerRecord:
    return AnswerRecord(
        answer_id,
        "development",
        "cleartrail",
        "What is the price?",
        "Think about the trip.",
    )


def _pipeline(provider: ScriptProvider, policy: RetryPolicy | None = None) -> tuple:
    wrapped = RetryingProvider(provider, policy=policy or RetryPolicy(sleep=lambda _delay: None))
    catalogue = CsvFactCatalogue.from_path(DATA / "brand_facts.csv")
    pipeline = V1ClaimPipeline(
        wrapped,
        FactToolExecutor(catalogue),
        catalogue.fact_ids(),
        PROMPT,
    )
    return pipeline, wrapped


def _final(answer_id: str = "A01") -> ModelTurn:
    return ModelTurn(tool_calls=(), content=json.dumps({"answer_id": answer_id, "claims": []}))


def test_transient_failures_retry_once_and_auth_does_not() -> None:
    sleeps: list[float] = []
    policy = RetryPolicy(sleep=sleeps.append)
    provider = ScriptProvider([ProviderTimeout(), _final()])
    pipeline, wrapped = _pipeline(provider, policy)
    outcome = pipeline.classify(_answer("A01"))
    assert outcome.status == "ok"
    assert wrapped.attempts == 2
    assert sleeps == [0.2]

    exhausted = ScriptProvider([ProviderRateLimit(), ProviderRateLimit()])
    pipeline, wrapped = _pipeline(exhausted, RetryPolicy(sleep=lambda _delay: None))
    assert pipeline.classify(_answer("A01")).error_code == "provider_failure"
    assert wrapped.attempts == 2

    denied = ScriptProvider([ProviderAuthError()])
    pipeline, wrapped = _pipeline(denied, RetryPolicy(sleep=lambda _delay: None))
    assert pipeline.classify(_answer("A01")).error_code == "authentication_failed"
    assert wrapped.attempts == 1


def test_malformed_output_is_not_retried() -> None:
    provider = ScriptProvider([ModelTurn(tool_calls=(), content="not json")])
    pipeline, wrapped = _pipeline(provider)
    outcome = pipeline.classify(_answer("A01"))
    assert outcome.error_code == "invalid_output"
    assert outcome.claims == ()
    assert wrapped.attempts == 1


def test_provider_calls_stay_within_three_in_flight() -> None:
    gate = ProviderGate(limit=3)
    provider = RetryingProvider(SlowProvider(0.15), gate=gate)
    catalogue = CsvFactCatalogue.from_path(DATA / "brand_facts.csv")
    pipeline = V1ClaimPipeline(provider, FactToolExecutor(catalogue), catalogue.fact_ids(), PROMPT)
    answers = tuple(_answer(f"A{index:02d}") for index in range(1, 7))
    service = EvaluationService(
        {"v1": pipeline},
        catalogue,
        MemoryAnswers(answers),
        MemoryGold(),
        MemoryStore(),
        provider="openai",
        model="gpt-4o-mini",
        settings={},
        source_revision=None,
        file_hashes={},
    )
    started = time.perf_counter()
    run = service.start_run("v1", "development")
    elapsed = time.perf_counter() - started
    assert run.answer_count == 6
    assert run.answer_error_count == 0
    assert [answer.answer_id for answer in run.answers] == [answer.answer_id for answer in answers]
    assert gate.max_in_flight <= 3
    assert gate.max_in_flight >= 2
    assert elapsed < 0.7


class RaisingClassifier:
    def classify(self, answer: AnswerRecord) -> AnswerOutcome:
        if answer.answer_id == "A02":
            raise RuntimeError("boom")
        return AnswerOutcome(answer.answer_id, "v1", "ok", ())


def test_one_answer_failure_does_not_drop_the_others() -> None:
    catalogue = CsvFactCatalogue.from_path(DATA / "brand_facts.csv")
    answers = (_answer("A03"), _answer("A01"), _answer("A02"))
    service = EvaluationService(
        {"v1": RaisingClassifier()},
        catalogue,
        MemoryAnswers(answers),
        MemoryGold(),
        MemoryStore(),
        provider="openai",
        model="gpt-4o-mini",
        settings={},
        source_revision=None,
        file_hashes={},
    )
    run = service.start_run("v1", "development")
    assert [answer.answer_id for answer in run.answers] == ["A01", "A02", "A03"]
    assert [answer.status for answer in run.answers] == ["ok", "error", "ok"]
    assert run.answers[1].error_message == "The answer failed"
    assert run.answer_count == 3
