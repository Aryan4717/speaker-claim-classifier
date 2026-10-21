import ast
import json
import threading
from datetime import date
from pathlib import Path

import langfuse
import pytest

from speaker_claims.adapters.langfuse_tracer import LangfuseTracer
from speaker_claims.application.claim_pipeline import V1ClaimPipeline
from speaker_claims.application.evaluation import EvaluationService
from speaker_claims.application.fact_tool import FactToolExecutor
from speaker_claims.application.tracing import NoOpTracer
from speaker_claims.composition import build_tracer
from speaker_claims.config import Settings
from speaker_claims.domain.claims import AnswerRecord
from speaker_claims.domain.errors import UnknownBrandError
from speaker_claims.domain.facts import BrandFact
from speaker_claims.domain.gold import GoldClaim
from speaker_claims.domain.outcome import AnswerOutcome
from speaker_claims.domain.runs import StoredRun
from speaker_claims.ports.llm import ChatMessage, ModelTurn, ToolCall

ROOT = Path(__file__).resolve().parents[1]
TEXT = "Think about the trip. The price is INR 1,299."
QUOTE = "The price is INR 1,299."
START = TEXT.index(QUOTE)
PROMPT = "Return JSON only."


class RecordingTracer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0
        self.traces: list[dict[str, object]] = []
        self.generations: list[dict[str, object]] = []
        self.tools: list[dict[str, object]] = []
        self.scores: list[tuple[str, float, float]] = []
        self.finished: list[dict[str, object]] = []
        self.flushes = 0

    def start_answer(
        self,
        *,
        answer_id: str,
        brand_id: str,
        split: str,
        agent_version: str,
        model: str,
        run_id: str | None,
    ) -> str:
        with self._lock:
            self._next += 1
            trace_id = f"trace-{self._next}"
            self.traces.append(
                {
                    "trace_id": trace_id,
                    "answer_id": answer_id,
                    "brand_id": brand_id,
                    "split": split,
                    "agent_version": agent_version,
                    "model": model,
                    "run_id": run_id,
                }
            )
            return trace_id

    def record_generation(
        self,
        trace_id: str | None,
        *,
        model: str,
        messages: tuple[dict[str, object], ...],
        output: str | None,
    ) -> None:
        with self._lock:
            self.generations.append(
                {
                    "trace_id": trace_id,
                    "model": model,
                    "messages": messages,
                    "output": output,
                }
            )

    def record_tool(
        self,
        trace_id: str | None,
        *,
        name: str,
        arguments: dict[str, object] | None,
        output: dict[str, object] | None,
        error_code: str | None,
    ) -> None:
        with self._lock:
            self.tools.append(
                {
                    "trace_id": trace_id,
                    "name": name,
                    "arguments": arguments,
                    "output": output,
                    "error_code": error_code,
                }
            )

    def finish_answer(
        self,
        trace_id: str | None,
        *,
        status: str,
        error_code: str | None,
    ) -> None:
        with self._lock:
            self.finished.append(
                {"trace_id": trace_id, "status": status, "error_code": error_code}
            )

    def score_answer(
        self,
        trace_id: str | None,
        *,
        extraction_f1: float,
        joint_f1: float,
    ) -> None:
        if trace_id is None:
            return
        with self._lock:
            self.scores.append((trace_id, extraction_f1, joint_f1))

    def flush(self) -> None:
        with self._lock:
            self.flushes += 1


class MemoryCatalogue:
    def __init__(self, facts: tuple[BrandFact, ...]) -> None:
        self._facts = facts

    def get_brand_facts(self, brand_id: str) -> tuple[BrandFact, ...]:
        matched = tuple(fact for fact in self._facts if fact.brand_id == brand_id)
        if not matched:
            raise UnknownBrandError(brand_id)
        return matched


class MemoryAnswers:
    def __init__(self, answers: tuple[AnswerRecord, ...]) -> None:
        self._answers = answers

    def by_split(self, split: str) -> tuple[AnswerRecord, ...]:
        return tuple(answer for answer in self._answers if answer.split == split)


class MemoryGold:
    def __init__(self, claims: tuple[GoldClaim, ...] = ()) -> None:
        self._claims = claims
        self.loads = 0

    def load(self) -> tuple[GoldClaim, ...]:
        self.loads += 1
        return self._claims


class GuardGold:
    def load(self) -> tuple[GoldClaim, ...]:
        raise AssertionError("check run loaded gold")


class MemoryStore:
    def __init__(self) -> None:
        self.saved: list[StoredRun] = []

    def save(self, run: StoredRun) -> None:
        self.saved.append(run)

    def get(self, run_id: object) -> StoredRun | None:
        return None


class ScriptProvider:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = list(turns)

    def complete(self, messages: tuple[ChatMessage, ...], timeout_seconds: float) -> ModelTurn:
        return self._turns.pop(0)


class OkClassifier:
    def classify(self, answer: AnswerRecord) -> AnswerOutcome:
        return AnswerOutcome(answer.answer_id, "v1", "ok", ())


class RaisingClassifier:
    def classify(self, answer: AnswerRecord) -> AnswerOutcome:
        if answer.answer_id == "A02":
            raise RuntimeError("boom")
        return AnswerOutcome(answer.answer_id, "v1", "ok", ())


def _fact() -> BrandFact:
    return BrandFact(
        fact_id="cleartrail.mini.price_2026",
        brand_id="cleartrail",
        model_id="mini",
        field="price_2026",
        raw_value="1299",
        value_type="number",
        unit="INR",
        condition="India list price",
        valid_from=date(2026, 1, 1),
        valid_to=date(2026, 12, 31),
    )


def _answer(split: str, answer_id: str = "A01", brand_id: str = "cleartrail") -> AnswerRecord:
    return AnswerRecord(answer_id, split, brand_id, "What is the price?", TEXT)


def _gold() -> GoldClaim:
    return GoldClaim(
        answer_id="A01",
        claim_id=f"A01:{START}:{START + len(QUOTE)}",
        start_char=START,
        end_char=START + len(QUOTE),
        quote=QUOTE,
        expected_label="supported",
        evidence_fact_ids=("cleartrail.mini.price_2026",),
        explanation="The catalogue lists this price.",
        difficulty_tags=(),
    )


def _pipeline(provider: ScriptProvider) -> V1ClaimPipeline:
    catalogue = MemoryCatalogue((_fact(),))
    return V1ClaimPipeline(
        provider,
        FactToolExecutor(catalogue),
        frozenset({"cleartrail.mini.price_2026"}),
        PROMPT,
    )


def _service(
    classifier: object,
    answers: tuple[AnswerRecord, ...],
    gold: object,
    tracer: object,
) -> EvaluationService:
    return EvaluationService(
        {"v1": classifier},
        MemoryCatalogue((_fact(),)),
        MemoryAnswers(answers),
        gold,
        MemoryStore(),
        provider="openai",
        model="gpt-4o-mini",
        settings={},
        source_revision=None,
        file_hashes={},
        tracer=tracer,
    )


def _turns() -> list[ModelTurn]:
    tool = ModelTurn(
        tool_calls=(ToolCall("call_1", "get_brand_facts", '{"brand_id": "cleartrail"}'),),
        content=None,
    )
    final = ModelTurn(
        tool_calls=(),
        content=json.dumps(
            {
                "answer_id": "A01",
                "claims": [
                    {
                        "quote": QUOTE,
                        "label": "supported",
                        "fact_ids": ["cleartrail.mini.price_2026"],
                        "reason": "The catalogue price matches.",
                    }
                ],
            }
        ),
    )
    return [tool, final]


def test_development_trace_records_calls_and_scores() -> None:
    tracer = RecordingTracer()
    gold = MemoryGold((_gold(),))
    run = _service(
        _pipeline(ScriptProvider(_turns())),
        (_answer("development"),),
        gold,
        tracer,
    ).start_run("v1", "development")
    assert run.answers[0].trace_id == "trace-1"
    assert tracer.traces[0]["run_id"] == str(run.run_id)
    assert tracer.traces[0]["split"] == "development"
    assert len(tracer.generations) == 2
    assert tracer.generations[0]["model"] == "gpt-4o-mini"
    assert tracer.tools[0]["name"] == "get_brand_facts"
    assert tracer.tools[0]["error_code"] is None
    assert tracer.scores == [("trace-1", 1.0, 1.0)]
    assert tracer.flushes == 1
    blob = json.dumps(
        {"generations": tracer.generations, "tools": tracer.tools, "traces": tracer.traces}
    )
    assert QUOTE in blob
    assert "expected_label" not in blob
    assert gold.loads == 1


def test_check_trace_has_no_scores() -> None:
    tracer = RecordingTracer()
    run = _service(
        _pipeline(ScriptProvider(_turns())),
        (_answer("check"),),
        GuardGold(),
        tracer,
    ).start_run("v1", "check")
    assert run.quality is None
    assert run.answers[0].trace_id == "trace-1"
    assert tracer.traces[0]["split"] == "check"
    assert len(tracer.generations) == 2
    assert tracer.tools[0]["name"] == "get_brand_facts"
    assert tracer.scores == []


def test_classify_traces_without_scores_and_unknown_brand_stays_a_trace() -> None:
    tracer = RecordingTracer()
    service = _service(OkClassifier(), (), MemoryGold(), tracer)
    outcome = service.classify(_answer("development"), "v1")
    assert outcome.trace_id == "trace-1"
    assert tracer.scores == []
    assert tracer.finished[0]["status"] == "ok"
    assert tracer.flushes == 1
    with pytest.raises(UnknownBrandError):
        service.classify(_answer("development", brand_id="northwind"), "v1")
    assert tracer.finished[1]["error_code"] == "unknown_brand"
    assert tracer.scores == []


def test_one_answer_failure_still_traces_the_run() -> None:
    tracer = RecordingTracer()
    answers = (_answer("development", "A02"), _answer("development", "A01"))
    run = _service(RaisingClassifier(), answers, MemoryGold(), tracer).start_run(
        "v1",
        "development",
    )
    assert [answer.answer_id for answer in run.answers] == ["A01", "A02"]
    assert [answer.status for answer in run.answers] == ["ok", "error"]
    stored_ids = {answer.trace_id for answer in run.answers}
    assert stored_ids == {trace["trace_id"] for trace in tracer.traces}
    assert None not in stored_ids
    finished = {(item["status"], item["error_code"]) for item in tracer.finished}
    assert finished == {("ok", None), ("error", "provider_failure")}
    assert len(tracer.scores) == 2


class BoomClient:
    def start_observation(self, **kwargs: object) -> object:
        raise RuntimeError("langfuse down")

    def create_score(self, **kwargs: object) -> None:
        raise RuntimeError("langfuse down")

    def flush(self) -> None:
        raise RuntimeError("langfuse down")


def test_langfuse_outage_does_not_fail_the_run() -> None:
    tracer = LangfuseTracer(client=BoomClient())
    run = _service(OkClassifier(), (_answer("development"),), MemoryGold(), tracer).start_run(
        "v1",
        "development",
    )
    assert run.answer_count == 1
    assert run.answers[0].trace_id is None
    assert run.answer_error_count == 0


class Observation:
    def __init__(self, trace_id: str, span_id: str) -> None:
        self.trace_id = trace_id
        self.id = span_id
        self.updates: list[dict[str, object]] = []
        self.ended = False

    def update(self, **kwargs: object) -> "Observation":
        self.updates.append(kwargs)
        return self

    def end(self) -> None:
        self.ended = True


class StubClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.observations: list[Observation] = []
        self.scores: list[dict[str, object]] = []
        self.flushed = 0

    def start_observation(self, **kwargs: object) -> Observation:
        self.calls.append(kwargs)
        observation = Observation("trace-1", f"span-{len(self.calls)}")
        self.observations.append(observation)
        return observation

    def create_score(self, **kwargs: object) -> None:
        self.scores.append(kwargs)

    def flush(self) -> None:
        self.flushed += 1


def test_adapter_maps_generations_tools_and_scores() -> None:
    client = StubClient()
    tracer = LangfuseTracer(client=client)
    trace_id = tracer.start_answer(
        answer_id="A01",
        brand_id="cleartrail",
        split="development",
        agent_version="v1",
        model="gpt-4o-mini",
        run_id="run-1",
    )
    assert trace_id == "trace-1"
    assert client.calls[0]["as_type"] == "span"
    assert "ai_answer" not in json.dumps(client.calls[0])
    tracer.record_generation(
        trace_id,
        model="gpt-4o-mini",
        messages=({"role": "user", "content": "hello"},),
        output="{}",
    )
    tracer.record_tool(
        trace_id,
        name="get_brand_facts",
        arguments={"brand_id": "cleartrail"},
        output={"brand_id": "cleartrail"},
        error_code=None,
    )
    assert client.calls[1]["as_type"] == "generation"
    assert client.calls[1]["trace_context"] == {
        "trace_id": "trace-1",
        "parent_span_id": "span-1",
    }
    assert client.calls[2]["as_type"] == "tool"
    assert client.calls[2]["name"] == "get_brand_facts"
    tracer.finish_answer(trace_id, status="ok", error_code=None)
    assert client.observations[0].ended is True
    tracer.score_answer(trace_id, extraction_f1=0.8, joint_f1=0.7)
    assert [score["name"] for score in client.scores] == ["extraction_f1", "joint_f1"]
    assert client.scores[0]["value"] == 0.8
    assert client.scores[0]["data_type"] == "NUMERIC"
    assert client.scores[1]["trace_id"] == "trace-1"
    tracer.flush()
    assert client.flushed == 1


def test_missing_keys_select_the_noop_tracer() -> None:
    settings = _settings(public_key="", secret_key="")
    assert isinstance(build_tracer(settings), NoOpTracer)
    partial = _settings(public_key="pk-test", secret_key="")
    assert isinstance(build_tracer(partial), NoOpTracer)


def test_both_keys_construct_the_langfuse_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(langfuse, "Langfuse", FakeLangfuse)
    settings = _settings(public_key="pk-test", secret_key="sk-test")
    tracer = build_tracer(settings)
    assert isinstance(tracer, LangfuseTracer)
    assert captured == {
        "public_key": "pk-test",
        "secret_key": "sk-test",
        "host": "https://cloud.langfuse.com",
    }


def test_application_modules_do_not_import_langfuse() -> None:
    offenders: list[str] = []
    for path in (ROOT / "src" / "speaker_claims").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            if any(name == "langfuse" or name.startswith("langfuse.") for name in modules):
                offenders.append(path.name)
    assert offenders == ["langfuse_tracer.py"]


def _settings(*, public_key: str, secret_key: str) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test",
        database_url="postgresql://user:password@localhost:5432/speaker_claims",
        langfuse_public_key=public_key,
        langfuse_secret_key=secret_key,
        langfuse_host="https://cloud.langfuse.com",
    )
