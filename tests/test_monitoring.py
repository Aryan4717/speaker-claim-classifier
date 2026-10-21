import ast
import json
from pathlib import Path

import pytest
import sentry_sdk
from fastapi.testclient import TestClient

from speaker_claims.adapters.sentry_monitor import SentryMonitor, scrub_event
from speaker_claims.api.app import create_app
from speaker_claims.application.claim_pipeline import V1ClaimPipeline
from speaker_claims.application.evaluation import EvaluationService
from speaker_claims.application.fact_tool import FactToolExecutor
from speaker_claims.application.monitoring import NoOpMonitor, ReportedFailure, monitor_scope
from speaker_claims.composition import build_monitor
from speaker_claims.config import Settings
from speaker_claims.domain.claims import AnswerRecord
from speaker_claims.domain.errors import ProviderTimeout, UnknownBrandError
from speaker_claims.domain.gold import GoldClaim
from speaker_claims.domain.outcome import AnswerOutcome
from speaker_claims.domain.runs import StoredRun
from speaker_claims.ports.llm import ChatMessage, ModelTurn

ROOT = Path(__file__).resolve().parents[1]
TEXT = "Think about the trip. The price is INR 1,299."


class RecordingMonitor:
    def __init__(self) -> None:
        self.contexts: list[dict[str, object]] = []
        self.breadcrumbs: list[dict[str, object]] = []
        self.captured: list[BaseException] = []

    def set_context(
        self,
        *,
        run_id: str | None,
        answer_id: str | None,
        split: str | None,
        agent_version: str | None,
    ) -> None:
        self.contexts.append(
            {
                "run_id": run_id,
                "answer_id": answer_id,
                "split": split,
                "agent_version": agent_version,
            }
        )

    def breadcrumb(
        self,
        *,
        category: str,
        message: str,
        data: dict[str, object] | None = None,
    ) -> None:
        self.breadcrumbs.append({"category": category, "message": message, "data": data or {}})

    def capture_exception(self, exc: BaseException) -> None:
        self.captured.append(exc)


class MemoryCatalogue:
    def get_brand_facts(self, brand_id: str) -> tuple[object, ...]:
        if brand_id == "northwind":
            raise UnknownBrandError(brand_id)
        return ()


class MemoryAnswers:
    def __init__(self, answers: tuple[AnswerRecord, ...]) -> None:
        self._answers = answers

    def by_split(self, split: str) -> tuple[AnswerRecord, ...]:
        return tuple(answer for answer in self._answers if answer.split == split)


class MemoryGold:
    def load(self) -> tuple[GoldClaim, ...]:
        return ()


class MemoryStore:
    def save(self, run: StoredRun) -> None:
        return None

    def get(self, run_id: object) -> StoredRun | None:
        return None


class TimeoutProvider:
    def complete(self, messages: tuple[ChatMessage, ...], timeout_seconds: float) -> ModelTurn:
        raise ProviderTimeout()


class Scripted:
    def __init__(self, code: str) -> None:
        self.code = code

    def classify(self, answer: AnswerRecord) -> AnswerOutcome:
        return AnswerOutcome(
            answer.answer_id,
            "v1",
            "error",
            (),
            self.code,
            "The model request failed",
        )


class Raising:
    def classify(self, answer: AnswerRecord) -> AnswerOutcome:
        raise RuntimeError("boom")


def _answer(
    answer_id: str = "A01",
    brand_id: str = "cleartrail",
    split: str = "development",
) -> AnswerRecord:
    return AnswerRecord(answer_id, split, brand_id, "What is the price?", TEXT)


def _service(
    classifier: object,
    monitor: RecordingMonitor,
    answers: tuple[AnswerRecord, ...] = (),
) -> EvaluationService:
    return EvaluationService(
        {"v1": classifier},
        MemoryCatalogue(),
        MemoryAnswers(answers),
        MemoryGold(),
        MemoryStore(),
        provider="openai",
        model="gpt-4o-mini",
        settings={},
        source_revision=None,
        file_hashes={},
        monitor=monitor,
    )


def _pipeline() -> V1ClaimPipeline:
    catalogue = MemoryCatalogue()
    return V1ClaimPipeline(
        TimeoutProvider(),
        FactToolExecutor(catalogue),
        frozenset(),
        "Return JSON.",
    )


def test_pipeline_timeout_is_captured_once() -> None:
    monitor = RecordingMonitor()
    answer = _answer()
    with monitor_scope(
        monitor,
        run_id="run-1",
        answer_id=answer.answer_id,
        split="development",
        agent_version="v1",
    ):
        outcome = _pipeline().classify(answer)
    assert outcome.error_code == "timeout"
    assert len(monitor.captured) == 1
    assert isinstance(monitor.captured[0], ProviderTimeout)
    service_monitor = RecordingMonitor()
    run = _service(_pipeline(), service_monitor, (answer,)).start_run("v1", "development")
    assert len(service_monitor.captured) == 1
    assert isinstance(service_monitor.captured[0], ProviderTimeout)
    assert service_monitor.contexts[0]["run_id"] == str(run.run_id)
    assert service_monitor.contexts[0]["answer_id"] == "A01"
    blob = json.dumps(service_monitor.contexts)
    assert TEXT not in blob
    assert "expected_label" not in blob


def test_run_reports_timeout_unknown_brand_and_unhandled_separately() -> None:
    monitor = RecordingMonitor()
    answers = (
        _answer("A03"),
        _answer("A02", brand_id="northwind"),
        _answer("A01"),
    )

    class Mixed:
        def classify(self, answer: AnswerRecord) -> AnswerOutcome:
            if answer.answer_id == "A03":
                raise RuntimeError("boom")
            return AnswerOutcome(answer.answer_id, "v1", "error", (), "invalid_output", "bad json")

    run = _service(Mixed(), monitor, answers).start_run("v1", "development")
    assert [answer.answer_id for answer in run.answers] == ["A01", "A02", "A03"]
    assert [answer.error_code for answer in run.answers] == [
        "invalid_output",
        "unknown_brand",
        "provider_failure",
    ]
    assert [type(exc) for exc in monitor.captured] == [RuntimeError]
    assert [item["message"] for item in monitor.breadcrumbs] == ["unknown_brand"]


def test_scripted_timeout_is_captured_and_invalid_output_is_not() -> None:
    monitor = RecordingMonitor()
    _service(Scripted("timeout"), monitor).classify(_answer(), "v1")
    assert [exc.code for exc in monitor.captured if isinstance(exc, ReportedFailure)] == ["timeout"]
    quiet = RecordingMonitor()
    _service(Scripted("invalid_output"), quiet).classify(_answer(), "v1")
    assert quiet.captured == []


def test_client_errors_are_breadcrumbs() -> None:
    monitor = RecordingMonitor()
    client = TestClient(create_app(_service(Scripted("timeout"), monitor)))
    invalid = client.post("/classify", json={"answer_id": "A01", "expected_label": "supported"})
    assert invalid.status_code == 422
    unknown = client.post(
        "/classify",
        json={
            "answer_id": "A01",
            "brand_id": "northwind",
            "question": "Price?",
            "ai_answer": TEXT,
            "agent_version": "v1",
        },
    )
    assert unknown.status_code == 404
    missing = client.get("/evaluation-runs/not-a-uuid")
    assert missing.status_code == 422
    timeout = client.post(
        "/classify",
        json={
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "Price?",
            "ai_answer": TEXT,
            "agent_version": "v1",
        },
    )
    assert timeout.status_code == 504
    messages = [item["message"] for item in monitor.breadcrumbs]
    assert messages == ["validation", "client_error", "client_error"]
    assert [exc.code for exc in monitor.captured if isinstance(exc, ReportedFailure)] == ["timeout"]
    dumped = json.dumps(monitor.breadcrumbs)
    assert TEXT not in dumped
    assert "expected_label" not in dumped


def test_unhandled_exception_is_captured_as_500() -> None:
    monitor = RecordingMonitor()
    client = TestClient(create_app(_service(Raising(), monitor)), raise_server_exceptions=False)
    response = client.post(
        "/classify",
        json={
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "Price?",
            "ai_answer": TEXT,
            "agent_version": "v1",
        },
    )
    assert response.status_code == 500
    assert len(monitor.captured) == 1
    assert isinstance(monitor.captured[0], RuntimeError)


def test_scrubber_removes_secrets_and_answer_text() -> None:
    event = {
        "request": {
            "headers": {"Authorization": "Bearer sk-test", "Cookie": "session=abc"},
            "cookies": {"session": "abc"},
            "data": {"ai_answer": TEXT, "OPENAI_API_KEY": "sk-live"},
        },
        "extra": {
            "DATABASE_URL": "postgresql://user:password@localhost/db",
            "LANGFUSE_SECRET_KEY": "sk-lf",
            "note": "kept",
        },
    }
    cleaned = scrub_event(event, {})
    request = cleaned["request"]
    assert request["headers"]["Authorization"] == "[Filtered]"
    assert request["headers"]["Cookie"] == "[Filtered]"
    assert "cookies" not in request
    assert "data" not in request
    assert cleaned["extra"]["DATABASE_URL"] == "[Filtered]"
    assert cleaned["extra"]["LANGFUSE_SECRET_KEY"] == "[Filtered]"
    assert cleaned["extra"]["note"] == "kept"
    assert TEXT not in json.dumps(cleaned)


class FakeSdk:
    def __init__(self) -> None:
        self.inits: list[dict[str, object]] = []
        self.contexts: list[tuple[str, dict[str, object]]] = []
        self.breadcrumbs: list[dict[str, object]] = []
        self.captured: list[BaseException] = []

    def init(self, **kwargs: object) -> None:
        self.inits.append(kwargs)

    def set_context(self, name: str, data: dict[str, object]) -> None:
        self.contexts.append((name, data))

    def add_breadcrumb(self, **kwargs: object) -> None:
        self.breadcrumbs.append(kwargs)

    def capture_exception(self, exc: BaseException) -> None:
        self.captured.append(exc)


def test_adapter_initializes_sentry_and_filters_breadcrumb_data() -> None:
    sdk = FakeSdk()
    monitor = SentryMonitor(dsn="https://public@sentry.example/1", environment="local", sdk=sdk)
    before_send = sdk.inits[0]["before_send"]
    assert sdk.inits[0]["send_default_pii"] is False
    assert before_send is scrub_event
    monitor.set_context(run_id="run-1", answer_id="A01", split="check", agent_version="v1")
    assert sdk.contexts[0][1] == {
        "run_id": "run-1",
        "answer_id": "A01",
        "split": "check",
        "agent_version": "v1",
    }
    monitor.breadcrumb(
        category="http",
        message="client_error",
        data={"status_code": 404, "ai_answer": TEXT},
    )
    assert sdk.breadcrumbs[0]["data"] == {"status_code": 404}
    monitor.capture_exception(ProviderTimeout())
    assert isinstance(sdk.captured[0], ProviderTimeout)


def test_empty_dsn_selects_the_noop_monitor() -> None:
    settings = _settings(dsn="")
    assert isinstance(build_monitor(settings), NoOpMonitor)


def test_dsn_selects_the_sentry_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_init(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(sentry_sdk, "init", fake_init)
    settings = _settings(dsn="https://public@sentry.example/1")
    monitor = build_monitor(settings)
    assert isinstance(monitor, SentryMonitor)
    assert captured["dsn"] == "https://public@sentry.example/1"
    assert captured["environment"] == "local"
    assert captured["before_send"] is scrub_event
    assert captured["send_default_pii"] is False


def test_pipeline_and_scorer_do_not_import_sentry() -> None:
    modules = _closure("speaker_claims.application.claim_pipeline")
    modules.extend(_closure("speaker_claims.application.scoring"))
    assert all(not name.startswith("sentry_sdk") for name in modules)
    offenders: list[str] = []
    for path in (ROOT / "src" / "speaker_claims").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name == "sentry_sdk" or name.startswith("sentry_sdk.") for name in names):
                offenders.append(path.name)
    assert offenders == ["sentry_monitor.py"]


def _closure(module: str) -> list[str]:
    pending = [module]
    seen: set[str] = set()
    found: list[str] = []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        path = _module_path(current)
        if path is None:
            continue
        for name in _imported(path):
            found.append(name)
            if name.startswith("speaker_claims"):
                pending.append(name)
    return found


def _module_path(module: str) -> Path | None:
    relative = Path(*module.split("."))
    direct = ROOT / "src" / relative.with_suffix(".py")
    package = ROOT / "src" / relative / "__init__.py"
    if direct.exists():
        return direct
    if package.exists():
        return package
    return None


def _imported(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _settings(*, dsn: str) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test",
        database_url="postgresql://user:password@localhost:5432/speaker_claims",
        sentry_dsn=dsn,
        sentry_environment="local",
    )
