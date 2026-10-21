from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from speaker_claims.ports.tracer import TracerPort


@dataclass(frozen=True)
class ActiveTrace:
    tracer: TracerPort
    trace_id: str
    model: str


_current: ContextVar[ActiveTrace | None] = ContextVar("speaker_claim_trace", default=None)


def current_trace() -> ActiveTrace | None:
    return _current.get()


class NoOpTracer:
    """Stand-in used when Langfuse keys are absent. It satisfies TracerPort."""

    def start_answer(
        self,
        *,
        answer_id: str,
        brand_id: str,
        split: str,
        agent_version: str,
        model: str,
        run_id: str | None,
    ) -> str | None:
        return None

    def record_generation(
        self,
        trace_id: str | None,
        *,
        model: str,
        messages: tuple[dict[str, object], ...],
        output: str | None,
    ) -> None:
        return None

    def record_tool(
        self,
        trace_id: str | None,
        *,
        name: str,
        arguments: dict[str, object] | None,
        output: dict[str, object] | None,
        error_code: str | None,
    ) -> None:
        return None

    def finish_answer(
        self,
        trace_id: str | None,
        *,
        status: str,
        error_code: str | None,
    ) -> None:
        return None

    def score_answer(
        self,
        trace_id: str | None,
        *,
        extraction_f1: float,
        joint_f1: float,
    ) -> None:
        return None

    def flush(self) -> None:
        return None


@contextmanager
def open_answer(
    tracer: TracerPort,
    *,
    answer_id: str,
    brand_id: str,
    split: str,
    agent_version: str,
    model: str,
    run_id: str | None,
) -> Iterator[str | None]:
    """Bind the answer trace for the current thread. Workers inherit their own copy."""
    trace_id = tracer.start_answer(
        answer_id=answer_id,
        brand_id=brand_id,
        split=split,
        agent_version=agent_version,
        model=model,
        run_id=run_id,
    )
    if trace_id is None:
        yield None
        return
    token = _current.set(ActiveTrace(tracer=tracer, trace_id=trace_id, model=model))
    try:
        yield trace_id
    finally:
        _current.reset(token)
