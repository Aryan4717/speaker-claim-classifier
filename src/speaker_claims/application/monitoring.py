from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from speaker_claims.ports.monitor import ErrorMonitorPort


class ReportedFailure(Exception):
    """Outcome that had no live exception to forward."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class _Scope:
    monitor: ErrorMonitorPort


_scope: ContextVar[_Scope | None] = ContextVar("error_monitor", default=None)
_reported: ContextVar[bool] = ContextVar("error_monitor_reported", default=False)


class NoOpMonitor:
    """Used when SENTRY_DSN is empty. It satisfies ErrorMonitorPort."""

    def set_context(
        self,
        *,
        run_id: str | None,
        answer_id: str | None,
        split: str | None,
        agent_version: str | None,
    ) -> None:
        return None

    def breadcrumb(
        self,
        *,
        category: str,
        message: str,
        data: dict[str, object] | None = None,
    ) -> None:
        return None

    def capture_exception(self, exc: BaseException) -> None:
        return None


@contextmanager
def monitor_scope(
    monitor: ErrorMonitorPort,
    *,
    run_id: str | None,
    answer_id: str | None,
    split: str | None,
    agent_version: str | None,
) -> Iterator[None]:
    """Bind the monitor for this thread and publish the safe answer context."""
    monitor.set_context(
        run_id=run_id,
        answer_id=answer_id,
        split=split,
        agent_version=agent_version,
    )
    scope_token = _scope.set(_Scope(monitor))
    reported_token = _reported.set(False)
    try:
        yield
    finally:
        _reported.reset(reported_token)
        _scope.reset(scope_token)


def capture_failure(exc: BaseException) -> None:
    """Capture from the active scope. A missing scope leaves the caller unchanged."""
    scope = _scope.get()
    if scope is None:
        return
    scope.monitor.capture_exception(exc)
    _reported.set(True)


def already_reported() -> bool:
    return _reported.get()
