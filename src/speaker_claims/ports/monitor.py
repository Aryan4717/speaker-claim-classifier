from __future__ import annotations

from typing import Protocol


class ErrorMonitorPort(Protocol):
    """Report failures without exposing the monitoring SDK to claim logic."""

    def set_context(
        self,
        *,
        run_id: str | None,
        answer_id: str | None,
        split: str | None,
        agent_version: str | None,
    ) -> None:
        """Attach the answer identity. Do not include answer text or gold labels."""

    def breadcrumb(
        self,
        *,
        category: str,
        message: str,
        data: dict[str, object] | None = None,
    ) -> None:
        """Record an expected client outcome."""

    def capture_exception(self, exc: BaseException) -> None:
        """Record a provider failure, exhausted timeout, or unhandled exception."""
