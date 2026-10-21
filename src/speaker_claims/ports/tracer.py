from __future__ import annotations

from typing import Protocol


class TracerPort(Protocol):
    """One trace per answer. Scores are a separate call so check runs can omit them."""

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
        """Open a trace. None means this call is not recorded."""

    def record_generation(
        self,
        trace_id: str | None,
        *,
        model: str,
        messages: tuple[dict[str, object], ...],
        output: str | None,
    ) -> None:
        """Record one model turn."""

    def record_tool(
        self,
        trace_id: str | None,
        *,
        name: str,
        arguments: dict[str, object] | None,
        output: dict[str, object] | None,
        error_code: str | None,
    ) -> None:
        """Record one tool call."""

    def finish_answer(
        self,
        trace_id: str | None,
        *,
        status: str,
        error_code: str | None,
    ) -> None:
        """Close the answer trace."""

    def score_answer(
        self,
        trace_id: str | None,
        *,
        extraction_f1: float,
        joint_f1: float,
    ) -> None:
        """Attach development scores. Check runs do not call this."""

    def flush(self) -> None:
        """Send queued traces. A no-op tracer ignores this."""
