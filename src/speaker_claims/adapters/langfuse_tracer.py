from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any


class LangfuseTracer:
    """Translate trace calls into the Langfuse SDK. This is the only importer of langfuse."""

    def __init__(
        self,
        *,
        public_key: str = "",
        secret_key: str = "",
        host: str = "",
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            from langfuse import Langfuse

            kwargs: dict[str, str] = {"public_key": public_key, "secret_key": secret_key}
            if host:
                kwargs["host"] = host
            self._client = Langfuse(**kwargs)
        self._spans: dict[str, Any] = {}
        self._lock = Lock()

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
        def create() -> str:
            span = self._client.start_observation(
                name="classify-answer",
                as_type="span",
                input={"answer_id": answer_id, "brand_id": brand_id},
                metadata={
                    "answer_id": answer_id,
                    "brand_id": brand_id,
                    "split": split,
                    "agent_version": agent_version,
                    "model": model,
                    "run_id": run_id,
                },
                version=agent_version,
            )
            with self._lock:
                self._spans[span.trace_id] = span
            return span.trace_id

        created = self._quiet(create)
        if isinstance(created, str):
            return created
        return None

    def record_generation(
        self,
        trace_id: str | None,
        *,
        model: str,
        messages: tuple[dict[str, object], ...],
        output: str | None,
    ) -> None:
        parent = self._parent(trace_id)
        if parent is None or trace_id is None:
            return

        def create() -> None:
            generation = self._client.start_observation(
                trace_context={"trace_id": trace_id, "parent_span_id": parent.id},
                name="chat.completion",
                as_type="generation",
                model=model,
                input=list(messages),
            )
            generation.update(output=output)
            generation.end()

        self._quiet(create)

    def record_tool(
        self,
        trace_id: str | None,
        *,
        name: str,
        arguments: dict[str, object] | None,
        output: dict[str, object] | None,
        error_code: str | None,
    ) -> None:
        parent = self._parent(trace_id)
        if parent is None or trace_id is None:
            return

        def create() -> None:
            tool = self._client.start_observation(
                trace_context={"trace_id": trace_id, "parent_span_id": parent.id},
                name=name,
                as_type="tool",
                input=arguments,
            )
            if error_code is None:
                tool.update(output=output)
            else:
                tool.update(output={"error_code": error_code}, level="ERROR")
            tool.end()

        self._quiet(create)

    def finish_answer(
        self,
        trace_id: str | None,
        *,
        status: str,
        error_code: str | None,
    ) -> None:
        if trace_id is None:
            return
        with self._lock:
            span = self._spans.pop(trace_id, None)
        if span is None:
            return

        def close() -> None:
            if status == "error":
                span.update(
                    output={"status": status, "error_code": error_code},
                    level="ERROR",
                    status_message=error_code,
                )
            else:
                span.update(output={"status": status, "error_code": error_code})
            span.end()

        self._quiet(close)

    def score_answer(
        self,
        trace_id: str | None,
        *,
        extraction_f1: float,
        joint_f1: float,
    ) -> None:
        if trace_id is None:
            return
        self._score(trace_id, "extraction_f1", extraction_f1)
        self._score(trace_id, "joint_f1", joint_f1)

    def flush(self) -> None:
        self._quiet(self._client.flush)

    def _score(self, trace_id: str, name: str, value: float) -> None:
        def create() -> None:
            self._client.create_score(
                name=name,
                value=value,
                trace_id=trace_id,
                data_type="NUMERIC",
            )

        self._quiet(create)

    def _parent(self, trace_id: str | None) -> Any | None:
        if trace_id is None:
            return None
        with self._lock:
            return self._spans.get(trace_id)

    def _quiet(self, action: Callable[[], Any]) -> Any | None:
        try:
            return action()
        except Exception:
            return None
