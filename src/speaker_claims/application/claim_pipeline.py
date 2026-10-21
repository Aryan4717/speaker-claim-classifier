from __future__ import annotations

import json
from abc import ABC, abstractmethod

from speaker_claims.application.fact_tool import FactToolExecutor
from speaker_claims.application.quote_locator import locate_quotes
from speaker_claims.application.span_validation import AnswerClaims, build_span_chain
from speaker_claims.domain.claims import AnswerRecord
from speaker_claims.domain.errors import DomainError, ProviderFailure, ProviderTimeout
from speaker_claims.domain.outcome import AnswerOutcome
from speaker_claims.ports.llm import ChatMessage, LlmPort, ToolCall

MAX_MODEL_TURNS = 3
REQUEST_TIMEOUT_SECONDS = 15.0


class ClaimPipeline(ABC):
    """Run one answer through tool calls and quote location. Subclasses supply the prompt."""

    def __init__(
        self,
        provider: LlmPort,
        executor: FactToolExecutor,
        known_fact_ids: frozenset[str],
    ) -> None:
        self._provider = provider
        self._executor = executor
        self._known_fact_ids = known_fact_ids
        self._chain = build_span_chain()

    @property
    @abstractmethod
    def version(self) -> str:
        """Agent version stored with the outcome."""

    @abstractmethod
    def opening_messages(self, answer: AnswerRecord) -> tuple[ChatMessage, ...]:
        """The one step a later version may replace."""

    def classify(self, answer: AnswerRecord) -> AnswerOutcome:
        messages = list(self.opening_messages(answer))
        for turn in range(MAX_MODEL_TURNS):
            try:
                result = self._provider.complete(tuple(messages), REQUEST_TIMEOUT_SECONDS)
            except ProviderTimeout:
                return self._error(answer, "timeout", "The model request timed out")
            except ProviderFailure:
                return self._error(answer, "provider_failure", "The model request failed")
            if result.tool_calls:
                if turn == MAX_MODEL_TURNS - 1:
                    return self._error(
                        answer,
                        "turn_exhausted",
                        "The model did not finish within three turns",
                    )
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=result.content or "",
                        tool_calls=result.tool_calls,
                    )
                )
                tool_error = self._append_tool_results(answer, messages, result.tool_calls)
                if tool_error is not None:
                    return tool_error
                continue
            return self._finish(answer, result.content)
        return self._error(answer, "turn_exhausted", "The model did not finish within three turns")

    def _append_tool_results(
        self,
        answer: AnswerRecord,
        messages: list[ChatMessage],
        tool_calls: tuple[ToolCall, ...],
    ) -> AnswerOutcome | None:
        for call in tool_calls:
            arguments = _parse_arguments(call.arguments_json)
            if arguments is None:
                return self._error(
                    answer,
                    "invalid_output",
                    "Tool arguments were not a JSON object",
                )
            try:
                payload = self._executor.execute(call.name, arguments)
            except DomainError as exc:
                return self._error(answer, exc.code, str(exc))
            messages.append(
                ChatMessage(
                    role="tool",
                    content=json.dumps(payload, ensure_ascii=False),
                    tool_call_id=call.call_id,
                )
            )
        return None

    def _finish(self, answer: AnswerRecord, content: str | None) -> AnswerOutcome:
        if content is None or not content.strip():
            return self._error(answer, "invalid_output", "The model returned no claims JSON")
        try:
            quoted = _parse_claims(content, answer.answer_id)
            claims = locate_quotes(answer, quoted)
            self._chain.validate(
                AnswerClaims(
                    answer_id=answer.answer_id,
                    ai_answer=answer.ai_answer,
                    claims=claims,
                    known_fact_ids=self._known_fact_ids,
                )
            )
        except DomainError as exc:
            return self._error(answer, exc.code, str(exc))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return self._error(
                answer,
                "invalid_output",
                "The model JSON did not match the claim schema",
            )
        return AnswerOutcome(
            answer_id=answer.answer_id,
            agent_version=self.version,
            status="ok",
            claims=claims,
        )

    def _error(self, answer: AnswerRecord, code: str, message: str) -> AnswerOutcome:
        return AnswerOutcome(
            answer_id=answer.answer_id,
            agent_version=self.version,
            status="error",
            claims=(),
            error_code=code,
            error_message=message,
        )


class V1ClaimPipeline(ClaimPipeline):
    def __init__(
        self,
        provider: LlmPort,
        executor: FactToolExecutor,
        known_fact_ids: frozenset[str],
        prompt: str,
    ) -> None:
        super().__init__(provider, executor, known_fact_ids)
        self._prompt = prompt

    @property
    def version(self) -> str:
        return "v1"

    def opening_messages(self, answer: AnswerRecord) -> tuple[ChatMessage, ...]:
        user = json.dumps(
            {
                "answer_id": answer.answer_id,
                "brand_id": answer.brand_id,
                "question": answer.question,
                "ai_answer": answer.ai_answer,
            },
            ensure_ascii=False,
        )
        return (
            ChatMessage(role="system", content=self._prompt),
            ChatMessage(role="user", content=user),
        )


def _parse_arguments(raw: str) -> dict[str, object] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value


def _parse_claims(content: str, answer_id: str) -> list[tuple[str, str, tuple[str, ...], str]]:
    payload = json.loads(_json_text(content))
    if not isinstance(payload, dict):
        raise TypeError("claims payload must be an object")
    if payload.get("answer_id") != answer_id:
        raise ValueError("answer_id does not match the input")
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        raise TypeError("claims must be a list")
    parsed: list[tuple[str, str, tuple[str, ...], str]] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            raise TypeError("each claim must be an object")
        quote = item["quote"]
        label = item["label"]
        reason = item["reason"]
        fact_ids = item["fact_ids"]
        if not all(isinstance(value, str) for value in (quote, label, reason)):
            raise TypeError("quote, label, and reason must be strings")
        fact_ids_are_strings = isinstance(fact_ids, list) and all(
            isinstance(fact_id, str) for fact_id in fact_ids
        )
        if not fact_ids_are_strings:
            raise TypeError("fact_ids must be a list of strings")
        parsed.append((quote, label, tuple(fact_ids), reason))
    return parsed


def _json_text(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
