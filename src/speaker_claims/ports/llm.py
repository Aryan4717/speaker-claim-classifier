from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""


@dataclass(frozen=True)
class ModelTurn:
    """One model response. Tool calls and a final JSON body are mutually exclusive."""

    tool_calls: tuple[ToolCall, ...]
    content: str | None


class LlmPort(Protocol):
    def complete(self, messages: tuple[ChatMessage, ...], timeout_seconds: float) -> ModelTurn:
        """Request one model turn. Raise ProviderTimeout or ProviderFailure on transport errors."""
