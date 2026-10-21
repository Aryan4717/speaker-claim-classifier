from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from speaker_claims.domain.errors import (
    ProviderRateLimit,
    ProviderTimeout,
    ProviderTransient,
)
from speaker_claims.ports.llm import ChatMessage, LlmPort, ModelTurn

MAX_IN_FLIGHT = 3
MAX_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 0.2

_RETRYABLE = (ProviderTimeout, ProviderRateLimit, ProviderTransient)


class RetryPolicy:
    """One retry for a transient provider failure. Malformed output never reaches this policy."""

    def __init__(
        self,
        max_attempts: int = MAX_ATTEMPTS,
        delay_seconds: float = RETRY_DELAY_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_attempts = max_attempts
        self.delay_seconds = delay_seconds
        self._sleep = sleep

    def retryable(self, exc: BaseException) -> bool:
        return isinstance(exc, _RETRYABLE)

    def wait(self) -> None:
        self._sleep(self.delay_seconds)


class ProviderGate:
    """Cap active provider requests, including retries."""

    def __init__(self, limit: int = MAX_IN_FLIGHT) -> None:
        self._slots = threading.BoundedSemaphore(limit)
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    @contextmanager
    def slot(self) -> Iterator[None]:
        self._slots.acquire()
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            yield
        finally:
            with self._lock:
                self.in_flight -= 1
            self._slots.release()


class RetryingProvider:
    """LlmPort adapter. The pipeline stays unaware of retries and the in-flight cap."""

    def __init__(
        self,
        inner: LlmPort,
        policy: RetryPolicy | None = None,
        gate: ProviderGate | None = None,
    ) -> None:
        self._inner = inner
        self._policy = policy if policy is not None else RetryPolicy()
        self._gate = gate if gate is not None else ProviderGate()
        self.attempts = 0

    def complete(self, messages: tuple[ChatMessage, ...], timeout_seconds: float) -> ModelTurn:
        attempt = 0
        while True:
            attempt += 1
            self.attempts += 1
            try:
                with self._gate.slot():
                    return self._inner.complete(messages, timeout_seconds)
            except Exception as exc:
                if attempt >= self._policy.max_attempts or not self._policy.retryable(exc):
                    raise
                self._policy.wait()
