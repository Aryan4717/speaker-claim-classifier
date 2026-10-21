from __future__ import annotations

from collections.abc import Callable
from typing import Any

_SECRET_KEYS = frozenset(
    {
        "authorization",
        "cookie",
        "cookies",
        "openai_api_key",
        "database_url",
        "langfuse_public_key",
        "langfuse_secret_key",
        "langfuse_host",
        "sentry_dsn",
        "api_key",
        "secret_key",
    }
)
_HIDDEN_DATA = frozenset(
    {
        "ai_answer",
        "question",
        "quote",
        "expected_label",
        "explanation",
        "reason",
        "claims",
        "prompt",
    }
)


def scrub_event(event: dict[str, Any], hint: object) -> dict[str, Any]:
    """Drop credentials, cookies, and request bodies before an event is sent."""
    del hint
    request = event.get("request")
    if isinstance(request, dict):
        request.pop("data", None)
        request.pop("cookies", None)
        headers = request.get("headers")
        if isinstance(headers, dict):
            for key in list(headers):
                if _secret_key(str(key)):
                    headers[key] = "[Filtered]"
    return _scrub(event)


class SentryMonitor:
    """Translate port calls into sentry_sdk. This is the only importer of that SDK."""

    def __init__(
        self,
        *,
        dsn: str,
        environment: str = "local",
        sdk: Any | None = None,
    ) -> None:
        if sdk is None:
            import sentry_sdk

            sdk = sentry_sdk
        self._sdk = sdk
        self._quiet(
            lambda: sdk.init(
                dsn=dsn,
                environment=environment,
                before_send=scrub_event,
                send_default_pii=False,
            )
        )

    def set_context(
        self,
        *,
        run_id: str | None,
        answer_id: str | None,
        split: str | None,
        agent_version: str | None,
    ) -> None:
        data = {
            "run_id": run_id,
            "answer_id": answer_id,
            "split": split,
            "agent_version": agent_version,
        }
        self._quiet(lambda: self._sdk.set_context("answer", data))

    def breadcrumb(
        self,
        *,
        category: str,
        message: str,
        data: dict[str, object] | None = None,
    ) -> None:
        safe = _public_data(data)
        self._quiet(
            lambda: self._sdk.add_breadcrumb(
                category=category,
                message=message,
                data=safe,
                level="info",
            )
        )

    def capture_exception(self, exc: BaseException) -> None:
        self._quiet(lambda: self._sdk.capture_exception(exc))

    def _quiet(self, action: Callable[[], Any]) -> None:
        if self._sdk is None:
            return
        try:
            action()
        except Exception:
            return


def _public_data(data: dict[str, object] | None) -> dict[str, object]:
    if not data:
        return {}
    visible: dict[str, object] = {}
    for key, value in data.items():
        if key in _HIDDEN_DATA or _secret_key(key):
            continue
        visible[key] = value
    return visible


def _secret_key(key: str) -> bool:
    return key.lower().replace("-", "_") in _SECRET_KEYS


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if _secret_key(str(key)):
                cleaned[str(key)] = "[Filtered]"
            else:
                cleaned[str(key)] = _scrub(item)
        return cleaned
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    return value
