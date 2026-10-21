from __future__ import annotations

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from speaker_claims.domain.errors import (
    ProviderAuthError,
    ProviderFailure,
    ProviderRateLimit,
    ProviderTimeout,
    ProviderTransient,
)
from speaker_claims.ports.llm import ChatMessage, ModelTurn, ToolCall

DEFAULT_MODEL = "gpt-4o-mini"

_TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "get_brand_facts",
            "description": (
                "Return the supplied catalogue rows for one brand. "
                "This is the only factual authority."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "brand_id": {
                        "type": "string",
                        "description": "Catalogue brand id, for example cleartrail.",
                    }
                },
                "required": ["brand_id"],
                "additionalProperties": False,
            },
        },
    },
)


class OpenAIProvider:
    """Adapt the OpenAI chat client to LlmPort. This is the only module that imports openai."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        client: OpenAI | None = None,
    ) -> None:
        self._client = client if client is not None else OpenAI(api_key=api_key)
        self._model = model

    def complete(self, messages: tuple[ChatMessage, ...], timeout_seconds: float) -> ModelTurn:
        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[_message(message) for message in messages],
                tools=_TOOLS,
                tool_choice="auto",
                temperature=0,
                timeout=timeout_seconds,
            )
        except APITimeoutError as exc:
            raise ProviderTimeout() from exc
        except RateLimitError as exc:
            raise ProviderRateLimit() from exc
        except (AuthenticationError, PermissionDeniedError) as exc:
            raise ProviderAuthError() from exc
        except InternalServerError as exc:
            raise ProviderTransient() from exc
        except APIConnectionError as exc:
            raise ProviderTransient() from exc
        except APIStatusError as exc:
            if exc.status_code in {500, 502, 503}:
                raise ProviderTransient() from exc
            raise ProviderFailure() from exc
        except APIError as exc:
            raise ProviderFailure() from exc
        choice = completion.choices[0].message
        raw_calls = choice.tool_calls or []
        calls = tuple(
            ToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments_json=call.function.arguments or "{}",
            )
            for call in raw_calls
        )
        return ModelTurn(tool_calls=calls, content=choice.content)


def _message(message: ChatMessage) -> dict[str, object]:
    if message.role == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
    if message.tool_calls:
        return {
            "role": "assistant",
            "content": message.content or None,
            "tool_calls": [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_json},
                }
                for call in message.tool_calls
            ],
        }
    return {"role": message.role, "content": message.content}
