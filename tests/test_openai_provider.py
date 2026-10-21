from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError

from speaker_claims.adapters.openai_provider import OpenAIProvider
from speaker_claims.domain.errors import ProviderTimeout
from speaker_claims.ports.llm import ChatMessage


class StubCompletions:
    def __init__(self, result: object) -> None:
        self.result = result
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _client(result: object) -> SimpleNamespace:
    completions = StubCompletions(result)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions), completions=completions)
    return client


def test_provider_maps_a_tool_call_and_passes_the_timeout() -> None:
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="get_brand_facts",
                    arguments='{"brand_id": "cleartrail"}',
                ),
            )
        ],
    )
    completion = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    client = _client(completion)
    turn = OpenAIProvider(api_key="sk-test", client=client).complete(
        (ChatMessage(role="user", content="hello"),),
        timeout_seconds=15,
    )
    assert turn.tool_calls[0].name == "get_brand_facts"
    assert turn.tool_calls[0].arguments_json == '{"brand_id": "cleartrail"}'
    assert client.completions.kwargs["timeout"] == 15
    assert client.completions.kwargs["model"] == "gpt-4o-mini"
    tools = client.completions.kwargs["tools"]
    assert tools[0]["function"]["name"] == "get_brand_facts"


def test_provider_timeout_does_not_include_the_sdk_message() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    client = _client(APITimeoutError(request))
    with pytest.raises(ProviderTimeout) as caught:
        OpenAIProvider(api_key="sk-test", client=client).complete(
            (ChatMessage(role="user", content="hello"),),
            timeout_seconds=15,
        )
    assert "api.openai.com" not in str(caught.value)
