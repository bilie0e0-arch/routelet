import json
from unittest.mock import MagicMock, patch

from routelet.adapters.anthropic import AnthropicAdapter
from routelet.adapters.openai import OpenAIAdapter
from routelet.core.request import Message, NormalizedRequest, ToolDefinition


def _make_request(tools: bool = False) -> NormalizedRequest:
    return NormalizedRequest(
        messages=[
            Message(role="system", content="You are a coding assistant."),
            Message(role="user", content="Fix the bug."),
        ],
        tools=[
            ToolDefinition(
                name="bash",
                description="Run shell command",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            )
        ]
        if tools
        else [],
    )


def test_openai_adapter_calls_api_and_returns_response(mocker: MagicMock) -> None:
    fake_response = MagicMock()
    fake_response.choices[0].message.content = "hello"
    fake_response.choices[0].message.tool_calls = None
    fake_response.usage.prompt_tokens = 10
    fake_response.usage.completion_tokens = 5
    fake_response.model = "gpt-4o-2024-08-06"

    with patch("openai.OpenAI") as mock_client_class:
        mock_client_class.return_value.chat.completions.create.return_value = fake_response
        adapter = OpenAIAdapter(api_key="test-key")
        resp = adapter.call("gpt-4o-2024-08-06", _make_request())

    assert resp.content == "hello"
    assert resp.tool_calls == []
    assert resp.usage["input_tokens"] == 10
    assert resp.usage["output_tokens"] == 5


def test_openai_adapter_normalizes_tool_calls(mocker: MagicMock) -> None:
    fake_tc = MagicMock()
    fake_tc.id = "call_abc"
    fake_tc.function.name = "bash"
    fake_tc.function.arguments = '{"command": "ls"}'

    fake_response = MagicMock()
    fake_response.choices[0].message.content = None
    fake_response.choices[0].message.tool_calls = [fake_tc]
    fake_response.usage.prompt_tokens = 20
    fake_response.usage.completion_tokens = 8
    fake_response.model = "gpt-4o-2024-08-06"

    with patch("openai.OpenAI") as mock_client_class:
        mock_client_class.return_value.chat.completions.create.return_value = fake_response
        adapter = OpenAIAdapter(api_key="test-key")
        resp = adapter.call("gpt-4o-2024-08-06", _make_request(tools=True))

    assert resp.tool_calls == [{"id": "call_abc", "name": "bash", "arguments": '{"command": "ls"}'}]


def test_anthropic_adapter_normalizes_text_response(mocker: MagicMock) -> None:
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = "Fixed."

    fake_response = MagicMock()
    fake_response.content = [fake_block]
    fake_response.usage.input_tokens = 15
    fake_response.usage.output_tokens = 3
    fake_response.model = "claude-sonnet-4-6-20260101"

    with patch("anthropic.Anthropic") as mock_client_class:
        mock_client_class.return_value.messages.create.return_value = fake_response
        adapter = AnthropicAdapter(api_key="test-key")
        resp = adapter.call("claude-sonnet-4-6-20260101", _make_request())

    assert resp.content == "Fixed."
    assert resp.tool_calls == []


def test_anthropic_adapter_normalizes_tool_use_response(mocker: MagicMock) -> None:
    fake_block = MagicMock()
    fake_block.type = "tool_use"
    fake_block.id = "toolu_01"
    fake_block.name = "bash"
    fake_block.input = {"command": "ls"}

    fake_response = MagicMock()
    fake_response.content = [fake_block]
    fake_response.usage.input_tokens = 20
    fake_response.usage.output_tokens = 10
    fake_response.model = "claude-sonnet-4-6-20260101"

    with patch("anthropic.Anthropic") as mock_client_class:
        mock_client_class.return_value.messages.create.return_value = fake_response
        adapter = AnthropicAdapter(api_key="test-key")
        resp = adapter.call("claude-sonnet-4-6-20260101", _make_request(tools=True))

    assert resp.tool_calls == [
        {"id": "toolu_01", "name": "bash", "arguments": json.dumps({"command": "ls"})}
    ]


def test_compat_adapters_use_correct_base_urls() -> None:
    from routelet.adapters.compat import CerebrasAdapter, GoogleAdapter, GroqAdapter, OllamaAdapter

    with patch("openai.OpenAI") as mock_class:
        GroqAdapter(api_key="k")
        _, kwargs = mock_class.call_args
        assert "groq.com" in kwargs.get("base_url", "")

    with patch("openai.OpenAI") as mock_class:
        CerebrasAdapter(api_key="k")
        _, kwargs = mock_class.call_args
        assert "cerebras" in kwargs.get("base_url", "")

    with patch("openai.OpenAI") as mock_class:
        OllamaAdapter()
        _, kwargs = mock_class.call_args
        assert "localhost" in kwargs.get("base_url", "")

    with patch("openai.OpenAI") as mock_class:
        GoogleAdapter(api_key="k")
        _, kwargs = mock_class.call_args
        assert "generativelanguage" in kwargs.get("base_url", "")
