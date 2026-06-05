from routelet.core.request import Message, NormalizedRequest, NormalizedResponse, ToolDefinition


def test_normalized_request_constructs() -> None:
    req = NormalizedRequest(
        messages=[Message(role="user", content="hello")],
        tools=[],
    )
    assert req.temperature == 0.0
    assert req.max_tokens == 4096
    assert req.stream is False


def test_message_with_tool_call_id() -> None:
    msg = Message(role="tool", content="result", tool_call_id="call_123")
    assert msg.tool_call_id == "call_123"


def test_tool_definition() -> None:
    tool = ToolDefinition(
        name="bash",
        description="Run a shell command",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )
    assert tool.name == "bash"


def test_normalized_response_constructs() -> None:
    resp = NormalizedResponse(
        content="hi",
        tool_calls=[],
        usage={"input_tokens": 10, "output_tokens": 5},
        model="claude-sonnet-4-6-20260101",
        latency_ms=120.0,
    )
    assert resp.content == "hi"
