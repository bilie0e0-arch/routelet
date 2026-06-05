import asyncio
from unittest.mock import MagicMock

from routelet.core.request import NormalizedResponse
from routelet.surfaces.openai_client import RouteletClient
from routelet.telemetry.schema import RoutingDecision


def _mock_router(content: str = "hello") -> MagicMock:
    resp = NormalizedResponse(
        content=content,
        tool_calls=[],
        usage={"input_tokens": 5, "output_tokens": 2},
        model="m",
        latency_ms=10.0,
    )
    decision = MagicMock(spec=RoutingDecision)
    router = MagicMock()
    router.route.return_value = (resp, decision)
    return router


def test_routelet_client_create_returns_content() -> None:
    client = RouteletClient(_mock_router())
    result = client.chat.completions.create(
        model="any",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result.choices[0].message.content == "hello"


def test_routelet_client_create_with_tools() -> None:
    client = RouteletClient(_mock_router())
    result = client.chat.completions.create(
        model="any",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "bash", "description": "run", "parameters": {}},
            }
        ],
    )
    assert result is not None


def test_langchain_surface_invokes_router(mocker: MagicMock) -> None:
    from langchain_core.messages import HumanMessage

    from routelet.surfaces.langchain import RouteletLangChain

    router = _mock_router("langchain response")
    lc = RouteletLangChain(router=router)
    result = lc.invoke([HumanMessage(content="hello")])
    assert result.content == "langchain response"
    router.route.assert_called_once()


def test_pre_tool_use_hook_logs_tool_name() -> None:
    from routelet.surfaces.claude_sdk import make_pre_tool_use_hook

    log: list[str] = []
    hook_fn = make_pre_tool_use_hook(on_tool=lambda name: log.append(name))

    input_data = {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_use_id": "toolu_01",
        "session_id": "sess_abc",
        "cwd": "/home/user",
    }
    result = asyncio.get_event_loop().run_until_complete(hook_fn(input_data, "toolu_01", None))

    assert log == ["Bash"]
    assert result == {}


def test_post_tool_use_hook_returns_empty() -> None:
    from routelet.surfaces.claude_sdk import make_post_tool_use_hook

    hook_fn = make_post_tool_use_hook(on_result=lambda name, resp: None)
    input_data = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_response": "file1.py\nfile2.py",
        "tool_use_id": "toolu_01",
        "session_id": "sess_abc",
        "cwd": "/home/user",
    }
    result = asyncio.get_event_loop().run_until_complete(hook_fn(input_data, "toolu_01", None))
    assert result == {}
