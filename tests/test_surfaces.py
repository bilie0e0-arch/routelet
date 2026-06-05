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
