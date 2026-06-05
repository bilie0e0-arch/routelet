from unittest.mock import AsyncMock, MagicMock

from routelet.core.request import Message, NormalizedRequest, NormalizedResponse
from routelet.core.router import Router
from routelet.policies.naive import CheapPolicy


def _make_response(model: str) -> NormalizedResponse:
    return NormalizedResponse(
        content="done",
        tool_calls=[],
        usage={"input_tokens": 50, "output_tokens": 10},
        model=model,
        latency_ms=100.0,
    )


def test_router_calls_policy_then_adapter_then_telemetry(mocker: MagicMock) -> None:
    policy = CheapPolicy()
    adapter = MagicMock()
    adapter.call.return_value = _make_response("llama-3.3-8b-instant")
    telemetry = MagicMock()
    telemetry.log = AsyncMock()

    adapters = {"groq": adapter}
    model_to_provider = {"llama-3.3-8b-instant": "groq"}

    router = Router(
        policy=policy,
        adapters=adapters,
        model_to_provider=model_to_provider,
        telemetry=telemetry,
    )
    req = NormalizedRequest(messages=[Message(role="user", content="hello")])
    resp, decision = router.route(req)

    assert resp.content == "done"
    assert decision.chosen_model == "llama-3.3-8b-instant"
    adapter.call.assert_called_once()


def test_router_decision_includes_cost(mocker: MagicMock) -> None:
    from routelet.constants import MODELS

    policy = CheapPolicy()
    adapter = MagicMock()
    adapter.call.return_value = _make_response(MODELS["cheap_groq"])
    telemetry = MagicMock()
    telemetry.log = AsyncMock()

    adapters = {"groq": adapter}
    model_to_provider = {MODELS["cheap_groq"]: "groq"}

    router = Router(
        policy=policy,
        adapters=adapters,
        model_to_provider=model_to_provider,
        telemetry=telemetry,
    )
    req = NormalizedRequest(messages=[Message(role="user", content="hello")])
    _, decision = router.route(req)

    assert decision.list_price_cost_usd >= 0.0
    assert decision.input_tokens == 50
    assert decision.output_tokens == 10
