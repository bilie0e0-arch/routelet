from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from routelet.core.request import Message, NormalizedRequest, ToolDefinition
from routelet.core.router import Router


class _Completions:
    def __init__(self, router: Router):
        self._router = router

    def create(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        request = _build_request(messages, tools, kwargs)
        response, _ = self._router.route(request)
        return _to_openai_response(response)


class _Chat:
    def __init__(self, router: Router):
        self.completions = _Completions(router)


class RouteletClient:
    """Drop-in replacement for openai.OpenAI(). Use client.chat.completions.create(...)."""

    def __init__(self, router: Router):
        self.chat = _Chat(router)


def _build_request(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    kwargs: dict[str, Any],
) -> NormalizedRequest:
    norm_messages = [
        Message(
            role=m["role"],
            content=m.get("content", ""),
            tool_call_id=m.get("tool_call_id"),
            name=m.get("name"),
        )
        for m in messages
    ]
    norm_tools = []
    if tools:
        for t in tools:
            fn = t.get("function", t)
            norm_tools.append(
                ToolDefinition(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters", {}),
                )
            )
    return NormalizedRequest(
        messages=norm_messages,
        tools=norm_tools,
        temperature=float(kwargs.get("temperature", 0.0)),
        max_tokens=int(kwargs.get("max_tokens", 4096)),
    )


def _to_openai_response(response: Any) -> SimpleNamespace:
    """Return a minimal OpenAI-shaped response object matching the openai SDK shape."""
    tool_calls = (
        [
            SimpleNamespace(
                id=tc["id"],
                type="function",
                function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
            )
            for tc in response.tool_calls
        ]
        if response.tool_calls
        else None
    )
    msg = SimpleNamespace(content=response.content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=response.usage["input_tokens"],
        completion_tokens=response.usage["output_tokens"],
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=response.model)
