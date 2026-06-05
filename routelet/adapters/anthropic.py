from __future__ import annotations

import json
import time
from typing import Any

import anthropic

from routelet.adapters.base import AdapterBase
from routelet.core.request import NormalizedRequest, NormalizedResponse


class AnthropicAdapter(AdapterBase):
    def __init__(self, api_key: str):
        self._client = anthropic.Anthropic(api_key=api_key)

    def call(self, model: str, request: NormalizedRequest) -> NormalizedResponse:
        system, messages = self._split_system(request)
        tools = self._to_anthropic_tools(request)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        start = time.monotonic()
        response = self._client.messages.create(**kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        content = None
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {"id": block.id, "name": block.name, "arguments": json.dumps(block.input)}
                )

        return NormalizedResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            model=response.model,
            latency_ms=latency_ms,
        )

    def _split_system(self, request: NormalizedRequest) -> tuple[str | None, list[dict[str, Any]]]:
        system = None
        messages: list[dict[str, Any]] = []
        for m in request.messages:
            if m.role == "system":
                system = m.content if isinstance(m.content, str) else ""
            elif m.role == "tool":
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content,
                            }
                        ],
                    }
                )
            else:
                messages.append({"role": m.role, "content": m.content})
        return system, messages

    def _to_anthropic_tools(self, request: NormalizedRequest) -> list[dict[str, Any]] | None:
        if not request.tools:
            return None
        return [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in request.tools
        ]
