from __future__ import annotations

import time
from typing import Any

import openai

from routelet.adapters.base import AdapterBase
from routelet.core.request import NormalizedRequest, NormalizedResponse


class OpenAIAdapter(AdapterBase):
    def __init__(self, api_key: str, base_url: str | None = None):
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def call(self, model: str, request: NormalizedRequest) -> NormalizedResponse:
        messages = self._to_openai_messages(request)
        tools: Any = self._to_openai_tools(request) or openai.NOT_GIVEN

        start = time.monotonic()
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        latency_ms = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        return NormalizedResponse(
            content=choice.message.content,
            tool_calls=[
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (choice.message.tool_calls or [])
            ],
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            model=response.model,
            latency_ms=latency_ms,
        )

    def _to_openai_messages(self, request: NormalizedRequest) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in request.messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    def _to_openai_tools(self, request: NormalizedRequest) -> list[dict[str, Any]] | None:
        if not request.tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in request.tools
        ]
