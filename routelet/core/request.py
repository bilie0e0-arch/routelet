from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | list[dict[str, Any]]
    tool_call_id: str | None = None  # set when role == "tool"
    name: str | None = None  # set when role == "tool" (tool name)


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass
class NormalizedRequest:
    messages: list[Message]
    tools: list[ToolDefinition] = field(default_factory=list)
    framework_context: dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class NormalizedResponse:
    content: str | None
    tool_calls: list[dict[str, Any]]  # [{"id": str, "name": str, "arguments": str}, ...]
    usage: dict[str, int]  # {"input_tokens": int, "output_tokens": int}
    model: str  # actual model ID used
    latency_ms: float
