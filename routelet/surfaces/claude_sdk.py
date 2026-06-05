"""
Claude Agent SDK hook factories for telemetry + step-type signal collection.

These hooks do NOT reroute model calls (PreToolUse fires after the model call).
They feed tool-name context into routelet telemetry so the Phase 2 classifier
learns which tool calls correspond to which step types.

Registration example:
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
    from routelet.surfaces.claude_sdk import make_pre_tool_use_hook, make_post_tool_use_hook

    options = ClaudeAgentOptions(hooks={
        "PreToolUse":  [HookMatcher(hooks=[make_pre_tool_use_hook(on_tool=my_logger)])],
        "PostToolUse": [HookMatcher(hooks=[make_post_tool_use_hook(on_result=my_result_logger)])],
    })
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

HookFn = Callable[[dict[str, Any], str | None, Any], Any]


def make_pre_tool_use_hook(on_tool: Callable[[str], None] | None = None) -> HookFn:
    """Returns a PreToolUse hook. Calls on_tool(tool_name) before each tool. Returns {} to allow."""

    async def _hook(
        input_data: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        tool_name = input_data.get("tool_name", "unknown")
        if on_tool is not None:
            on_tool(tool_name)
        return {}

    return _hook


def make_post_tool_use_hook(
    on_result: Callable[[str, Any], None] | None = None,
) -> HookFn:
    """Returns a PostToolUse hook. Calls on_result(tool_name, response) after each tool."""

    async def _hook(
        input_data: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        tool_name = input_data.get("tool_name", "unknown")
        tool_response = input_data.get("tool_response")
        if on_result is not None:
            on_result(tool_name, tool_response)
        return {}

    return _hook
