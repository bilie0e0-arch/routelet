from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from routelet.core.request import Message, NormalizedRequest
from routelet.core.router import Router


class RouteletLangChain(BaseChatModel):  # type: ignore[misc]
    """LangChain ChatModel wrapper that routes each call through routelet."""

    router: Any  # Router — typed as Any to avoid Pydantic schema issues

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, router: Router, **kwargs: Any):
        super().__init__(router=router, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "routelet"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        request = NormalizedRequest(messages=_lc_to_normalized(messages))
        response, _ = self.router.route(request)
        ai_message = AIMessage(content=response.content or "")
        return ChatResult(generations=[ChatGeneration(message=ai_message)])


def _lc_to_normalized(messages: list[BaseMessage]) -> list[Message]:
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append(Message(role="system", content=str(m.content)))
        elif isinstance(m, HumanMessage):
            out.append(Message(role="user", content=str(m.content)))
        elif isinstance(m, AIMessage):
            out.append(Message(role="assistant", content=str(m.content)))
        elif isinstance(m, ToolMessage):
            out.append(Message(role="tool", content=str(m.content), tool_call_id=m.tool_call_id))
        else:
            out.append(Message(role="user", content=str(m.content)))
    return out
