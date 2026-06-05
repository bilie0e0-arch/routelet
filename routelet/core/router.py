from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping
from typing import Any

from routelet.adapters.base import AdapterBase
from routelet.core.policy import PolicyBase
from routelet.core.request import NormalizedRequest, NormalizedResponse
from routelet.prices import compute_cost
from routelet.telemetry.schema import RoutingDecision


class Router:
    def __init__(
        self,
        policy: PolicyBase,
        adapters: Mapping[str, AdapterBase],
        model_to_provider: Mapping[str, str],
        telemetry: Any | None = None,
    ):
        self._policy = policy
        self._adapters = adapters
        self._model_to_provider = model_to_provider
        self._telemetry = telemetry

    def route(self, request: NormalizedRequest) -> tuple[NormalizedResponse, RoutingDecision]:
        model = self._policy.select_model(request)
        provider = self._model_to_provider[model]
        adapter = self._adapters[provider]

        response = adapter.call(model, request)

        decision = RoutingDecision(
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
            step_type=None,
            classifier_confidence=None,
            ood_score=None,
            chosen_model=model,
            list_price_cost_usd=compute_cost(
                model, response.usage["input_tokens"], response.usage["output_tokens"]
            ),
            input_tokens=response.usage["input_tokens"],
            output_tokens=response.usage["output_tokens"],
            latency_ms=response.latency_ms,
            downstream_success=None,
        )

        if self._telemetry is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._telemetry.log(decision))
            except RuntimeError:
                pass  # no running loop — telemetry skipped in sync context

        return response, decision
