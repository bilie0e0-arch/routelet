from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoutingDecision:
    request_id: str
    timestamp: float
    step_type: str | None  # None until Phase 2 classifier is live
    classifier_confidence: float | None
    ood_score: float | None
    chosen_model: str
    list_price_cost_usd: float
    input_tokens: int
    output_tokens: int
    latency_ms: float
    downstream_success: bool | None  # filled in after full task completes
