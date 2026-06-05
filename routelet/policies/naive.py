from __future__ import annotations

import random

from routelet.constants import MODELS
from routelet.core.policy import PolicyBase
from routelet.core.request import NormalizedRequest


class TopTierPolicy(PolicyBase):
    """Always routes to the strongest available model."""

    def select_model(self, request: NormalizedRequest) -> str:
        return MODELS["strong_anthropic"]


class CheapPolicy(PolicyBase):
    """Always routes to the cheapest model."""

    def select_model(self, request: NormalizedRequest) -> str:
        return MODELS["cheap_groq"]


class RandomPolicy(PolicyBase):
    """Picks uniformly at random from candidate models. Seed for reproducibility."""

    def __init__(self, seed: int | None = None, candidate_models: list[str] | None = None):
        self._rng = random.Random(seed)
        self._candidates = (
            candidate_models if candidate_models is not None else list(MODELS.values())
        )

    def select_model(self, request: NormalizedRequest) -> str:
        return self._rng.choice(self._candidates)
