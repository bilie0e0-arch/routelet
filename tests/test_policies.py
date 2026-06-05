from routelet.constants import MODELS
from routelet.core.request import Message, NormalizedRequest
from routelet.policies.naive import CheapPolicy, RandomPolicy, TopTierPolicy


def _req() -> NormalizedRequest:
    return NormalizedRequest(messages=[Message(role="user", content="hello")])


def test_top_tier_always_returns_strong_model() -> None:
    policy = TopTierPolicy()
    assert policy.select_model(_req()) == MODELS["strong_anthropic"]


def test_cheap_always_returns_cheap_model() -> None:
    policy = CheapPolicy()
    assert policy.select_model(_req()) == MODELS["cheap_groq"]


def test_random_returns_a_known_model() -> None:
    policy = RandomPolicy(seed=42)
    result = policy.select_model(_req())
    assert result in MODELS.values()


def test_random_is_deterministic_with_seed() -> None:
    p1 = RandomPolicy(seed=42)
    p2 = RandomPolicy(seed=42)
    results = [p1.select_model(_req()) for _ in range(10)]
    results2 = [p2.select_model(_req()) for _ in range(10)]
    assert results == results2
