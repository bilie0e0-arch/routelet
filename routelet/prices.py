import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_prices() -> dict:  # type: ignore[type-arg]
    path = Path(__file__).parent.parent / "prices.json"
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost using list prices from prices.json. Returns 0.0 for unknown models."""
    models = load_prices().get("models", {})
    entry = models.get(model)
    if entry is None:
        return 0.0
    return float(
        input_tokens * entry["input_per_1m"] / 1_000_000
        + output_tokens * entry["output_per_1m"] / 1_000_000
    )
