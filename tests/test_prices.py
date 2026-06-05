import json
from pathlib import Path

from routelet.prices import compute_cost, load_prices

REQUIRED_FIELDS = {"input_per_1m", "output_per_1m"}


def test_prices_json_loads() -> None:
    data = json.loads(Path("prices.json").read_text())
    assert "snapshot_date" in data
    assert "models" in data
    assert len(data["models"]) > 0


def test_prices_json_all_models_have_required_fields() -> None:
    data = json.loads(Path("prices.json").read_text())
    for name, entry in data["models"].items():
        assert entry.keys() >= REQUIRED_FIELDS, f"{name} missing fields"


def test_compute_cost_known_model() -> None:
    # gpt-4o costs $2.50/1M input, $10.00/1M output
    cost = compute_cost("gpt-4o-2024-08-06", input_tokens=1000, output_tokens=500)
    assert abs(cost - (1000 * 2.50 / 1_000_000 + 500 * 10.00 / 1_000_000)) < 1e-9


def test_compute_cost_unknown_model_returns_zero() -> None:
    cost = compute_cost("unknown-model-xyz", input_tokens=1000, output_tokens=500)
    assert cost == 0.0


def test_load_prices_has_snapshot_date() -> None:
    prices = load_prices()
    assert "snapshot_date" in prices
