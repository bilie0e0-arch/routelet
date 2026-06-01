import json
from pathlib import Path

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
