from routelet.constants import FALLBACK_MODEL, MODELS


def test_judge_model_not_in_candidates() -> None:
    for name, model_id in MODELS.items():
        assert "gpt-4o" not in model_id, (
            f"Candidate model '{name}' ({model_id}) is an OpenAI model "
            "and must be removed — GPT-4o is the judge."
        )


def test_fallback_model_in_candidates() -> None:
    assert FALLBACK_MODEL in MODELS.values()
