"""
20-task pilot benchmark using naive policies.
Run from repo root: python -m benchmark.pilot

Requires env vars:
    GROQ_API_KEY
    ANTHROPIC_API_KEY
"""

import asyncio
import os

from datasets import load_dataset

from routelet.adapters.anthropic import AnthropicAdapter
from routelet.adapters.compat import GroqAdapter
from routelet.constants import MODELS
from routelet.core.request import Message, NormalizedRequest
from routelet.core.router import Router
from routelet.policies.naive import CheapPolicy, TopTierPolicy
from routelet.telemetry.sqlite import SQLiteTelemetry


async def main() -> None:
    adapters = {
        "groq": GroqAdapter(api_key=os.environ["GROQ_API_KEY"]),
        "anthropic": AnthropicAdapter(api_key=os.environ["ANTHROPIC_API_KEY"]),
    }
    model_to_provider = {
        MODELS["cheap_groq"]: "groq",
        MODELS["strong_anthropic"]: "anthropic",
    }

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    tasks = list(dataset)[:20]

    async with SQLiteTelemetry("pilot_telemetry.db") as tel:
        for policy_cls, label in [(CheapPolicy, "cheap"), (TopTierPolicy, "top_tier")]:
            router = Router(
                policy=policy_cls(),
                adapters=adapters,
                model_to_provider=model_to_provider,
                telemetry=tel,
            )
            for task in tasks:
                req = NormalizedRequest(
                    messages=[Message(role="user", content=task["problem_statement"])]
                )
                resp, decision = router.route(req)
                print(
                    f"[{label}] task={task['instance_id']} "
                    f"model={decision.chosen_model} "
                    f"cost=${decision.list_price_cost_usd:.6f}"
                )
            # Flush telemetry before the next policy loop closes the router
            await router.flush_telemetry()


if __name__ == "__main__":
    asyncio.run(main())
