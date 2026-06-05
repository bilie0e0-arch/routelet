import time
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from routelet.telemetry.schema import RoutingDecision
from routelet.telemetry.sqlite import SQLiteTelemetry


def make_decision() -> RoutingDecision:
    return RoutingDecision(
        request_id="req_001",
        timestamp=time.time(),
        step_type=None,
        classifier_confidence=None,
        ood_score=None,
        chosen_model="llama-3.3-8b-instant",
        list_price_cost_usd=0.00001,
        input_tokens=100,
        output_tokens=20,
        latency_ms=80.0,
        downstream_success=None,
    )


@pytest.mark.asyncio  # type: ignore[misc]
async def test_sqlite_telemetry_logs_decision(tmp_path: Path) -> None:
    db_path = tmp_path / "telemetry.db"
    async with SQLiteTelemetry(str(db_path)) as tel:
        await tel.log(make_decision())

    async with (
        aiosqlite.connect(str(db_path)) as db,
        db.execute("SELECT chosen_model FROM routing_decisions") as cursor,
    ):
        rows: list[Any] = await cursor.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "llama-3.3-8b-instant"


@pytest.mark.asyncio  # type: ignore[misc]
async def test_sqlite_telemetry_creates_table_idempotently(tmp_path: Path) -> None:
    db_path = tmp_path / "telemetry.db"
    async with SQLiteTelemetry(str(db_path)) as tel:
        await tel.log(make_decision())
    async with SQLiteTelemetry(str(db_path)) as tel:
        await tel.log(make_decision())
