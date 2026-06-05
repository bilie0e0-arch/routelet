from __future__ import annotations

from types import TracebackType

import aiosqlite

from routelet.telemetry.schema import RoutingDecision

_CREATE = """
CREATE TABLE IF NOT EXISTS routing_decisions (
    request_id TEXT PRIMARY KEY,
    timestamp REAL,
    step_type TEXT,
    classifier_confidence REAL,
    ood_score REAL,
    chosen_model TEXT,
    list_price_cost_usd REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency_ms REAL,
    downstream_success INTEGER
)
"""

_INSERT = """
INSERT INTO routing_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?)
"""


class SQLiteTelemetry:
    def __init__(self, db_path: str = "routelet_telemetry.db"):
        self._path = db_path
        self._db: aiosqlite.Connection | None = None

    async def __aenter__(self) -> SQLiteTelemetry:
        self._db = await aiosqlite.connect(self._path)
        await self._db.execute(_CREATE)
        await self._db.commit()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._db:
            await self._db.close()

    async def log(self, d: RoutingDecision) -> None:
        if self._db is None:
            raise RuntimeError("SQLiteTelemetry must be used as an async context manager")
        await self._db.execute(
            _INSERT,
            (
                d.request_id,
                d.timestamp,
                d.step_type,
                d.classifier_confidence,
                d.ood_score,
                d.chosen_model,
                d.list_price_cost_usd,
                d.input_tokens,
                d.output_tokens,
                d.latency_ms,
                int(d.downstream_success) if d.downstream_success is not None else None,
            ),
        )
        await self._db.commit()
