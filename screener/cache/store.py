from __future__ import annotations

import json
import sqlite3
import time
from datetime import date
from pathlib import Path
from typing import Any, Optional


class CacheStore:
    """TTL-based cache keyed by (ticker, endpoint, as_of_date). A re-run on the
    same calendar day hits this cache instead of the network."""

    def __init__(self, cache_dir: Path | str, ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.db_path = self.cache_dir / "cache.sqlite3"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                ticker TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                as_of_date TEXT NOT NULL,
                payload TEXT NOT NULL,
                stored_at REAL NOT NULL,
                PRIMARY KEY (ticker, endpoint, as_of_date)
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _key_date(as_of_date: Optional[date]) -> str:
        return (as_of_date or date.today()).isoformat()

    def get(self, ticker: str, endpoint: str, as_of_date: Optional[date] = None) -> Optional[Any]:
        key_date = self._key_date(as_of_date)
        row = self._conn.execute(
            "SELECT payload, stored_at FROM cache WHERE ticker=? AND endpoint=? AND as_of_date=?",
            (ticker, endpoint, key_date),
        ).fetchone()
        if row is None:
            return None
        payload, stored_at = row
        if time.time() - stored_at > self.ttl_seconds:
            return None
        return json.loads(payload)

    def set(self, ticker: str, endpoint: str, value: Any, as_of_date: Optional[date] = None) -> None:
        key_date = self._key_date(as_of_date)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (ticker, endpoint, as_of_date, payload, stored_at) VALUES (?,?,?,?,?)",
            (ticker, endpoint, key_date, json.dumps(value), time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
