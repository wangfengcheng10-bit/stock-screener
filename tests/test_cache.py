import time
from datetime import date
from pathlib import Path

from screener.cache.store import CacheStore


def test_cache_roundtrip(tmp_path: Path):
    store = CacheStore(tmp_path, ttl_hours=1)
    assert store.get("AAPL", "fundamentals") is None
    store.set("AAPL", "fundamentals", {"revenue": 100})
    assert store.get("AAPL", "fundamentals") == {"revenue": 100}


def test_cache_expires(tmp_path: Path):
    store = CacheStore(tmp_path, ttl_hours=0)
    store.set("AAPL", "fundamentals", {"revenue": 100})
    time.sleep(1.1)
    assert store.get("AAPL", "fundamentals") is None


def test_cache_keyed_by_as_of_date(tmp_path: Path):
    store = CacheStore(tmp_path, ttl_hours=24)
    store.set("AAPL", "fundamentals", {"revenue": 100}, as_of_date=date(2026, 1, 1))
    assert store.get("AAPL", "fundamentals", as_of_date=date(2026, 1, 2)) is None
    assert store.get("AAPL", "fundamentals", as_of_date=date(2026, 1, 1)) == {"revenue": 100}
