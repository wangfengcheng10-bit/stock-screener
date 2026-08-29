from datetime import date

from screener.models import FundamentalSnapshot
from screener.pipeline.fetch import fetch_all_fundamentals
from screener.providers.base import FundamentalsProvider


class _StubFundamentalsProvider(FundamentalsProvider):
    name = "stub"

    def __init__(self, fail_tickers: set[str] | None = None):
        self._fail_tickers = fail_tickers or set()
        self.calls: list[str] = []

    async def get_snapshot(self, ticker: str) -> FundamentalSnapshot:
        self.calls.append(ticker)
        if ticker in self._fail_tickers:
            return FundamentalSnapshot(ticker=ticker, fetch_failed=True, fetch_error="no data")
        return FundamentalSnapshot(ticker=ticker, market_cap=1_000_000.0)


async def test_failed_tickers_collected_not_raised():
    provider = _StubFundamentalsProvider(fail_tickers={"BADCO"})
    result = await fetch_all_fundamentals(["AAPL", "BADCO"], provider, cache=None, use_cache=False)
    assert "AAPL" in result.snapshots
    assert "BADCO" not in result.snapshots
    assert result.failures[0].ticker == "BADCO"


async def test_second_run_hits_cache(tmp_path):
    from screener.cache.store import CacheStore

    cache = CacheStore(tmp_path, ttl_hours=24)
    provider = _StubFundamentalsProvider()

    await fetch_all_fundamentals(["AAPL"], provider, cache=cache, use_cache=True)
    assert provider.calls == ["AAPL"]

    await fetch_all_fundamentals(["AAPL"], provider, cache=cache, use_cache=True)
    assert provider.calls == ["AAPL"]  # no second network call — served from cache
