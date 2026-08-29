from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from screener.cache.store import CacheStore
from screener.models import FundamentalSnapshot
from screener.providers.base import FundamentalsProvider

logger = logging.getLogger(__name__)


@dataclass
class FetchFailure:
    ticker: str
    reason: str


@dataclass
class FetchResult:
    snapshots: dict[str, FundamentalSnapshot] = field(default_factory=dict)
    failures: list[FetchFailure] = field(default_factory=list)


async def fetch_all_fundamentals(
    tickers: list[str],
    provider: FundamentalsProvider,
    cache: CacheStore | None = None,
    max_concurrent: int = 8,
    use_cache: bool = True,
) -> FetchResult:
    """Bounded-concurrency fetch. Failed tickers never abort the run — they're
    collected with a reason and excluded from the scored universe downstream."""
    semaphore = asyncio.Semaphore(max_concurrent)
    result = FetchResult()
    endpoint = f"fundamentals:{provider.name}"

    async def _fetch_one(ticker: str) -> None:
        async with semaphore:
            if use_cache and cache is not None:
                cached = cache.get(ticker, endpoint)
                if cached is not None:
                    result.snapshots[ticker] = FundamentalSnapshot.model_validate(cached)
                    return
            try:
                snapshot = await provider.get_snapshot(ticker)
            except Exception as exc:
                result.failures.append(FetchFailure(ticker=ticker, reason=str(exc)))
                logger.warning("Fundamentals fetch failed for %s: %s", ticker, exc)
                return

            if snapshot.fetch_failed:
                result.failures.append(FetchFailure(ticker=ticker, reason=snapshot.fetch_error or "unknown"))
                return

            result.snapshots[ticker] = snapshot
            if cache is not None:
                cache.set(ticker, endpoint, snapshot.model_dump(mode="json"))

    await asyncio.gather(*(_fetch_one(t) for t in tickers))
    return result
