from __future__ import annotations

from abc import ABC, abstractmethod

import httpx

from screener.models import FundamentalSnapshot, Holding


class HoldingsProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def get_holdings(self, etf_ticker: str) -> list[Holding]: ...


class FundamentalsProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def get_snapshot(self, ticker: str) -> FundamentalSnapshot: ...


def is_retryable_http_error(exc: BaseException) -> bool:
    """A 404/other 4xx means 'this endpoint is wrong', not 'try again' —
    only rate limits and server errors are worth retrying."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)
