import pytest

from screener.models import AssetClass, Holding
from screener.pipeline.holdings import resolve_holdings
from screener.providers.base import HoldingsProvider


class _FailingProvider(HoldingsProvider):
    name = "issuer"

    async def get_holdings(self, etf_ticker: str) -> list[Holding]:
        raise RuntimeError("404 Not Found")


class _StubProvider(HoldingsProvider):
    name = "fmp"

    def __init__(self, holdings: list[Holding]):
        self._holdings = holdings

    async def get_holdings(self, etf_ticker: str) -> list[Holding]:
        return self._holdings


async def test_cascade_falls_through_on_issuer_failure():
    stub_holdings = [
        Holding(ticker="AAPL", name="Apple", weight_pct=10.0, asset_class=AssetClass.EQUITY, source="fmp"),
        Holding(ticker="MSFT", name="Microsoft", weight_pct=9.0, asset_class=AssetClass.EQUITY, source="fmp"),
    ]
    cascade = [_FailingProvider(), _StubProvider(stub_holdings)]
    holdings, report = await resolve_holdings("QQQ", cascade)
    assert report.source_used == "fmp"
    assert {h.ticker for h in holdings} == {"AAPL", "MSFT"}


async def test_cascade_raises_when_all_providers_fail():
    cascade = [_FailingProvider(), _FailingProvider()]
    with pytest.raises(RuntimeError):
        await resolve_holdings("QQQ", cascade)


async def test_drops_non_equity_and_dedupes_multi_class():
    holdings_in = [
        Holding(ticker="GOOG", name="Alphabet C", weight_pct=3.0, asset_class=AssetClass.EQUITY, source="fmp"),
        Holding(ticker="GOOGL", name="Alphabet A", weight_pct=4.0, asset_class=AssetClass.EQUITY, source="fmp"),
        Holding(ticker="USD", name="Cash", weight_pct=1.0, asset_class=AssetClass.CASH, source="fmp"),
    ]
    cascade = [_StubProvider(holdings_in)]
    holdings, report = await resolve_holdings("QQQ", cascade)
    tickers = {h.ticker for h in holdings}
    assert tickers == {"GOOGL"}
    assert report.dropped_non_equity == 1
    assert report.dropped_duplicate_class == [("GOOGL", "GOOG")]


async def test_ticker_normalization_applied():
    holdings_in = [
        Holding(ticker="BRK.B", name="Berkshire", weight_pct=2.0, asset_class=AssetClass.EQUITY, source="fmp"),
    ]
    cascade = [_StubProvider(holdings_in)]
    holdings, _ = await resolve_holdings("SPY", cascade)
    assert holdings[0].ticker == "BRK-B"
