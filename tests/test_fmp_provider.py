from __future__ import annotations

import httpx
import pytest

from screener.providers.fmp import FMPFundamentalsProvider, FMPHoldingsProvider


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    """Routes GETs by matching a substring in the URL, so one fake client can
    stand in for all the parallel calls FMPFundamentalsProvider makes."""

    def __init__(self, routes: dict[str, object]):
        self._routes = routes
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url: str, params=None):
        self.calls.append((url, params or {}))
        for key, payload in self._routes.items():
            if key in url:
                return _FakeResponse(payload)
        raise AssertionError(f"unexpected URL: {url}")

    async def aclose(self):
        pass


async def test_holdings_provider_maps_rows_and_drops_blank_tickers():
    client = _FakeClient(
        {
            "etf-holdings": [
                {"asset": "AAPL", "name": "Apple", "weightPercentage": 7.5, "sector": "Technology"},
                {"asset": "", "name": "Cash", "weightPercentage": 1.0},
            ]
        }
    )
    provider = FMPHoldingsProvider(api_key="test-key", client=client)
    holdings = await provider.get_holdings("QQQ")
    assert len(holdings) == 1
    assert holdings[0].ticker == "AAPL"
    assert holdings[0].weight_pct == 7.5


async def test_fundamentals_provider_builds_snapshot():
    client = _FakeClient(
        {
            "income-statement": [
                {"date": "2026-06-30", "period": "Q2", "revenue": 1000, "netIncome": 200, "epsdiluted": 1.5}
            ],
            "balance-sheet-statement": [{"date": "2026-06-30", "totalDebt": 500, "cashAndCashEquivalents": 100}],
            "cash-flow-statement": [{"date": "2026-06-30", "operatingCashFlow": 300, "capitalExpenditure": -50}],
            "analyst-estimates": [{"date": "2027-12-31", "estimatedRevenueAvg": 5000, "estimatedEpsAvg": 6.0}],
            "earnings-surprises": [{"date": "2026-06-30", "actualEarningResult": 1.6, "estimatedEarning": 1.5}],
            "profile": [{"sector": "Technology", "industry": "Software", "mktCap": 1_000_000, "currency": "USD"}],
        }
    )
    provider = FMPFundamentalsProvider(api_key="test-key", client=client)
    snapshot = await provider.get_snapshot("AAPL")
    assert snapshot.fetch_failed is False
    assert snapshot.sector == "Technology"
    assert snapshot.income_quarterly[0].revenue == 1000
    assert snapshot.income_quarterly[0].period.fiscal_quarter == 2
    assert snapshot.earnings_surprises[0].surprise_pct == pytest.approx((1.6 - 1.5) / 1.5 * 100)

    # quarterly and annual cash flow must be two distinct requests, not one reused payload
    cashflow_periods = sorted(p.get("period") for url, p in client.calls if "cash-flow-statement" in url)
    assert cashflow_periods == ["annual", "quarter"]
    assert snapshot.cash_flow_annual[0].operating_cash_flow == 300


async def test_fundamentals_provider_skips_rows_missing_date_instead_of_crashing():
    client = _FakeClient(
        {
            "income-statement": [],
            "balance-sheet-statement": [],
            "cash-flow-statement": [],
            "analyst-estimates": [{"estimatedRevenueAvg": 5000}],  # no "date"
            "earnings-surprises": [{"actualEarningResult": 1.6, "estimatedEarning": 1.5}],  # no "date"
            "profile": [{}],
        }
    )
    provider = FMPFundamentalsProvider(api_key="test-key", client=client)
    snapshot = await provider.get_snapshot("AAPL")
    assert snapshot.analyst_estimates == []
    assert snapshot.earnings_surprises == []


async def test_fundamentals_provider_marks_fetch_failed_on_http_error():
    class _FailingClient:
        async def get(self, url, params=None):
            request = httpx.Request("GET", url)
            response = httpx.Response(404, request=request)  # non-retryable, fails fast
            raise httpx.HTTPStatusError("not found", request=request, response=response)

        async def aclose(self):
            pass

    provider = FMPFundamentalsProvider(api_key="test-key", client=_FailingClient())
    snapshot = await provider.get_snapshot("AAPL")
    assert snapshot.fetch_failed is True
