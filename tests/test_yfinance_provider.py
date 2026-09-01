from __future__ import annotations

import pandas as pd
import pytest

from screener.providers import yfinance_provider as yfp


class _FakeTicker:
    def __init__(self):
        self.info = {
            "sector": "Technology",
            "industry": "Software",
            "marketCap": 1_000_000,
            "currency": "USD",
            "revenueGrowth": 0.1,
            "forwardEps": 5.0,
        }
        cols = [pd.Timestamp("2026-06-30"), pd.Timestamp("2026-03-31"), pd.Timestamp("2025-12-31"), pd.Timestamp("2025-09-30")]
        self.quarterly_financials = pd.DataFrame(
            {cols[0]: [1000.0, 200.0], cols[1]: [900.0, 180.0], cols[2]: [950.0, 190.0], cols[3]: [920.0, 185.0]},
            index=["Total Revenue", "Net Income"],
        )
        self.financials = pd.DataFrame()
        self.quarterly_balance_sheet = pd.DataFrame()
        self.quarterly_cashflow = pd.DataFrame()
        annual_cols = [pd.Timestamp("2025-12-31"), pd.Timestamp("2024-12-31")]
        self.cashflow = pd.DataFrame(
            {annual_cols[0]: [300.0, -50.0], annual_cols[1]: [250.0, -40.0]},
            index=["Operating Cash Flow", "Capital Expenditure"],
        )


async def test_yfinance_snapshot_builds_from_quarterly_financials(monkeypatch):
    monkeypatch.setattr(yfp.yf, "Ticker", lambda ticker: _FakeTicker())
    snapshot = await yfp.YFinanceFundamentalsProvider().get_snapshot("AAPL")
    assert snapshot.fetch_failed is False
    assert snapshot.sector == "Technology"
    assert snapshot.income_quarterly[0].revenue == 1000.0
    assert snapshot.income_quarterly[0].net_income == 200.0
    # TTM revenue (sum of the 4 fake quarters) triggers the forward-estimate derivation
    assert snapshot.analyst_estimates[0].consensus_eps == 5.0
    # annual cash flow is populated separately from the quarterly series
    assert len(snapshot.cash_flow_annual) == 2
    assert snapshot.cash_flow_annual[0].operating_cash_flow == 300.0
    assert snapshot.cash_flow_annual[0].free_cash_flow == 250.0  # ocf + negative capex


async def test_yfinance_snapshot_marks_fetch_failed_on_empty_financials(monkeypatch):
    class _EmptyTicker:
        info = {}
        quarterly_financials = pd.DataFrame()
        financials = pd.DataFrame()
        quarterly_balance_sheet = pd.DataFrame()
        quarterly_cashflow = pd.DataFrame()
        cashflow = pd.DataFrame()

    monkeypatch.setattr(yfp.yf, "Ticker", lambda ticker: _EmptyTicker())
    snapshot = await yfp.YFinanceFundamentalsProvider().get_snapshot("BADCO")
    assert snapshot.fetch_failed is True


async def test_yfinance_snapshot_handles_exception_from_yf(monkeypatch):
    def _raise(ticker):
        raise RuntimeError("network down")

    monkeypatch.setattr(yfp.yf, "Ticker", _raise)
    snapshot = await yfp.YFinanceFundamentalsProvider().get_snapshot("AAPL")
    assert snapshot.fetch_failed is True
    assert "network down" in snapshot.fetch_error


async def test_yfinance_holdings_provider_refuses_partial_universe():
    with pytest.raises(NotImplementedError):
        await yfp.YFinanceHoldingsProvider().get_holdings("QQQ")
