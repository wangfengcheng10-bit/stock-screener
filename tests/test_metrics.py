import statistics
from datetime import date

import pytest
from conftest import annual_cash_flow, annual_income, balance, cash_flow, quarterly_income

from screener.models import EarningsSurprise, FiscalPeriod, FundamentalSnapshot
from screener.pipeline import metrics as m


def test_rev_growth_ttm_yoy():
    snap = FundamentalSnapshot(ticker="T", income_quarterly=quarterly_income([110] * 4 + [100] * 4, [10] * 8))
    assert m.rev_growth_ttm_yoy(snap) == pytest.approx(0.10)


def test_rev_growth_ttm_yoy_missing_with_insufficient_history():
    snap = FundamentalSnapshot(ticker="T", income_quarterly=quarterly_income([110] * 4, [10] * 4))
    assert m.rev_growth_ttm_yoy(snap) is None


def test_rev_cagr_3y():
    snap = FundamentalSnapshot(ticker="T", income_annual=annual_income([133.1, 121, 110, 100, 90], [10] * 5))
    assert m.rev_cagr_3y(snap) == pytest.approx(0.10, rel=1e-3)


def test_growth_acceleration_and_consistency():
    revenues = [140, 130, 120, 110, 100, 100, 100, 100]
    snap = FundamentalSnapshot(ticker="T", income_quarterly=quarterly_income(revenues, [10] * 8))
    expected_series = [0.40, 0.30, 0.20, 0.10]
    assert m.growth_acceleration(snap) == pytest.approx(expected_series[0] - expected_series[3])
    assert m.growth_consistency(snap) == pytest.approx(-statistics.pstdev(expected_series))


def test_net_margin_ttm():
    snap = FundamentalSnapshot(ticker="T", income_quarterly=quarterly_income([110] * 4, [11] * 4))
    assert m.net_margin_ttm(snap) == pytest.approx(0.10)


def test_net_debt_ebitda_and_current_ratio_and_debt_equity():
    snap = FundamentalSnapshot(
        ticker="T",
        income_quarterly=quarterly_income([100] * 4, [10] * 4),  # ebitda_ttm = 0.25*100*4 = 100
        balance_sheet=balance(total_debt=500, cash=100, equity=1000, assets=2000, current_assets=300, current_liabilities=150),
    )
    assert m.net_debt_ebitda(snap) == pytest.approx(4.0)
    assert m.current_ratio(snap) == pytest.approx(2.0)
    assert m.debt_equity(snap) == pytest.approx(0.5)


def test_fcf_margin_and_conversion():
    snap = FundamentalSnapshot(
        ticker="T",
        income_quarterly=quarterly_income([100] * 4, [20] * 4),
        cash_flow=cash_flow([30] * 4, [-10] * 4),  # ocf_ttm=120, capex_ttm=-40, fcf_ttm=80
    )
    assert m.fcf_margin(snap) == pytest.approx(0.20)  # 80 / revenue_ttm(400)
    assert m.fcf_conversion(snap) == pytest.approx(1.0)  # 80 / net_income_ttm(80)


def test_dilution_penalizes_share_count_increase():
    snap = FundamentalSnapshot(
        ticker="T",
        income_annual=annual_income([100] * 5, [10] * 5, diluted_shares=[1100, 1075, 1050, 1000, 950]),
    )
    assert m.dilution(snap) == pytest.approx(-0.10)  # +10% dilution -> -0.10 metric


def test_earnings_surprise_history_averages_available_surprises():
    surprises = [
        EarningsSurprise(period=FiscalPeriod(fiscal_year=2026, period_end=date(2026, 6, 30)), surprise_pct=5.0),
        EarningsSurprise(period=FiscalPeriod(fiscal_year=2026, period_end=date(2026, 3, 31)), surprise_pct=3.0),
    ]
    snap = FundamentalSnapshot(ticker="T", earnings_surprises=surprises)
    assert m.earnings_surprise_history(snap) == pytest.approx(4.0)


def test_ocf_cagr_3y():
    snap = FundamentalSnapshot(ticker="T", cash_flow_annual=annual_cash_flow([133.1, 121, 110, 100, 90], [-10] * 5))
    assert m.ocf_cagr_3y(snap) == pytest.approx(0.10, rel=1e-3)


def test_ocf_cagr_3y_missing_when_base_year_not_positive():
    snap = FundamentalSnapshot(ticker="T", cash_flow_annual=annual_cash_flow([133.1, 121, 110, -50, 90], [-10] * 5))
    assert m.ocf_cagr_3y(snap) is None


def test_fcf_consistency_counts_positive_years():
    # FCF per year = ocf + capex: 90, 40, -5, -20, 10 -> 3 positive years
    snap = FundamentalSnapshot(
        ticker="T",
        cash_flow_annual=annual_cash_flow([100, 50, 5, 0, 20], [-10, -10, -10, -20, -10]),
    )
    assert m.fcf_consistency(snap) == pytest.approx(3.0)


def test_annual_cashflow_metrics_missing_when_no_annual_data():
    snap = FundamentalSnapshot(ticker="T")
    assert m.ocf_cagr_3y(snap) is None
    assert m.fcf_consistency(snap) is None


def test_empty_snapshot_all_metrics_return_none_not_crash():
    snap = FundamentalSnapshot(ticker="EMPTY")
    for metric_def in m.METRIC_REGISTRY:
        assert metric_def.compute(snap) is None
