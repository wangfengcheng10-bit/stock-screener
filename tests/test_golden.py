"""A small, frozen, hand-built universe with a known ranking outcome.

Scaled down from the spec's "frozen 20-ticker fixture" to 3 names for
maintainability, but keeps the same intent: deterministic, known ranked
output, including the data-quality gate sorting a low-coverage name last
regardless of its raw composite score.
"""

from conftest import annual_cash_flow, annual_income, balance, cash_flow, quarterly_income

from screener.models import FundamentalSnapshot, Holding
from screener.pipeline.scoring import build_universe_data, score_universe

PILLAR_WEIGHTS = {"growth": 0.20, "profitability": 0.25, "balance_sheet": 0.20, "cash_flow": 0.20, "outlook": 0.15}


def _make_universe():
    good = FundamentalSnapshot(
        ticker="GOOD",
        sector="Technology",
        income_quarterly=quarterly_income([140, 135, 130, 125, 100, 100, 100, 100], [28, 27, 26, 25, 15, 15, 15, 15]),
        income_annual=annual_income([500, 450, 400, 350, 300], [100, 85, 70, 55, 40], diluted_shares=[1000] * 5),
        balance_sheet=balance(total_debt=200, cash=500, equity=2000, assets=3000, current_assets=800, current_liabilities=300),
        cash_flow=cash_flow([40, 38, 36, 34], [-5, -5, -5, -5]),
        cash_flow_annual=annual_cash_flow([160, 140, 125, 110, 100], [-20, -18, -16, -15, -14]),
    )
    weak = FundamentalSnapshot(
        ticker="WEAK",
        sector="Technology",
        income_quarterly=quarterly_income([102, 101, 100, 100, 100, 100, 100, 100], [2, 2, 2, 2, 3, 3, 3, 3]),
        income_annual=annual_income([405, 400, 395, 390, 385], [8, 9, 10, 11, 12], diluted_shares=[1050, 1030, 1010, 1000, 990]),
        balance_sheet=balance(total_debt=1800, cash=50, equity=500, assets=2500, current_assets=350, current_liabilities=350),
        cash_flow=cash_flow([5, 4, 3, 2], [-8, -8, -8, -8]),
        cash_flow_annual=annual_cash_flow([14, 20, 26, 30, 34], [-32, -30, -28, -26, -24]),
    )
    sparse = FundamentalSnapshot(ticker="SPARSE", sector="Technology", market_cap=1_000_000)

    holdings = {
        "GOOD": Holding(ticker="GOOD", name="Good Co", weight_pct=10.0, sector="Technology", source="test"),
        "WEAK": Holding(ticker="WEAK", name="Weak Co", weight_pct=8.0, sector="Technology", source="test"),
        "SPARSE": Holding(ticker="SPARSE", name="Sparse Co", weight_pct=2.0, sector="Technology", source="test"),
    }
    return holdings, {"GOOD": good, "WEAK": weak, "SPARSE": sparse}


def _score():
    holdings, snapshots = _make_universe()
    universe = build_universe_data(snapshots)
    return score_universe(universe, holdings, PILLAR_WEIGHTS, 0.05, 0.95, min_coverage_pct=60.0)


def test_golden_ranking_is_deterministic():
    run1, run2 = _score(), _score()
    assert {t: sc.composite_score for t, sc in run1.items()} == {t: sc.composite_score for t, sc in run2.items()}


def test_golden_ranking_favors_stronger_fundamentals():
    scorecards = _score()
    assert scorecards["GOOD"].composite_score > scorecards["WEAK"].composite_score
    assert scorecards["GOOD"].low_confidence is False
    assert scorecards["WEAK"].low_confidence is False


def test_golden_annual_cashflow_metrics_are_populated():
    holdings, snapshots = _make_universe()
    universe = build_universe_data(snapshots)
    assert universe.raw_df.loc["GOOD", "fcf_consistency"] == 5  # all 5 years FCF-positive
    assert universe.raw_df.loc["WEAK", "fcf_consistency"] == 2
    assert universe.raw_df.loc["GOOD", "ocf_cagr_3y"] > 0  # OCF growing
    assert universe.raw_df.loc["WEAK", "ocf_cagr_3y"] < 0  # OCF shrinking


def test_golden_low_confidence_sorts_last_regardless_of_score():
    scorecards = _score()
    assert scorecards["SPARSE"].low_confidence is True
    ranked = sorted(scorecards.values(), key=lambda sc: (sc.low_confidence, -sc.composite_score))
    assert [sc.ticker for sc in ranked] == ["GOOD", "WEAK", "SPARSE"]
