from conftest import annual_income, balance, quarterly_income

from screener.models import FundamentalSnapshot, Holding
from screener.pipeline.scoring import build_universe_data, score_universe
from screener.pipeline.sectors import SectorGroup, applicable_metrics

PILLAR_WEIGHTS = {"growth": 0.20, "profitability": 0.25, "balance_sheet": 0.20, "cash_flow": 0.20, "outlook": 0.15}


def test_financials_group_excludes_fcf_and_gross_margin_includes_substitutes():
    keys = {mdef.key for mdef in applicable_metrics(SectorGroup.FINANCIALS)}
    assert "fcf_margin" not in keys
    assert "fcf_conversion" not in keys
    assert "gross_margin_ttm" not in keys
    assert "roic" not in keys
    assert "roe" in keys
    assert "capital_adequacy_proxy" in keys


def test_standard_group_is_unmodified():
    from screener.pipeline.metrics import METRIC_REGISTRY

    keys = {mdef.key for mdef in applicable_metrics(SectorGroup.STANDARD)}
    assert keys == {mdef.key for mdef in METRIC_REGISTRY}


def test_bank_fixture_scorecard_has_no_cash_flow_pillar():
    bank_snap = FundamentalSnapshot(
        ticker="BANK",
        sector="Financial Services",
        income_quarterly=quarterly_income([100] * 4, [20] * 4),
        income_annual=annual_income([100] * 5, [20] * 5),
        balance_sheet=balance(total_debt=5000, cash=1000, equity=2000, assets=20000, current_assets=3000, current_liabilities=1500),
    )
    holdings = {"BANK": Holding(ticker="BANK", name="Bank Co", weight_pct=5.0, sector="Financial Services", source="test")}
    universe = build_universe_data({"BANK": bank_snap})

    assert "fcf_margin" not in universe.applicable_by_ticker["BANK"]
    assert "roe" in universe.applicable_by_ticker["BANK"]

    scorecards = score_universe(universe, holdings, PILLAR_WEIGHTS, 0.05, 0.95, min_coverage_pct=60.0)
    assert "cash_flow" not in scorecards["BANK"].pillar_scores
