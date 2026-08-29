from __future__ import annotations

from pathlib import Path

import pandas as pd

from screener.models import ScoreCard

PILLAR_ORDER = ("growth", "profitability", "balance_sheet", "cash_flow", "outlook")


def scorecards_to_dataframe(scorecards: list[ScoreCard], raw_df: pd.DataFrame | None = None) -> pd.DataFrame:
    ranked = sorted(scorecards, key=lambda sc: (sc.low_confidence, -sc.composite_score))
    rows = []
    for i, sc in enumerate(ranked, start=1):
        row = {
            "rank": i,
            "ticker": sc.ticker,
            "name": sc.name,
            "sector": sc.sector,
            "etf_weight_pct": sc.etf_weight_pct,
            "composite_score": sc.composite_score,
            "grade": sc.grade,
        }
        for pillar in PILLAR_ORDER:
            ps = sc.pillar_scores.get(pillar)
            row[f"{pillar}_score"] = ps.score if ps else None
        if raw_df is not None and sc.ticker in raw_df.index:
            for col in ("rev_growth_ttm_yoy", "net_margin_ttm", "net_debt_ebitda", "fcf_margin", "forward_revenue_growth", "pe_ttm", "ev_ebitda"):
                if col in raw_df.columns:
                    row[col] = raw_df.loc[sc.ticker, col]
        row["data_coverage_pct"] = sc.data_coverage_pct
        row["low_confidence"] = sc.low_confidence
        row["as_of"] = sc.as_of.isoformat()
        rows.append(row)
    return pd.DataFrame(rows)


def export(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".xlsx":
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported export format: {suffix!r} (use .csv, .parquet, or .xlsx)")
