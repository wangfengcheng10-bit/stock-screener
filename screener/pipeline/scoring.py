from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import pandas as pd

from screener.models import FundamentalSnapshot, Holding, MetricSet, MetricValue, PillarScore, ScoreCard
from screener.pipeline.metrics import CONTEXT_METRICS, METRIC_REGISTRY_BY_KEY
from screener.pipeline.sectors import applicable_metrics, classify_sector_group

GRADE_THRESHOLDS: list[tuple[float, str]] = [(90, "A+"), (80, "A"), (70, "B"), (60, "C"), (50, "D"), (0, "F")]


def grade_for_score(score: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "F"


@dataclass
class UniverseData:
    """Raw metric values (rows=ticker, cols=metric key) plus, per ticker,
    which metric keys are actually applicable to that company's sector."""

    raw_df: pd.DataFrame
    applicable_by_ticker: dict[str, list[str]]
    unscored_prerevenue: list[str] = field(default_factory=list)
    unscored_fetch_failed: list[str] = field(default_factory=list)


def build_universe_data(
    snapshots: dict[str, FundamentalSnapshot],
    include_prerevenue: bool = False,
) -> UniverseData:
    rows: dict[str, dict[str, Optional[float]]] = {}
    applicable: dict[str, list[str]] = {}
    unscored_prerevenue: list[str] = []

    for ticker, snap in snapshots.items():
        if snap.pre_revenue and not include_prerevenue:
            unscored_prerevenue.append(ticker)
            continue
        group = classify_sector_group(snap.sector)
        metric_defs = applicable_metrics(group)
        applicable[ticker] = [m.key for m in metric_defs]
        row = {m.key: m.compute(snap) for m in metric_defs}
        for ctx in CONTEXT_METRICS:
            row[ctx.key] = ctx.compute(snap)
        rows[ticker] = row

    raw_df = pd.DataFrame.from_dict(rows, orient="index") if rows else pd.DataFrame()
    return UniverseData(raw_df=raw_df, applicable_by_ticker=applicable, unscored_prerevenue=unscored_prerevenue)


def winsorize_series(s: pd.Series, lower: float, upper: float) -> pd.Series:
    valid = s.dropna()
    if valid.empty:
        return s
    lo, hi = valid.quantile(lower), valid.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def percentile_rank(s: pd.Series, higher_is_better: bool) -> pd.Series:
    """Direction-aware percentile rank in [0, 100]. Missing values stay NaN --
    the neutral-50 substitution happens one layer up, per-company, so it can
    be tracked in that company's missing_metric_count."""
    valid = s.dropna()
    if valid.empty:
        return pd.Series(index=s.index, dtype=float)
    ranks = valid.rank(pct=True) * 100
    if not higher_is_better:
        ranks = 100 - ranks
    return ranks.reindex(s.index)


def compute_percentiles(
    raw_df: pd.DataFrame,
    winsorize_lower: float,
    winsorize_upper: float,
    sector_by_ticker: Optional[dict[str, Optional[str]]] = None,
) -> pd.DataFrame:
    """Percentile-rank every metric column. If sector_by_ticker is given
    (--sector-neutral), ranking happens within each raw GICS sector rather
    than across the whole universe."""
    pct_df = pd.DataFrame(index=raw_df.index, columns=raw_df.columns, dtype=float)
    metric_defs_by_key = {**METRIC_REGISTRY_BY_KEY, **{m.key: m for m in CONTEXT_METRICS}}
    # substitute-only metric keys (roe, capital_adequacy_proxy) aren't in the base
    # registry; pull their direction from sectors.py's substitute definitions.
    from screener.pipeline.sectors import SUBSTITUTE_METRICS_BY_GROUP

    for group_defs in SUBSTITUTE_METRICS_BY_GROUP.values():
        for m in group_defs:
            metric_defs_by_key.setdefault(m.key, m)

    sectors = pd.Series(sector_by_ticker) if sector_by_ticker else None

    for col in raw_df.columns:
        higher_is_better = metric_defs_by_key[col].higher_is_better
        if sectors is not None:
            for _, idx in raw_df.groupby(sectors.reindex(raw_df.index)).groups.items():
                sub = raw_df.loc[idx, col]
                wins = winsorize_series(sub, winsorize_lower, winsorize_upper)
                pct_df.loc[idx, col] = percentile_rank(wins, higher_is_better)
        else:
            wins = winsorize_series(raw_df[col], winsorize_lower, winsorize_upper)
            pct_df[col] = percentile_rank(wins, higher_is_better)
    return pct_df


def score_universe(
    universe: UniverseData,
    holdings: dict[str, Holding],
    pillar_weights: dict[str, float],
    winsorize_lower: float,
    winsorize_upper: float,
    min_coverage_pct: float,
    sector_by_ticker: Optional[dict[str, Optional[str]]] = None,
) -> dict[str, ScoreCard]:
    raw_df = universe.raw_df
    if raw_df.empty:
        return {}

    from screener.pipeline.sectors import SUBSTITUTE_METRICS_BY_GROUP

    metric_defs_by_key = {**METRIC_REGISTRY_BY_KEY, **{m.key: m for m in CONTEXT_METRICS}}
    for group_defs in SUBSTITUTE_METRICS_BY_GROUP.values():
        for m in group_defs:
            metric_defs_by_key.setdefault(m.key, m)

    scored_cols = [k for k in raw_df.columns if metric_defs_by_key[k].scored]
    pct_df = compute_percentiles(raw_df, winsorize_lower, winsorize_upper, sector_by_ticker)

    scorecards: dict[str, ScoreCard] = {}
    for ticker, applicable_keys in universe.applicable_by_ticker.items():
        applicable_keys = [k for k in applicable_keys if k in scored_cols]
        total_metrics = len(applicable_keys)
        missing_count = 0
        metric_values: dict[str, MetricValue] = {}

        for key in applicable_keys:
            raw_val = raw_df.loc[ticker, key]
            is_missing = pd.isna(raw_val)
            pct = pct_df.loc[ticker, key] if not is_missing else float("nan")
            if pd.isna(pct):
                is_missing = True
                pct = 50.0
            if is_missing:
                missing_count += 1
            metric_values[key] = MetricValue(
                name=key,
                raw_value=None if pd.isna(raw_val) else float(raw_val),
                percentile=float(pct),
                higher_is_better=metric_defs_by_key[key].higher_is_better,
                is_missing=is_missing,
            )

        pillar_scores: dict[str, PillarScore] = {}
        pillars_present = sorted({metric_defs_by_key[k].pillar for k in applicable_keys})
        for pillar in pillars_present:
            pillar_keys = [k for k in applicable_keys if metric_defs_by_key[k].pillar == pillar]
            pillar_score = sum(metric_values[k].percentile for k in pillar_keys) / len(pillar_keys)
            pillar_scores[pillar] = PillarScore(
                name=pillar,
                score=round(pillar_score, 2),
                metrics_available=sum(1 for k in pillar_keys if not metric_values[k].is_missing),
                metrics_total=len(pillar_keys),
            )

        weight_sum = sum(pillar_weights.get(p, 0.0) for p in pillars_present)
        composite = (
            sum(pillar_scores[p].score * pillar_weights.get(p, 0.0) for p in pillars_present) / weight_sum
            if weight_sum > 0
            else 50.0
        )
        coverage_pct = ((total_metrics - missing_count) / total_metrics * 100) if total_metrics else 0.0

        holding = holdings.get(ticker)
        scorecards[ticker] = ScoreCard(
            ticker=ticker,
            name=holding.name if holding else ticker,
            sector=holding.sector if holding else None,
            etf_weight_pct=holding.weight_pct if holding else 0.0,
            composite_score=round(composite, 2),
            grade=grade_for_score(composite),
            pillar_scores=pillar_scores,
            data_coverage_pct=round(coverage_pct, 2),
            low_confidence=coverage_pct < min_coverage_pct,
            as_of=date.today(),
        )

    return scorecards


def build_metric_set(ticker: str, universe: UniverseData, pct_df: pd.DataFrame, metric_defs_by_key: dict) -> MetricSet:
    """Per-metric breakdown for one ticker -- backs the `explain` CLI command."""
    applicable_keys = universe.applicable_by_ticker.get(ticker, [])
    metrics: dict[str, MetricValue] = {}
    for key in applicable_keys:
        raw_val = universe.raw_df.loc[ticker, key]
        pct = pct_df.loc[ticker, key] if key in pct_df.columns else float("nan")
        is_missing = pd.isna(raw_val) or pd.isna(pct)
        metrics[key] = MetricValue(
            name=key,
            raw_value=None if pd.isna(raw_val) else float(raw_val),
            percentile=50.0 if pd.isna(pct) else float(pct),
            higher_is_better=metric_defs_by_key[key].higher_is_better,
            is_missing=is_missing,
        )
    return MetricSet(ticker=ticker, metrics=metrics)
