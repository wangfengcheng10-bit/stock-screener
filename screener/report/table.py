from __future__ import annotations

from rich.console import Console
from rich.table import Table

from screener.models import ScoreCard


def render_scorecard_table(console: Console, scorecards: list[ScoreCard], title: str) -> None:
    """Ranked by composite score descending, with low-confidence rows sorted
    to the bottom rather than dropped -- per the data-quality gate."""
    ranked = sorted(scorecards, key=lambda sc: (sc.low_confidence, -sc.composite_score))

    table = Table(title=title)
    table.add_column("Rank", justify="right")
    table.add_column("Ticker")
    table.add_column("Name")
    table.add_column("Sector")
    table.add_column("Wt %", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Grade", justify="center")
    for pillar in ("growth", "profitability", "balance_sheet", "cash_flow", "outlook"):
        table.add_column(pillar[:4].title(), justify="right")
    table.add_column("Coverage %", justify="right")

    for i, sc in enumerate(ranked, start=1):
        row = [
            str(i),
            sc.ticker,
            sc.name,
            sc.sector or "-",
            f"{sc.etf_weight_pct:.2f}",
            f"{sc.composite_score:.1f}",
            sc.grade,
        ]
        for pillar in ("growth", "profitability", "balance_sheet", "cash_flow", "outlook"):
            ps = sc.pillar_scores.get(pillar)
            row.append(f"{ps.score:.0f}" if ps else "-")
        coverage = f"{sc.data_coverage_pct:.0f}" + ("*" if sc.low_confidence else "")
        row.append(coverage)
        table.add_row(*row)

    console.print(table)
    if any(sc.low_confidence for sc in ranked):
        console.print("[dim]* low_confidence: metric coverage below the configured minimum[/dim]")
