from __future__ import annotations

from datetime import date, timedelta

from screener.models import BalanceSheetPeriod, CashFlowPeriod, FiscalPeriod, IncomeStatementPeriod


def quarterly_income(revenues, net_incomes, start=date(2026, 6, 30)):
    """periods[0] is the most recent quarter, matching provider ordering."""
    periods = []
    for i, (rev, ni) in enumerate(zip(revenues, net_incomes)):
        period_end = start - timedelta(days=91 * i)
        periods.append(
            IncomeStatementPeriod(
                period=FiscalPeriod(fiscal_year=period_end.year, fiscal_quarter=((period_end.month - 1) // 3) + 1, period_end=period_end),
                revenue=rev,
                net_income=ni,
                gross_profit=rev * 0.5 if rev is not None else None,
                operating_income=rev * 0.2 if rev is not None else None,
                ebit=rev * 0.2 if rev is not None else None,
                ebitda=rev * 0.25 if rev is not None else None,
                interest_expense=rev * 0.02 if rev is not None else None,
                diluted_shares_outstanding=1000.0,
                diluted_eps=(ni / 1000.0) if ni is not None else None,
            )
        )
    return periods


def annual_income(revenues, net_incomes, diluted_shares=None, start_year=2026):
    diluted_shares = diluted_shares or [1000.0] * len(revenues)
    periods = []
    for i, (rev, ni, shares) in enumerate(zip(revenues, net_incomes, diluted_shares)):
        year = start_year - i
        periods.append(
            IncomeStatementPeriod(
                period=FiscalPeriod(fiscal_year=year, period_end=date(year, 12, 31)),
                revenue=rev,
                net_income=ni,
                gross_profit=rev * 0.5 if rev is not None else None,
                operating_income=rev * 0.2 if rev is not None else None,
                diluted_shares_outstanding=shares,
                diluted_eps=(ni / shares) if ni is not None and shares else None,
            )
        )
    return periods


def balance(total_debt, cash, equity, assets, current_assets, current_liabilities, period_end=date(2026, 6, 30)):
    return [
        BalanceSheetPeriod(
            period=FiscalPeriod(fiscal_year=period_end.year, period_end=period_end),
            total_debt=total_debt,
            cash_and_equivalents=cash,
            total_equity=equity,
            total_assets=assets,
            current_assets=current_assets,
            current_liabilities=current_liabilities,
        )
    ]


def annual_cash_flow(ocf_list, capex_list, start_year=2026):
    periods = []
    for i, (ocf, capex) in enumerate(zip(ocf_list, capex_list)):
        year = start_year - i
        fcf = (ocf + capex) if ocf is not None and capex is not None else None
        periods.append(
            CashFlowPeriod(
                period=FiscalPeriod(fiscal_year=year, period_end=date(year, 12, 31)),
                operating_cash_flow=ocf,
                capex=capex,
                free_cash_flow=fcf,
            )
        )
    return periods


def cash_flow(ocf_list, capex_list, start=date(2026, 6, 30)):
    periods = []
    for i, (ocf, capex) in enumerate(zip(ocf_list, capex_list)):
        period_end = start - timedelta(days=91 * i)
        fcf = (ocf + capex) if ocf is not None and capex is not None else None
        periods.append(
            CashFlowPeriod(
                period=FiscalPeriod(fiscal_year=period_end.year, period_end=period_end),
                operating_cash_flow=ocf,
                capex=capex,
                free_cash_flow=fcf,
            )
        )
    return periods
