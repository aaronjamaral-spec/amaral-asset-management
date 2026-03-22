"""
utils/data_utils.py
--------------------
Data cleaning and transformation helpers for AAM research.
Standardizes financial data pulled from yfinance and EDGAR
into consistent formats for modeling and analysis.
"""

import pandas as pd
import numpy as np
from typing import Union


def clean_financial_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw financial statement DataFrame from yfinance.
    - Converts index to datetime
    - Fills NaN with 0 for numeric columns
    - Sorts columns chronologically (oldest to newest)
    - Converts values to $M where appropriate

    Parameters
    ----------
    df : pd.DataFrame
        Raw financial statement (income_stmt, balance_sheet, cashflow)

    Returns
    -------
    pd.DataFrame cleaned and sorted
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = pd.to_datetime(df.columns, errors="coerce")
    df = df.sort_index(axis=1)
    df = df.fillna(0)
    return df


def to_millions(series: pd.Series) -> pd.Series:
    """Convert a series from raw dollars to $M, rounded to 1 decimal."""
    return (series / 1_000_000).round(1)


def safe_divide(numerator: Union[float, pd.Series],
                denominator: Union[float, pd.Series],
                default: float = np.nan) -> Union[float, pd.Series]:
    """Divide safely, returning default on zero/NaN denominator."""
    if isinstance(denominator, pd.Series):
        return numerator.div(denominator.replace(0, np.nan)).fillna(default)
    return numerator / denominator if denominator else default


def calc_cagr(start_val: float, end_val: float, years: float) -> float:
    """Calculate compound annual growth rate."""
    if start_val <= 0 or years <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / years) - 1


def calc_margins(income_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate key margin metrics from a cleaned income statement.

    Returns
    -------
    pd.DataFrame with gross margin, EBITDA margin, net margin by period
    """
    margins = pd.DataFrame(index=income_df.columns)
    rev = income_df.loc["Total Revenue"] if "Total Revenue" in income_df.index else None
    if rev is None:
        return margins

    if "Gross Profit" in income_df.index:
        margins["gross_margin"] = safe_divide(income_df.loc["Gross Profit"], rev)
    if "EBITDA" in income_df.index:
        margins["ebitda_margin"] = safe_divide(income_df.loc["EBITDA"], rev)
    if "Net Income" in income_df.index:
        margins["net_margin"] = safe_divide(income_df.loc["Net Income"], rev)

    margins["revenue_m"] = to_millions(rev)
    return (margins * 100).round(1).assign(revenue_m=margins["revenue_m"])


def summarize_balance_sheet(bs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract key balance sheet metrics: net debt, book value,
    current ratio, and leverage ratio.
    """
    result = {}
    def get(row):
        return bs_df.loc[row] if row in bs_df.index else pd.Series(0, index=bs_df.columns)

    cash       = get("Cash And Cash Equivalents")
    total_debt = get("Total Debt")
    equity     = get("Stockholders Equity")
    curr_assets  = get("Current Assets")
    curr_liab    = get("Current Liabilities")

    df = pd.DataFrame({
        "net_debt_m":      to_millions(total_debt - cash),
        "book_value_m":    to_millions(equity),
        "current_ratio":   safe_divide(curr_assets, curr_liab).round(2),
        "debt_to_equity":  safe_divide(total_debt, equity).round(2),
    })
    return df


def normalize_returns(prices: pd.DataFrame, base: float = 100) -> pd.DataFrame:
    """
    Normalize a price DataFrame to a common base for comparison charts.
    Useful for plotting portfolio vs. benchmark on the same scale.
    """
    return prices / prices.iloc[0] * base


def rolling_stats(returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """
    Calculate rolling annualized return, volatility, and Sharpe ratio.
    """
    rf = 0.05 / 252
    excess = returns - rf
    return pd.DataFrame({
        "rolling_return":   returns.rolling(window).mean() * 252 * 100,
        "rolling_vol":      returns.rolling(window).std() * np.sqrt(252) * 100,
        "rolling_sharpe":   (excess.rolling(window).mean() * 252) /
                            (returns.rolling(window).std() * np.sqrt(252)),
    })


def format_large_number(val: float, decimals: int = 1) -> str:
    """Format a large number into human-readable string (K, M, B)."""
    if abs(val) >= 1e9:
        return f"${val/1e9:.{decimals}f}B"
    elif abs(val) >= 1e6:
        return f"${val/1e6:.{decimals}f}M"
    elif abs(val) >= 1e3:
        return f"${val/1e3:.{decimals}f}K"
    return f"${val:.{decimals}f}"
