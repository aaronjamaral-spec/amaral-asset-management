"""
screening/yfinance_pull.py
--------------------------
Market data pipeline using yfinance.
Pulls price history, financial statements, and key metrics
for any ticker in the AAM universe.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.config import UNIVERSE, DATA


def get_price_history(ticker: str, years: int = None) -> pd.DataFrame:
    """
    Pull daily OHLCV price history for a ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'ACVA')
    years : int
        Number of years of history (default from config)

    Returns
    -------
    pd.DataFrame
        DatetimeIndex with Open, High, Low, Close, Volume, Adj Close
    """
    years = years or DATA["price_history_years"]
    start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    df.index = pd.to_datetime(df.index)
    return df


def get_financials(ticker: str) -> dict:
    """
    Pull key financial statement data for a ticker.

    Returns
    -------
    dict with keys: info, income_stmt, balance_sheet, cash_flow, earnings
    """
    t = yf.Ticker(ticker)
    return {
        "info":          t.info,
        "income_stmt":   t.financials,
        "balance_sheet": t.balance_sheet,
        "cash_flow":     t.cashflow,
        "earnings":      t.earnings_history,
    }


def get_key_metrics(ticker: str) -> dict:
    """
    Extract the most relevant valuation and quality metrics
    from yfinance info dict.

    Returns
    -------
    dict of clean key metrics for screening / memo use
    """
    info = yf.Ticker(ticker).info

    def safe(key, default=None):
        val = info.get(key, default)
        return val if val not in (None, "None", "N/A") else default

    return {
        "ticker":              ticker.upper(),
        "company_name":        safe("longName"),
        "sector":              safe("sector"),
        "industry":            safe("industry"),
        "market_cap_m":        round(safe("marketCap", 0) / 1e6, 1),
        "enterprise_value_m":  round(safe("enterpriseValue", 0) / 1e6, 1),
        "price":               safe("currentPrice"),
        "52w_high":            safe("fiftyTwoWeekHigh"),
        "52w_low":             safe("fiftyTwoWeekLow"),
        "price_to_52w_high":   round(safe("currentPrice", 0) / safe("fiftyTwoWeekHigh", 1), 2),
        "pe_trailing":         safe("trailingPE"),
        "pe_forward":          safe("forwardPE"),
        "ev_ebitda":           safe("enterpriseToEbitda"),
        "ev_revenue":          safe("enterpriseToRevenue"),
        "pb_ratio":            safe("priceToBook"),
        "ps_ratio":            safe("priceToSalesTrailing12Months"),
        "revenue_ttm_m":       round(safe("totalRevenue", 0) / 1e6, 1),
        "ebitda_m":            round(safe("ebitda", 0) / 1e6, 1),
        "net_income_m":        round(safe("netIncomeToCommon", 0) / 1e6, 1),
        "free_cash_flow_m":    round(safe("freeCashflow", 0) / 1e6, 1),
        "gross_margin":        round(safe("grossMargins", 0) * 100, 1),
        "ebitda_margin":       round(safe("ebitdaMargins", 0) * 100, 1),
        "net_margin":          round(safe("profitMargins", 0) * 100, 1),
        "roe":                 round(safe("returnOnEquity", 0) * 100, 1),
        "roa":                 round(safe("returnOnAssets", 0) * 100, 1),
        "revenue_growth_yoy":  round(safe("revenueGrowth", 0) * 100, 1),
        "earnings_growth_yoy": round(safe("earningsGrowth", 0) * 100, 1),
        "debt_to_equity":      safe("debtToEquity"),
        "current_ratio":       safe("currentRatio"),
        "avg_volume":          safe("averageVolume"),
        "short_float_pct":     round(safe("shortPercentOfFloat", 0) * 100, 1),
        "beta":                safe("beta"),
        "description":         safe("longBusinessSummary", "")[:300],
    }


def get_universe_snapshot(tickers: list) -> pd.DataFrame:
    """
    Pull key metrics for a list of tickers and return as a DataFrame.
    Useful for building a quick comp table or screening output.

    Parameters
    ----------
    tickers : list of str

    Returns
    -------
    pd.DataFrame sorted by EV/EBITDA ascending
    """
    rows = []
    for t in tickers:
        try:
            metrics = get_key_metrics(t)
            rows.append(metrics)
            print(f"  {t} OK")
        except Exception as e:
            print(f"  {t} ERROR: {e}")
    df = pd.DataFrame(rows)
    return df.sort_values("ev_ebitda", ascending=True).reset_index(drop=True)


if __name__ == "__main__":
    # Quick test
    ticker = "ACVA"
    print(f"\n=== Key metrics: {ticker} ===")
    metrics = get_key_metrics(ticker)
    for k, v in metrics.items():
        if k != "description":
            print(f"  {k:<25} {v}")
