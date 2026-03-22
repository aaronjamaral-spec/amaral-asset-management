"""
screening/factor_screen.py
--------------------------
Multi-factor screener for AAM small/mid cap universe.
Screens for deep value + quality characteristics aligned
with the fund's investment philosophy.
Outputs a ranked DataFrame ready for further research.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils.config import UNIVERSE, PORTFOLIO


# Default screen tickers — Russell 2500 proxy list
# In practice, replace with a proper index constituent pull
# or use a screener API for the full universe
SAMPLE_TICKERS = [
    "ACVA", "DRVN", "ACCO", "MSGS", "PAHC", "GATO", "PRDO",
    "AMSF", "HURN", "SPOK", "LIQT", "HAFC", "MGPI", "RCUS",
    "FLGT", "FXNC", "TTGT", "LGND", "NVEE", "NSIT",
]


def screen_ticker(ticker: str) -> dict | None:
    """
    Pull key screening metrics for a single ticker.
    Returns None if data unavailable or ticker fails universe filter.
    """
    try:
        info = yf.Ticker(ticker).info

        def safe(key, default=None):
            val = info.get(key)
            return val if val not in (None, "None", 0) else default

        market_cap_m = (safe("marketCap") or 0) / 1e6
        avg_vol = safe("averageVolume") or 0
        price = safe("currentPrice") or 0
        avg_dollar_vol_m = (avg_vol * price) / 1e6

        # Universe filter
        if not (
            UNIVERSE["min_market_cap_m"] <= market_cap_m <= UNIVERSE["max_market_cap_m"]
            and avg_dollar_vol_m >= UNIVERSE["min_avg_volume_m"]
        ):
            return None

        ev_ebitda    = safe("enterpriseToEbitda")
        pe_forward   = safe("forwardPE")
        pb           = safe("priceToBook")
        ps           = safe("priceToSalesTrailing12Months")
        fcf_yield    = None
        fcf          = safe("freeCashflow")
        mktcap       = safe("marketCap")
        if fcf and mktcap:
            fcf_yield = round((fcf / mktcap) * 100, 1)

        return {
            "ticker":           ticker.upper(),
            "company":          (safe("shortName") or "")[:28],
            "sector":           safe("sector", "Unknown"),
            "market_cap_m":     round(market_cap_m, 0),
            "ev_ebitda":        round(ev_ebitda, 1) if ev_ebitda else None,
            "pe_forward":       round(pe_forward, 1) if pe_forward else None,
            "pb_ratio":         round(pb, 2) if pb else None,
            "ps_ratio":         round(ps, 2) if ps else None,
            "fcf_yield_pct":    fcf_yield,
            "gross_margin_pct": round((safe("grossMargins") or 0) * 100, 1),
            "ebitda_margin_pct":round((safe("ebitdaMargins") or 0) * 100, 1),
            "roe_pct":          round((safe("returnOnEquity") or 0) * 100, 1),
            "rev_growth_pct":   round((safe("revenueGrowth") or 0) * 100, 1),
            "debt_to_equity":   round(safe("debtToEquity") or 0, 2),
            "short_float_pct":  round((safe("shortPercentOfFloat") or 0) * 100, 1),
            "52w_from_high_pct":round(
                ((price / (safe("fiftyTwoWeekHigh") or price)) - 1) * 100, 1
            ),
            "beta":             safe("beta"),
        }
    except Exception as e:
        print(f"  {ticker}: {e}")
        return None


def run_screen(
    tickers: list = None,
    max_ev_ebitda: float = 12.0,
    min_fcf_yield: float = 3.0,
    max_debt_to_equity: float = 3.0,
    min_market_cap_m: float = None,
    max_market_cap_m: float = None,
) -> pd.DataFrame:
    """
    Run the AAM factor screen across a list of tickers.

    Parameters
    ----------
    tickers : list, optional
        List of ticker symbols. Defaults to SAMPLE_TICKERS.
    max_ev_ebitda : float
        Maximum EV/EBITDA (value filter)
    min_fcf_yield : float
        Minimum FCF yield % (quality/value filter)
    max_debt_to_equity : float
        Maximum debt/equity ratio (balance sheet filter)
    min_market_cap_m / max_market_cap_m : float
        Override universe market cap bounds

    Returns
    -------
    pd.DataFrame sorted by EV/EBITDA ascending
    """
    tickers = tickers or SAMPLE_TICKERS
    min_cap = min_market_cap_m or UNIVERSE["min_market_cap_m"]
    max_cap = max_market_cap_m or UNIVERSE["max_market_cap_m"]

    print(f"\nAAM Factor Screen — {len(tickers)} tickers")
    print(f"Universe: ${min_cap:,.0f}M – ${max_cap:,.0f}M market cap")
    print(f"Filters: EV/EBITDA ≤ {max_ev_ebitda}x | FCF yield ≥ {min_fcf_yield}% | D/E ≤ {max_debt_to_equity}x\n")

    results = []
    for ticker in tqdm(tickers, desc="Screening"):
        row = screen_ticker(ticker)
        if row:
            results.append(row)

    if not results:
        print("No results passed universe filter.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Apply value/quality filters
    mask = pd.Series([True] * len(df))
    if max_ev_ebitda:
        mask &= (df["ev_ebitda"].fillna(999) <= max_ev_ebitda)
    if min_fcf_yield:
        mask &= (df["fcf_yield_pct"].fillna(0) >= min_fcf_yield)
    if max_debt_to_equity:
        mask &= (df["debt_to_equity"].fillna(0) <= max_debt_to_equity)

    df_filtered = df[mask].sort_values("ev_ebitda", ascending=True).reset_index(drop=True)
    print(f"\n{len(df_filtered)} names passed all filters (from {len(df)} in universe)\n")
    return df_filtered


if __name__ == "__main__":
    results = run_screen()
    if not results.empty:
        cols = ["ticker", "company", "sector", "market_cap_m",
                "ev_ebitda", "pe_forward", "fcf_yield_pct",
                "gross_margin_pct", "roe_pct", "debt_to_equity"]
        print(results[cols].to_string(index=False))
