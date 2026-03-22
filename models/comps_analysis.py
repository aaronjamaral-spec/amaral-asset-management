"""
models/comps_analysis.py
-------------------------
Comparable company (trading comps) analysis for AAM equity research.
Pulls live multiples via yfinance and outputs a clean comp table
with implied valuation ranges for the target company.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from utils.config import COLORS


def get_comp_metrics(ticker: str) -> dict:
    """Pull key trading multiples for a single ticker."""
    try:
        info = yf.Ticker(ticker).info
        def safe(key):
            val = info.get(key)
            return val if val not in (None, "None", 0) else np.nan

        return {
            "ticker":           ticker.upper(),
            "company":          (info.get("shortName") or "")[:25],
            "market_cap_m":     round((safe("marketCap") or 0) / 1e6, 0),
            "ev_ebitda":        round(safe("enterpriseToEbitda"), 1) if safe("enterpriseToEbitda") else np.nan,
            "ev_revenue":       round(safe("enterpriseToRevenue"), 2) if safe("enterpriseToRevenue") else np.nan,
            "pe_forward":       round(safe("forwardPE"), 1) if safe("forwardPE") else np.nan,
            "pb_ratio":         round(safe("priceToBook"), 2) if safe("priceToBook") else np.nan,
            "ebitda_margin":    round((safe("ebitdaMargins") or 0) * 100, 1),
            "rev_growth_pct":   round((safe("revenueGrowth") or 0) * 100, 1),
            "gross_margin_pct": round((safe("grossMargins") or 0) * 100, 1),
            "roe_pct":          round((safe("returnOnEquity") or 0) * 100, 1),
            "net_debt_m":       round(((safe("totalDebt") or 0) - (safe("totalCash") or 0)) / 1e6, 0),
        }
    except Exception as e:
        print(f"  Error on {ticker}: {e}")
        return None


def build_comps_table(
    target: str,
    peers: list,
    target_ebitda_m: float = None,
    target_revenue_m: float = None,
    target_net_debt_m: float = None,
    target_shares_m: float = None,
) -> pd.DataFrame:
    """
    Build a trading comps table for a target company vs. peers.

    Parameters
    ----------
    target : str
        Target company ticker
    peers : list of str
        Peer company tickers
    target_ebitda_m : float, optional
        Target company LTM EBITDA in $M (for implied value calc)
    target_revenue_m : float, optional
        Target company LTM revenue in $M
    target_net_debt_m : float, optional
        Target net debt in $M (positive = net debt, negative = net cash)
    target_shares_m : float, optional
        Target diluted shares outstanding in millions

    Returns
    -------
    pd.DataFrame with target + peers, medians, and implied values
    """
    all_tickers = [target] + peers
    rows = []
    for t in all_tickers:
        print(f"  Pulling {t}...")
        row = get_comp_metrics(t)
        if row:
            row["is_target"] = (t.upper() == target.upper())
            rows.append(row)

    df = pd.DataFrame(rows)

    # Peer medians (exclude target)
    peers_df = df[~df["is_target"]]
    medians = peers_df[["ev_ebitda", "ev_revenue", "pe_forward",
                         "pb_ratio", "ebitda_margin", "rev_growth_pct",
                         "gross_margin_pct", "roe_pct"]].median()

    median_row = {"ticker": "MEDIAN", "company": "--- Peer median ---",
                  "is_target": False}
    median_row.update(medians.round(2).to_dict())
    df = pd.concat([df, pd.DataFrame([median_row])], ignore_index=True)

    # Implied values for target
    if all([target_ebitda_m, target_net_debt_m, target_shares_m]):
        implied_ev_ebitda = medians["ev_ebitda"] * target_ebitda_m
        implied_price_ebitda = (implied_ev_ebitda - target_net_debt_m) / target_shares_m
        print(f"\n  Implied EV (EV/EBITDA median): ${implied_ev_ebitda:,.0f}M")
        print(f"  Implied price/share:           ${implied_price_ebitda:.2f}")

    if all([target_revenue_m, target_net_debt_m, target_shares_m]):
        implied_ev_rev = medians["ev_revenue"] * target_revenue_m
        implied_price_rev = (implied_ev_rev - target_net_debt_m) / target_shares_m
        print(f"  Implied EV (EV/Revenue median): ${implied_ev_rev:,.0f}M")
        print(f"  Implied price/share:            ${implied_price_rev:.2f}")

    return df.drop(columns=["is_target"])


def print_comps(df: pd.DataFrame):
    """Print a formatted comps table."""
    cols = ["ticker", "company", "market_cap_m", "ev_ebitda",
            "ev_revenue", "pe_forward", "ebitda_margin",
            "rev_growth_pct", "gross_margin_pct"]
    available = [c for c in cols if c in df.columns]
    print(f"\n{'='*85}")
    print(f"  Trading Comps Table")
    print(f"{'='*85}")
    print(df[available].to_string(index=False))
    print(f"{'='*85}\n")


if __name__ == "__main__":
    target  = "ACVA"
    peers   = ["KAR", "CPRT", "IAA", "ADESA"]

    print(f"\nBuilding comps for {target} vs peers...")
    df = build_comps_table(
        target=target,
        peers=peers,
        target_ebitda_m=72,
        target_revenue_m=485,
        target_net_debt_m=-120,
        target_shares_m=155,
    )
    print_comps(df)
