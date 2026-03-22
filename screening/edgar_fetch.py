"""
screening/edgar_fetch.py
------------------------
SEC EDGAR filing retrieval for AAM research workflow.
Fetches 10-K, 10-Q, 8-K, and proxy (DEF 14A) filings
for any US-listed company. No API key required.
"""

import requests
import pandas as pd
from pathlib import Path
from utils.config import DATA

HEADERS = {"User-Agent": DATA["edgar_user_agent"]}
BASE_URL = "https://data.sec.gov"


def get_cik(ticker: str) -> str:
    """
    Resolve a ticker symbol to its SEC CIK number.

    Parameters
    ----------
    ticker : str

    Returns
    -------
    str : zero-padded 10-digit CIK
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR")


def get_filings(ticker: str, form_type: str = "10-K", n: int = 5) -> pd.DataFrame:
    """
    Get recent filings of a given type for a ticker.

    Parameters
    ----------
    ticker : str
    form_type : str
        Filing type: '10-K', '10-Q', '8-K', 'DEF 14A', etc.
    n : int
        Number of most recent filings to return

    Returns
    -------
    pd.DataFrame with columns: form, filed, accession_number, url
    """
    cik = get_cik(ticker)
    url = f"{BASE_URL}/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json()

    recent = data.get("filings", {}).get("recent", {})
    df = pd.DataFrame({
        "form":             recent.get("form", []),
        "filed":            recent.get("filingDate", []),
        "accession_number": recent.get("accessionNumber", []),
        "primary_doc":      recent.get("primaryDocument", []),
    })

    filtered = df[df["form"] == form_type].head(n).copy()
    filtered["url"] = filtered.apply(
        lambda row: (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{row['accession_number'].replace('-', '')}/"
            f"{row['primary_doc']}"
        ), axis=1
    )
    return filtered.reset_index(drop=True)


def get_company_facts(ticker: str) -> dict:
    """
    Get structured financial facts from EDGAR XBRL data.
    Returns raw JSON — useful for pulling specific line items
    across multiple periods programmatically.
    """
    cik = get_cik(ticker)
    url = f"{BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def get_revenue_history(ticker: str) -> pd.DataFrame:
    """
    Pull annual revenue history from EDGAR XBRL data.

    Returns
    -------
    pd.DataFrame with columns: end (fiscal year end), revenue ($M), form
    """
    facts = get_company_facts(ticker)
    us_gaap = facts.get("facts", {}).get("us-gaap", {})

    # Try common revenue tags in order
    revenue_tags = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ]

    for tag in revenue_tags:
        if tag in us_gaap:
            units = us_gaap[tag].get("units", {}).get("USD", [])
            df = pd.DataFrame(units)
            if df.empty:
                continue
            # Filter for annual 10-K filings only
            annual = df[df["form"] == "10-K"][["end", "val", "form"]].copy()
            annual = annual.drop_duplicates(subset="end").sort_values("end")
            annual["revenue_m"] = (annual["val"] / 1e6).round(1)
            annual = annual.rename(columns={"end": "fiscal_year_end"})
            return annual[["fiscal_year_end", "revenue_m", "form"]].reset_index(drop=True)

    raise ValueError(f"Could not find revenue data for {ticker} in EDGAR XBRL")


if __name__ == "__main__":
    ticker = "ACVA"
    print(f"\n=== Recent 10-K filings: {ticker} ===")
    filings = get_filings(ticker, form_type="10-K", n=3)
    print(filings[["form", "filed", "url"]].to_string())

    print(f"\n=== Revenue history: {ticker} ===")
    rev = get_revenue_history(ticker)
    print(rev.to_string(index=False))
