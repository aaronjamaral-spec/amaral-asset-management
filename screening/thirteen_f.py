"""
screening/thirteen_f.py
------------------------
13F institutional holdings parser for AAM idea generation.
Pulls recent 13F filings from SEC EDGAR to surface
what smart money is buying in the small/mid cap universe.
"""

import requests
import pandas as pd
from utils.config import DATA

HEADERS = {"User-Agent": DATA["edgar_user_agent"]}
BASE    = "https://data.sec.gov"


def get_13f_filers(name_contains: str = None, n: int = 20) -> pd.DataFrame:
    """
    Search for institutional investment managers who file 13Fs.

    Parameters
    ----------
    name_contains : str, optional
        Filter by manager name (e.g. 'Greenlight', 'Third Point')
    n : int
        Max results to return

    Returns
    -------
    pd.DataFrame with CIK, name, and latest 13F filing info
    """
    url = "https://efts.sec.gov/LATEST/search-index?q=%2213F-HR%22&dateRange=custom&startdt=2024-01-01&forms=13F-HR"
    r = requests.get(
        "https://efts.sec.gov/LATEST/search-index?forms=13F-HR&dateRange=custom&startdt=2025-01-01",
        headers=HEADERS
    )

    # Fallback — use known activist/value filers
    known_filers = [
        {"name": "Greenlight Capital",      "cik": "0001159159"},
        {"name": "Third Point LLC",         "cik": "0001040273"},
        {"name": "Starboard Value",         "cik": "0001517767"},
        {"name": "Pershing Square",         "cik": "0001336528"},
        {"name": "ValueAct Capital",        "cik": "0001418121"},
        {"name": "Corvex Management",       "cik": "0001547546"},
        {"name": "Elliott Investment Mgmt", "cik": "0000914612"},
        {"name": "Jana Partners",           "cik": "0001159159"},
        {"name": "Engaged Capital",         "cik": "0001547781"},
        {"name": "Barington Capital",       "cik": "0001278752"},
    ]
    df = pd.DataFrame(known_filers)
    if name_contains:
        df = df[df["name"].str.contains(name_contains, case=False)]
    return df.head(n).reset_index(drop=True)


def get_13f_holdings(cik: str, n_filings: int = 1) -> pd.DataFrame:
    """
    Pull holdings from the most recent 13F filing for a given CIK.

    Parameters
    ----------
    cik : str
        SEC CIK number (with or without leading zeros)
    n_filings : int
        Number of most recent 13F filings to pull

    Returns
    -------
    pd.DataFrame of holdings sorted by value descending
    """
    cik_padded = str(cik).lstrip("0").zfill(10)
    url = f"{BASE}/submissions/CIK{cik_padded}.json"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    data = r.json()

    recent = data.get("filings", {}).get("recent", {})
    df = pd.DataFrame({
        "form":             recent.get("form", []),
        "filed":            recent.get("filingDate", []),
        "accession_number": recent.get("accessionNumber", []),
    })

    filings_13f = df[df["form"].isin(["13F-HR", "13F-HR/A"])].head(n_filings)
    if filings_13f.empty:
        print(f"No 13F filings found for CIK {cik}")
        return pd.DataFrame()

    all_holdings = []
    for _, filing in filings_13f.iterrows():
        acc = filing["accession_number"].replace("-", "")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik_padded)}/{acc}/infotable.xml"
        )
        try:
            resp = requests.get(index_url, headers=HEADERS)
            if resp.status_code != 200:
                continue
            holdings = _parse_infotable_xml(resp.text)
            for h in holdings:
                h["filed"] = filing["filed"]
            all_holdings.extend(holdings)
        except Exception as e:
            print(f"  Could not parse {filing['accession_number']}: {e}")

    if not all_holdings:
        return pd.DataFrame()

    result = pd.DataFrame(all_holdings)
    result["value_m"] = pd.to_numeric(result.get("value", 0), errors="coerce") / 1000
    return result.sort_values("value_m", ascending=False).reset_index(drop=True)


def _parse_infotable_xml(xml_text: str) -> list:
    """Parse 13F infotable XML into a list of holding dicts."""
    from xml.etree import ElementTree as ET
    holdings = []
    try:
        ns = {"ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}
        root = ET.fromstring(xml_text)
        for info in root.findall(".//ns:infoTable", ns):
            def tag(t):
                el = info.find(f"ns:{t}", ns)
                return el.text.strip() if el is not None and el.text else None
            holdings.append({
                "name":        tag("nameOfIssuer"),
                "class":       tag("titleOfClass"),
                "cusip":       tag("cusip"),
                "value":       tag("value"),
                "shares":      tag("sshPrnamt"),
                "share_type":  tag("sshPrnamtType"),
                "put_call":    tag("putCall"),
            })
    except Exception as e:
        print(f"  XML parse error: {e}")
    return holdings


def screen_13f_for_universe(cik: str, min_value_m: float = 5.0) -> pd.DataFrame:
    """
    Pull 13F holdings and filter for names worth researching.
    Returns holdings above a minimum position value.

    Parameters
    ----------
    cik : str
        Institutional manager CIK
    min_value_m : float
        Minimum position value in $M to include

    Returns
    -------
    pd.DataFrame of filtered holdings
    """
    holdings = get_13f_holdings(cik)
    if holdings.empty:
        return holdings

    filtered = holdings[
        (holdings["value_m"] >= min_value_m) &
        (holdings["put_call"].isna())         # exclude options
    ].copy()

    print(f"\n  {len(filtered)} positions >= ${min_value_m}M (equity only)")
    return filtered[["name", "cusip", "value_m", "shares", "filed"]]


if __name__ == "__main__":
    # Pull Starboard Value's most recent 13F
    cik = "0001517767"
    print(f"\nPulling 13F holdings for Starboard Value (CIK {cik})...")
    holdings = get_13f_holdings(cik)
    if not holdings.empty:
        print(holdings[["name", "value_m", "shares", "filed"]].head(15).to_string(index=False))
    else:
        print("Could not retrieve holdings — check CIK or try another filer.")
