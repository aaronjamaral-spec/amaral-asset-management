"""
performance/attribution.py
---------------------------
Portfolio performance attribution vs. Russell 2500 benchmark.
Calculates total return, alpha, Sharpe ratio, max drawdown,
and sector-level attribution for the AAM portfolio.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from utils.config import COLORS

BENCHMARK_TICKER = "RWGL"   # Russell 2500 ETF proxy


def get_returns(ticker: str, start: str, end: str = None) -> pd.Series:
    """Pull daily adjusted close returns for a ticker."""
    end = end or datetime.today().strftime("%Y-%m-%d")
    prices = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)["Close"]
    return prices.pct_change().dropna()


def portfolio_return(
    positions: dict,
    start_date: str,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Calculate weighted portfolio returns.

    Parameters
    ----------
    positions : dict
        {ticker: weight} e.g. {"ACVA": 0.115, "DRVN": 0.07, ...}
        Weights should sum to ≤ 1.0 (remainder assumed cash at 0%)
    start_date : str
        Portfolio inception date "YYYY-MM-DD"
    end_date : str, optional

    Returns
    -------
    pd.DataFrame with daily portfolio and benchmark returns
    """
    end_date = end_date or datetime.today().strftime("%Y-%m-%d")

    # Pull all ticker returns
    returns = {}
    for ticker in positions:
        try:
            returns[ticker] = get_returns(ticker, start_date, end_date)
        except Exception as e:
            print(f"Warning: could not pull {ticker}: {e}")

    # Pull benchmark
    try:
        returns["BENCHMARK"] = get_returns(BENCHMARK_TICKER, start_date, end_date)
    except Exception:
        print(f"Warning: benchmark {BENCHMARK_TICKER} unavailable — using SPY proxy")
        returns["BENCHMARK"] = get_returns("SPY", start_date, end_date)

    df = pd.DataFrame(returns).dropna()

    # Weighted portfolio return
    portfolio_cols = [t for t in positions if t in df.columns]
    weights = np.array([positions[t] for t in portfolio_cols])
    df["PORTFOLIO"] = df[portfolio_cols].values @ weights

    return df[["PORTFOLIO", "BENCHMARK"]]


def attribution_summary(returns_df: pd.DataFrame) -> dict:
    """
    Compute key performance metrics from a returns DataFrame.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Must have 'PORTFOLIO' and 'BENCHMARK' columns of daily returns.

    Returns
    -------
    dict of summary statistics
    """
    port = returns_df["PORTFOLIO"]
    bench = returns_df["BENCHMARK"]
    rf = 0.05 / 252   # Risk-free rate (annualized 5% / 252)

    # Cumulative returns
    cum_port  = (1 + port).cumprod()
    cum_bench = (1 + bench).cumprod()

    # Annualized metrics
    n_days = len(port)
    ann_port  = (cum_port.iloc[-1]) ** (252 / n_days) - 1
    ann_bench = (cum_bench.iloc[-1]) ** (252 / n_days) - 1
    ann_alpha = ann_port - ann_bench

    # Volatility and Sharpe
    ann_vol   = port.std() * np.sqrt(252)
    sharpe    = (ann_port - 0.05) / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    rolling_max = cum_port.cummax()
    drawdown = (cum_port - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Beta
    cov = np.cov(port.values, bench.values)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0

    # Information ratio
    active_returns = port - bench
    ir = (active_returns.mean() * 252) / (active_returns.std() * np.sqrt(252)) if active_returns.std() > 0 else 0

    return {
        "total_return_portfolio": round((cum_port.iloc[-1] - 1) * 100, 2),
        "total_return_benchmark": round((cum_bench.iloc[-1] - 1) * 100, 2),
        "ann_return_portfolio":   round(ann_port * 100, 2),
        "ann_return_benchmark":   round(ann_bench * 100, 2),
        "ann_alpha_bps":          round(ann_alpha * 10_000, 0),
        "ann_volatility":         round(ann_vol * 100, 2),
        "sharpe_ratio":           round(sharpe, 2),
        "max_drawdown_pct":       round(max_dd * 100, 2),
        "beta":                   round(beta, 2),
        "information_ratio":      round(ir, 2),
        "n_trading_days":         n_days,
    }


def print_attribution(returns_df: pd.DataFrame):
    """Print a formatted attribution summary."""
    stats = attribution_summary(returns_df)
    print(f"\n{'='*52}")
    print(f"  AAM Portfolio — Performance Attribution")
    print(f"  {stats['n_trading_days']} trading days")
    print(f"{'='*52}")
    print(f"  {'Metric':<32} {'Portfolio':>9} {'Benchmark':>9}")
    print(f"  {'-'*50}")
    print(f"  {'Total return':<32} {stats['total_return_portfolio']:>8.1f}% {stats['total_return_benchmark']:>8.1f}%")
    print(f"  {'Annualized return':<32} {stats['ann_return_portfolio']:>8.1f}% {stats['ann_return_benchmark']:>8.1f}%")
    print(f"  {'Alpha (annualized)':<32} {stats['ann_alpha_bps']:>7.0f}bps {'':>9}")
    print(f"  {'Annualized volatility':<32} {stats['ann_volatility']:>8.1f}% {'':>9}")
    print(f"  {'Sharpe ratio':<32} {stats['sharpe_ratio']:>9.2f} {'':>9}")
    print(f"  {'Max drawdown':<32} {stats['max_drawdown_pct']:>8.1f}% {'':>9}")
    print(f"  {'Beta vs. benchmark':<32} {stats['beta']:>9.2f} {'':>9}")
    print(f"  {'Information ratio':<32} {stats['information_ratio']:>9.2f} {'':>9}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    # Example — replace with real positions and inception date
    positions = {
        "ACVA":  0.115,
        "DRVN":  0.070,
        "ACCO":  0.055,
        "MSGS":  0.090,
    }
    start = "2025-01-01"

    print(f"Pulling returns since {start}...")
    returns = portfolio_return(positions, start_date=start)
    print_attribution(returns)
