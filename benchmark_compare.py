"""
performance/benchmark_compare.py
----------------------------------
Benchmark comparison and alpha decomposition for AAM portfolio.
Plots cumulative returns, rolling alpha, and sector attribution
vs. the Russell 2500 benchmark.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime
from utils.config import COLORS

BENCHMARK = "IWC"   # iShares Micro-Cap ETF — Russell 2500 proxy


def cumulative_returns_chart(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    title: str = "AAM Portfolio vs. Russell 2500",
    save_path: str = None,
):
    """
    Plot cumulative return comparison: portfolio vs. benchmark.
    Highlights outperformance/underperformance periods.
    """
    cum_port  = (1 + portfolio_returns).cumprod() - 1
    cum_bench = (1 + benchmark_returns).cumprod() - 1
    active    = cum_port - cum_bench

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(title, fontsize=14, fontweight="bold", color=COLORS["navy"])

    # Cumulative returns
    ax1.plot(cum_port.index,  cum_port  * 100,
             color=COLORS["navy"], linewidth=2, label="AAM Portfolio")
    ax1.plot(cum_bench.index, cum_bench * 100,
             color=COLORS["gray"], linewidth=1.5,
             linestyle="--", label="Russell 2500")
    ax1.fill_between(active.index,
                     (cum_port * 100), (cum_bench * 100),
                     where=(cum_port >= cum_bench),
                     alpha=0.15, color=COLORS["green"],
                     label="Outperformance")
    ax1.fill_between(active.index,
                     (cum_port * 100), (cum_bench * 100),
                     where=(cum_port < cum_bench),
                     alpha=0.15, color=COLORS["red"],
                     label="Underperformance")
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax1.set_ylabel("Cumulative return (%)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color=COLORS["gray"], linewidth=0.5)

    # Active return (alpha)
    ax2.bar(active.index, active * 100,
            color=[COLORS["green"] if x >= 0 else COLORS["red"]
                   for x in active],
            alpha=0.7, width=1)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.set_ylabel("Active return (%)", fontsize=10)
    ax2.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved to {save_path}")
    return fig


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Generate a monthly returns table (months as columns, years as rows).
    Useful for performance reporting.
    """
    monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    df = monthly.to_frame("return")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot["Annual"] = (1 + monthly).resample("Y").prod() - 1
    return (pivot * 100).round(2)


def print_monthly_table(portfolio_returns: pd.Series,
                        benchmark_returns: pd.Series):
    """Print monthly return tables for portfolio and benchmark."""
    port_monthly  = monthly_returns_table(portfolio_returns)
    bench_monthly = monthly_returns_table(benchmark_returns)

    print(f"\n  AAM Portfolio — Monthly Returns (%)")
    print(f"  {'='*90}")
    print(port_monthly.to_string())

    print(f"\n  Russell 2500 — Monthly Returns (%)")
    print(f"  {'='*90}")
    print(bench_monthly.to_string())

    print(f"\n  Active Returns (Portfolio minus Benchmark, %)")
    print(f"  {'='*90}")
    active = (port_monthly - bench_monthly).round(2)
    print(active.to_string())


if __name__ == "__main__":
    from datetime import timedelta

    start = (datetime.today() - timedelta(days=252)).strftime("%Y-%m-%d")
    tickers = {"ACVA": 0.115, "DRVN": 0.07, "ACCO": 0.055, "MSGS": 0.09}

    print(f"Pulling data from {start}...")
    prices = yf.download(
        list(tickers.keys()) + [BENCHMARK],
        start=start, progress=False, auto_adjust=True
    )["Close"]

    returns  = prices.pct_change().dropna()
    weights  = np.array([tickers[t] for t in list(tickers.keys())])
    port_ret = returns[list(tickers.keys())].values @ weights

    port_series  = pd.Series(port_ret, index=returns.index, name="Portfolio")
    bench_series = returns[BENCHMARK].rename("Benchmark")

    fig = cumulative_returns_chart(port_series, bench_series)
    print_monthly_table(port_series, bench_series)
    plt.show()
