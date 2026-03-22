"""
performance/risk_metrics.py
----------------------------
Portfolio risk metrics for AAM — VaR, CVaR, drawdown analysis,
rolling Sharpe, beta, and correlation matrix.
All metrics calculated from daily return series.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.config import COLORS


def value_at_risk(returns: pd.Series, confidence: float = 0.95,
                  horizon: int = 1) -> float:
    """
    Historical VaR at given confidence level and horizon (days).
    Returns the loss (positive number = $ loss per $1 invested).
    """
    scaled = returns * np.sqrt(horizon)
    return float(-np.percentile(scaled, (1 - confidence) * 100))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR (Expected Shortfall) — average loss beyond the VaR threshold."""
    var = value_at_risk(returns, confidence)
    return float(-returns[returns <= -var].mean())


def max_drawdown(returns: pd.Series) -> tuple:
    """
    Calculate maximum drawdown and its duration.

    Returns
    -------
    (max_dd_pct, peak_date, trough_date, recovery_date or None)
    """
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max

    trough_idx = drawdown.idxmin()
    peak_idx = drawdown[:trough_idx].index[
        cum[:trough_idx].argmax()
    ]

    # Find recovery
    post_trough = cum[trough_idx:]
    peak_value = cum[peak_idx]
    recovery = post_trough[post_trough >= peak_value]
    recovery_date = recovery.index[0] if not recovery.empty else None

    return (
        float(drawdown.min()),
        peak_idx,
        trough_idx,
        recovery_date,
    )


def rolling_sharpe(returns: pd.Series, window: int = 63,
                   rf_annual: float = 0.05) -> pd.Series:
    """
    Rolling Sharpe ratio over a given window (default 63 days = ~1 quarter).
    """
    rf_daily = rf_annual / 252
    excess = returns - rf_daily
    roll_mean = excess.rolling(window).mean() * 252
    roll_std  = returns.rolling(window).std() * np.sqrt(252)
    return (roll_mean / roll_std).replace([np.inf, -np.inf], np.nan)


def beta(portfolio_returns: pd.Series,
         benchmark_returns: pd.Series) -> float:
    """Calculate portfolio beta vs. benchmark."""
    aligned = pd.concat([portfolio_returns, benchmark_returns],
                        axis=1).dropna()
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0


def full_risk_report(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    confidence: float = 0.95,
    rf_annual: float = 0.05,
) -> dict:
    """
    Generate a complete risk metrics report.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns
    benchmark_returns : pd.Series
        Daily benchmark returns
    confidence : float
        VaR/CVaR confidence level (default 95%)
    rf_annual : float
        Annual risk-free rate

    Returns
    -------
    dict of all risk metrics
    """
    aligned = pd.concat(
        [portfolio_returns.rename("port"),
         benchmark_returns.rename("bench")], axis=1
    ).dropna()

    port  = aligned["port"]
    bench = aligned["bench"]

    var_1d   = value_at_risk(port, confidence, horizon=1)
    var_10d  = value_at_risk(port, confidence, horizon=10)
    cvar_1d  = conditional_var(port, confidence)

    dd, peak, trough, recovery = max_drawdown(port)

    ann_port  = port.mean() * 252
    ann_bench = bench.mean() * 252
    ann_vol   = port.std() * np.sqrt(252)
    sharpe    = (ann_port - rf_annual) / ann_vol if ann_vol > 0 else 0
    b         = beta(port, bench)
    treynor   = (ann_port - rf_annual) / b if b != 0 else 0

    active    = port - bench
    info_ratio = (active.mean() * 252) / (active.std() * np.sqrt(252)) \
                  if active.std() > 0 else 0

    return {
        "var_1d_pct":        round(var_1d * 100, 2),
        "var_10d_pct":       round(var_10d * 100, 2),
        "cvar_1d_pct":       round(cvar_1d * 100, 2),
        "max_drawdown_pct":  round(dd * 100, 2),
        "drawdown_peak":     str(peak.date()) if hasattr(peak, 'date') else str(peak),
        "drawdown_trough":   str(trough.date()) if hasattr(trough, 'date') else str(trough),
        "drawdown_recovery": str(recovery.date()) if recovery and hasattr(recovery, 'date') else "Not yet recovered",
        "ann_return_pct":    round(ann_port * 100, 2),
        "ann_volatility_pct":round(ann_vol * 100, 2),
        "sharpe_ratio":      round(sharpe, 2),
        "beta":              round(b, 2),
        "treynor_ratio":     round(treynor, 4),
        "information_ratio": round(info_ratio, 2),
        "ann_alpha_bps":     round((ann_port - ann_bench) * 10_000, 0),
        "confidence_level":  confidence,
        "n_trading_days":    len(port),
    }


def print_risk_report(metrics: dict):
    """Print a formatted risk metrics report."""
    print(f"\n{'='*52}")
    print(f"  AAM Risk Metrics Report")
    print(f"  Confidence level: {metrics['confidence_level']*100:.0f}%")
    print(f"  Trading days: {metrics['n_trading_days']}")
    print(f"{'='*52}")
    print(f"  {'VaR (1-day)':<35} {metrics['var_1d_pct']:>6.2f}%")
    print(f"  {'VaR (10-day)':<35} {metrics['var_10d_pct']:>6.2f}%")
    print(f"  {'CVaR / Expected shortfall':<35} {metrics['cvar_1d_pct']:>6.2f}%")
    print(f"  {'Max drawdown':<35} {metrics['max_drawdown_pct']:>6.2f}%")
    print(f"  {'  Peak':<35} {metrics['drawdown_peak']:>12}")
    print(f"  {'  Trough':<35} {metrics['drawdown_trough']:>12}")
    print(f"  {'  Recovery':<35} {metrics['drawdown_recovery']:>12}")
    print(f"  {'Annualized return':<35} {metrics['ann_return_pct']:>6.2f}%")
    print(f"  {'Annualized volatility':<35} {metrics['ann_volatility_pct']:>6.2f}%")
    print(f"  {'Sharpe ratio':<35} {metrics['sharpe_ratio']:>6.2f}x")
    print(f"  {'Beta':<35} {metrics['beta']:>6.2f}x")
    print(f"  {'Information ratio':<35} {metrics['information_ratio']:>6.2f}x")
    print(f"  {'Alpha (annualized)':<35} {metrics['ann_alpha_bps']:>4.0f} bps")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta

    start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    port_tickers = {"ACVA": 0.115, "DRVN": 0.07, "ACCO": 0.055, "MSGS": 0.09}

    prices = yf.download(
        list(port_tickers.keys()) + ["IWC"],
        start=start, progress=False, auto_adjust=True
    )["Close"]

    returns = prices.pct_change().dropna()
    weights = np.array([port_tickers[t] for t in list(port_tickers.keys())])
    port_ret = returns[list(port_tickers.keys())].values @ weights
    port_series  = pd.Series(port_ret, index=returns.index)
    bench_series = returns["IWC"]

    metrics = full_risk_report(port_series, bench_series)
    print_risk_report(metrics)
