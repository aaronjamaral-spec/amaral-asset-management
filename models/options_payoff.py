"""
models/options_payoff.py
-------------------------
Options payoff diagram and P&L analysis for AAM trade structuring.
Supports long calls, bull call spreads, covered calls,
protective puts, and LEAPS analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.config import COLORS


def long_call(S: np.ndarray, K: float, premium: float) -> np.ndarray:
    return np.maximum(S - K, 0) - premium

def long_put(S: np.ndarray, K: float, premium: float) -> np.ndarray:
    return np.maximum(K - S, 0) - premium

def covered_call(S: np.ndarray, K: float, premium: float,
                 cost_basis: float) -> np.ndarray:
    return (S - cost_basis) + np.minimum(K - S, 0) + premium

def bull_call_spread(S: np.ndarray, K_long: float, K_short: float,
                     premium_long: float, premium_short: float) -> np.ndarray:
    return long_call(S, K_long, premium_long) - long_call(S, K_short, -premium_short)

def protective_put(S: np.ndarray, K: float, premium: float,
                   cost_basis: float) -> np.ndarray:
    return (S - cost_basis) + long_put(S, K, premium)


class OptionsPayoff:
    """
    Generate payoff diagrams and summary stats for options strategies.

    Usage
    -----
    op = OptionsPayoff(current_price=18.40, ticker="ACVA")

    # Long call
    op.plot_long_call(strike=20, premium=1.50, expiry="Dec 2025")

    # Bull call spread
    op.plot_bull_spread(K_long=18, K_short=24,
                        p_long=2.20, p_short=0.80, expiry="Jun 2025")

    # LEAPS analysis
    op.plot_leaps(strike=20, premium=3.50, expiry="Jan 2027",
                  price_target=28.0)
    """

    def __init__(self, current_price: float, ticker: str = ""):
        self.S0 = current_price
        self.ticker = ticker.upper()
        self.price_range = np.linspace(self.S0 * 0.5, self.S0 * 2.0, 500)

    def _base_plot(self, title: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axhline(0, color=COLORS["gray"], linewidth=0.8, linestyle="--")
        ax.axvline(self.S0, color=COLORS["gray"], linewidth=0.8,
                   linestyle="--", label=f"Current price ${self.S0:.2f}")
        ax.set_xlabel("Stock price at expiration ($)", fontsize=11)
        ax.set_ylabel("Profit / Loss ($)", fontsize=11)
        ax.set_title(f"{self.ticker}  |  {title}", fontsize=13,
                     fontweight="bold", color=COLORS["navy"])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x:,.0f}"))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x:.0f}"))
        ax.grid(True, alpha=0.3)
        return fig, ax

    def plot_long_call(self, strike: float, premium: float,
                       expiry: str = "", contracts: int = 1):
        """Plot P&L for a long call position."""
        pnl = long_call(self.price_range, strike, premium) * 100 * contracts
        breakeven = strike + premium
        max_loss = -premium * 100 * contracts

        fig, ax = self._base_plot(
            f"Long Call | K=${strike} | Premium=${premium:.2f} | {expiry}")
        ax.plot(self.price_range, pnl, color=COLORS["navy"], linewidth=2.5)
        ax.fill_between(self.price_range, pnl, 0,
                        where=(pnl >= 0), alpha=0.15, color=COLORS["green"])
        ax.fill_between(self.price_range, pnl, 0,
                        where=(pnl < 0), alpha=0.15, color=COLORS["red"])
        ax.axvline(breakeven, color=COLORS["gold"], linewidth=1.2,
                   linestyle=":", label=f"Breakeven ${breakeven:.2f}")
        ax.axvline(strike, color=COLORS["blue"], linewidth=1,
                   linestyle=":", label=f"Strike ${strike:.2f}")
        ax.legend(fontsize=9)

        print(f"\n  Long Call Summary — {self.ticker} {expiry}")
        print(f"  Strike:       ${strike:.2f}")
        print(f"  Premium:      ${premium:.2f} ({premium/self.S0*100:.1f}% of stock)")
        print(f"  Breakeven:    ${breakeven:.2f} ({(breakeven/self.S0-1)*100:.1f}% upside needed)")
        print(f"  Max loss:     ${max_loss:,.0f} (premium paid)")
        print(f"  Upside:       Unlimited above ${breakeven:.2f}")

        plt.tight_layout()
        return fig, ax

    def plot_bull_spread(self, K_long: float, K_short: float,
                         p_long: float, p_short: float,
                         expiry: str = "", contracts: int = 1):
        """Plot P&L for a bull call spread."""
        net_debit = p_long - p_short
        pnl = bull_call_spread(self.price_range, K_long, K_short,
                                p_long, p_short) * 100 * contracts
        breakeven = K_long + net_debit
        max_profit = (K_short - K_long - net_debit) * 100 * contracts
        max_loss = -net_debit * 100 * contracts

        fig, ax = self._base_plot(
            f"Bull Call Spread | ${K_long}/{K_short} | Net debit ${net_debit:.2f} | {expiry}")
        ax.plot(self.price_range, pnl, color=COLORS["navy"], linewidth=2.5)
        ax.fill_between(self.price_range, pnl, 0,
                        where=(pnl >= 0), alpha=0.15, color=COLORS["green"])
        ax.fill_between(self.price_range, pnl, 0,
                        where=(pnl < 0), alpha=0.15, color=COLORS["red"])
        ax.axvline(breakeven, color=COLORS["gold"], linewidth=1.2,
                   linestyle=":", label=f"Breakeven ${breakeven:.2f}")
        ax.axvline(K_long,  color=COLORS["blue"], linewidth=1,
                   linestyle=":", label=f"Long strike ${K_long:.2f}")
        ax.axvline(K_short, color=COLORS["blue"], linewidth=1,
                   linestyle=":", label=f"Short strike ${K_short:.2f}")
        ax.legend(fontsize=9)

        rr = abs(max_profit / max_loss) if max_loss != 0 else np.inf
        print(f"\n  Bull Call Spread Summary — {self.ticker} {expiry}")
        print(f"  Long ${K_long} / Short ${K_short}")
        print(f"  Net debit:    ${net_debit:.2f} per share")
        print(f"  Breakeven:    ${breakeven:.2f}")
        print(f"  Max profit:   ${max_profit:,.0f}")
        print(f"  Max loss:     ${max_loss:,.0f}")
        print(f"  Risk/reward:  {rr:.1f}x")

        plt.tight_layout()
        return fig, ax

    def plot_leaps(self, strike: float, premium: float,
                   expiry: str, price_target: float = None):
        """
        Analyze a LEAPS (long-dated call) position.
        Shows payoff vs. owning stock outright.
        """
        pnl_leaps = long_call(self.price_range, strike, premium) * 100
        pnl_stock = (self.price_range - self.S0) * 100
        breakeven = strike + premium
        leverage = self.S0 / premium

        fig, ax = self._base_plot(
            f"LEAPS Analysis | K=${strike} | Premium=${premium:.2f} | {expiry}")
        ax.plot(self.price_range, pnl_leaps, color=COLORS["navy"],
                linewidth=2.5, label=f"LEAPS (1 contract = ${premium*100:.0f})")
        ax.plot(self.price_range, pnl_stock, color=COLORS["gray"],
                linewidth=1.5, linestyle="--",
                label=f"100 shares (${self.S0*100:,.0f} outlay)")
        ax.axvline(breakeven, color=COLORS["gold"], linewidth=1.2,
                   linestyle=":", label=f"Breakeven ${breakeven:.2f}")
        if price_target:
            ax.axvline(price_target, color=COLORS["green"], linewidth=1.5,
                       linestyle="-.",
                       label=f"Price target ${price_target:.2f}")
        ax.legend(fontsize=9)

        print(f"\n  LEAPS Summary — {self.ticker} {expiry}")
        print(f"  Strike:       ${strike:.2f}")
        print(f"  Premium:      ${premium:.2f} (${premium*100:.0f}/contract)")
        print(f"  Breakeven:    ${breakeven:.2f} ({(breakeven/self.S0-1)*100:.1f}% upside)")
        print(f"  Leverage:     {leverage:.1f}x vs. owning stock")
        print(f"  Capital at risk: ${premium*100:.0f} vs. ${self.S0*100:,.0f} for stock")
        if price_target:
            leaps_pnl_at_target = max(price_target - strike, 0) - premium
            stock_pnl_at_target = price_target - self.S0
            print(f"  At target ${price_target:.2f}:")
            print(f"    LEAPS P&L: ${leaps_pnl_at_target*100:,.0f} "
                  f"({leaps_pnl_at_target/premium*100:.0f}% return on premium)")
            print(f"    Stock P&L: ${stock_pnl_at_target*100:,.0f} "
                  f"({stock_pnl_at_target/self.S0*100:.0f}% return)")

        plt.tight_layout()
        return fig, ax


if __name__ == "__main__":
    op = OptionsPayoff(current_price=18.40, ticker="ACVA")
    fig1, _ = op.plot_long_call(strike=20, premium=1.50, expiry="Dec 2025")
    fig2, _ = op.plot_bull_spread(K_long=18, K_short=24,
                                   p_long=2.20, p_short=0.80, expiry="Jun 2025")
    fig3, _ = op.plot_leaps(strike=20, premium=3.50,
                             expiry="Jan 2027", price_target=28.0)
    plt.show()
