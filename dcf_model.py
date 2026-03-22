"""
models/dcf_model.py
--------------------
Discounted Cash Flow (DCF) valuation model for AAM equity research.
Supports 3-scenario analysis (bull / base / bear) and sensitivity tables.
Pulls live financial data from yfinance where possible.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import dataclass, field
from typing import Optional
from utils.config import VALUATION, COLORS


@dataclass
class DCFAssumptions:
    """Holds all assumptions for a single DCF scenario."""
    label: str
    revenue_growth_rates: list      # Year-by-year revenue growth (e.g. [0.15, 0.12, 0.10, 0.08, 0.07])
    ebitda_margin: float            # Stabilized EBITDA margin
    capex_pct_revenue: float        # CapEx as % of revenue
    nwc_change_pct_revenue: float   # Change in NWC as % of revenue
    terminal_growth_rate: float     # Terminal growth rate (Gordon Growth)
    wacc: float                     # Discount rate
    da_pct_revenue: float = 0.04    # D&A as % of revenue
    tax_rate: float = field(default_factory=lambda: VALUATION["tax_rate"])


class DCFModel:
    """
    Five-year DCF with terminal value (Gordon Growth Model).
    Supports bull / base / bear scenario analysis.

    Usage
    -----
    dcf = DCFModel(
        ticker="ACVA",
        base_revenue_ttm=500,     # $M TTM revenue
        net_debt=150,             # $M net debt
        shares_outstanding=100,   # M shares
    )
    dcf.run()
    dcf.summary()
    dcf.plot_waterfall()
    dcf.sensitivity_table()
    """

    def __init__(
        self,
        ticker: str,
        base_revenue_ttm: float,
        net_debt: float,
        shares_outstanding: float,
        scenarios: Optional[list[DCFAssumptions]] = None,
        current_price: Optional[float] = None,
    ):
        self.ticker = ticker.upper()
        self.base_revenue = base_revenue_ttm
        self.net_debt = net_debt
        self.shares = shares_outstanding
        self.current_price = current_price
        self.results = {}

        # Default scenarios if none provided
        self.scenarios = scenarios or self._default_scenarios()

    def _default_scenarios(self) -> list[DCFAssumptions]:
        """Return standard bull / base / bear assumptions."""
        return [
            DCFAssumptions(
                label="Bear",
                revenue_growth_rates=[0.05, 0.04, 0.03, 0.03, 0.02],
                ebitda_margin=0.10,
                capex_pct_revenue=0.05,
                nwc_change_pct_revenue=0.02,
                terminal_growth_rate=0.02,
                wacc=0.12,
            ),
            DCFAssumptions(
                label="Base",
                revenue_growth_rates=[0.10, 0.09, 0.08, 0.07, 0.06],
                ebitda_margin=0.15,
                capex_pct_revenue=0.04,
                nwc_change_pct_revenue=0.015,
                terminal_growth_rate=0.025,
                wacc=0.10,
            ),
            DCFAssumptions(
                label="Bull",
                revenue_growth_rates=[0.18, 0.16, 0.14, 0.12, 0.10],
                ebitda_margin=0.22,
                capex_pct_revenue=0.035,
                nwc_change_pct_revenue=0.01,
                terminal_growth_rate=0.03,
                wacc=0.09,
            ),
        ]

    def _run_scenario(self, a: DCFAssumptions) -> dict:
        """Run a single DCF scenario and return results dict."""
        years = len(a.revenue_growth_rates)
        revenues, ebitdas, fcfs = [], [], []
        rev = self.base_revenue

        for g in a.revenue_growth_rates:
            rev = rev * (1 + g)
            revenues.append(rev)
            ebitda = rev * a.ebitda_margin
            ebitdas.append(ebitda)
            da = rev * a.da_pct_revenue
            ebit = ebitda - da
            nopat = ebit * (1 - a.tax_rate)
            capex = rev * a.capex_pct_revenue
            nwc_change = rev * a.nwc_change_pct_revenue
            fcf = nopat + da - capex - nwc_change
            fcfs.append(fcf)

        # Discount FCFs
        discount_factors = [(1 / (1 + a.wacc) ** (i + 1)) for i in range(years)]
        pv_fcfs = [fcf * df for fcf, df in zip(fcfs, discount_factors)]

        # Terminal value (Gordon Growth)
        terminal_fcf = fcfs[-1] * (1 + a.terminal_growth_rate)
        terminal_value = terminal_fcf / (a.wacc - a.terminal_growth_rate)
        pv_terminal = terminal_value * discount_factors[-1]

        # Enterprise and equity value
        enterprise_value = sum(pv_fcfs) + pv_terminal
        equity_value = enterprise_value - self.net_debt
        intrinsic_value_per_share = equity_value / self.shares

        return {
            "label":                    a.label,
            "revenues":                 revenues,
            "ebitdas":                  ebitdas,
            "fcfs":                     fcfs,
            "pv_fcfs":                  pv_fcfs,
            "pv_fcf_total":             sum(pv_fcfs),
            "terminal_value":           terminal_value,
            "pv_terminal":              pv_terminal,
            "enterprise_value":         enterprise_value,
            "equity_value":             equity_value,
            "intrinsic_value_per_share":intrinsic_value_per_share,
            "tv_pct_of_ev":             pv_terminal / enterprise_value * 100,
            "wacc":                     a.wacc,
            "terminal_growth":          a.terminal_growth_rate,
            "ebitda_margin":            a.ebitda_margin,
        }

    def run(self):
        """Run all scenarios and store results."""
        for scenario in self.scenarios:
            self.results[scenario.label] = self._run_scenario(scenario)
        print(f"\nDCF complete for {self.ticker} — {len(self.results)} scenarios run.")
        return self

    def summary(self):
        """Print a clean summary table of all scenario outputs."""
        if not self.results:
            print("Run .run() first.")
            return

        print(f"\n{'='*60}")
        print(f"  DCF Summary — {self.ticker}")
        print(f"  Base revenue: ${self.base_revenue:,.0f}M | Net debt: ${self.net_debt:,.0f}M")
        print(f"  Shares outstanding: {self.shares:.1f}M")
        if self.current_price:
            print(f"  Current price: ${self.current_price:.2f}")
        print(f"{'='*60}")
        print(f"  {'Metric':<35} {'Bear':>8} {'Base':>8} {'Bull':>8}")
        print(f"  {'-'*59}")

        def row(label, key, fmt="dollar"):
            vals = [self.results[s][key] for s in ["Bear", "Base", "Bull"]]
            if fmt == "dollar":
                formatted = [f"${v:,.0f}M" if abs(v) >= 1 else f"${v:.1f}M" for v in vals]
            elif fmt == "pct":
                formatted = [f"{v:.1f}%" for v in vals]
            elif fmt == "share":
                formatted = [f"${v:.2f}" for v in vals]
            else:
                formatted = [f"{v:.1f}x" for v in vals]
            print(f"  {label:<35} {formatted[0]:>8} {formatted[1]:>8} {formatted[2]:>8}")

        row("Enterprise value", "enterprise_value")
        row("Equity value", "equity_value")
        row("Intrinsic value / share", "intrinsic_value_per_share", "share")
        row("PV of FCFs", "pv_fcf_total")
        row("PV of terminal value", "pv_terminal")
        row("Terminal value % of EV", "tv_pct_of_ev", "pct")
        row("EBITDA margin (stabilized)", "ebitda_margin", "pct")
        row("WACC", "wacc", "pct")

        if self.current_price:
            print(f"\n  {'Upside / downside to current price':}")
            for label in ["Bear", "Base", "Bull"]:
                iv = self.results[label]["intrinsic_value_per_share"]
                upside = (iv / self.current_price - 1) * 100
                sign = "+" if upside >= 0 else ""
                print(f"    {label:<8}: {sign}{upside:.1f}%")
        print(f"{'='*60}\n")

    def sensitivity_table(self, scenario: str = "Base"):
        """
        Print a WACC × terminal growth rate sensitivity table
        for the base case intrinsic value per share.
        """
        if scenario not in self.results:
            print(f"Scenario '{scenario}' not found. Run .run() first.")
            return

        res = self.results[scenario]
        waccs = [res["wacc"] - 0.02, res["wacc"] - 0.01, res["wacc"],
                 res["wacc"] + 0.01, res["wacc"] + 0.02]
        tgrs  = [res["terminal_growth"] - 0.01, res["terminal_growth"] - 0.005,
                 res["terminal_growth"], res["terminal_growth"] + 0.005,
                 res["terminal_growth"] + 0.01]

        s = self.scenarios[[s.label for s in self.scenarios].index(scenario)]
        print(f"\n  Sensitivity: Intrinsic value/share ({scenario} case)")
        print(f"  Rows = WACC | Columns = Terminal growth rate\n")
        header = f"  {'WACC \\ TGR':<12}" + "".join(f"  {t*100:.1f}%" for t in tgrs)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for w in waccs:
            row_vals = []
            for tg in tgrs:
                test = DCFAssumptions(
                    label="test",
                    revenue_growth_rates=s.revenue_growth_rates,
                    ebitda_margin=s.ebitda_margin,
                    capex_pct_revenue=s.capex_pct_revenue,
                    nwc_change_pct_revenue=s.nwc_change_pct_revenue,
                    terminal_growth_rate=tg,
                    wacc=w,
                    da_pct_revenue=s.da_pct_revenue,
                    tax_rate=s.tax_rate,
                )
                r = self._run_scenario(test)
                row_vals.append(r["intrinsic_value_per_share"])
            row_str = f"  {w*100:.1f}%{'':<8}" + "".join(f"  ${v:>6.2f}" for v in row_vals)
            print(row_str)
        print()


if __name__ == "__main__":
    # Example: run a DCF on ACVA
    dcf = DCFModel(
        ticker="ACVA",
        base_revenue_ttm=485,
        net_debt=-120,          # net cash position
        shares_outstanding=155,
        current_price=18.40,
    )
    dcf.run()
    dcf.summary()
    dcf.sensitivity_table("Base")
