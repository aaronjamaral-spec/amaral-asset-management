"""
models/scenario_model.py
-------------------------
Bull / base / bear scenario analysis for AAM equity research.
Builds a structured scenario table showing revenue, EBITDA,
FCF, and implied valuation under each case.
Designed to complement the DCF model.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from utils.config import VALUATION


@dataclass
class Scenario:
    label: str
    revenue_growth: float       # Annual revenue growth rate
    ebitda_margin: float        # EBITDA margin (stabilized)
    ev_ebitda_exit: float       # Exit EV/EBITDA multiple
    probability: float          # Probability weight (should sum to 1.0 across scenarios)
    color: str = "gray"


class ScenarioModel:
    """
    Three-scenario valuation model for AAM investment memos.
    Outputs implied share price under each scenario and
    a probability-weighted intrinsic value.

    Usage
    -----
    sm = ScenarioModel(
        ticker="ACVA",
        base_revenue_ttm=485,
        net_debt=-120,
        shares_outstanding=155,
        current_price=18.40,
        projection_years=3,
    )
    sm.run()
    sm.summary()
    """

    def __init__(
        self,
        ticker: str,
        base_revenue_ttm: float,
        net_debt: float,
        shares_outstanding: float,
        current_price: float,
        projection_years: int = 3,
        scenarios: list = None,
        tax_rate: float = None,
        capex_pct: float = 0.04,
        da_pct: float = 0.04,
    ):
        self.ticker = ticker.upper()
        self.base_revenue = base_revenue_ttm
        self.net_debt = net_debt
        self.shares = shares_outstanding
        self.price = current_price
        self.years = projection_years
        self.tax_rate = tax_rate or VALUATION["tax_rate"]
        self.capex_pct = capex_pct
        self.da_pct = da_pct
        self.results = []

        self.scenarios = scenarios or [
            Scenario("Bear", revenue_growth=0.03, ebitda_margin=0.10,
                     ev_ebitda_exit=6.0,  probability=0.25, color="red"),
            Scenario("Base", revenue_growth=0.10, ebitda_margin=0.15,
                     ev_ebitda_exit=10.0, probability=0.50, color="blue"),
            Scenario("Bull", revenue_growth=0.18, ebitda_margin=0.22,
                     ev_ebitda_exit=14.0, probability=0.25, color="green"),
        ]

    def _run_scenario(self, s: Scenario) -> dict:
        fwd_revenue = self.base_revenue * ((1 + s.revenue_growth) ** self.years)
        fwd_ebitda  = fwd_revenue * s.ebitda_margin
        fwd_fcf     = fwd_ebitda * (1 - self.tax_rate) - (fwd_revenue * self.capex_pct)
        exit_ev     = fwd_ebitda * s.ev_ebitda_exit
        equity_val  = exit_ev - self.net_debt
        price_iv    = equity_val / self.shares
        upside      = (price_iv / self.price - 1) * 100

        return {
            "scenario":         s.label,
            "probability":      s.probability,
            "rev_growth_pa":    s.revenue_growth,
            "fwd_revenue_m":    round(fwd_revenue, 1),
            "fwd_ebitda_m":     round(fwd_ebitda, 1),
            "ebitda_margin":    s.ebitda_margin,
            "exit_ev_ebitda":   s.ev_ebitda_exit,
            "exit_ev_m":        round(exit_ev, 1),
            "equity_value_m":   round(equity_val, 1),
            "implied_price":    round(price_iv, 2),
            "upside_pct":       round(upside, 1),
        }

    def run(self):
        self.results = [self._run_scenario(s) for s in self.scenarios]
        return self

    def summary(self):
        if not self.results:
            print("Run .run() first.")
            return

        df = pd.DataFrame(self.results)
        weighted_iv = (df["implied_price"] * df["probability"]).sum()
        weighted_upside = (weighted_iv / self.price - 1) * 100

        print(f"\n{'='*68}")
        print(f"  Scenario Analysis — {self.ticker}")
        print(f"  Base revenue: ${self.base_revenue:,.0f}M | "
              f"Net debt: ${self.net_debt:,.0f}M | "
              f"Current price: ${self.price:.2f}")
        print(f"  Projection horizon: {self.years} years")
        print(f"{'='*68}")
        print(f"  {'':12} {'Bear':>10} {'Base':>10} {'Bull':>10}")
        print(f"  {'-'*44}")
        print(f"  {'Probability':<12} "
              f"{'25%':>10} {'50%':>10} {'25%':>10}")
        print(f"  {'Rev growth':<12} "
              + "".join(f"{r['rev_growth_pa']*100:>9.0f}%" for r in self.results))
        print(f"  {'Fwd revenue':<12} "
              + "".join(f"${r['fwd_revenue_m']:>8,.0f}M" for r in self.results))
        print(f"  {'EBITDA margin':<12} "
              + "".join(f"{r['ebitda_margin']*100:>9.0f}%" for r in self.results))
        print(f"  {'Exit EV/EBITDA':<12} "
              + "".join(f"{r['exit_ev_ebitda']:>9.1f}x" for r in self.results))
        print(f"  {'Implied price':<12} "
              + "".join(f"${r['implied_price']:>9.2f}" for r in self.results))
        print(f"  {'Upside':<12} "
              + "".join(f"{r['upside_pct']:>9.1f}%" for r in self.results))
        print(f"{'='*68}")
        print(f"  Probability-weighted IV:  ${weighted_iv:.2f}  "
              f"({weighted_upside:+.1f}% vs. current price)")
        print(f"{'='*68}\n")

        return df


if __name__ == "__main__":
    sm = ScenarioModel(
        ticker="ACVA",
        base_revenue_ttm=485,
        net_debt=-120,
        shares_outstanding=155,
        current_price=18.40,
        projection_years=3,
    )
    sm.run()
    sm.summary()
