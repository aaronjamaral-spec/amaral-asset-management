"""
utils/config.py
---------------
Central configuration for Amaral Asset Management research infrastructure.
Add your API keys here. Never commit this file with real keys — use a
.env file or environment variables in production.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
FRED_API_KEY = os.getenv("FRED_API_KEY", "d11ef2371ab1996d1af18c6c02b6491d")

# Fund Universe Parameters
UNIVERSE = {
    "min_market_cap_m": 500,
    "max_market_cap_m": 10_000,
    "min_avg_volume_m": 1,
    "exchanges": ["NYSE", "NASDAQ", "NYSEAmerican"],
    "benchmark_ticker": "IWC",
    "benchmark_name": "Russell 2500",
}

# Portfolio Parameters
PORTFOLIO = {
    "max_positions": 15,
    "min_positions": 8,
    "max_single_position": 0.125,
    "max_sector_concentration": 0.35,
    "max_cash": 0.20,
    "target_gross_return": 0.15,
    "max_drawdown": 0.20,
    "min_rr_ratio": 2.0,
    "min_discount_to_iv": 0.30,
}

# Valuation Defaults
VALUATION = {
    "default_wacc": 0.10,
    "default_terminal_growth": 0.03,
    "default_projection_years": 5,
    "tax_rate": 0.21,
}

# Data
DATA = {
    "edgar_user_agent": "Amaral Asset Management research@amaralassetmgmt.com",
    "price_history_years": 5,
}

# Plotting colors
COLORS = {
    "navy":       "#1B3A6B",
    "blue":       "#2E75B6",
    "gold":       "#B8960C",
    "light_gold": "#C9A227",
    "gray":       "#888888",
    "green":      "#2E7D32",
    "red":        "#C62828",
}
