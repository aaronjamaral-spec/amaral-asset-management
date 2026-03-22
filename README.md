# Amaral Asset Management
### AI-Enabled Investment Research Portfolio

![Strategy](https://img.shields.io/badge/Strategy-Deep%20Value%20%2B%20Event--Driven-1B3A6B)
![Universe](https://img.shields.io/badge/Universe-%24500M%E2%80%93%2410B%20Market%20Cap-2E75B6)
![Benchmark](https://img.shields.io/badge/Benchmark-Russell%202500-B8960C)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This repository contains the full analytical infrastructure for Amaral Asset Management (AAM), a model investment fund built to demonstrate institutional-quality equity research, financial modeling, and AI-augmented investment processes.

**Fund parameters:**
- **Strategy:** Deep Value + Event-Driven
- **Universe:** $500M–$10B market capitalization, US-listed equities
- **Benchmark:** Russell 2500 Total Return Index (RUTWOG)
- **Portfolio:** 10–15 concentrated positions
- **Target return:** 15%+ gross per annum over a full market cycle
- **Max drawdown:** 20% from peak NAV

All investment decisions are governed by a formal Investment Policy Statement (IPS) located in `/ips`.

---

## Repository Structure

```
amaral-asset-management/
│
├── ips/                        # Investment Policy Statement
│   └── AAM_IPS_v1.pdf
│
├── models/                     # Valuation and financial models
│   ├── dcf_model.py            # Discounted cash flow model
│   ├── comps_analysis.py       # Comparable company analysis
│   ├── options_payoff.py       # Options payoff diagrams and P&L
│   └── scenario_model.py       # Bull / base / bear scenario analysis
│
├── screening/                  # Idea generation and data sourcing
│   ├── factor_screen.py        # Multi-factor small/mid cap screener
│   ├── edgar_fetch.py          # SEC EDGAR filing retrieval
│   ├── thirteen_f.py           # 13F institutional holdings parser
│   └── yfinance_pull.py        # Market data pipeline via yfinance
│
├── research/                   # Stock pitch notebooks
│   ├── pitch_template.ipynb    # Standardized pitch notebook template
│   └── [TICKER]_pitch.ipynb    # Individual company research notebooks
│
├── performance/                # Portfolio analytics
│   ├── attribution.py          # Return attribution vs. Russell 2500
│   ├── benchmark_compare.py    # Benchmark comparison and alpha calculation
│   └── risk_metrics.py         # VaR, drawdown, Sharpe, beta
│
├── utils/                      # Shared utilities
│   ├── config.py               # API keys, constants, universe parameters
│   ├── data_utils.py           # Data cleaning and transformation helpers
│   └── plot_utils.py           # Standardized charting (matplotlib/seaborn)
│
├── requirements.txt            # Python dependencies
└── README.md
```

---

## AI-Augmented Research Workflow

This portfolio is built around a structured AI-assisted research process. Key integrations:

| Tool | Role in workflow |
|---|---|
| **Claude (Anthropic)** | 10-K/transcript analysis, memo drafting, model review, research synthesis |
| **ChatGPT** | Idea generation, news scanning, alternative perspectives |
| **Python + VS Code** | Financial modeling, data pipelines, performance attribution |
| **SEC EDGAR** | Primary filings, ownership data, insider transactions |
| **FRED API** | Macro data, interest rates, economic indicators |
| **yfinance** | Market data, price history, financial statements |

All AI-generated analysis is verified against primary sources and treated as a starting point, not a conclusion.

---

## Setup

### Requirements
- Python 3.10+
- pip packages in `requirements.txt`
- Free FRED API key (register at https://fred.stlouisfed.org/docs/api/api_key.html)

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/amaral-asset-management.git
cd amaral-asset-management

# Install dependencies
pip install -r requirements.txt

# Add your FRED API key to utils/config.py
# FRED_API_KEY = "your_key_here"
```

### Quick start — pull market data for a ticker

```python
from screening.yfinance_pull import get_financials
df = get_financials("ACVA")
print(df.head())
```

### Quick start — run the factor screener

```python
from screening.factor_screen import run_screen
results = run_screen(min_market_cap=500, max_market_cap=10000)
print(results.sort_values("ev_ebitda").head(20))
```

### Quick start — run a DCF

```python
from models.dcf_model import DCFModel
dcf = DCFModel(ticker="ACVA", wacc=0.10, terminal_growth=0.03)
dcf.run()
dcf.summary()
```

---

## Research Process

Every position in this portfolio passes through a five-stage process before capital is committed:

1. **Idea generation** — Factor screener, 13F filings, spinoff monitoring, AI-assisted news scanning
2. **Preliminary review** — 30-minute screen: valuation range, business quality, red flags
3. **Deep fundamental research** — 10-K/10-Q, earnings transcripts, industry, management
4. **Quantitative modeling** — DCF, comps, scenario analysis, options payoff
5. **Investment memo + IC decision** — Standardized memo, IC Log entry, position initiated

---

## Investment Memos

Individual stock pitches are in `/research` as Jupyter notebooks. Each follows the standardized memo template covering:
- Executive summary and thesis
- Business description and competitive position
- Variant perception: why is the market wrong?
- Valuation (DCF, EV/EBITDA comps, P/TBV)
- Catalysts with estimated timelines
- Key risks and mitigants
- Position structure (instrument, size, entry, target, bear case)

---

## Disclaimer

Amaral Asset Management is a model portfolio constructed for investment research demonstration and educational purposes. Nothing in this repository constitutes investment advice or an offer to manage third-party capital. All returns referenced are hypothetical and for illustrative purposes only.

---

*Built by [@yourusername](https://github.com/yourusername) · [Mithril Multi-Asset on Substack](https://substack.com) · CFA · CAIA*
