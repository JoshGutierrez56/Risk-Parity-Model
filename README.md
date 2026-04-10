# Risk Parity in SQL
### Equal Risk Contribution | DuckDB Analytics | Python ERC Solver | Multi-Asset

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/SQL-DuckDB-yellow)](https://duckdb.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What This Is

A risk parity analytics system built on a **DuckDB + Python hybrid architecture**,
directly targeting PanAgora's documented tech stack. Edward Qian (PanAgora's CIO
of Multi-Asset) coined the term "risk parity" in a 2004 paper and this project
implements his Equal Risk Contribution framework:

> *"The term 'risk parity' was coined by Edward Qian in a paper written in 2004.
> Qian is currently Chief Investment Officer of Multi Asset Investments at PanAgora."*
> — Wikipedia / PanAgora

PanAgora's job postings explicitly require Python + SQL fluency. This repo
demonstrates exactly that split: **SQL handles everything expressible as a
window function; Python handles only what SQL cannot** (iterative ERC optimisation).

---

## The SQL / Python Split

| Layer | Tool | What it does |
|-------|------|-------------|
| Data ingestion | Python (yfinance) | Download prices, load into DuckDB |
| Returns | **SQL** | `LN(close / LAG(close) OVER (...))` |
| Rolling covariance | **SQL** | `E[XY] − E[X]·E[Y]` via window aggregates |
| Inverse-vol weights | **SQL** | `1/σ / Σ(1/σ)` — closed form |
| NAV compounding | **SQL** | `EXP(SUM(log_ret) OVER (...))` |
| Drawdown | **SQL** | `nav / MAX(nav) OVER (...) − 1` |
| Rolling Sharpe | **SQL** | `AVG / STDDEV × SQRT(252)` window |
| Annual returns | **SQL** | `GROUP BY YEAR(date)` |
| ERC optimisation | Python (scipy) | Iterative — SQL cannot do this |
| Figures | Python (matplotlib) | Queries DuckDB views directly |

---

## Equal Risk Contribution (ERC)

The ERC portfolio satisfies:

```
RC_i = w_i × (Σw)_i / (wᵀΣw) = 1/N  ∀ i
```

Where RC_i is the fraction of total portfolio variance attributable to asset i.
Unlike the simpler inverse-volatility heuristic, ERC accounts for correlations —
which is critical in a multi-asset portfolio where equities and bonds are negatively
correlated.

**Verified properties (test suite):**

| Test | Result |
|------|--------|
| `solve_erc` converges on diagonal Σ | ✅ RC error < 5.6e-17 |
| `solve_erc` converges on correlated 4-asset Σ | ✅ RC error < 2.4e-6 |
| ERC = inv-vol on diagonal Σ (zero correlations) | ✅ |
| ERC ≠ inv-vol on correlated Σ (diff = 9.5%) | ✅ |
| ERC weights: all positive, sum to 1 | ✅ |

---

## Universe — 10-Asset Multi-Class

```
US Equity     SPY   S&P 500
Intl Equity   EFA   MSCI EAFE
EM Equity     EEM   MSCI Emerging Markets
LT Treasury   TLT   20+ Year US Treasury
EM Bonds      EMB   EM USD Bonds
Commodities   DBC   Broad Commodity Index
Gold          GLD   Gold ETF
US REIT       VNQ   Vanguard Real Estate
TIPS          TIP   Inflation-Linked Bonds
Cash Proxy    SHY   1-3 Year Treasury
```

This 10-asset universe spans Growth, Rates, Inflation, and Liquidity risk
factors — the same decomposition used in Bridgewater's All Weather strategy.

---

## Repository Structure

```
risk-parity-sql/
├── run_risk_parity.py              ← CLI entry point
│
├── sql/                            ← Pure SQL analytics (DuckDB)
│   ├── schema.sql                  ← Table and view definitions
│   ├── returns.sql                 ← Log returns + vol views
│   ├── covariance.sql              ← Rolling pairwise cov + inv-vol weights
│   └── backtest.sql                ← NAV, drawdown, Sharpe, summary
│
├── src/riskparity/
│   ├── db.py                       ← DuckDB connection wrapper
│   ├── loader.py                   ← yfinance → DuckDB ingestion
│   ├── optimizer.py                ← ERC solver (scipy SLSQP)
│   └── report.py                   ← 7 figures querying DuckDB views
│
└── tests/
    └── test_erc_math.py            ← ERC math tests (no DuckDB required)
```

---

## Setup & Usage

```bash
pip install -r requirements.txt   # includes duckdb

# Fully offline (synthetic correlated prices)
python run_risk_parity.py --synthetic

# Full pipeline (downloads via yfinance)
python run_risk_parity.py

# Print performance table only
python run_risk_parity.py --scorecard
```

**Outputs:**

```
outputs/figures/
  fig1_weights_risk_parity.png    ← stacked weight history
  fig2_risk_contribution.png      ← capital vs risk contribution bar chart
  fig3_cumulative_nav.png         ← all strategies vs 60/40
  fig4_drawdown.png               ← drawdown series (from SQL)
  fig5_rolling_sharpe.png         ← 252-day rolling Sharpe (from SQL)
  fig6_correlation.png            ← asset correlation matrix
  fig7_annual_returns.png         ← calendar-year heatmap
outputs/tables/
  performance_summary.csv         ← CAGR, Sharpe, MDD, Calmar, hit rate
```

---

## Requirements

```
duckdb>=0.10
pandas>=2.0
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
yfinance>=0.2.36   # optional; --synthetic bypasses
```

---

## Academic References

| Reference | Applied In |
|-----------|-----------|
| Qian (2004) — *Risk Parity Portfolios* (PanAgora white paper) | Core framework |
| Maillard, Roncalli & Teïletche (2010) — *ERC Portfolios* | ERC optimisation |
| Ledoit & Wolf (2004) — *Analytical LW Shrinkage* | Covariance estimation |
| Roncalli (2013) — *Introduction to Risk Parity* | Implementation reference |
