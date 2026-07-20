# Results

Generated with:

```bash
python run_risk_parity.py --synthetic --start 2020-01-01 --end 2024-01-01
```

The synthetic price generator is deterministic (seed 42). No market observations are represented.

- [`tables/performance_summary.csv`](tables/performance_summary.csv): 1,014-day strategy comparison
- [`tables/latest_risk_contributions.csv`](tables/latest_risk_contributions.csv): final-date diagonal risk-share diagnostic
- [`figures/`](figures/): weights, risk contributions, NAV, drawdown, rolling Sharpe, correlations, and annual returns

ERC produced a 6.94% CAGR, 5.06% annualized volatility, 1.306 Sharpe, and -8.18% maximum drawdown. Inverse volatility was slightly stronger on this one synthetic path. That comparison is useful for code validation, not evidence of investable alpha.
