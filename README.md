# Risk Parity Model

A hybrid Python/DuckDB implementation of equal-risk-contribution (ERC) portfolio construction, walk-forward monthly rebalancing, SQL-based performance measurement, and report generation.

The project now has unpacked source code, deterministic offline data, fail-fast SQL execution, tests, CI, and committed results.

## Verified result

The committed result bundle uses deterministic correlated synthetic prices from 2020-01-01 through 2024-01-01 (seed 42):

| Strategy | CAGR | Volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Inverse volatility | 7.20% | 5.20% | 1.318 | -7.85% |
| Equal risk contribution | 6.94% | 5.06% | 1.306 | -8.18% |
| 60/40 proxy | 6.35% | 10.43% | 0.622 | -17.68% |
| Equal weight | 3.33% | 7.35% | 0.468 | -11.08% |

These are **synthetic software-validation results**, not a historical ETF backtest. See [`results/README.md`](results/README.md) for the exact artifacts and caveats.

## Run it

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
python run_risk_parity.py --synthetic --start 2020-01-01 --end 2024-01-01
```

Omit `--synthetic` to request prices through yfinance.

## Architecture

```text
prices -> DuckDB returns/covariance views -> Python ERC optimizer
       -> DuckDB walk-forward NAV/metrics -> matplotlib reports
```

The repaired SQL layer now:

- executes whole SQL files rather than splitting semicolons inside comments;
- avoids unsupported nested window functions;
- selects one deterministic prior rebalance date per return date;
- computes diagonal risk shares with a correctly normalized numerator; and
- raises pipeline errors instead of silently returning empty tables.

## Limitations

- Synthetic correlations and return distributions are illustrative.
- The committed results exclude fees, slippage, taxes, and borrow constraints.
- Risk parity equalizes modeled risk contribution; it does not guarantee capital preservation or superior returns.
- This is research software, not investment advice.
