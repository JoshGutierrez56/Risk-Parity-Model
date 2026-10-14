# Methodology and validation contract

## Universe and covariance

The demonstration uses SPY, EFA, EEM, TLT, EMB, DBC, GLD, VNQ, TIP and SHY. SHY is a short-duration Treasury ETF, not risk-free cash. The baseline labeled `60_40` is 60% SPY / 40% TLT.

Inputs are positive, finite adjusted-close prices with one observation per date/ticker and a common calendar. The loader rejects incomplete panels, duplicate dates and invalid prices. It does not fill gaps or silently shrink the universe. Downloads are requested for the selected period without reusing an unrelated cached range; a failed download never becomes synthetic data.

The SQL return table stores daily log returns. Covariance is `COVAR_POP` over up to 60 paired return observations, multiplied by 252, with at least 30 observations. Windows are partitioned by **both** asset identifiers after the pairwise date join. This is a population estimator, not sample covariance or shrinkage.

The Python optimizer requires a complete, finite, symmetric covariance matrix with positive diagonal and no materially negative eigenvalue. Missing entries are not assumed to be zero correlation. ERC failure or a maximum risk-share deviation above 0.005 stops the run.

## Risk contributions

For weights `w` and covariance `Sigma`:

```text
portfolio_variance = w.T @ Sigma @ w
marginal_variance_i = (Sigma @ w)_i
risk_share_i = w_i * marginal_variance_i / portfolio_variance
```

Both the optimizer and the SQL diagnostic use the full covariance, including cross terms. No diagonal approximation is displayed as an ERC result. The solver targets 1/N shares with positive weights summing to one.

## Timing and holdings

1. On a decision date `d`, calculate covariance using observations through `d` and calculate target weights.
2. Those targets apply only to return dates strictly greater than `d`.
3. Between target changes, asset holdings drift with their prices. They are not silently restored to target weights each day.
4. On the next decision date, the old holdings earn that date's return. The new targets apply to the following return.

Within each target segment, each asset's initial dollar allocation grows by its cumulative simple gross return. Summing those dollar values gives segment wealth; successive segment-wealth ratios give portfolio simple returns. Across segments, the portfolio compounds `1 + simple_return`. This avoids treating a weighted average of asset log returns as an exact portfolio return.

This is **idealized close-to-close accounting**, not a claim of feasible execution at the close after observing the closing price. A live strategy needs explicit decision/execution timestamps, latency, slippage and costs. The prototype models none of those.

### Hand-derived example

Initial prices are 100 for both assets; initial weights are 50/50. Asset A then goes to 110 and 121 while asset B stays at 100:

| Observation | A dollar holding | B dollar holding | Portfolio value |
|---|---:|---:|---:|
| Initial | 50 | 50 | 100 |
| First return | 55 | 50 | 105 |
| Second return | 60.5 | 50 | 110.5 |

The second portfolio return is `110.5 / 105 - 1`, not 5%. A new 100% A target set at the second close affects only the next return. This literal holding calculation is asserted by the [accounting fixture](../tests/test_sql_accounting.py).

Missing returns for held assets raise an error; they are not dropped and renormalized. Input validation requires a complete price calendar before returns are calculated.

## Performance definitions

- **NAV:** compounded simple returns, starting at 1.
- **Drawdown:** NAV divided by its running peak, including initial NAV = 1, minus 1. A loss on the first earned day is therefore a drawdown.
- **CAGR:** `final_nav ** (252 / number_of_earned_returns) - 1`; all earned return observations, including the first, count.
- **Annualized volatility:** sample standard deviation of daily simple returns times sqrt(252).
- **Sharpe:** arithmetic mean daily simple return divided by sample standard deviation, times sqrt(252); zero risk-free rate.
- **Rolling Sharpe:** available only with a full 252-return window.
- **Annual figures:** compound the observed returns within each year; partial years remain partial periods, not full-calendar-year estimates.

There are no modeled transaction costs, financing charges, taxes or leverage. These definitions do not establish investment merit.

## Repair and provenance notes

The legacy archive remains unchanged:

```text
risk_parity_sql.zip
SHA-256: e5a3afc0b2231c287bd2a26e376b6674d959c4b543f622c8e32a96b12f9ef4be
```

Its twelve source/SQL/test files were first extracted byte-identically. The current working tree then received bounded correctness repairs:

- Removed an unused schema definition with an unquoted reserved column name and repaired invalid nested-window SQL.
- Corrected pairwise covariance window grain.
- Replaced the diagonal/missing-weight risk diagnostic with full-covariance risk shares.
- Replaced same-date target activation and weighted-log-return accounting with strict-prior targets and drifting holdings.
- Included initial NAV in drawdown and every earned observation in CAGR.
- Stopped swallowing SQL failures, stopped filling missing covariance with zero, and made data-download failure explicit.
- Repaired ambiguous correlation-plot SQL and made the CLI print safely in non-Unicode consoles.

The archive is historical evidence, **not the current installable version**. Use the root source and do not extract the archive over it.

## Verification scope

Local verification on October 14, 2026 used Python 3.10.11 and DuckDB 1.5.4. The 18 tests comprise ten ERC mathematics tests and eight SQL/data-contract checks. The documented offline command generated 720 target-weight rows across 18 decision dates and four strategies, 359 earned returns per strategy, ten figure files and a summary CSV.

Synthetic seed 42 is deterministic, but outputs are simulated; none of their return metrics are promoted as historical evidence. The online yfinance path, alternative dependency versions, production scheduling and real-world execution were not part of that smoke test.
