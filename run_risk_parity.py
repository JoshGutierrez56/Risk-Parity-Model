#!/usr/bin/env python3
"""
run_risk_parity.py
==================
Risk Parity Pipeline — DuckDB + Python hybrid.

Architecture
------------
  yfinance → DuckDB (prices, returns)
    SQL: rolling covariance, inverse-vol weights, inv-vol NAV, Sharpe, drawdown
  DuckDB → Python ERC optimizer (scipy SLSQP)
    Python: solve RC_i = 1/N iteratively, write weights back to DuckDB
  DuckDB: portfolio_weights → daily NAV → all performance metrics in SQL
  DuckDB → matplotlib: all 7 figures query SQL views directly

Usage
-----
    python run_risk_parity.py              # downloads via yfinance
    python run_risk_parity.py --synthetic  # fully offline, no internet
    python run_risk_parity.py --scorecard  # print metrics table and exit
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("risk_parity")
OUT_DIR  = Path("outputs")
SQL_DIR  = Path("sql")


def main(args: argparse.Namespace) -> None:
    sys.path.insert(0, "src")
    from riskparity.db        import DB
    from riskparity.loader    import download_and_load, load_prices, UNIVERSE
    from riskparity.optimizer import RiskParityOptimizer
    from riskparity.report    import generate_report

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 62)
    logger.info("  Risk Parity Analytics — DuckDB + Python")
    logger.info("  Equal-risk-contribution research prototype")
    logger.info("=" * 62)

    # ── 1. Initialise DuckDB ─────────────────────────────────────────
    db = DB("data/risk_parity.duckdb")
    db.initialise(str(SQL_DIR / "schema.sql"))

    # ── 2. Load price data ───────────────────────────────────────────
    tickers = list(UNIVERSE.keys())

    if args.synthetic:
        from riskparity.loader import _synthetic_prices
        logger.warning("Synthetic mode — generating correlated price data")
        prices = _synthetic_prices(tickers, args.start, args.end)
        load_prices(db, prices, tickers)
    else:
        download_and_load(db, tickers=tickers, start=args.start, end=args.end)

    # ── 3. Compute returns and covariance in SQL ─────────────────────
    logger.info("Computing returns and rolling covariance in SQL...")
    db.load_views(
        str(SQL_DIR / "returns.sql"),
        str(SQL_DIR / "covariance.sql"),
    )

    # Verify
    n_ret = db.query("SELECT COUNT(*) AS n FROM returns").iloc[0, 0]
    logger.info("Returns computed: %d rows in DuckDB", n_ret)

    # ── 4. Rebalancing schedule (monthly) ────────────────────────────
    dates_df = db.query("""
        SELECT DISTINCT CAST(date AS VARCHAR) AS date
        FROM rolling_cov_60d
        WHERE ticker_i = ticker_j
        ORDER BY date
    """)

    all_dates   = dates_df["date"].tolist()
    rebal_dates = all_dates[::21]   # approximately monthly (every 21 trading days)
    logger.info("Rebalancing dates: %d", len(rebal_dates))

    if not rebal_dates:
        logger.error("No rebalancing dates available — check data coverage")
        db.close()
        sys.exit(1)

    # ── 5. Run ERC optimizer (Python), write weights back to DuckDB ──
    logger.info("Running ERC optimizer at each rebalancing date...")
    optimizer = RiskParityOptimizer(db, tickers)
    strategies = ["risk_parity", "inv_vol", "equal_weight", "60_40"]
    weights_df = optimizer.run(rebal_dates, strategies=strategies)

    logger.info("Portfolio weights written to DuckDB: %d rows", len(weights_df))

    # ── 6. Compute portfolio NAV and metrics in SQL ───────────────────
    logger.info("Computing portfolio NAV and metrics in SQL...")
    db.load_views(str(SQL_DIR / "backtest.sql"))

    # ── 7. Print performance summary (from SQL) ───────────────────────
    perf = db.query("SELECT * FROM performance_summary")

    if args.scorecard:
        if not perf.empty:
            print("\nRisk Parity Performance Summary ")
            print(perf.to_string(index=False))
            print()
        db.close()
        return

    if not perf.empty:
        print("\nPerformance Summary (from DuckDB SQL) ")
        print(perf.to_string(index=False))
        print()

    # Latest risk contribution
    latest_date = rebal_dates[-1]
    rc = optimizer.risk_contribution_table(latest_date, "risk_parity")
    if not rc.empty:
        print(f"Risk Contributions at {latest_date} ")
        print(rc.to_string(index=False))
        print()

    # ── 8. Generate all figures ───────────────────────────────────────
    logger.info("Generating figures from DuckDB views...")
    generate_report(db, OUT_DIR, latest_date, strategy="risk_parity")

    db.close()

    logger.info("=" * 62)
    logger.info("  Complete.  Outputs: %s/", OUT_DIR)
    logger.info("  Database:   data/risk_parity.duckdb")
    logger.info("=" * 62)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Risk Parity — DuckDB + Python ERC")
    p.add_argument("--start",     default="2010-01-01")
    p.add_argument("--end",       default="2024-01-01")
    p.add_argument("--synthetic", action="store_true",
                   help="Generate synthetic prices (no internet required)")
    p.add_argument("--scorecard", action="store_true",
                   help="Print performance table and exit (no figures)")
    main(p.parse_args())
