"""
report.py
=========
All visualisations — data pulled directly from DuckDB SQL views.

Figures
-------
1.  Weight Allocation History  — stacked area chart, strategy comparison
2.  Risk Contribution  — how evenly risk is distributed at latest date
3.  Cumulative NAV  — all strategies vs 60/40
4.  Drawdown Series  — pulled from SQL drawdown_series view
5.  Rolling Sharpe  — pulled from SQL rolling_sharpe view
6.  Asset Correlation Matrix  — from DuckDB covariance data
7.  Annual Returns Heatmap  — calendar-year returns by strategy
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from riskparity.db import DB

logger = logging.getLogger(__name__)

STRATEGY_COLORS = {
    "risk_parity":  "#1a1a2e",
    "inv_vol":      "#2166ac",
    "equal_weight": "#d6604d",
    "60_40":        "#888888",
}
STRATEGY_LABELS = {
    "risk_parity":  "Risk Parity (ERC)",
    "inv_vol":      "Inverse Volatility",
    "equal_weight": "Equal Weight",
    "60_40":        "60/40 Benchmark",
}
ASSET_COLORS = {
    "SPY": "#c0392b", "EFA": "#e74c3c", "EEM": "#e67e22",
    "TLT": "#2980b9", "EMB": "#1abc9c", "TIP": "#16a085",
    "DBC": "#8e44ad", "GLD": "#f39c12",
    "VNQ": "#27ae60", "SHY": "#95a5a6",
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top":  False,   "axes.spines.right": False,
    "font.size": 10,             "axes.titlesize": 12,
    "axes.titleweight": "bold",  "figure.dpi": 130,
})


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    p = out / name
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", p)


# ── Figure 1: Weight allocation history ────────────────────────────────────

def fig1_weight_history(db: DB, strategy: str, out: Path) -> None:
    df = db.query("""
        SELECT date, ticker, weight
        FROM portfolio_weights
        WHERE strategy = ?
        ORDER BY date, ticker
    """.replace("?", f"'{strategy}'"))

    if df.empty:
        return

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="ticker", values="weight").fillna(0)

    fig, ax = plt.subplots(figsize=(14, 5))
    bottom  = np.zeros(len(pivot))
    for col in pivot.columns:
        color = ASSET_COLORS.get(col, "#888888")
        ax.fill_between(pivot.index, bottom, bottom + pivot[col].values,
                        label=col, color=color, alpha=0.85)
        bottom += pivot[col].values

    ax.set_ylim(0, 1)
    ax.set_ylabel("Portfolio Weight")
    ax.set_title(f"Figure 1: Weight Allocation History — {STRATEGY_LABELS.get(strategy, strategy)}", pad=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(ncol=5, frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    _save(fig, out, f"fig1_weights_{strategy}.png")


# ── Figure 2: Risk contribution bar chart ──────────────────────────────────

def fig2_risk_contribution(db: DB, date: str, strategy: str, out: Path) -> None:
    df = db.query(f"""
        SELECT ticker, weight, risk_share
        FROM risk_contribution_diagnostic
        WHERE CAST(date AS VARCHAR) = '{date}'
          AND strategy = '{strategy}'
        ORDER BY risk_share DESC
    """)

    if df.empty:
        logger.warning("No risk contribution data for %s %s", date, strategy)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: weights
    colors = [ASSET_COLORS.get(t, "#888") for t in df["ticker"]]
    axes[0].barh(df["ticker"], df["weight"] * 100, color=colors, edgecolor="white")
    axes[0].axvline(100 / len(df), color="black", ls="--", lw=1, label="Equal weight")
    axes[0].set_xlabel("Weight (%)")
    axes[0].set_title("Capital Weight", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    # Right: risk contributions
    axes[1].barh(df["ticker"], df["risk_share"] * 100, color=colors, edgecolor="white")
    axes[1].axvline(100 / len(df), color="#c1121f", ls="--", lw=1.5, label="Equal risk target")
    axes[1].set_xlabel("Risk Contribution (%)")
    axes[1].set_title("Risk Contribution", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Figure 2: Capital vs Risk Contribution — {STRATEGY_LABELS.get(strategy, strategy)}\n"
        f"Date: {date}  |  ERC target: {100/len(df):.1f}% per asset",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, out, f"fig2_risk_contribution_{strategy}.png")


# ── Figure 3: Cumulative NAV comparison ────────────────────────────────────

def fig3_cumulative_nav(db: DB, out: Path) -> None:
    df = db.query("""
        SELECT date, strategy, nav
        FROM portfolio_nav_series
        ORDER BY date
    """)

    if df.empty:
        return

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="strategy", values="nav")

    fig, ax = plt.subplots(figsize=(14, 6))
    for strat in pivot.columns:
        color = STRATEGY_COLORS.get(strat, "#888")
        label = STRATEGY_LABELS.get(strat, strat)
        lw    = 2.8 if strat == "risk_parity" else 1.8
        ls    = "-" if strat == "risk_parity" else ("--" if strat == "60_40" else "-")
        ax.plot(pivot.index, pivot[strat], color=color, lw=lw, ls=ls, label=label)

    ax.set_ylabel("Growth of $1")
    ax.set_title("Figure 3: Cumulative NAV — Risk Parity vs Alternatives", pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save(fig, out, "fig3_cumulative_nav.png")


# ── Figure 4: Drawdown series ───────────────────────────────────────────────

def fig4_drawdown(db: DB, out: Path) -> None:
    df = db.query("""
        SELECT date, strategy, drawdown
        FROM drawdown_series
        ORDER BY date
    """)

    if df.empty:
        return

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="strategy", values="drawdown") * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    for strat in pivot.columns:
        color = STRATEGY_COLORS.get(strat, "#888")
        label = STRATEGY_LABELS.get(strat, strat)
        lw    = 2.5 if strat == "risk_parity" else 1.5
        ax.plot(pivot.index, pivot[strat], color=color, lw=lw, label=label)

    ax.fill_between(
        pivot.index,
        pivot.get("risk_parity", pd.Series(0, index=pivot.index)),
        0, alpha=0.15, color=STRATEGY_COLORS["risk_parity"],
    )
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Figure 4: Drawdown — Risk Parity vs Alternatives", pad=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, out, "fig4_drawdown.png")


# ── Figure 5: Rolling Sharpe ────────────────────────────────────────────────

def fig5_rolling_sharpe(db: DB, out: Path) -> None:
    df = db.query("""
        SELECT date, strategy, rolling_sharpe_252d
        FROM rolling_sharpe
        WHERE rolling_sharpe_252d IS NOT NULL
        ORDER BY date
    """)

    if df.empty:
        return

    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="strategy", values="rolling_sharpe_252d")

    fig, ax = plt.subplots(figsize=(14, 5))
    for strat in pivot.columns:
        color = STRATEGY_COLORS.get(strat, "#888")
        label = STRATEGY_LABELS.get(strat, strat)
        lw    = 2.5 if strat == "risk_parity" else 1.5
        ax.plot(pivot.index, pivot[strat], color=color, lw=lw, label=label)

    ax.axhline(0,   color="black",   lw=0.7, ls="--")
    ax.axhline(0.5, color="#276749", lw=1.0, ls=":",  alpha=0.7, label="Sharpe = 0.5")
    ax.set_ylabel("Rolling Sharpe (252-day)")
    ax.set_title("Figure 5: Rolling 12-Month Sharpe Ratio", pad=10)
    ax.legend(frameon=False, fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    _save(fig, out, "fig5_rolling_sharpe.png")


# ── Figure 6: Asset correlation matrix ─────────────────────────────────────

def fig6_correlation(db: DB, out: Path, date: str) -> None:
    df = db.query(f"""
        SELECT c.ticker_i, c.ticker_j,
               c.cov_60d_ann / NULLIF(
                   SQRT(c1.cov_60d_ann * c2.cov_60d_ann), 0
               ) AS corr
        FROM rolling_cov_60d c
        JOIN rolling_cov_60d c1
            ON c.date = c1.date AND c.ticker_i = c1.ticker_i AND c1.ticker_i = c1.ticker_j
        JOIN rolling_cov_60d c2
            ON c.date = c2.date AND c.ticker_j = c2.ticker_i AND c2.ticker_i = c2.ticker_j
        WHERE CAST(c.date AS VARCHAR) = '{date}'
    """)

    if df.empty:
        logger.warning("No correlation data for %s", date)
        return

    pivot = df.pivot(index="ticker_i", columns="ticker_j", values="corr").fillna(0)
    np.fill_diagonal(pivot.values, 1.0)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(pivot.values, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1)
    tickers = pivot.index.tolist()
    ax.set_xticks(range(len(tickers))); ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_yticks(range(len(tickers))); ax.set_yticklabels(tickers)

    for i in range(len(tickers)):
        for j in range(len(tickers)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(val) > 0.5 else "black")

    plt.colorbar(im, ax=ax, fraction=0.046, label="Correlation")
    ax.set_title(f"Figure 6: Asset Correlation Matrix  ({date})\n"
                 "Negative correlations between equities and bonds are risk parity's engine", pad=10)
    fig.tight_layout()
    _save(fig, out, "fig6_correlation.png")


# ── Figure 7: Annual returns heatmap ────────────────────────────────────────

def fig7_annual_returns(db: DB, out: Path) -> None:
    df = db.query("""
        SELECT
            YEAR(CAST(date AS DATE))   AS year,
            strategy,
            EXP(SUM(port_log_ret)) - 1 AS annual_ret
        FROM portfolio_daily_returns
        GROUP BY YEAR(CAST(date AS DATE)), strategy
        ORDER BY year, strategy
    """)

    if df.empty:
        return

    pivot = df.pivot(index="year", columns="strategy", values="annual_ret") * 100
    strategies = [s for s in STRATEGY_LABELS if s in pivot.columns]
    pivot = pivot[strategies]

    fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 2), max(6, len(pivot) * 0.55 + 2)))
    im = ax.imshow(pivot.values, cmap=plt.cm.RdYlGn, vmin=-25, vmax=25, aspect="auto")

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies], rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int))

    for i in range(len(pivot.index)):
        for j in range(len(strategies)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if abs(val) > 15 else "black")

    plt.colorbar(im, ax=ax, fraction=0.03, label="Annual Return (%)")
    ax.set_title("Figure 7: Annual Returns Heatmap — Strategy Comparison\n"
                 "(Green = positive, Red = negative)", pad=10)
    fig.tight_layout()
    _save(fig, out, "fig7_annual_returns.png")


# ── Master report ────────────────────────────────────────────────────────────

def generate_report(
    db:       DB,
    out_dir:  Path,
    latest_date: str,
    strategy: str = "risk_parity",
) -> pd.DataFrame:
    """
    Generate all figures and return the performance summary table.
    All data is queried from DuckDB — no pandas computation here.
    """
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    for strat in ["risk_parity", "inv_vol", "equal_weight", "60_40"]:
        fig1_weight_history(db, strat, fig_dir)

    fig2_risk_contribution(db, latest_date, strategy, fig_dir)
    fig3_cumulative_nav(db, fig_dir)
    fig4_drawdown(db, fig_dir)
    fig5_rolling_sharpe(db, fig_dir)
    fig6_correlation(db, fig_dir, latest_date)
    fig7_annual_returns(db, fig_dir)

    # Performance summary from SQL
    perf = db.query("SELECT * FROM performance_summary")
    if not perf.empty:
        perf.to_csv(tbl_dir / "performance_summary.csv", index=False)
        logger.info("\n%s", perf.to_string(index=False))

    return perf
