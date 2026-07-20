"""
optimizer.py
============
Equal Risk Contribution (ERC) optimizer — the Python layer.

Why Python and not SQL?
-----------------------
SQL window functions handle everything that is a closed-form expression:
returns, rolling covariance, inverse-volatility weights, NAV, Sharpe,
drawdown. ERC is different: it requires iterative numerical optimisation
because the risk contribution is a nonlinear function of weights.

    Risk contribution of asset i:
        RC_i = w_i × (Σw)_i / (wᵀΣw)

    ERC constraint: RC_i = RC_j  ∀ i,j  →  RC_i = 1/N  ∀ i

This cannot be solved in closed form for N > 2.

We solve it here with scipy SLSQP, feeding data fetched from DuckDB.
The weights are then written back to the portfolio_weights table so
all downstream analytics (NAV, Sharpe, drawdown) run in pure SQL.

Reference: Maillard, Roncalli & Teïletche (2010)
           "The Properties of Equally Weighted Risk Contribution Portfolios"
           Journal of Portfolio Management.

Edward Qian (PanAgora CIO) introduced risk parity concepts in 2004;
this implementation follows the formal ERC framework from the 2010 paper.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from riskparity.db import DB

logger = logging.getLogger(__name__)


# ── Risk contribution functions ─────────────────────────────────────────────

def portfolio_variance(w: np.ndarray, Sigma: np.ndarray) -> float:
    return float(w @ Sigma @ w)


def risk_contributions(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """RC_i = w_i × (Σw)_i / (wᵀΣw)"""
    port_var = portfolio_variance(w, Sigma)
    if port_var <= 0:
        return np.ones(len(w)) / len(w)
    marginal = Sigma @ w
    return w * marginal / port_var


def erc_objective(w: np.ndarray, Sigma: np.ndarray) -> float:
    """
    Minimise sum of squared deviations from equal risk contribution.
    At the optimum, all RC_i = 1/N.
    """
    rc  = risk_contributions(w, Sigma)
    n   = len(w)
    return float(np.sum((rc - 1.0 / n) ** 2))


def erc_gradient(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Numerical gradient (scipy can compute analytically if needed)."""
    eps = 1e-6
    g   = np.zeros(len(w))
    f0  = erc_objective(w, Sigma)
    for i in range(len(w)):
        w_h    = w.copy()
        w_h[i] += eps
        g[i]   = (erc_objective(w_h, Sigma) - f0) / eps
    return g


# ── ERC solver ──────────────────────────────────────────────────────────────

def solve_erc(
    Sigma:   np.ndarray,
    w0:      Optional[np.ndarray] = None,
    tol:     float = 1e-10,
    maxiter: int   = 1000,
) -> Tuple[np.ndarray, bool]:
    """
    Solve the Equal Risk Contribution problem.

    Parameters
    ----------
    Sigma   : (N × N) covariance matrix
    w0      : initial weights (inverse-vol if None)
    tol     : solver tolerance
    maxiter : max iterations

    Returns
    -------
    weights : np.ndarray  — ERC weights (sum to 1, all ≥ 0)
    success : bool
    """
    n = Sigma.shape[0]

    # Warm-start: inverse-vol weights
    if w0 is None:
        vols = np.sqrt(np.diag(Sigma))
        vols = np.where(vols > 0, vols, vols.mean())
        w0   = (1.0 / vols) / (1.0 / vols).sum()

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds      = [(1e-6, 1.0)] * n   # all weights strictly positive

    result = minimize(
        fun=erc_objective,
        x0=w0,
        args=(Sigma,),
        jac=erc_gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": maxiter},
    )

    w_opt = np.clip(result.x, 0, 1)
    w_opt /= w_opt.sum()
    return w_opt, result.success


# ── Walk-forward weight generation ──────────────────────────────────────────

class RiskParityOptimizer:
    """
    Walk-forward ERC weight generation, reading covariance from DuckDB
    and writing weights back so all downstream SQL views can use them.

    Parameters
    ----------
    db      : active DB connection
    tickers : asset universe (must match DuckDB contents)
    """

    def __init__(self, db: DB, tickers: List[str]) -> None:
        self.db      = db
        self.tickers = tickers

    def fetch_covariance(self, date: str) -> Optional[pd.DataFrame]:
        """
        Pull the 60-day rolling covariance matrix for a single date from DuckDB.
        Returns None if insufficient data.
        """
        sql = """
            SELECT ticker_i, ticker_j, cov_60d_ann
            FROM rolling_cov_60d
            WHERE date = ?
              AND ticker_i IN ({placeholders})
              AND ticker_j IN ({placeholders})
        """.format(placeholders=",".join(["?"] * len(self.tickers)))

        params = [date] + self.tickers + self.tickers
        try:
            df = self.db.conn.execute(sql, params).df()
        except Exception as exc:
            logger.warning("Covariance fetch failed for %s: %s", date, exc)
            return None

        if df.empty:
            return None

        pivot = df.pivot(index="ticker_i", columns="ticker_j", values="cov_60d_ann")
        pivot = pivot.reindex(index=self.tickers, columns=self.tickers)

        # Fill missing with zero (uncorrelated assumption)
        pivot = pivot.fillna(0.0)

        # Ensure positive semi-definite via diagonal bump
        min_eig = np.linalg.eigvalsh(pivot.values).min()
        if min_eig < 0:
            pivot += pd.DataFrame(
                np.eye(len(self.tickers)) * (-min_eig + 1e-8),
                index=self.tickers, columns=self.tickers,
            )
        return pivot

    def run(
        self,
        rebal_dates: List[str],
        strategies:  List[str] = ("risk_parity", "inv_vol", "equal_weight"),
    ) -> pd.DataFrame:
        """
        Compute weights for all strategies at each rebalancing date.

        Writes results into portfolio_weights table in DuckDB.

        Returns
        -------
        pd.DataFrame  — all weights for inspection
        """
        all_rows = []
        n = len(self.tickers)

        for date in rebal_dates:
            cov_df = self.fetch_covariance(date)

            if cov_df is None:
                logger.warning("No covariance at %s — skipping", date)
                continue

            Sigma = cov_df.values
            vols  = np.sqrt(np.diag(Sigma))
            vols  = np.where(vols > 0, vols, 1e-4)

            weights_by_strategy: Dict[str, np.ndarray] = {}

            # ── Equal Risk Contribution ─────────────────────────────
            if "risk_parity" in strategies:
                w_erc, success = solve_erc(Sigma)
                if not success:
                    logger.warning("ERC solver did not converge at %s", date)
                weights_by_strategy["risk_parity"] = w_erc

            # ── Inverse-volatility ─────────────────────────────────
            if "inv_vol" in strategies:
                w_iv = (1.0 / vols) / (1.0 / vols).sum()
                weights_by_strategy["inv_vol"] = w_iv

            # ── Equal weight ───────────────────────────────────────
            if "equal_weight" in strategies:
                weights_by_strategy["equal_weight"] = np.ones(n) / n

            # ── 60-40 proxy (SPY + TLT if in universe) ────────────
            if "60_40" in strategies:
                w_6040 = np.zeros(n)
                ticker_arr = np.array(self.tickers)
                spy_idx = np.where(ticker_arr == "SPY")[0]
                tlt_idx = np.where(ticker_arr == "TLT")[0]
                if len(spy_idx) and len(tlt_idx):
                    w_6040[spy_idx[0]] = 0.60
                    w_6040[tlt_idx[0]] = 0.40
                else:
                    w_6040 = np.ones(n) / n
                weights_by_strategy["60_40"] = w_6040

            # ── Write rows ─────────────────────────────────────────
            for strategy, w_arr in weights_by_strategy.items():
                for ticker, weight in zip(self.tickers, w_arr):
                    all_rows.append({
                        "date":     date,
                        "ticker":   ticker,
                        "strategy": strategy,
                        "weight":   float(weight),
                    })

        weights_df = pd.DataFrame(all_rows)
        if not weights_df.empty:
            self.db.load_df(weights_df, "portfolio_weights")
            logger.info(
                "Wrote %d weight rows (%d dates × %d strategies)",
                len(weights_df),
                len(rebal_dates),
                len(strategies),
            )
        return weights_df

    def risk_contribution_table(self, date: str, strategy: str = "risk_parity") -> pd.DataFrame:
        """
        Compute and display risk contributions for a single date.
        Calls DuckDB's risk_contribution_diagnostic view.
        """
        sql = """
            SELECT ticker, weight, risk_share_diag
            FROM risk_contribution_diagnostic
            WHERE date = ? AND strategy = ?
            ORDER BY risk_share_diag DESC
        """
        try:
            df = self.db.conn.execute(sql, [date, strategy]).df()
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            # Fallback: compute from weights in memory
            w_sql = f"""
                SELECT ticker, weight FROM portfolio_weights
                WHERE date = ? AND strategy = ?
            """
            w_df = self.db.conn.execute(w_sql, [date, strategy]).df()
            cov  = self.fetch_covariance(date)
            if cov is None or w_df.empty:
                return pd.DataFrame()
            w = w_df.set_index("ticker").reindex(self.tickers)["weight"].fillna(0).values
            rc  = risk_contributions(w, cov.values)
            df  = pd.DataFrame({"ticker": self.tickers, "weight": w, "risk_share": rc})

        return df
