"""
tests/test_erc_math.py
======================
Unit tests for the ERC optimiser math — no DuckDB required.

These validate the core risk parity mathematics:
  1. Risk contributions sum to 1
  2. ERC solution achieves equal risk contribution (RC_i ≈ 1/N)
  3. Inverse-volatility weights are proportional to 1/σ
  4. Portfolio variance is positive semi-definite
  5. ERC weights > 0 (all assets held)
"""
import sys
sys.path.insert(0, "src")

import numpy as np
import pytest

from riskparity.optimizer import (
    erc_objective,
    portfolio_variance,
    risk_contributions,
    solve_erc,
)


@pytest.fixture
def diagonal_cov():
    """3-asset diagonal covariance: vols = [10%, 20%, 30%] annualised."""
    vols = np.array([0.10, 0.20, 0.30])
    return np.diag(vols ** 2)


@pytest.fixture
def correlated_cov():
    """4-asset covariance with realistic cross-asset correlations."""
    vols = np.array([0.18, 0.06, 0.22, 0.15])   # EQ, Bond, Commodity, REIT
    corr = np.array([
        [1.00, -0.30,  0.15,  0.60],
        [-0.30,  1.00, -0.10, -0.20],
        [0.15, -0.10,  1.00,  0.10],
        [0.60, -0.20,  0.10,  1.00],
    ])
    sigma = np.diag(vols) @ corr @ np.diag(vols)
    return sigma


# ── Sanity checks ────────────────────────────────────────────────────────────

def test_risk_contributions_sum_to_one(diagonal_cov):
    w = np.array([1/3, 1/3, 1/3])
    rc = risk_contributions(w, diagonal_cov)
    assert abs(rc.sum() - 1.0) < 1e-10, f"RC sum = {rc.sum()}, expected 1.0"


def test_portfolio_variance_positive(diagonal_cov):
    w = np.array([0.5, 0.3, 0.2])
    pv = portfolio_variance(w, diagonal_cov)
    assert pv > 0


def test_erc_objective_zero_at_optimum(diagonal_cov):
    """
    For a diagonal covariance, inv-vol weights ARE the ERC solution.
    The objective should be zero (all RC_i = 1/N exactly).
    """
    vols = np.sqrt(np.diag(diagonal_cov))
    w_erc = (1 / vols) / (1 / vols).sum()
    obj = erc_objective(w_erc, diagonal_cov)
    assert obj < 1e-10, f"ERC objective at inv-vol = {obj}, expected ~0"


# ── ERC solver tests ─────────────────────────────────────────────────────────

def test_erc_weights_sum_to_one(diagonal_cov):
    w, _ = solve_erc(diagonal_cov)
    assert abs(w.sum() - 1.0) < 1e-6


def test_erc_weights_all_positive(diagonal_cov):
    w, _ = solve_erc(diagonal_cov)
    assert np.all(w > 0), f"Negative ERC weights: {w}"


def test_erc_achieves_equal_risk_diagonal(diagonal_cov):
    """On diagonal Σ, ERC = inverse-vol, so RC_i = 1/N exactly."""
    w, success = solve_erc(diagonal_cov)
    assert success, "ERC solver did not converge"
    rc = risk_contributions(w, diagonal_cov)
    n  = len(w)
    np.testing.assert_allclose(rc, np.ones(n) / n, atol=1e-4,
                               err_msg=f"Risk contributions not equal: {rc}")


def test_erc_achieves_equal_risk_correlated(correlated_cov):
    """On correlated Σ, ERC still achieves RC_i ≈ 1/N (within tolerance)."""
    w, success = solve_erc(correlated_cov)
    assert success, "ERC solver did not converge on correlated matrix"
    rc = risk_contributions(w, correlated_cov)
    n  = len(w)
    # Stricter check: each RC within 0.5% of target
    np.testing.assert_allclose(rc, np.ones(n) / n, atol=0.005,
                               err_msg=f"Risk contributions not equal: {rc}")


def test_erc_vs_equal_weight_different_vols(diagonal_cov):
    """
    ERC should give more weight to low-vol assets vs equal weight.
    With vols [10%, 20%, 30%], ERC should have w[0] > w[1] > w[2].
    """
    w, _ = solve_erc(diagonal_cov)
    assert w[0] > w[1] > w[2], \
        f"ERC should prefer low-vol assets; got {w}"


def test_erc_vs_inv_vol_close_on_diagonal(diagonal_cov):
    """ERC ≈ inv-vol when correlations are all zero (diagonal Σ)."""
    vols  = np.sqrt(np.diag(diagonal_cov))
    w_inv = (1 / vols) / (1 / vols).sum()
    w_erc, _ = solve_erc(diagonal_cov)
    np.testing.assert_allclose(w_erc, w_inv, atol=1e-4,
                               err_msg="ERC should equal inv-vol on diagonal covariance")


def test_erc_correlated_differs_from_inv_vol(correlated_cov):
    """
    With correlations, ERC ≠ inv-vol. This is the whole point of ERC
    over the simpler inverse-volatility heuristic.
    """
    vols  = np.sqrt(np.diag(correlated_cov))
    w_inv = (1 / vols) / (1 / vols).sum()
    w_erc, _ = solve_erc(correlated_cov)
    diff = np.abs(w_erc - w_inv).max()
    assert diff > 0.01, \
        f"ERC and inv-vol should differ on correlated Σ; max diff = {diff}"
