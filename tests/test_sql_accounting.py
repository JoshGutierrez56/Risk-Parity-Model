"""Hand-derived accounting fixtures, independent of the portfolio SQL formulas."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb
import numpy as np
import pandas as pd
import pytest

from riskparity.db import DB
from riskparity.loader import download_and_load, load_prices
from riskparity.optimizer import RiskParityOptimizer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def db():
    connection = DB(":memory:")
    connection.initialise(str(ROOT / "sql/schema.sql"))
    yield connection
    connection.close()


def load_sql(db, *names):
    db.load_views(*(str(ROOT / "sql" / name) for name in names))


def test_all_sql_files_bind_and_bad_sql_fails(db, tmp_path):
    load_sql(db, "returns.sql", "covariance.sql", "backtest.sql")
    bad = tmp_path / "bad.sql"
    bad.write_text("SELECT non_existent_column", encoding="utf-8")
    with pytest.raises(duckdb.Error):
        db.execute_file(str(bad))


def test_rolling_covariance_uses_sixty_observations_per_pair(db):
    dates = pd.bdate_range("2020-01-01", periods=66)
    x = np.arange(65, dtype=float) / 10000 - 0.003
    y = (np.arange(65) % 7).astype(float) / 1000 - 0.002
    rows = [{"date": d.date(), "ticker": t, "ret": float(r)}
            for t, series in (("A", x), ("B", y))
            for d, r in zip(dates[1:], series)]
    db.load_df(pd.DataFrame(rows), "returns")
    load_sql(db, "covariance.sql")
    frame = db.query("SELECT * FROM rolling_cov_60d WHERE date = (SELECT MAX(date) FROM returns)")
    actual = frame.pivot(index="ticker_i", columns="ticker_j", values="cov_60d_ann").values
    expected = np.cov(np.stack([x[-60:], y[-60:]]), bias=True) * 252
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert frame["n_obs"].tolist() == [60] * 4
    assert db.query("SELECT MIN(date) AS d FROM rolling_cov_60d").iloc[0, 0] == dates[30]


def test_decision_lag_drifting_holdings_and_compounding(db):
    dates = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"SPY": [100, 110, 121, 60.5, 121],
                           "TLT": [100, 100, 100, 100, 100]}, index=dates)
    load_prices(db, prices, ["SPY", "TLT"])
    load_sql(db, "returns.sql")
    # Jan 1: buy half of each. Jan 3 close: move entirely into SPY.
    # Dollar holdings after Jan 2 are 55/50, NOT reset to 50/50.
    rows = [(dates[0].date(), "SPY", "test", 0.5),
            (dates[0].date(), "TLT", "test", 0.5),
            (dates[2].date(), "SPY", "test", 1.0),
            (dates[2].date(), "TLT", "test", 0.0)]
    db.conn.executemany("INSERT INTO portfolio_weights VALUES (?, ?, ?, ?)", rows)
    load_sql(db, "backtest.sql")
    actual = db.query("SELECT * FROM portfolio_nav_series ORDER BY date")
    # Portfolio values from literal share holdings: 105, 110.5, 55.25, 110.5.
    np.testing.assert_allclose(actual["nav"], [1.05, 1.105, 0.5525, 1.105])
    np.testing.assert_allclose(actual["port_simple_ret"], [0.05, 110.5 / 105 - 1, -0.5, 1.0])
    assert len(actual) == 4  # no earned return on the first decision date
    summary = db.query("SELECT * FROM performance_summary").iloc[0]
    assert summary["n_days"] == 4
    assert summary["max_drawdown"] == pytest.approx(-0.5)
    assert summary["cagr"] == pytest.approx(1.105 ** (252 / 4) - 1)


def test_first_loss_counts_against_initial_nav(db):
    dates = pd.bdate_range("2020-01-01", periods=3)
    load_prices(db, pd.DataFrame({"SPY": [100, 80, 88]}, index=dates), ["SPY"])
    load_sql(db, "returns.sql")
    db.conn.execute("INSERT INTO portfolio_weights VALUES (?, 'SPY', 'test', 1)", [dates[0].date()])
    load_sql(db, "backtest.sql")
    drawdown = db.query("SELECT drawdown FROM drawdown_series ORDER BY date")
    np.testing.assert_allclose(drawdown["drawdown"], [-0.2, -0.12])
    assert db.query("SELECT max_drawdown FROM drawdown_by_ticker").iloc[0, 0] == pytest.approx(-0.2)


def test_full_covariance_risk_shares(db):
    load_sql(db, "covariance.sql")
    db.conn.execute("DROP VIEW rolling_cov_60d")
    db.conn.execute("CREATE TABLE rolling_cov_60d(date DATE, ticker_i VARCHAR, ticker_j VARCHAR, cov_60d_ann DOUBLE)")
    db.conn.executemany("INSERT INTO rolling_cov_60d VALUES ('2020-01-01', ?, ?, ?)",
                        [("A", "A", 4), ("A", "B", 1), ("B", "A", 1), ("B", "B", 9)])
    db.conn.executemany("INSERT INTO portfolio_weights VALUES ('2020-01-01', ?, 'test', 0.5)", [("A",), ("B",)])
    frame = db.query("SELECT * FROM risk_contribution_diagnostic ORDER BY ticker")
    # Sigma*w=(2.5,5); w*(Sigma*w)=(1.25,2.5); total variance=3.75.
    np.testing.assert_allclose(frame["risk_share"], [1 / 3, 2 / 3])


def test_missing_held_return_is_not_silently_dropped(db):
    dates = pd.bdate_range("2020-01-01", periods=3)
    load_prices(db, pd.DataFrame({"SPY": [100, 101, 102], "TLT": [100, 99, 98]}, index=dates), ["SPY", "TLT"])
    load_sql(db, "returns.sql")
    db.conn.execute("INSERT INTO portfolio_weights VALUES (?, 'SPY', 'test', 1)", [dates[0].date()])
    db.conn.execute("DELETE FROM returns WHERE ticker='SPY' AND date=?", [dates[1].date()])
    load_sql(db, "backtest.sql")
    with pytest.raises(duckdb.Error, match="Missing or invalid return"):
        db.query("SELECT * FROM portfolio_daily_returns")


def test_incomplete_prices_and_covariance_are_rejected(db):
    dates = pd.bdate_range("2020-01-01", periods=2)
    with pytest.raises(ValueError, match="complete positive finite"):
        load_prices(db, pd.DataFrame({"SPY": [100, np.nan]}, index=dates), ["SPY"])
    db.conn.execute("CREATE TABLE rolling_cov_60d(date DATE, ticker_i VARCHAR, ticker_j VARCHAR, cov_60d_ann DOUBLE)")
    db.conn.execute("INSERT INTO rolling_cov_60d VALUES ('2020-01-01', 'SPY', 'SPY', 0.04)")
    with pytest.raises(ValueError, match="Incomplete"):
        RiskParityOptimizer(db, ["SPY", "TLT"]).fetch_covariance("2020-01-01")


def test_download_failure_does_not_substitute_synthetic(db, monkeypatch):
    def fail(**kwargs):
        raise ConnectionError("fixture failure")
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=fail))
    with pytest.raises(RuntimeError, match="no synthetic substitution"):
        download_and_load(db, ["SPY"])
    assert db.query("SELECT COUNT(*) AS n FROM prices").iloc[0, 0] == 0
