"""Aligned ETF price ingestion. Synthetic data is available only by explicit request."""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from riskparity.db import DB

logger = logging.getLogger(__name__)

UNIVERSE = {
    "SPY": ("US Equity", "Equity", "US"),
    "EFA": ("Intl Equity", "Equity", "DM Ex-US"),
    "EEM": ("EM Equity", "Equity", "EM"),
    "TLT": ("LT Treasury", "FixedIncome", "US"),
    "EMB": ("EM Bonds", "FixedIncome", "EM"),
    "DBC": ("Broad Commodity", "Commodity", "Global"),
    "GLD": ("Gold", "Commodity", "Global"),
    "VNQ": ("US REIT", "RealEstate", "US"),
    "TIP": ("TIPS", "FixedIncome", "US"),
    "SHY": ("Short Treasury", "FixedIncome", "US"),
}


def load_prices(db: DB, prices: pd.DataFrame, tickers: List[str]) -> None:
    """Validate a complete price panel, then atomically replace generated inputs.

    No forward-fill or partial-universe substitution. Existing generated returns
    and weights are removed so reruns cannot mix different samples/data modes.
    """
    if prices.empty or set(prices.columns) != set(tickers):
        raise ValueError("Prices must include exactly the requested tickers")
    prices = prices.loc[:, tickers].copy().sort_index()
    prices.index = pd.DatetimeIndex(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index = prices.index.normalize()
    if prices.index.has_duplicates or prices.index.isna().any():
        raise ValueError("Prices must have unique valid dates")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Prices must form a complete positive finite panel; choose a common coverage period")
    prices.index.name = "Date"
    prices.columns.name = None
    long = (prices.reset_index().melt(id_vars=["Date"], var_name="ticker", value_name="close")
            .rename(columns={"Date": "date"})
            .assign(date=lambda frame: frame["date"].dt.date))
    assets = pd.DataFrame([
        {"ticker": t, "name": UNIVERSE[t][0], "asset_class": UNIVERSE[t][1], "region": UNIVERSE[t][2]}
        for t in tickers
    ])
    db.conn.execute("BEGIN TRANSACTION")
    try:
        for table in ("portfolio_weights", "returns", "prices", "assets"):
            db.conn.execute(f"DELETE FROM {table}")
        db.load_df(long[["date", "ticker", "close"]], "prices")
        db.load_df(assets, "assets")
        db.conn.execute("COMMIT")
    except Exception:
        db.conn.execute("ROLLBACK")
        raise


def download_and_load(
    db: DB,
    tickers: Optional[List[str]] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> None:
    """Download adjusted closes for the requested period, or fail explicitly.

    No stale-cache reuse and no silent synthetic fallback. Pass --synthetic at
    the CLI when deliberately testing offline.
    """
    tickers = list(UNIVERSE) if tickers is None else tickers
    import yfinance as yf
    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                          progress=False, threads=True)
        prices = raw["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        load_prices(db, prices, tickers)
    except Exception as exc:
        raise RuntimeError(
            "Market-data download or validation failed; no synthetic substitution was made. "
            "Use --synthetic explicitly for the offline demonstration."
        ) from exc


def _synthetic_prices(
    tickers: List[str],
    start:   str,
    end:     Optional[str],
) -> pd.DataFrame:
    """Generate correlated synthetic prices for offline testing."""
    import numpy as np
    rng   = np.random.default_rng(42)
    dates = pd.bdate_range(start=start, end=end or "2024-01-01")
    n, k  = len(dates), len(tickers)

    # Rough asset-class correlation block structure
    asset_classes = [UNIVERSE[t][1] for t in tickers]
    corr = np.eye(k)
    for i in range(k):
        for j in range(k):
            if i != j:
                if asset_classes[i] == asset_classes[j]:
                    corr[i, j] = 0.70
                elif set([asset_classes[i], asset_classes[j]]) == {"Equity", "RealEstate"}:
                    corr[i, j] = 0.55
                elif "FixedIncome" in [asset_classes[i], asset_classes[j]]:
                    corr[i, j] = -0.20
                else:
                    corr[i, j] = 0.10

    # Annual vols roughly matching real assets
    vols = {"Equity": 0.18, "FixedIncome": 0.06, "Commodity": 0.22,
            "RealEstate": 0.20}
    sigma = np.array([vols.get(ac, 0.15) for ac in asset_classes]) / np.sqrt(252)

    L   = np.linalg.cholesky(np.clip(corr, -1, 1))
    ret = rng.normal(0, 1, (n, k)) @ L.T * sigma + 0.0004
    prices = pd.DataFrame(
        100.0 * np.cumprod(1 + ret, axis=0),
        index=dates,
        columns=tickers,
    )
    prices.index.name = "Date"
    return prices
