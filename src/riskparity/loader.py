"""
loader.py
=========
Download multi-asset price data via yfinance and load into DuckDB.

Universe
--------
We use a diversified multi-asset ETF universe — the natural habitat
for risk parity strategies, which allocate across asset classes by
risk contribution rather than dollar weight.

    US Equities:   SPY  (S&P 500)
    Int'l Equity:  EFA  (MSCI EAFE)
    EM Equity:     EEM  (MSCI Emerging Markets)
    US Bonds:      TLT  (20+ Year Treasury)
    EM Bonds:      EMB  (EM USD Bonds)
    Commodities:   DBC  (Broad Commodity)
    Gold:          GLD
    REITs:         VNQ  (US Real Estate)
    TIPS:          TIP  (Inflation-Linked Bonds)
    Cash proxy:    SHY  (1-3 Year Treasury)

This 10-asset universe spans the four main risk factors:
    Growth, Rates, Inflation, Liquidity
— the same decomposition Bridgewater uses for All Weather.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from riskparity.db import DB

logger = logging.getLogger(__name__)

UNIVERSE = {
    "SPY": ("US Equity",      "Equity",       "US"),
    "EFA": ("Intl Equity",    "Equity",       "DM Ex-US"),
    "EEM": ("EM Equity",      "Equity",       "EM"),
    "TLT": ("LT Treasury",    "FixedIncome",  "US"),
    "EMB": ("EM Bonds",       "FixedIncome",  "EM"),
    "DBC": ("Broad Commodity","Commodity",    "Global"),
    "GLD": ("Gold",           "Commodity",    "Global"),
    "VNQ": ("US REIT",        "RealEstate",   "US"),
    "TIP": ("TIPS",           "FixedIncome",  "US"),
    "SHY": ("Cash Proxy",     "FixedIncome",  "US"),
}


def download_and_load(
    db:      DB,
    tickers: Optional[List[str]] = None,
    start:   str = "2010-01-01",
    end:     Optional[str] = None,
) -> None:
    """
    Download adjusted close prices and load into DuckDB.

    Populates:  prices, assets, returns (via SQL views)

    Parameters
    ----------
    db      : active DB connection
    tickers : subset of UNIVERSE keys; all if None
    start   : start date
    end     : end date (today if None)
    """
    if tickers is None:
        tickers = list(UNIVERSE.keys())

    # ── Download prices ────────────────────────────────────────────────
    cache = Path("data/raw/prices.parquet")
    if cache.exists():
        prices_wide = pd.read_parquet(cache)
        logger.info("Prices loaded from cache (%d rows)", len(prices_wide))
    else:
        try:
            import yfinance as yf
            logger.info("Downloading %d tickers (%s → %s)...", len(tickers), start, end or "today")
            raw = yf.download(tickers, start=start, end=end,
                              auto_adjust=True, progress=True, threads=True)
            if isinstance(raw.columns, pd.MultiIndex):
                prices_wide = raw["Close"]
            else:
                prices_wide = raw
            prices_wide.index = pd.DatetimeIndex(prices_wide.index)
            if hasattr(prices_wide.index, "tz") and prices_wide.index.tz:
                prices_wide.index = prices_wide.index.tz_localize(None)
            cache.parent.mkdir(parents=True, exist_ok=True)
            prices_wide.to_parquet(cache)
        except Exception as exc:
            logger.error("yfinance download failed: %s — using synthetic prices", exc)
            prices_wide = _synthetic_prices(tickers, start, end)

    # ── Melt to long format and load ──────────────────────────────────
    prices_long = (prices_wide
                   .reset_index()
                   .melt(id_vars=["Date"], var_name="ticker", value_name="close")
                   .rename(columns={"Date": "date"})
                   .dropna(subset=["close"])
                   .assign(date=lambda df: pd.to_datetime(df["date"]).dt.date))

    prices_long = prices_long[prices_long["ticker"].isin(tickers)]
    db.load_df(prices_long[["date", "ticker", "close"]], "prices")
    logger.info("Loaded %d price rows into DuckDB", len(prices_long))

    # ── Load asset metadata ────────────────────────────────────────────
    assets = pd.DataFrame([
        {"ticker": t, "name": v[0], "asset_class": v[1], "region": v[2]}
        for t, v in UNIVERSE.items() if t in tickers
    ])
    db.load_df(assets, "assets")
    logger.info("Loaded %d asset records", len(assets))


def load_price_frame(db: DB, prices_wide: pd.DataFrame, tickers: List[str]) -> None:
    """Load a validated wide price/wealth-index frame and asset metadata."""
    frame = prices_wide.copy()
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index.name = "Date"
    prices_long = (
        frame.reset_index()
        .melt(id_vars=["Date"], var_name="ticker", value_name="close")
        .rename(columns={"Date": "date"})
        .dropna(subset=["close"])
        .assign(date=lambda df: pd.to_datetime(df["date"]).dt.date)
    )
    prices_long = prices_long[prices_long["ticker"].isin(tickers)]
    db.load_df(prices_long[["date", "ticker", "close"]], "prices")
    assets = pd.DataFrame([
        {"ticker": ticker, "name": UNIVERSE[ticker][0],
         "asset_class": UNIVERSE[ticker][1], "region": UNIVERSE[ticker][2]}
        for ticker in tickers if ticker in UNIVERSE
    ])
    db.load_df(assets, "assets")


def download_wrds_and_load(
    db: DB,
    tickers: List[str],
    start: str,
    end: str,
    dataset: str = "legacy",
) -> None:
    """Load a bounded CRSP total-return index through the WRDS connector."""
    from riskparity.wrds_loader import CRSPRequest, fetch_total_return_index

    request = CRSPRequest.create(tickers, start, end, dataset)
    prices_wide = fetch_total_return_index(request)
    load_price_frame(db, prices_wide, tickers)


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
