"""Bounded CRSP/WRDS total-return loader for the risk-parity pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Iterable

import pandas as pd


DATASETS = {
    "legacy": ("crsp.dsf", "crsp.dsenames"),
    "annual": ("crsp_a_stock.dsf", "crsp_a_stock.dsenames"),
    "quarterly": ("crsp_q_stock.dsf", "crsp_q_stock.dsenames"),
}
_TICKER = re.compile(r"^[A-Z][A-Z0-9.-]{0,9}$")


@dataclass(frozen=True)
class CRSPRequest:
    tickers: tuple[str, ...]
    start: date
    end: date
    dataset: str = "legacy"

    @classmethod
    def create(
        cls,
        tickers: Iterable[str],
        start: str | date,
        end: str | date,
        dataset: str = "legacy",
    ) -> "CRSPRequest":
        normalized = tuple(dict.fromkeys(str(t).strip().upper() for t in tickers))
        if not normalized or len(normalized) > 20:
            raise ValueError("WRDS requests require 1-20 tickers")
        if any(not _TICKER.fullmatch(t) for t in normalized):
            raise ValueError("ticker contains unsupported characters")
        start_date = pd.Timestamp(start).date()
        end_date = pd.Timestamp(end).date()
        if start_date > end_date:
            raise ValueError("start must not be after end")
        if (end_date - start_date).days > 365 * 35:
            raise ValueError("WRDS request window is capped at 35 years")
        if dataset not in DATASETS:
            raise ValueError(f"unsupported CRSP dataset: {dataset}")
        return cls(normalized, start_date, end_date, dataset)


def build_crsp_query(request: CRSPRequest) -> str:
    """Build an allowlisted query after all identifiers have been validated."""
    returns_table, names_table = DATASETS[request.dataset]
    ticker_sql = ", ".join(f"'{ticker}'" for ticker in request.tickers)
    return f"""
        SELECT UPPER(names.ticker) AS ticker,
               daily.date,
               daily.ret,
               daily.dlret
        FROM {returns_table} AS daily
        JOIN {names_table} AS names
          ON daily.permno = names.permno
         AND daily.date BETWEEN names.namedt
                            AND COALESCE(names.nameendt, daily.date)
        WHERE UPPER(names.ticker) IN ({ticker_sql})
          AND daily.date BETWEEN '{request.start.isoformat()}'
                             AND '{request.end.isoformat()}'
        ORDER BY ticker, daily.date
    """.strip()


def total_return_index(rows: pd.DataFrame, tickers: Iterable[str]) -> pd.DataFrame:
    """Convert CRSP RET/DLRET rows into a wide total-return wealth index."""
    required = {"ticker", "date", "ret", "dlret"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"CRSP rows missing columns: {sorted(missing)}")
    frame = rows.loc[:, ["ticker", "date", "ret", "dlret"]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    frame["dlret"] = pd.to_numeric(frame["dlret"], errors="coerce")
    frame = frame[frame["ret"].notna() | frame["dlret"].notna()]
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("duplicate CRSP ticker/date rows")
    frame["total_ret"] = (
        (1.0 + frame["ret"].fillna(0.0))
        * (1.0 + frame["dlret"].fillna(0.0))
        - 1.0
    )
    frame["close"] = frame.groupby("ticker")["total_ret"].transform(
        lambda values: 100.0 * (1.0 + values).cumprod()
    )
    wide = frame.pivot(index="date", columns="ticker", values="close").sort_index()
    requested = [str(t).upper() for t in tickers]
    available = [ticker for ticker in requested if ticker in wide.columns]
    if not available:
        raise ValueError("WRDS returned no requested tickers")
    wide = wide.loc[:, available]
    wide.index.name = "Date"
    return wide


def connect_wrds():
    """Open an interactive/local-config WRDS connection without logging secrets."""
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - optional live dependency
        raise RuntimeError("Install requirements-live.txt before using WRDS") from exc
    return wrds.Connection(verbose=False)


def fetch_total_return_index(request: CRSPRequest, connection=None) -> pd.DataFrame:
    """Execute one bounded CRSP query and return a wide wealth index."""
    owned = connection is None
    db = connection or connect_wrds()
    try:
        rows = db.raw_sql(build_crsp_query(request), date_cols=["date"])
        return total_return_index(rows, request.tickers)
    finally:
        if owned and hasattr(db, "close"):
            db.close()
