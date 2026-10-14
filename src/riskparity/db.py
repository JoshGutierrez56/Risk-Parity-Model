"""
db.py
=====
DuckDB connection manager and SQL file executor.

DuckDB is a high-performance analytical SQL database — no server required.
It operates in-process (like SQLite) but supports full analytical SQL:
window functions, CTEs, lateral joins, ARRAY types, and parallel query
execution. This makes it ideal for quantitative finance pipelines where
the data layer is SQL but the optimisation layer is Python.

The project separates SQL analytics from numerical optimization in Python.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DB = "data/risk_parity.duckdb"


class DB:
    """
    Thin wrapper around a DuckDB connection.

    Parameters
    ----------
    path : str  — path to the .duckdb file (created if absent)
    """

    def __init__(self, path: str = _DEFAULT_DB) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(path)
        logger.info("DuckDB connected: %s", path)

    # ------------------------------------------------------------------
    # SQL execution helpers
    # ------------------------------------------------------------------

    def execute_file(self, sql_path: str) -> None:
        """Read and execute a .sql file (may contain multiple statements)."""
        # Let DuckDB parse the file: splitting on ';' also splits comments.
        # Propagate errors so an incomplete pipeline cannot report success.
        self.conn.execute(Path(sql_path).read_text(encoding="utf-8"))

    def query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute a SELECT and return a DataFrame."""
        if params:
            return self.conn.execute(sql, list(params.values())).df()
        return self.conn.execute(sql).df()

    def execute(self, sql: str) -> None:
        """Execute a non-SELECT statement."""
        self.conn.execute(sql)

    def load_df(self, df: pd.DataFrame, table: str) -> None:
        """Load a pandas DataFrame into a DuckDB table."""
        self.conn.register("_tmp_df", df)
        self.conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM _tmp_df")
        self.conn.unregister("_tmp_df")
        logger.debug("Loaded %d rows into %s", len(df), table)

    def table_exists(self, name: str) -> bool:
        result = self.conn.execute(
            f"SELECT COUNT(*) FROM information_schema.tables "
            f"WHERE table_name = '{name}'"
        ).fetchone()
        return result[0] > 0 if result else False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self, schema_path: str = "sql/schema.sql") -> None:
        """Create all tables and views from schema.sql."""
        logger.info("Initialising schema from %s", schema_path)
        self.execute_file(schema_path)

    def load_views(self, *sql_paths: str) -> None:
        """Execute one or more SQL files to register views."""
        for path in sql_paths:
            logger.info("Loading SQL views from %s", path)
            self.execute_file(path)

    def close(self) -> None:
        self.conn.close()
        logger.info("DuckDB connection closed.")
