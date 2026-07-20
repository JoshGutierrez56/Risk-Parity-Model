-- =============================================================================
-- schema.sql
-- Risk Parity Analytics Database — DuckDB
-- =============================================================================
-- All computation that can be expressed in SQL lives here.
-- Python touches only: (1) data ingestion via yfinance, (2) ERC optimisation
-- (iterative — SQL cannot do it), and (3) matplotlib visualisation.
--
-- Edward Qian coined "risk parity" at PanAgora in 2004.
-- This implementation replicates the core analytics in their style.
-- =============================================================================

-- Prices: one row per (date, ticker)
CREATE TABLE IF NOT EXISTS prices (
    date   DATE    NOT NULL,
    ticker VARCHAR NOT NULL,
    close  DOUBLE  NOT NULL,
    PRIMARY KEY (date, ticker)
);

-- Daily log returns: computed from prices
CREATE TABLE IF NOT EXISTS returns (
    date   DATE    NOT NULL,
    ticker VARCHAR NOT NULL,
    ret    DOUBLE,      -- log return: ln(P_t / P_{t-1})
    PRIMARY KEY (date, ticker)
);

-- Asset metadata
CREATE TABLE IF NOT EXISTS assets (
    ticker      VARCHAR PRIMARY KEY,
    name        VARCHAR,
    asset_class VARCHAR,   -- Equity, FixedIncome, Commodity, RealEstate
    region      VARCHAR
);

-- Rolling covariance pairs: (date, ticker_i, ticker_j, cov_ij)
-- Populated by the SQL covariance view
CREATE TABLE IF NOT EXISTS rolling_cov (
    date     DATE    NOT NULL,
    ticker_i VARCHAR NOT NULL,
    ticker_j VARCHAR NOT NULL,
    cov_ij   DOUBLE,
    window_days INTEGER,
    PRIMARY KEY (date, ticker_i, ticker_j, window_days)
);

-- Portfolio weights produced by the Python ERC optimizer
CREATE TABLE IF NOT EXISTS portfolio_weights (
    date     DATE    NOT NULL,
    ticker   VARCHAR NOT NULL,
    strategy VARCHAR NOT NULL,   -- 'risk_parity', 'inv_vol', 'equal_weight', '60_40'
    weight   DOUBLE  NOT NULL,
    PRIMARY KEY (date, ticker, strategy)
);

-- Daily portfolio NAV (computed in SQL from weights + returns)
CREATE TABLE IF NOT EXISTS portfolio_nav (
    date     DATE    NOT NULL,
    strategy VARCHAR NOT NULL,
    nav      DOUBLE  NOT NULL,
    ret      DOUBLE,
    PRIMARY KEY (date, strategy)
);
