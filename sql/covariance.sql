-- =============================================================================
-- covariance.sql
-- Rolling pairwise covariance matrix using SQL window functions.
--
-- Key insight: cov(X,Y) = E[XY] - E[X]·E[Y]
-- We can express both terms as window aggregates, making the full
-- covariance matrix computable entirely in SQL.
--
-- For a universe of N assets, this produces N² rows per date.
-- =============================================================================

-- ── Rolling pairwise covariance (60-day window) ─────────────────────────────
CREATE OR REPLACE VIEW rolling_cov_60d AS
WITH windowed AS (
    SELECT
        r1.date,
        r1.ticker AS ticker_i,
        r2.ticker AS ticker_j,
        r1.ret    AS ret_i,
        r2.ret    AS ret_j,
        -- Running means in the 60-day window
        AVG(r1.ret) OVER (
            PARTITION BY r1.ticker
            ORDER BY r1.date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        ) AS mean_i,
        AVG(r2.ret) OVER (
            PARTITION BY r2.ticker
            ORDER BY r2.date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        ) AS mean_j,
        -- E[X·Y] in the window
        AVG(r1.ret * r2.ret) OVER (
            PARTITION BY r1.ticker, r2.ticker
            ORDER BY r1.date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        ) AS mean_xy,
        COUNT(r1.ret) OVER (
            PARTITION BY r1.ticker, r2.ticker
            ORDER BY r1.date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        ) AS n_obs
    FROM returns r1
    JOIN returns r2 USING (date)
    WHERE r1.ret IS NOT NULL AND r2.ret IS NOT NULL
)
SELECT
    date,
    ticker_i,
    ticker_j,
    -- Annualised covariance: cov = (E[XY] - E[X]·E[Y]) × 252
    (mean_xy - mean_i * mean_j) * 252 AS cov_60d_ann,
    n_obs
FROM windowed
WHERE n_obs >= 30;   -- require at least 30 observations


-- ── Rolling diagonal variances (annualised) ─────────────────────────────────
CREATE OR REPLACE VIEW rolling_variance_60d AS
SELECT
    date,
    ticker_i                        AS ticker,
    cov_60d_ann                     AS var_60d_ann,
    SQRT(cov_60d_ann)               AS vol_60d_ann
FROM rolling_cov_60d
WHERE ticker_i = ticker_j;


-- ── Inverse-volatility weights (closed-form; no optimisation needed) ─────────
-- This is the simplest risk parity implementation: each asset gets a weight
-- proportional to 1/σ, equalising the volatility contribution exactly when
-- correlations are all zero.
--
-- w_i = (1/σ_i) / Σ_j (1/σ_j)
-- =============================================================================
CREATE OR REPLACE VIEW inv_vol_weights AS
WITH inv_vols AS (
    SELECT
        date,
        ticker,
        1.0 / NULLIF(vol_60d_ann, 0) AS inv_vol
    FROM rolling_variance_60d
    WHERE vol_60d_ann > 0
),
totals AS (
    SELECT
        date,
        SUM(inv_vol) AS total_inv_vol
    FROM inv_vols
    GROUP BY date
)
SELECT
    iv.date,
    iv.ticker,
    'inv_vol'            AS strategy,
    iv.inv_vol / t.total_inv_vol AS weight
FROM inv_vols iv
JOIN totals t USING (date);


-- ── Equal-weight weights (baseline) ─────────────────────────────────────────
CREATE OR REPLACE VIEW equal_weights AS
WITH universe AS (
    SELECT
        date,
        ticker,
        COUNT(ticker) OVER (PARTITION BY date) AS n_assets
    FROM rolling_variance_60d
)
SELECT
    date,
    ticker,
    'equal_weight'     AS strategy,
    1.0 / n_assets     AS weight
FROM universe;


-- ── Risk contribution diagnostic ────────────────────────────────────────────
-- Given weights w and covariance Σ:
--   Portfolio variance = wᵀΣw
--   Risk contribution of asset i = w_i * (Σw)_i / (wᵀΣw)
--
-- For the inv-vol weights, we compute this as a diagnostic.
-- Full ERC requires iteration (done in Python).
-- =============================================================================
CREATE OR REPLACE VIEW risk_contribution_diagnostic AS
WITH port_var AS (
    SELECT
        w.date,
        w.strategy,
        -- wᵀΣw approximated using diagonal (variance only) for SQL tractability
        SUM(w.weight * w.weight * v.var_60d_ann) AS port_var_diag
    FROM portfolio_weights w
    JOIN rolling_variance_60d v USING (date, ticker)
    GROUP BY w.date, w.strategy
),
marginal AS (
    SELECT
        w.date,
        w.ticker,
        w.strategy,
        w.weight,
        v.var_60d_ann,
        -- Marginal risk contribution (diagonal approximation)
        w.weight * v.var_60d_ann AS mrc_diag
    FROM portfolio_weights w
    JOIN rolling_variance_60d v USING (date, ticker)
)
SELECT
    m.date,
    m.ticker,
    m.strategy,
    m.weight,
    m.weight * m.mrc_diag / NULLIF(pv.port_var_diag, 0) AS risk_share_diag
FROM marginal m
JOIN port_var pv USING (date, strategy);
