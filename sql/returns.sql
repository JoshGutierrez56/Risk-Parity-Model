-- =============================================================================
-- returns.sql
-- Compute log returns from price levels and produce summary statistics.
-- All computation is in SQL — no Python needed for this step.
-- =============================================================================

-- ── 1. Populate returns table ───────────────────────────────────────────────
INSERT OR REPLACE INTO returns
SELECT
    date,
    ticker,
    LN(close / LAG(close) OVER (PARTITION BY ticker ORDER BY date)) AS ret
FROM prices;


-- ── 2. Annualised return statistics per ticker ──────────────────────────────
CREATE OR REPLACE VIEW asset_stats AS
WITH base AS (
    SELECT
        ticker,
        COUNT(*) AS n_obs,
        AVG(ret) AS mean_daily,
        STDDEV(ret) AS vol_daily,
        MIN(ret) AS min_daily_ret,
        MAX(ret) AS max_daily_ret
    FROM returns
    WHERE ret IS NOT NULL
    GROUP BY ticker
),
moments AS (
    SELECT
        r.ticker,
        AVG(POWER((r.ret - b.mean_daily) / NULLIF(b.vol_daily, 0), 3)) AS skewness
    FROM returns r
    JOIN base b USING (ticker)
    WHERE r.ret IS NOT NULL
    GROUP BY r.ticker
)
SELECT
    b.ticker,
    b.n_obs,
    b.mean_daily * 252 AS mean_ann,
    b.vol_daily * SQRT(252) AS vol_ann,
    (b.mean_daily * 252) / NULLIF(b.vol_daily * SQRT(252), 0) AS sharpe_approx,
    b.min_daily_ret,
    b.max_daily_ret,
    m.skewness
FROM base b
JOIN moments m USING (ticker);


-- ── 3. Rolling 60-day realised volatility ───────────────────────────────────
CREATE OR REPLACE VIEW rolling_vol_60d AS
SELECT
    date,
    ticker,
    STDDEV(ret) OVER (
        PARTITION BY ticker
        ORDER BY date
        ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
    ) * SQRT(252) AS vol_60d_ann,
    COUNT(ret) OVER (
        PARTITION BY ticker
        ORDER BY date
        ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
    ) AS obs_in_window
FROM returns
WHERE ret IS NOT NULL;


-- ── 4. Rolling 252-day realised volatility ──────────────────────────────────
CREATE OR REPLACE VIEW rolling_vol_252d AS
SELECT
    date,
    ticker,
    STDDEV(ret) OVER (
        PARTITION BY ticker
        ORDER BY date
        ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
    ) * SQRT(252) AS vol_252d_ann,
    COUNT(ret) OVER (
        PARTITION BY ticker
        ORDER BY date
        ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
    ) AS obs_in_window
FROM returns
WHERE ret IS NOT NULL;


-- ── 5. Maximum drawdown (pure SQL) ──────────────────────────────────────────
CREATE OR REPLACE VIEW drawdown_by_ticker AS
WITH nav AS (
    SELECT
        date,
        ticker,
        EXP(SUM(ret) OVER (PARTITION BY ticker ORDER BY date)) AS cum_nav
    FROM returns
    WHERE ret IS NOT NULL
),
peak AS (
    SELECT
        date,
        ticker,
        cum_nav,
        MAX(cum_nav) OVER (PARTITION BY ticker ORDER BY date ROWS UNBOUNDED PRECEDING) AS peak_nav
    FROM nav
)
SELECT
    ticker,
    MIN(cum_nav / peak_nav - 1) AS max_drawdown
FROM peak
GROUP BY ticker;
