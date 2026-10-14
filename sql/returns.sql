-- Daily log returns for covariance estimation. Require an aligned price panel:
-- a missing observation must not become an unlabelled multi-day return.
SELECT CASE WHEN EXISTS (
    SELECT date FROM prices GROUP BY date
    HAVING COUNT(*) <> (SELECT COUNT(DISTINCT ticker) FROM prices)
) OR EXISTS (
    SELECT 1 FROM prices WHERE close <= 0 OR NOT isfinite(close)
) THEN error('Prices must form a complete, positive, finite date/ticker panel')
ELSE TRUE END;

DELETE FROM returns;
INSERT INTO returns
SELECT date, ticker,
       LN(close / LAG(close) OVER (PARTITION BY ticker ORDER BY date)) AS ret
FROM prices;

CREATE OR REPLACE VIEW asset_stats AS
WITH standardized AS (
    SELECT ticker, ret,
           (ret - AVG(ret) OVER (PARTITION BY ticker))
           / NULLIF(STDDEV_POP(ret) OVER (PARTITION BY ticker), 0) AS z
    FROM returns WHERE ret IS NOT NULL
)
SELECT ticker, COUNT(*) AS n_obs,
       AVG(ret) * 252 AS mean_ann,
       STDDEV_SAMP(ret) * SQRT(252) AS vol_ann,
       AVG(ret) * SQRT(252) / NULLIF(STDDEV_SAMP(ret), 0) AS sharpe_approx,
       MIN(ret) AS min_daily_ret, MAX(ret) AS max_daily_ret,
       AVG(POWER(z, 3)) AS skewness
FROM standardized GROUP BY ticker;

CREATE OR REPLACE VIEW rolling_vol_60d AS
SELECT date, ticker,
       STDDEV_SAMP(ret) OVER w60 * SQRT(252) AS vol_60d_ann,
       COUNT(ret) OVER w60 AS obs_in_window
FROM returns WHERE ret IS NOT NULL
WINDOW w60 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW);

CREATE OR REPLACE VIEW rolling_vol_252d AS
SELECT date, ticker,
       STDDEV_SAMP(ret) OVER w252 * SQRT(252) AS vol_252d_ann,
       COUNT(ret) OVER w252 AS obs_in_window
FROM returns WHERE ret IS NOT NULL
WINDOW w252 AS (PARTITION BY ticker ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW);

CREATE OR REPLACE VIEW drawdown_by_ticker AS
WITH nav AS (
    SELECT date, ticker,
           EXP(SUM(ret) OVER (PARTITION BY ticker ORDER BY date)) AS cum_nav
    FROM returns WHERE ret IS NOT NULL
), peak AS (
    SELECT *, GREATEST(1.0, MAX(cum_nav) OVER (
        PARTITION BY ticker ORDER BY date ROWS UNBOUNDED PRECEDING
    )) AS peak_nav FROM nav
)
SELECT ticker, MIN(cum_nav / peak_nav - 1) AS max_drawdown
FROM peak GROUP BY ticker;
