-- =============================================================================
-- backtest.sql
-- Walk-forward portfolio simulation and performance metrics — pure SQL.
--
-- Given the portfolio_weights table (populated by Python ERC optimizer)
-- and the returns table (populated by returns.sql), compute:
--   1. Daily portfolio returns
--   2. Cumulative NAV (geometric compounding)
--   3. Rolling 252-day Sharpe ratio
--   4. Drawdown series
--   5. Summary performance metrics
-- =============================================================================


-- ── 1. Daily portfolio returns ───────────────────────────────────────────────
-- port_ret_t = Σ_i  w_{i,t-1} × ret_{i,t}
-- We use the weight from the *previous* rebalancing date (hold-then-rebalance).
CREATE OR REPLACE VIEW portfolio_daily_returns AS
WITH weight_dates AS (
    -- Find one unambiguous most-recent weight date per return date/strategy.
    SELECT
        rd.date AS ret_date,
        s.strategy,
        (
            SELECT MAX(pw.date)
            FROM portfolio_weights pw
            WHERE pw.strategy = s.strategy AND pw.date <= rd.date
        ) AS weight_date
    FROM (SELECT DISTINCT date FROM returns WHERE ret IS NOT NULL) rd
    CROSS JOIN (SELECT DISTINCT strategy FROM portfolio_weights) s
),
weighted_returns AS (
    SELECT
        wd.ret_date   AS date,
        wd.strategy,
        SUM(pw.weight * r.ret) AS port_log_ret
    FROM weight_dates wd
    JOIN portfolio_weights pw
        ON pw.date = wd.weight_date AND pw.strategy = wd.strategy
    JOIN returns r
        ON r.date = wd.ret_date AND r.ticker = pw.ticker
    WHERE r.ret IS NOT NULL AND wd.weight_date IS NOT NULL
    GROUP BY wd.ret_date, wd.strategy
)
SELECT
    date,
    strategy,
    port_log_ret,
    EXP(port_log_ret) - 1 AS port_simple_ret   -- for Sharpe / metrics
FROM weighted_returns;


-- ── 2. Cumulative NAV ────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW portfolio_nav_series AS
WITH nav AS (
    SELECT
        date,
        strategy,
        port_simple_ret,
        EXP(
            SUM(port_log_ret) OVER (
                PARTITION BY strategy
                ORDER BY date
                ROWS UNBOUNDED PRECEDING
            )
        ) AS nav
    FROM portfolio_daily_returns
)
SELECT
    date,
    strategy,
    port_simple_ret,
    nav,
    MAX(nav) OVER (
        PARTITION BY strategy
        ORDER BY date
        ROWS UNBOUNDED PRECEDING
    ) AS peak_nav
FROM nav;


-- ── 3. Drawdown series ───────────────────────────────────────────────────────
CREATE OR REPLACE VIEW drawdown_series AS
SELECT
    date,
    strategy,
    nav,
    peak_nav,
    nav / peak_nav - 1 AS drawdown
FROM portfolio_nav_series;


-- ── 4. Rolling 252-day Sharpe ratio ─────────────────────────────────────────
CREATE OR REPLACE VIEW rolling_sharpe AS
SELECT
    date,
    strategy,
    AVG(port_simple_ret) OVER (
        PARTITION BY strategy
        ORDER BY date
        ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
    ) * 252 AS rolling_mean_ann,
    STDDEV(port_simple_ret) OVER (
        PARTITION BY strategy
        ORDER BY date
        ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
    ) * SQRT(252) AS rolling_vol_ann,
    CASE
        WHEN STDDEV(port_simple_ret) OVER (
            PARTITION BY strategy
            ORDER BY date
            ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
        ) > 0
        THEN (AVG(port_simple_ret) OVER (
                PARTITION BY strategy
                ORDER BY date
                ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
             ) * 252)
             /
             (STDDEV(port_simple_ret) OVER (
                PARTITION BY strategy
                ORDER BY date
                ROWS BETWEEN 251 PRECEDING AND CURRENT ROW
             ) * SQRT(252))
        ELSE NULL
    END AS rolling_sharpe_252d
FROM portfolio_daily_returns;


-- ── 5. Summary performance metrics ──────────────────────────────────────────
CREATE OR REPLACE VIEW performance_summary AS
WITH nav_final AS (
    SELECT strategy, MAX(date) AS last_date, MIN(date) AS first_date
    FROM portfolio_nav_series GROUP BY strategy
),
first_last AS (
    SELECT ns.strategy, ns.nav AS final_nav,
           DATEDIFF('day', nf.first_date, nf.last_date) / 365.25 AS years
    FROM portfolio_nav_series ns
    JOIN nav_final nf ON ns.strategy = nf.strategy AND ns.date = nf.last_date
),
stats AS (
    SELECT
        strategy,
        AVG(port_simple_ret) * 252         AS mean_ann,
        STDDEV(port_simple_ret) * SQRT(252) AS vol_ann,
        (AVG(port_simple_ret) * 252) /
            NULLIF(STDDEV(port_simple_ret) * SQRT(252), 0) AS sharpe,
        MIN(port_simple_ret)               AS worst_day,
        MAX(port_simple_ret)               AS best_day,
        COUNT(*)                           AS n_days,
        SUM(CASE WHEN port_simple_ret > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS hit_rate
    FROM portfolio_daily_returns
    GROUP BY strategy
),
mdd AS (
    SELECT strategy, MIN(drawdown) AS max_drawdown
    FROM drawdown_series GROUP BY strategy
)
SELECT
    s.strategy,
    POWER(fl.final_nav, 1.0 / NULLIF(fl.years, 0)) - 1 AS cagr,
    s.vol_ann,
    s.sharpe,
    mdd.max_drawdown,
    CASE WHEN mdd.max_drawdown < 0
         THEN (POWER(fl.final_nav, 1.0 / NULLIF(fl.years, 0)) - 1) / ABS(mdd.max_drawdown)
         ELSE NULL END AS calmar,
    s.hit_rate,
    s.n_days
FROM stats s
JOIN mdd     USING (strategy)
JOIN first_last fl USING (strategy)
ORDER BY sharpe DESC;
