-- Targets computed using the close of decision date d apply only to
-- returns with date > d. Holdings then drift until the next target date.
-- Idealized close-to-close accounting; no execution costs, tax or leverage.
CREATE OR REPLACE VIEW portfolio_daily_returns AS
WITH weight_dates AS (
    SELECT r.date AS ret_date, s.strategy,
           (SELECT MAX(w.date) FROM portfolio_weights w
            WHERE w.strategy = s.strategy AND w.date < r.date) AS weight_date
    FROM (SELECT DISTINCT date FROM returns) r
    CROSS JOIN (SELECT DISTINCT strategy FROM portfolio_weights) s
), asset_paths AS (
    SELECT wd.ret_date AS date, wd.strategy, wd.weight_date, pw.ticker, pw.weight,
           CASE WHEN pw.weight > 0 AND (r.ret IS NULL OR NOT isfinite(r.ret))
                THEN error('Missing or invalid return for a held asset')
                ELSE COALESCE(r.ret, 0) END AS asset_log_ret
    FROM weight_dates wd
    JOIN portfolio_weights pw ON pw.date = wd.weight_date AND pw.strategy = wd.strategy
    LEFT JOIN returns r ON r.date = wd.ret_date AND r.ticker = pw.ticker
), grown_assets AS (
    SELECT date, strategy, weight_date, ticker,
           weight * EXP(SUM(asset_log_ret) OVER (
               PARTITION BY strategy, weight_date, ticker ORDER BY date
               ROWS UNBOUNDED PRECEDING
           )) AS asset_value
    FROM asset_paths
), segments AS (
    SELECT date, strategy, weight_date, SUM(asset_value) AS segment_value
    FROM grown_assets GROUP BY date, strategy, weight_date
), daily AS (
    SELECT date, strategy,
           segment_value / LAG(segment_value, 1, 1.0) OVER (
               PARTITION BY strategy, weight_date ORDER BY date
           ) - 1 AS port_simple_ret
    FROM segments
)
SELECT date, strategy, port_simple_ret,
       LN(1 + port_simple_ret) AS port_log_ret
FROM daily;

CREATE OR REPLACE VIEW portfolio_nav_series AS
WITH nav AS (
    SELECT date, strategy, port_simple_ret,
           EXP(SUM(port_log_ret) OVER (
               PARTITION BY strategy ORDER BY date ROWS UNBOUNDED PRECEDING
           )) AS nav
    FROM portfolio_daily_returns
)
SELECT *, GREATEST(1.0, MAX(nav) OVER (
    PARTITION BY strategy ORDER BY date ROWS UNBOUNDED PRECEDING
)) AS peak_nav FROM nav;

CREATE OR REPLACE VIEW drawdown_series AS
SELECT date, strategy, nav, peak_nav, nav / peak_nav - 1 AS drawdown
FROM portfolio_nav_series;

CREATE OR REPLACE VIEW rolling_sharpe AS
SELECT date, strategy,
       AVG(port_simple_ret) OVER w * 252 AS rolling_mean_ann,
       STDDEV_SAMP(port_simple_ret) OVER w * SQRT(252) AS rolling_vol_ann,
       CASE WHEN COUNT(*) OVER w = 252
            THEN AVG(port_simple_ret) OVER w * SQRT(252)
                 / NULLIF(STDDEV_SAMP(port_simple_ret) OVER w, 0)
            ELSE NULL END AS rolling_sharpe_252d
FROM portfolio_daily_returns
WINDOW w AS (PARTITION BY strategy ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW);

-- CAGR uses all earned return observations, including the first, at 252/year.
-- Sharpe uses arithmetic daily returns, sample volatility and zero risk-free rate.
CREATE OR REPLACE VIEW performance_summary AS
WITH stats AS (
    SELECT strategy,
           EXP(SUM(port_log_ret) * 252.0 / COUNT(*)) - 1 AS cagr,
           STDDEV_SAMP(port_simple_ret) * SQRT(252) AS vol_ann,
           AVG(port_simple_ret) * SQRT(252)
               / NULLIF(STDDEV_SAMP(port_simple_ret), 0) AS sharpe,
           AVG(CASE WHEN port_simple_ret > 0 THEN 1.0 ELSE 0.0 END) AS hit_rate,
           COUNT(*) AS n_days
    FROM portfolio_daily_returns GROUP BY strategy
), mdd AS (
    SELECT strategy, MIN(drawdown) AS max_drawdown
    FROM drawdown_series GROUP BY strategy
)
SELECT s.strategy, s.cagr, s.vol_ann, s.sharpe, mdd.max_drawdown,
       s.cagr / NULLIF(ABS(mdd.max_drawdown), 0) AS calmar,
       s.hit_rate, s.n_days
FROM stats s JOIN mdd USING (strategy) ORDER BY sharpe DESC;
