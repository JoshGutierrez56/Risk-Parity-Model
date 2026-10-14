-- Each rolling window is partitioned by BOTH tickers after the pair join.
-- Population covariance of daily log returns, annualized by 252.
CREATE OR REPLACE VIEW rolling_cov_60d AS
WITH windowed AS (
    SELECT r1.date, r1.ticker AS ticker_i, r2.ticker AS ticker_j,
           COVAR_POP(r1.ret, r2.ret) OVER w * 252 AS cov_60d_ann,
           COUNT(*) OVER w AS n_obs
    FROM returns r1 JOIN returns r2 USING (date)
    WHERE r1.ret IS NOT NULL AND r2.ret IS NOT NULL
    WINDOW w AS (
        PARTITION BY r1.ticker, r2.ticker ORDER BY r1.date
        ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
    )
)
SELECT * FROM windowed WHERE n_obs >= 30;

CREATE OR REPLACE VIEW rolling_variance_60d AS
SELECT date, ticker_i AS ticker, cov_60d_ann AS var_60d_ann,
       SQRT(GREATEST(cov_60d_ann, 0)) AS vol_60d_ann
FROM rolling_cov_60d WHERE ticker_i = ticker_j;

CREATE OR REPLACE VIEW inv_vol_weights AS
WITH iv AS (
    SELECT date, ticker, 1.0 / vol_60d_ann AS inv_vol
    FROM rolling_variance_60d WHERE vol_60d_ann > 0
)
SELECT date, ticker, 'inv_vol' AS strategy,
       inv_vol / SUM(inv_vol) OVER (PARTITION BY date) AS weight
FROM iv;

CREATE OR REPLACE VIEW equal_weights AS
SELECT date, ticker, 'equal_weight' AS strategy,
       1.0 / COUNT(*) OVER (PARTITION BY date) AS weight
FROM rolling_variance_60d;

-- Full covariance diagnostic: w_i * (Sigma w)_i / (w' Sigma w).
CREATE OR REPLACE VIEW risk_contribution_diagnostic AS
WITH marginal AS (
    SELECT wi.date, wi.strategy, wi.ticker, wi.weight,
           SUM(c.cov_60d_ann * wj.weight) AS marginal_variance,
           COUNT(c.cov_60d_ann) AS n_pairs,
           COUNT(*) AS expected_pairs
    FROM portfolio_weights wi
    JOIN portfolio_weights wj ON wi.date = wj.date AND wi.strategy = wj.strategy
    LEFT JOIN rolling_cov_60d c ON c.date = wi.date
         AND c.ticker_i = wi.ticker AND c.ticker_j = wj.ticker
    GROUP BY wi.date, wi.strategy, wi.ticker, wi.weight
), components AS (
    SELECT *, CASE WHEN n_pairs <> expected_pairs
        THEN error('Incomplete covariance for risk contribution diagnostic')
        ELSE weight * marginal_variance END AS variance_contribution
    FROM marginal
)
SELECT date, strategy, ticker, weight,
       variance_contribution / NULLIF(SUM(variance_contribution)
          OVER (PARTITION BY date, strategy), 0) AS risk_share
FROM components;
