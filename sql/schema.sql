-- DuckDB storage for prices, log returns and close-date target weights.
CREATE TABLE IF NOT EXISTS prices (
    date DATE NOT NULL, ticker VARCHAR NOT NULL, close DOUBLE NOT NULL,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS returns (
    date DATE NOT NULL, ticker VARCHAR NOT NULL, ret DOUBLE,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS assets (
    ticker VARCHAR PRIMARY KEY, name VARCHAR, asset_class VARCHAR, region VARCHAR
);
CREATE TABLE IF NOT EXISTS portfolio_weights (
    date DATE NOT NULL, ticker VARCHAR NOT NULL, strategy VARCHAR NOT NULL,
    weight DOUBLE NOT NULL,
    PRIMARY KEY (date, ticker, strategy)
);
