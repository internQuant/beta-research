# beta-research

Research using different methods to estimate betas between assets.

## Running

- Clone repository
- Run notebook

## Overview

This research project uses financial data to estimate the beta of various assets relative to the S&P 500 index. Two methods are employed for beta estimation:
- Kalman Filter
- Rolling Ordinary Least Squares (OLS)

## Data

The data is fetched from Yahoo Finance and includes the following tickers:
- JPM
- ^GSPC (S&P 500)
- GLD
- WMT
- AAPL
- BRK-B
- BIL

## Methods

### Kalman Filter Beta
The Kalman Filter is used to estimate time-varying betas.

### Rolling OLS Beta
Rolling OLS is used to estimate betas over a specified window period.

## Visualization

The project includes visualizations of the estimated betas and portfolio returns.

## Usage

To run the analysis, execute the provided Python Jupyter Notebook. Ensure you have the necessary dependencies installed:
- pandas
- numpy
- matplotlib
- yfinance
- cycler
- jinja2

## Results

The results include:
- Estimated betas for each asset
- Portfolio returns and betas
- Hedged portfolio returns
- Performance metrics

## License

This project is licensed under the MIT License.

