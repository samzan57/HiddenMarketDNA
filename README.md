Discovering the Hidden Market DNA using PCA
A Rolling PCA Approach for Market Regimes, Risk Management and Market-Neutral Portfolios

Overview

Financial markets appear noisy and chaotic, yet their movements are often driven by a small number of latent common factors.
This project aims to uncover this hidden market structure using Principal Component Analysis (PCA) applied to sector-based ETFs.

By extending standard PCA to a rolling (time-varying) framework, we:

  * identify the dominant market forces,

  * detect volatility regimes,

  * and construct a market-neutral portfolio with dynamic risk adjustment.

This project is both a quantitative research exercise and a practical portfolio construction framework.

Objectives :

  * Extract latent market factors from sector ETFs

  * Interpret PCA components economically

  * Track the evolution of factors over time (Rolling PCA)

  * Detect market regimes using factor volatility

  * Build a market-neutral portfolio exploiting non-market factors

  * Dynamically adjust exposure in high-risk regimes

  * Evaluate performance with robust metrics

Economic Intuition

  * PC1 → Global market factor (systematic risk)

  * PC2 → Sector rotation / relative value dynamics

  * PC3+ → Idiosyncratic or residual structures

The key insight is that risk regimes can be detected endogenously by monitoring the volatility of PC1, rather than relying on external indicators.

Data:

Assets: US Sector ETFs
  XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU

Frequency: Daily

Source: Yahoo Finance (CSV)

Preprocessing:

  * Adjusted Close prices

  * Log-returns

  * Alignment and cleaning

Project Architecture
HiddenMarketDNA/
│
├── data/
│   └── raw/                     # ETF CSV files
│
├── src/
│   ├── data_loader.py           # Data loading & cleaning
│   ├── returns.py               # Log-return computation
│   ├── pca_engine.py            # PCA abstraction layer
│   ├── rolling_backtest.py      # Rolling PCA + regime detection
│   ├── portfolio_engine.py      # Market-neutral portfolio construction
│   ├── diagnostics.py           # PCA diagnostics
│   ├── performance.py           # Performance metrics
│   └── visualization.py         # Regime & portfolio visualization
│
├── notebooks/
│   ├── 01_exploration.ipynb      # Descriptive stats & correlations
│   ├── 02_pca_static.ipynb       # Static PCA & interpretation
│   ├── 03_pca_rolling.ipynb      # Factor evolution & regimes
│   └── 04_rolling_visual.ipynb   # Regime-aware portfolio visualization
│
└── main.py                      # End-to-end pipeline

Methodology
1. Returns Computation

   * Log-returns from adjusted prices

   * Strict positivity checks

   * Alignment across assets

2. PCA (Static)

   * Identification of dominant factors

   * Scree plots and explained variance

   * Economic interpretation of eigen-portfolios

3. Rolling PCA

   * PCA recomputed on a rolling window (252 days)

   * Time-varying factor structure

   * Captures regime changes and structural shifts

4. Regime Detection

   * Market regime defined by volatility of PC1

   * High PC1 volatility → stressed market

   * Regime threshold defined via empirical quantile

5. Portfolio Construction

   * Market-neutral exposure

   * Target factor: PC2

   * Zero net exposure to PC1

   * Dynamic exposure scaling in volatile regimes

6. Backtesting

   * Strict out-of-sample testing

   * Rolling weights and returns

   * No look-ahead bias
How to Run
  pip install -r requirements.txt
  python main.py
Technologies Used
  * Python

  * NumPy

  * Pandas

  * scikit-learn

  * Matplotlib / Seaborn
Possible Extensions
  * Multi-factor allocation (PC2 + PC3)

  * Dynamic window size selection

  * Alternative regime definitions

  * Extension to other asset classes

  * Transaction costs and leverage constraints

  * Live or paper trading integration
Key Takeaway
  Markets may look noisy, but their movements are often driven by a small number of hidden forces.
  PCA provides a powerful lens to uncover this market DNA, enabling better risk management and systematic portfolio construction.
Author
ZANTOKO Deo
Applied Mathematics & Quantitative Finance
