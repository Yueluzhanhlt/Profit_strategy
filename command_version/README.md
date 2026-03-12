# BTC Spot-Futures Statistical Arbitrage Research

This repository contains a high-frequency crypto research project focused on **BTC spot-futures basis mean reversion**, built from **millisecond-level trade and order-book-derived data** and developed into a **research-grade alpha prototype** through out-of-sample validation, robustness testing, and execution-aware backtesting.

## Project Overview

The project started from raw BTC spot and futures market data and followed a full research workflow:

1. **Raw data engineering**
   - Read and clean compressed millisecond-level market event streams
   - Aggregate asynchronous events into aligned **minute-level research bars**
   - Build microstructure features from trade and book-ticker data

2. **Volatility forecasting**
   - Train short-horizon volatility models using a **walk-forward out-of-sample framework**
   - Evaluate whether predicted volatility can improve basis mean-reversion trading

3. **Basis mean-reversion research**
   - Trade the BTC spot-futures basis:
     \[
     basis_t = \log(F_t) - \log(S_t)
     \]
   - Test z-score-based mean-reversion signals
   - Compare continuous trading vs regime-filtered trading

4. **Execution-aware backtesting**
   - Add transaction-cost assumptions
   - Add capital / position sizing
   - Add holding-period constraints
   - Stress-test strategy behavior under more realistic friction assumptions

---

## Core Research Question

Can BTC spot-futures basis mean reversion be improved using volatility-aware overlays and regime selection?

## Main Findings

### 1. Broad spot volatility was **not** an effective trading overlay
An initial hypothesis was that elevated spot volatility would help identify poor trading periods for the basis strategy.  
Out-of-sample testing showed that this was **not** an effective regime filter.

### 2. Basis-specific activity was the relevant regime variable
Signals built from the basis itself — such as:

- rolling basis realized variance
- rolling basis volatility
- spot-futures divergence measures

were much more informative for trade selection than broad spot volatility.

### 3. Trading only in high basis-activity regimes materially improved strategy quality
Restricting trades to high basis-activity regimes improved:

- Sharpe ratio
- drawdown
- trade quality
- cost robustness

relative to a continuous-trading baseline.

### 4. Strategy performance was friction-sensitive
The strategy remained attractive in lower-friction settings, but required stricter trade selection under more realistic transaction-cost assumptions.

---

## Repository Structure

```text
btc-spot-futures-stat-arb/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── default.yaml
├── src/
│   ├── data_clean.py
│   ├── vol_model.py
│   ├── basis_strategy.py
│   └── level1_backtest.py
├── scripts/
│   ├── run_data_pipeline.py
│   ├── run_vol_model.py
│   ├── run_basis_research.py
│   └── run_level1_backtest.py
├── notebooks/
│   ├── 01_vol_model_oos.ipynb
│   ├── 02_basis_baseline.ipynb
│   ├── 03_regime_discovery.ipynb
│   └── 04_level1_execution.ipynb
└── reports/
    ├── project_summary.md
    └── figures/
