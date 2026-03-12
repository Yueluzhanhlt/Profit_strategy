# Project Summary: BTC Spot-Futures Statistical Arbitrage Research

## Objective

The objective of this project is to study whether the **BTC spot-futures basis** contains a robust mean-reversion signal, and whether short-horizon predictive models and regime filters can improve trade selection under realistic transaction-cost assumptions.

---

## Data

The project starts from **millisecond-level raw market data** collected from BTC spot and futures streams.

### Primary datasets currently used
- **aggtrade**
- **bookticker**

These are cleaned and aggregated into **minute-level research bars** and microstructure features.

### Additional datasets collected for future execution realism
- **depthdiff**
- **snapshot**
- **trade**

These are intended for future work on **L2 order-book reconstruction** and more realistic execution assumptions.

---

## Research Workflow

### 1. Raw data engineering
- Read compressed `.jsonl.zst` market event files
- Handle malformed JSON lines and schema inconsistencies
- Align asynchronous spot and futures event streams
- Aggregate raw events into synchronized minute-level bars

### 2. Feature engineering
Built features including:
- last price / mid price
- VWAP
- volume / notional / trade count
- buy ratio
- spread
- relative spread
- order-book imbalance
- basis and basis divergence measures

### 3. Volatility forecasting
Implemented a **walk-forward out-of-sample framework**:
- 30 days training
- 1 day testing

Used this to model short-horizon volatility and evaluate whether predicted broad-market volatility improves trade selection.

### 4. Basis mean-reversion strategy
Constructed the traded object as:

\[
basis_t = \log(F_t) - \log(S_t)
\]

and used rolling z-score signals to identify mean-reversion opportunities.

### 5. Regime discovery
Tested whether elevated broad spot volatility could serve as an effective overlay for the strategy.

This hypothesis was **rejected** out of sample.

The research then shifted toward **basis-specific activity signals**, including:
- rolling basis realized variance
- rolling basis volatility
- spot-futures divergence measures

These proved substantially more informative for trade selection.

### 6. Execution-aware backtesting
Extended the strategy into a **Level 1 execution-aware backtest**, including:
- transaction-cost assumptions
- capital / NAV tracking
- signal-strength-based position sizing
- holding-period constraints

---

## Main Findings

### Finding 1: Broad spot volatility is not an effective overlay
A natural initial hypothesis was that elevated spot volatility would indicate poor trading conditions for basis mean reversion.  
Out-of-sample testing showed that this was not an effective regime filter.

### Finding 2: The relevant regime variable is basis-specific activity
Signals derived from the basis itself — rather than broad spot volatility — were much more informative.

In particular, restricting trades to **high basis-activity regimes** materially improved strategy quality.

### Finding 3: Strategy edge is concentrated in selective, high-quality regimes
The best-performing variants were highly selective:
- low time in market
- low average capital usage
- stronger Sharpe and lower drawdown relative to continuous trading

### Finding 4: The strategy is friction-sensitive
At low-to-moderate friction assumptions, the regime-filtered strategy remained attractive.  
At higher transaction-cost settings, stricter regime thresholds were required to maintain viability.

This suggests that the strategy is best understood as an **execution-sensitive alpha prototype** rather than a naïve continuous-trading signal.

---

## Robustness Checks Completed

The project currently includes robustness analysis across:

- transaction costs
- regime thresholds
- entry z-score
- exit z-score
- basis lookback windows
- monthly performance splits
- maximum holding constraints

This significantly strengthens the validity of the core research result.

---

## Current Best Interpretation

The core contribution of the project is not simply that a basis strategy was profitable in one backtest.

Instead, the main research contribution is:

1. a plausible initial hypothesis was tested and rejected,
2. the regime definition was reframed around the traded object itself,
3. the new mechanism was validated under multiple robustness dimensions.

This makes the project closer to a **quant research workflow** than a single isolated trading backtest.

---

## Current Limitations

The framework is still research-grade rather than production-grade.

It does not yet include:
- queue-aware maker execution
- exact partial-fill modeling
- order-priority simulation
- market-impact modeling
- funding / margin / liquidation modeling
- multi-asset portfolio optimization

---

## Ongoing Extension

Current work is focused on extending the framework toward:
- **L2 order-book reconstruction**
- more realistic taker-style execution
- bid/ask-aware fills
- richer fill assumptions
- richer transaction-cost modeling

---

## Practical Takeaway

This project demonstrates:
- high-frequency market data engineering
- microstructure-aware feature construction
- walk-forward OOS modeling
- alpha hypothesis testing
- statistical arbitrage research
- robustness analysis
- execution-aware backtesting

It is best viewed as a **research-grade alpha prototype for BTC spot-futures relative-value trading**.
