from typing import Optional, Dict, Tuple, List, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Small utilities
# =========================================================
def max_drawdown_from_curve(cum_curve: pd.Series) -> float:
    peak = cum_curve.cummax()
    dd = cum_curve - peak
    return float(dd.min())


def z_to_size_multiplier(abs_z: float, entry_z: float) -> float:
    """
    Simple signal-strength sizing schedule.
    """
    if np.isnan(abs_z) or abs_z < entry_z:
        return 0.0
    elif abs_z < entry_z + 0.5:
        return 0.50
    elif abs_z < entry_z + 1.0:
        return 0.75
    else:
        return 1.00


# =========================================================
# Core Level 1 execution-aware backtest
# =========================================================
def run_level1_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
    max_position_fraction: float = 0.10,
    basis_lookback_min: int = 240,
    freq_min: int = 1,
    entry_z: float = 1.5,
    exit_z: float = 1.0,
    max_holding_min: Optional[int] = 120,
    regime_signal_col: Optional[str] = "basis_rv_30m",
    regime_quantile: float = 0.80,
    regime_lookback_min: int = 30 * 24 * 60,
    high_only: bool = True,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
) -> pd.DataFrame:
    """
    Level 1 backtest:
    - capital / NAV tracking
    - signal-strength-based position sizing
    - simple next-bar execution
    - fixed fee + slippage assumptions
    - optional max holding period
    - optional regime gating
    """
    out = (
        df.copy()
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    # z-score
    w = int(round(basis_lookback_min / freq_min))
    if w <= 1:
        raise ValueError("basis_lookback_min too small")

    out["basis_mean"] = out["basis"].rolling(w, min_periods=w).mean()
    out["basis_std"] = out["basis"].rolling(w, min_periods=w).std()
    out["basis_z"] = (out["basis"] - out["basis_mean"]) / (out["basis_std"] + 1e-12)

    # regime filter
    if regime_signal_col is not None:
        vw = int(round(regime_lookback_min / freq_min))
        out["regime_threshold"] = (
            out[regime_signal_col]
            .rolling(vw, min_periods=max(100, int(0.5 * vw)))
            .quantile(regime_quantile)
            .shift(1)
        )

        if high_only:
            out["allow_entry"] = (out[regime_signal_col] >= out["regime_threshold"]).astype(float)
        else:
            out["allow_entry"] = (out[regime_signal_col] <= out["regime_threshold"]).astype(float)

        out.loc[out["regime_threshold"].isna(), "allow_entry"] = 0.0
    else:
        out["regime_threshold"] = np.nan
        out["allow_entry"] = 1.0

    # state
    pos_dir = np.zeros(len(out), dtype=float)      # -1, 0, +1
    pos_frac = np.zeros(len(out), dtype=float)     # fraction of NAV
    hold_bars = np.zeros(len(out), dtype=int)
    nav = np.zeros(len(out), dtype=float)
    nav[0] = initial_capital

    max_hold_bars = None if max_holding_min is None else int(round(max_holding_min / freq_min))
    total_cost_bps = fee_bps + slippage_bps

    for i in range(1, len(out)):
        prev_dir = pos_dir[i - 1]
        prev_frac = pos_frac[i - 1]
        prev_hold = hold_bars[i - 1]
        z = out.at[i, "basis_z"]

        cur_dir = prev_dir
        cur_frac = prev_frac
        cur_hold = prev_hold

        # accumulate holding time
        if prev_dir != 0:
            cur_hold += 1

        # exit rules
        if prev_dir != 0 and not np.isnan(z) and abs(z) < exit_z:
            cur_dir = 0.0
            cur_frac = 0.0
            cur_hold = 0
        elif prev_dir != 0 and max_hold_bars is not None and cur_hold >= max_hold_bars:
            cur_dir = 0.0
            cur_frac = 0.0
            cur_hold = 0

        # entry only when flat
        if cur_dir == 0 and not np.isnan(z) and bool(out.at[i, "allow_entry"]):
            abs_z = abs(z)
            mult = z_to_size_multiplier(abs_z, entry_z)
            if mult > 0:
                cur_frac = max_position_fraction * mult
                if z < -entry_z:
                    cur_dir = 1.0   # long futures / short spot
                    cur_hold = 0
                elif z > entry_z:
                    cur_dir = -1.0  # short futures / long spot
                    cur_hold = 0

        # next-bar style pnl from previous position
        basis_ret = out.at[i, "basis_ret"] if pd.notna(out.at[i, "basis_ret"]) else 0.0
        gross_ret = prev_dir * prev_frac * basis_ret

        # cost on position change
        turnover = abs(cur_dir * cur_frac - prev_dir * prev_frac)
        cost_ret = turnover * (total_cost_bps / 1e4)

        net_ret = gross_ret - cost_ret
        nav[i] = nav[i - 1] * (1.0 + net_ret)

        pos_dir[i] = cur_dir
        pos_frac[i] = cur_frac
        hold_bars[i] = cur_hold

    out["position_dir"] = pos_dir
    out["position_fraction"] = pos_frac
    out["holding_bars"] = hold_bars
    out["holding_min"] = out["holding_bars"] * freq_min

    out["nav"] = nav
    out["nav_ret"] = pd.Series(nav).pct_change().fillna(0.0)
    out["cum_return"] = out["nav"] / initial_capital - 1.0
    out["cum_pnl_dollars"] = out["nav"] - initial_capital
    out["position_notional"] = out["position_fraction"] * pd.Series(nav).shift(1).fillna(initial_capital)

    return out


# =========================================================
# Summary functions
# =========================================================
def summarize_level1(bt: pd.DataFrame, freq_min: int = 1) -> pd.Series:
    r = bt["nav_ret"].fillna(0.0)
    periods_per_year = 365 * 24 * (60 // freq_min)

    ann_return = r.mean() * periods_per_year
    ann_vol = r.std() * np.sqrt(periods_per_year) if r.std() > 0 else np.nan
    sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan

    cum_curve = bt["cum_return"].fillna(method="ffill").fillna(0.0)

    return pd.Series({
        "final_nav": bt["nav"].iloc[-1],
        "cum_return": bt["cum_return"].iloc[-1],
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown_from_curve(cum_curve),
        "avg_position_fraction": bt["position_fraction"].mean(),
        "pct_time_in_market": (bt["position_dir"] != 0).mean(),
    })


def summarize_by_month_level1(bt: pd.DataFrame, freq_min: int = 1) -> pd.DataFrame:
    x = bt.copy()
    x["month"] = pd.to_datetime(x["timestamp"], utc=True).dt.to_period("M").astype(str)

    rows = []
    for month, g in x.groupby("month"):
        r = g["nav_ret"].fillna(0.0)
        periods_per_year = 365 * 24 * (60 // freq_min)

        ann_return = r.mean() * periods_per_year
        ann_vol = r.std() * np.sqrt(periods_per_year) if r.std() > 0 else np.nan
        sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan

        rows.append({
            "month": month,
            "n_rows": len(g),
            "cum_return": g["cum_return"].iloc[-1] - (g["cum_return"].iloc[0] if len(g) else 0.0),
            "cum_pnl_dollars": g["cum_pnl_dollars"].iloc[-1] - (g["cum_pnl_dollars"].iloc[0] if len(g) else 0.0),
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown_from_curve(g["cum_return"].fillna(method="ffill").fillna(0.0)),
            "pct_time_in_market": (g["position_dir"] != 0).mean(),
            "avg_position_fraction": g["position_fraction"].mean(),
        })

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


# =========================================================
# Plotting
# =========================================================
def plot_level1_cum_return(bt_dict: Dict[str, pd.DataFrame], title: str = "Cumulative Return") -> None:
    plt.figure(figsize=(12, 6))
    for name, bt in bt_dict.items():
        plt.plot(bt["timestamp"], bt["cum_return"], label=name)
    plt.title(title)
    plt.xlabel("timestamp")
    plt.ylabel("cumulative return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_level1_cum_pnl(bt_dict: Dict[str, pd.DataFrame], title: str = "Cumulative PnL ($)") -> None:
    plt.figure(figsize=(12, 6))
    for name, bt in bt_dict.items():
        plt.plot(bt["timestamp"], bt["cum_pnl_dollars"], label=name)
    plt.title(title)
    plt.xlabel("timestamp")
    plt.ylabel("PnL ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# =========================================================
# Sensitivity helpers
# =========================================================
def run_level1_cost_sensitivity(
    df: pd.DataFrame,
    initial_capital: float = 1000.0,
    max_position_fraction: float = 0.05,
    cost_pairs: Iterable[Tuple[float, float]] = ((0.5, 0.5), (1.0, 1.0), (2.5, 2.5), (5.0, 5.0)),
    basis_lookback_min: int = 240,
    entry_z: float = 1.5,
    exit_z: float = 1.0,
    max_holding_min: int = 120,
    regime_signal_col: str = "basis_rv_30m",
    regime_quantile: float = 0.80,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows = []
    bt_store: Dict[str, pd.DataFrame] = {}

    for fee_bps, slippage_bps in cost_pairs:
        label = f"fee{fee_bps}_slip{slippage_bps}"

        bt = run_level1_backtest(
            df=df,
            initial_capital=initial_capital,
            max_position_fraction=max_position_fraction,
            basis_lookback_min=basis_lookback_min,
            freq_min=1,
            entry_z=entry_z,
            exit_z=exit_z,
            max_holding_min=max_holding_min,
            regime_signal_col=regime_signal_col,
            regime_quantile=regime_quantile,
            high_only=True,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        bt_store[label] = bt
        s = summarize_level1(bt, freq_min=1)
        s["fee_bps"] = fee_bps
        s["slippage_bps"] = slippage_bps
        s["total_bps"] = fee_bps + slippage_bps
        s["label"] = label
        rows.append(s)

    summary = pd.DataFrame(rows)
    summary = summary[
        [
            "label",
            "fee_bps",
            "slippage_bps",
            "total_bps",
            "final_nav",
            "cum_return",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "avg_position_fraction",
            "pct_time_in_market",
        ]
    ].sort_values("total_bps").reset_index(drop=True)

    return summary, bt_store


def run_max_hold_sensitivity_main(
    df: pd.DataFrame,
    initial_capital: float = 1000.0,
    max_position_fraction: float = 0.05,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    holds: Iterable[Optional[int]] = (30, 60, 120, 240, None),
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows = []
    bt_store: Dict[str, pd.DataFrame] = {}

    for h in holds:
        label = "none" if h is None else str(h)

        bt = run_level1_backtest(
            df=df,
            initial_capital=initial_capital,
            max_position_fraction=max_position_fraction,
            basis_lookback_min=240,
            freq_min=1,
            entry_z=1.5,
            exit_z=1.0,
            max_holding_min=h,
            regime_signal_col="basis_rv_30m",
            regime_quantile=0.80,
            high_only=True,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        bt_store[f"main_hold_{label}"] = bt
        s = summarize_level1(bt, freq_min=1)
        s["max_holding_min"] = h if h is not None else -1
        rows.append(s)

    summary = pd.DataFrame(rows)
    summary["max_holding_label"] = summary["max_holding_min"].replace({-1: "none"}).astype(str)
    summary = summary[
        [
            "max_holding_label",
            "final_nav",
            "cum_return",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "avg_position_fraction",
            "pct_time_in_market",
        ]
    ]

    return summary, bt_store


def run_entry_exit_sensitivity_level1(
    df: pd.DataFrame,
    entry_list: Iterable[float] = (1.5, 2.0, 2.5),
    exit_list: Iterable[float] = (0.5, 1.0),
    initial_capital: float = 1000.0,
    max_position_fraction: float = 0.05,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    basis_lookback_min: int = 240,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows = []
    bt_store: Dict[str, pd.DataFrame] = {}

    for entry_z in entry_list:
        for exit_z in exit_list:
            name = f"e{entry_z}_x{exit_z}"

            bt = run_level1_backtest(
                df=df,
                initial_capital=initial_capital,
                max_position_fraction=max_position_fraction,
                basis_lookback_min=basis_lookback_min,
                freq_min=1,
                entry_z=entry_z,
                exit_z=exit_z,
                max_holding_min=120,
                regime_signal_col="basis_rv_30m",
                regime_quantile=0.80,
                high_only=True,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )

            bt_store[name] = bt
            s = summarize_level1(bt, freq_min=1)
            s["entry_z"] = entry_z
            s["exit_z"] = exit_z
            rows.append(s)

    summary = pd.DataFrame(rows)
    summary = summary[
        [
            "entry_z",
            "exit_z",
            "final_nav",
            "cum_return",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "avg_position_fraction",
            "pct_time_in_market",
        ]
    ].sort_values(["entry_z", "exit_z"]).reset_index(drop=True)

    return summary, bt_store


def run_lookback_sensitivity_level1(
    df: pd.DataFrame,
    lookback_list: Iterable[int] = (120, 240, 360, 720),
    entry_z: float = 1.5,
    exit_z: float = 1.0,
    initial_capital: float = 1000.0,
    max_position_fraction: float = 0.05,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows = []
    bt_store: Dict[str, pd.DataFrame] = {}

    for lookback in lookback_list:
        name = f"lb{lookback}"

        bt = run_level1_backtest(
            df=df,
            initial_capital=initial_capital,
            max_position_fraction=max_position_fraction,
            basis_lookback_min=lookback,
            freq_min=1,
            entry_z=entry_z,
            exit_z=exit_z,
            max_holding_min=120,
            regime_signal_col="basis_rv_30m",
            regime_quantile=0.80,
            high_only=True,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )

        bt_store[name] = bt
        s = summarize_level1(bt, freq_min=1)
        s["basis_lookback_min"] = lookback
        rows.append(s)

    summary = pd.DataFrame(rows)
    summary = summary[
        [
            "basis_lookback_min",
            "final_nav",
            "cum_return",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "avg_position_fraction",
            "pct_time_in_market",
        ]
    ].sort_values("basis_lookback_min").reset_index(drop=True)

    return summary, bt_store


# =========================================================
# Realistic-cost grid search
# =========================================================
def run_level1_grid_search(
    df: pd.DataFrame,
    initial_capital: float = 1000.0,
    max_position_fraction: float = 0.05,
    fee_bps: float = 2.5,
    slippage_bps: float = 2.5,
    regime_signals: Iterable[str] = ("basis_rv_30m", "basis_vol_30m"),
    quantiles: Iterable[float] = (0.80, 0.85, 0.90, 0.95),
    entry_list: Iterable[float] = (1.5, 2.0, 2.5),
    exit_list: Iterable[float] = (0.5, 1.0),
    lookback_list: Iterable[int] = (120, 240, 360),
    max_holding_min: int = 120,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows = []
    bt_store: Dict[str, pd.DataFrame] = {}

    for sig in regime_signals:
        for q in quantiles:
            for entry_z in entry_list:
                for exit_z in exit_list:
                    for lookback in lookback_list:
                        name = f"{sig}_q{int(q*100)}_e{entry_z}_x{exit_z}_lb{lookback}"

                        bt = run_level1_backtest(
                            df=df,
                            initial_capital=initial_capital,
                            max_position_fraction=max_position_fraction,
                            basis_lookback_min=lookback,
                            freq_min=1,
                            entry_z=entry_z,
                            exit_z=exit_z,
                            max_holding_min=max_holding_min,
                            regime_signal_col=sig,
                            regime_quantile=q,
                            high_only=True,
                            fee_bps=fee_bps,
                            slippage_bps=slippage_bps,
                        )

                        bt_store[name] = bt
                        s = summarize_level1(bt, freq_min=1)
                        s["name"] = name
                        s["regime_signal"] = sig
                        s["quantile"] = q
                        s["entry_z"] = entry_z
                        s["exit_z"] = exit_z
                        s["basis_lookback_min"] = lookback
                        s["fee_bps"] = fee_bps
                        s["slippage_bps"] = slippage_bps
                        s["total_cost_bps"] = fee_bps + slippage_bps
                        rows.append(s)

    summary = pd.DataFrame(rows)
    summary = summary[
        [
            "name",
            "regime_signal",
            "quantile",
            "entry_z",
            "exit_z",
            "basis_lookback_min",
            "fee_bps",
            "slippage_bps",
            "total_cost_bps",
            "final_nav",
            "cum_return",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "avg_position_fraction",
            "pct_time_in_market",
        ]
    ].sort_values("sharpe", ascending=False).reset_index(drop=True)

    return summary, bt_store
