import os
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Helpers: load bucket parquet
# =========================================================
def _date_range_yyyymmdd(start_date: str, end_date: str) -> List[str]:
    dates = pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq="D")
    return [d.strftime("%Y%m%d") for d in dates]


def load_bucket_range(
    buckets_root: str,
    freq_min: int,
    start_date: str,
    end_date: str,
    mid_col_prefer: str = "mid_last",
    px_fallback_col: str = "px_last",
    prefix: str = "spot",
) -> pd.DataFrame:
    """
    Load bucket parquet files over a date range and return a standardized dataframe.

    Expected path:
      {buckets_root}/freq={freq_min}min/date=YYYYMMDD/buckets.parquet
    """
    dates = _date_range_yyyymmdd(start_date, end_date)

    frames = []
    for d in dates:
        path = os.path.join(buckets_root, f"freq={freq_min}min", f"date={d}", "buckets.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))

    if not frames:
        raise FileNotFoundError(
            f"No bucket parquet files found under {buckets_root} for [{start_date}, {end_date}]"
        )

    df = pd.concat(frames, ignore_index=True)
    df = (
        df.sort_values("timestamp")
          .drop_duplicates(subset=["timestamp"], keep="last")
          .reset_index(drop=True)
    )

    px = df[mid_col_prefer].copy() if mid_col_prefer in df.columns else pd.Series(np.nan, index=df.index)
    if px_fallback_col in df.columns:
        px = px.fillna(df[px_fallback_col])
    px = px.ffill()

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["timestamp"], utc=True),
        f"{prefix}_mid": px.astype(float),
    })

    extra_cols = [
        "mid_last",
        "px_last",
        "rel_spread_mean",
        "spread_mean",
        "imbalance_mean",
        "vol_notional",
        "n_trades",
        "buy_ratio",
    ]
    for c in extra_cols:
        if c in df.columns:
            out[f"{prefix}_{c}"] = df[c].values

    return out.sort_values("timestamp").reset_index(drop=True)


# =========================================================
# Spot / futures merge and basis dataframe
# =========================================================
def prepare_basis_dataframe(
    spot_buckets_root: str,
    fut_buckets_root: str,
    freq_min: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Load spot and futures buckets, align on timestamp, and construct basis dataframe.
    """
    spot = load_bucket_range(
        buckets_root=spot_buckets_root,
        freq_min=freq_min,
        start_date=start_date,
        end_date=end_date,
        prefix="spot",
    )
    fut = load_bucket_range(
        buckets_root=fut_buckets_root,
        freq_min=freq_min,
        start_date=start_date,
        end_date=end_date,
        prefix="fut",
    )

    df = spot.merge(fut, on="timestamp", how="inner")
    df = (
        df.sort_values("timestamp")
          .drop_duplicates(subset=["timestamp"], keep="last")
          .reset_index(drop=True)
    )

    df = df[(df["spot_mid"] > 0) & (df["fut_mid"] > 0)].copy()

    df["log_spot"] = np.log(df["spot_mid"])
    df["log_fut"] = np.log(df["fut_mid"])
    df["ret_spot"] = df["log_spot"].diff()
    df["ret_fut"] = df["log_fut"].diff()

    df["basis"] = df["log_fut"] - df["log_spot"]
    df["basis_ret"] = df["ret_fut"] - df["ret_spot"]

    return df.reset_index(drop=True)


# =========================================================
# Extra signals
# =========================================================
def add_historical_vol_signal(
    df: pd.DataFrame,
    hist_horizon_min: int = 30,
    freq_min: int = 1,
    out_col: str = "hist_vol_spot_30m",
) -> pd.DataFrame:
    out = df.copy()
    w = int(round(hist_horizon_min / freq_min))
    if w <= 0:
        raise ValueError("hist_horizon_min must be >= freq_min")

    out["ret_spot2"] = out["ret_spot"] ** 2
    out[out_col] = out["ret_spot2"].rolling(w, min_periods=w).sum()
    return out


def add_basis_specific_signals(
    df: pd.DataFrame,
    freq_min: int = 1,
    basis_vol_horizon_min: int = 30,
    div_horizon_min: int = 30,
) -> pd.DataFrame:
    out = df.copy()

    wb = int(round(basis_vol_horizon_min / freq_min))
    wd = int(round(div_horizon_min / freq_min))

    out["basis_ret2"] = out["basis_ret"] ** 2
    out["basis_vol_30m"] = out["basis_ret"].rolling(wb, min_periods=wb).std()
    out["basis_rv_30m"] = out["basis_ret2"].rolling(wb, min_periods=wb).sum()

    out["divergence_abs"] = (out["ret_fut"] - out["ret_spot"]).abs()
    out["div_30m_mean"] = out["divergence_abs"].rolling(wd, min_periods=wd).mean()
    out["div_30m_max"] = out["divergence_abs"].rolling(wd, min_periods=wd).max()

    out["basis_absret"] = out["basis_ret"].abs()
    out["basis_absret_30m_mean"] = out["basis_absret"].rolling(wb, min_periods=wb).mean()
    out["basis_absret_30m_max"] = out["basis_absret"].rolling(wb, min_periods=wb).max()

    return out


def merge_predicted_vol_signal(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    pred_col: str,
    timestamp_col: str = "timestamp",
    out_col: str = "pred_vol_spot_30m",
) -> pd.DataFrame:
    pred = pred_df.copy()
    pred[timestamp_col] = pd.to_datetime(pred[timestamp_col], utc=True)
    pred = pred[[timestamp_col, pred_col]].rename(columns={timestamp_col: "timestamp", pred_col: out_col})
    pred = (
        pred.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
    )

    return df.merge(pred, on="timestamp", how="left")


# =========================================================
# Core z-score strategy
# =========================================================
def run_basis_zscore_strategy(
    df: pd.DataFrame,
    basis_lookback_min: int = 240,
    freq_min: int = 1,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    cost_bps: float = 2.0,
    vol_filter_col: Optional[str] = None,
    vol_filter_quantile: float = 0.90,
    vol_filter_lookback_min: int = 30 * 24 * 60,
    high_vol_only: bool = False,
) -> pd.DataFrame:
    """
    Mean-reversion strategy on BTC spot-futures basis.

    If vol_filter_col is not None:
      - high_vol_only=False: allow entry only when signal <= rolling quantile
      - high_vol_only=True : allow entry only when signal >= rolling quantile
    """
    out = (
        df.copy()
          .sort_values("timestamp")
          .drop_duplicates(subset=["timestamp"], keep="last")
          .reset_index(drop=True)
    )

    w = int(round(basis_lookback_min / freq_min))
    if w <= 1:
        raise ValueError("basis_lookback_min too small")

    out["basis_mean"] = out["basis"].rolling(w, min_periods=w).mean()
    out["basis_std"] = out["basis"].rolling(w, min_periods=w).std()
    out["basis_z"] = (out["basis"] - out["basis_mean"]) / (out["basis_std"] + 1e-12)

    if vol_filter_col is not None:
        vw = int(round(vol_filter_lookback_min / freq_min))
        out["vol_filter_threshold"] = (
            out[vol_filter_col]
            .rolling(vw, min_periods=max(100, int(0.5 * vw)))
            .quantile(vol_filter_quantile)
            .shift(1)
        )

        if high_vol_only:
            out["allow_entry"] = (out[vol_filter_col] >= out["vol_filter_threshold"]).astype(float)
        else:
            out["allow_entry"] = (out[vol_filter_col] <= out["vol_filter_threshold"]).astype(float)

        out.loc[out["vol_filter_threshold"].isna(), "allow_entry"] = 0.0
    else:
        out["vol_filter_threshold"] = np.nan
        out["allow_entry"] = 1.0

    pos = np.zeros(len(out), dtype=float)

    for i in range(1, len(out)):
        prev_pos = pos[i - 1]
        z = out.at[i, "basis_z"]

        if pd.isna(z):
            pos[i] = prev_pos
            continue

        # exit
        if prev_pos == 1 and abs(z) < exit_z:
            prev_pos = 0
        elif prev_pos == -1 and abs(z) < exit_z:
            prev_pos = 0

        # entry
        if prev_pos == 0 and bool(out.at[i, "allow_entry"]):
            if z < -entry_z:
                prev_pos = 1.0   # long futures / short spot
            elif z > entry_z:
                prev_pos = -1.0  # short futures / long spot

        pos[i] = prev_pos

    out["position"] = pos
    out["gross_pnl"] = out["position"].shift(1).fillna(0.0) * out["basis_ret"].fillna(0.0)
    out["turnover_units"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["cost"] = out["turnover_units"] * (cost_bps / 1e4)
    out["net_pnl"] = out["gross_pnl"] - out["cost"]

    out["trade_open"] = ((out["position"] != 0) & (out["position"].shift(1).fillna(0) == 0)).astype(int)
    out["trade_close"] = ((out["position"] == 0) & (out["position"].shift(1).fillna(0) != 0)).astype(int)

    return out


# =========================================================
# Scaling variant
# =========================================================
def build_size_multiplier_from_quantiles(
    signal: pd.Series,
    freq_min: int = 1,
    lookback_min: int = 30 * 24 * 60,
    q1: float = 0.80,
    q2: float = 0.90,
    q3: float = 0.97,
    size_low: float = 1.0,
    size_mid: float = 0.7,
    size_high: float = 0.4,
    size_extreme: float = 0.15,
) -> pd.Series:
    w = int(round(lookback_min / freq_min))

    q1s = signal.rolling(w, min_periods=max(100, int(0.5 * w))).quantile(q1).shift(1)
    q2s = signal.rolling(w, min_periods=max(100, int(0.5 * w))).quantile(q2).shift(1)
    q3s = signal.rolling(w, min_periods=max(100, int(0.5 * w))).quantile(q3).shift(1)

    mult = pd.Series(np.nan, index=signal.index, dtype=float)
    ready = q1s.notna() & q2s.notna() & q3s.notna()

    mult.loc[ready] = size_low
    mult.loc[ready & (signal > q1s)] = size_mid
    mult.loc[ready & (signal > q2s)] = size_high
    mult.loc[ready & (signal > q3s)] = size_extreme

    return mult


def run_basis_zscore_strategy_with_scaling(
    df: pd.DataFrame,
    basis_lookback_min: int = 240,
    freq_min: int = 1,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    cost_bps: float = 2.0,
    scale_col: Optional[str] = None,
    scale_lookback_min: int = 30 * 24 * 60,
    q1: float = 0.80,
    q2: float = 0.90,
    q3: float = 0.97,
    size_low: float = 1.0,
    size_mid: float = 0.7,
    size_high: float = 0.4,
    size_extreme: float = 0.15,
) -> pd.DataFrame:
    out = (
        df.copy()
          .sort_values("timestamp")
          .drop_duplicates(subset=["timestamp"], keep="last")
          .reset_index(drop=True)
    )

    w = int(round(basis_lookback_min / freq_min))
    out["basis_mean"] = out["basis"].rolling(w, min_periods=w).mean()
    out["basis_std"] = out["basis"].rolling(w, min_periods=w).std()
    out["basis_z"] = (out["basis"] - out["basis_mean"]) / (out["basis_std"] + 1e-12)

    raw_pos = np.zeros(len(out), dtype=float)

    for i in range(1, len(out)):
        prev_pos = raw_pos[i - 1]
        z = out.at[i, "basis_z"]

        if pd.isna(z):
            raw_pos[i] = prev_pos
            continue

        if prev_pos == 1 and abs(z) < exit_z:
            prev_pos = 0
        elif prev_pos == -1 and abs(z) < exit_z:
            prev_pos = 0

        if prev_pos == 0:
            if z < -entry_z:
                prev_pos = 1
            elif z > entry_z:
                prev_pos = -1

        raw_pos[i] = prev_pos

    out["raw_position"] = raw_pos

    if scale_col is not None:
        out["size_multiplier"] = build_size_multiplier_from_quantiles(
            out[scale_col],
            freq_min=freq_min,
            lookback_min=scale_lookback_min,
            q1=q1,
            q2=q2,
            q3=q3,
            size_low=size_low,
            size_mid=size_mid,
            size_high=size_high,
            size_extreme=size_extreme,
        ).fillna(0.0)
    else:
        out["size_multiplier"] = 1.0

    out["position"] = out["raw_position"] * out["size_multiplier"]
    out["gross_pnl"] = out["position"].shift(1).fillna(0.0) * out["basis_ret"].fillna(0.0)
    out["turnover_units"] = out["position"].diff().abs().fillna(out["position"].abs())
    out["cost"] = out["turnover_units"] * (cost_bps / 1e4)
    out["net_pnl"] = out["gross_pnl"] - out["cost"]

    out["trade_open"] = ((out["position"] != 0) & (out["position"].shift(1).fillna(0) == 0)).astype(int)
    out["trade_close"] = ((out["position"] == 0) & (out["position"].shift(1).fillna(0) != 0)).astype(int)

    return out


# =========================================================
# Compare multiple variants
# =========================================================
def compare_three_versions(
    df: pd.DataFrame,
    freq_min: int = 1,
    basis_lookback_min: int = 240,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    cost_bps: float = 2.0,
    hist_vol_col: str = "hist_vol_spot_30m",
    pred_vol_col: str = "pred_vol_spot_30m",
    vol_filter_quantile: float = 0.90,
    vol_filter_lookback_min: int = 30 * 24 * 60,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    out_dict = {}

    bt_none = run_basis_zscore_strategy(
        df=df,
        basis_lookback_min=basis_lookback_min,
        freq_min=freq_min,
        entry_z=entry_z,
        exit_z=exit_z,
        cost_bps=cost_bps,
        vol_filter_col=None,
        vol_filter_quantile=vol_filter_quantile,
        vol_filter_lookback_min=vol_filter_lookback_min,
    )
    out_dict["no_filter"] = bt_none

    bt_hist = run_basis_zscore_strategy(
        df=df,
        basis_lookback_min=basis_lookback_min,
        freq_min=freq_min,
        entry_z=entry_z,
        exit_z=exit_z,
        cost_bps=cost_bps,
        vol_filter_col=hist_vol_col,
        vol_filter_quantile=vol_filter_quantile,
        vol_filter_lookback_min=vol_filter_lookback_min,
    )
    out_dict["hist_vol_filter"] = bt_hist

    bt_pred = run_basis_zscore_strategy(
        df=df,
        basis_lookback_min=basis_lookback_min,
        freq_min=freq_min,
        entry_z=entry_z,
        exit_z=exit_z,
        cost_bps=cost_bps,
        vol_filter_col=pred_vol_col,
        vol_filter_quantile=vol_filter_quantile,
        vol_filter_lookback_min=vol_filter_lookback_min,
    )
    out_dict["pred_vol_filter"] = bt_pred

    summary = pd.DataFrame({
        k: summarize_backtest(v, freq_min=freq_min) for k, v in out_dict.items()
    }).T

    return out_dict, summary


# =========================================================
# Performance summary
# =========================================================
def max_drawdown(cum_curve: pd.Series) -> float:
    peak = cum_curve.cummax()
    dd = cum_curve - peak
    return float(dd.min())


def summarize_backtest(bt: pd.DataFrame, freq_min: int = 1) -> pd.Series:
    r = bt["net_pnl"].fillna(0.0)
    g = bt["gross_pnl"].fillna(0.0)

    periods_per_year = 365 * 24 * (60 // freq_min)

    mean_r = r.mean()
    std_r = r.std()

    ann_return = mean_r * periods_per_year
    ann_vol = std_r * np.sqrt(periods_per_year) if std_r > 0 else np.nan
    sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan

    cum_net = r.cumsum()
    cum_gross = g.cumsum()

    return pd.Series({
        "n_rows": len(bt),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "cum_net_pnl": float(cum_net.iloc[-1]) if len(cum_net) else np.nan,
        "cum_gross_pnl": float(cum_gross.iloc[-1]) if len(cum_gross) else np.nan,
        "max_drawdown": max_drawdown(cum_net) if len(cum_net) else np.nan,
        "avg_position_abs": bt["position"].abs().mean(),
        "turnover_units_total": bt["turnover_units"].sum(),
        "n_trade_opens": bt["trade_open"].sum(),
        "n_trade_closes": bt["trade_close"].sum(),
        "pct_time_in_market": (bt["position"] != 0).mean(),
        "hit_rate_bar": (bt.loc[bt["position"].shift(1).fillna(0) != 0, "net_pnl"] > 0).mean(),
    })


# =========================================================
# Decile analysis
# =========================================================
def add_next_bar_baseline_pnl(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("timestamp").reset_index(drop=True)
    out["next_basis_ret"] = out["basis_ret"].shift(-1)
    return out


def decile_analysis_for_signal(
    df: pd.DataFrame,
    signal_col: str,
    basis_z_col: str = "basis_z",
    next_ret_col: str = "next_basis_ret",
    entry_z: float = 2.0,
    n_bins: int = 10,
) -> pd.DataFrame:
    x = df[[signal_col, basis_z_col, next_ret_col]].dropna().copy()
    x["decile"] = pd.qcut(x[signal_col], q=n_bins, labels=False, duplicates="drop") + 1

    x["mr_trade_flag"] = 0
    x.loc[x[basis_z_col] > entry_z, "mr_trade_flag"] = -1
    x.loc[x[basis_z_col] < -entry_z, "mr_trade_flag"] = 1
    x["mr_next_bar_pnl"] = x["mr_trade_flag"] * x[next_ret_col]

    out = x.groupby("decile").agg(
        n=(signal_col, "size"),
        signal_mean=(signal_col, "mean"),
        mr_trade_count=("mr_trade_flag", lambda s: (s != 0).sum()),
        mr_next_bar_pnl_mean=("mr_next_bar_pnl", "mean"),
        mr_next_bar_pnl_std=("mr_next_bar_pnl", "std"),
    ).reset_index()

    return out


# =========================================================
# Plotting
# =========================================================
def plot_cum_pnl(
    bt_dict: Dict[str, pd.DataFrame],
    pnl_col: str = "net_pnl",
    title: str = "Cumulative PnL",
) -> None:
    plt.figure(figsize=(12, 6))
    for name, bt in bt_dict.items():
        curve = bt[pnl_col].fillna(0).cumsum()
        plt.plot(bt["timestamp"], curve, label=name)
    plt.title(title)
    plt.xlabel("timestamp")
    plt.ylabel(f"cumulative {pnl_col}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_monthly_cum_pnl_compare(month_compare: pd.DataFrame) -> None:
    x = np.arange(len(month_compare))
    w = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - w, month_compare["baseline_cum_pnl"], width=w, label="baseline")
    plt.bar(x, month_compare["main_cum_pnl"], width=w, label="main")
    plt.bar(x + w, month_compare["cons_cum_pnl"], width=w, label="conservative")
    plt.xticks(x, month_compare["month"], rotation=45)
    plt.ylabel("monthly cumulative net pnl")
    plt.title("Monthly performance comparison")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
