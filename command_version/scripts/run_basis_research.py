#!/usr/bin/env python3
#python scripts/run_basis_research.py
#python scripts/run_basis_research.py \
#  --vol-filter-col basis_rv_30m \
#  --vol-filter-quantile 0.80 \
#  --high-only
#python scripts/run_basis_research.py \
#  --vol-filter-col basis_rv_30m \
#  --vol-filter-quantile 0.95 \
#  --high-only \
#  --entry-z 2.0 \
#  --exit-z 1.0 \
#  --basis-lookback-min 120

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.basis_strategy import (  # noqa: E402
    prepare_basis_dataframe,
    add_historical_vol_signal,
    add_basis_specific_signals,
    merge_predicted_vol_signal,
    run_basis_zscore_strategy,
    summarize_backtest,
)


DEFAULTS = {
    "spot": {
        "spot_buckets_root": "/Volumes/profit/feature_store/buckets",
        "fut_buckets_root": "/Volumes/profit/feature_store_futures/buckets",
        "pred_path": "/Volumes/profit/stat_arb_test/spot_vol_oos_pred_30m.parquet",
    }
}


def main():
    parser = argparse.ArgumentParser(
        description="Run basis mean-reversion research backtest."
    )

    parser.add_argument(
        "--market",
        choices=["spot"],
        default="spot",
        help="Current basis research uses spot vol predictions + spot/futures buckets.",
    )
    parser.add_argument("--spot-buckets-root", type=str, default=None)
    parser.add_argument("--fut-buckets-root", type=str, default=None)
    parser.add_argument("--pred-path", type=str, default=None)

    parser.add_argument("--freq-min", type=int, default=1)
    parser.add_argument("--start-date", type=str, default="2025-11-01")
    parser.add_argument("--end-date", type=str, default="2026-03-10")

    parser.add_argument("--basis-lookback-min", type=int, default=240)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--cost-bps", type=float, default=2.0)

    parser.add_argument(
        "--vol-filter-col",
        type=str,
        default=None,
        help='Optional filter signal, e.g. "basis_rv_30m", "basis_vol_30m", "pred_vol_spot_30m"',
    )
    parser.add_argument("--vol-filter-quantile", type=float, default=0.90)
    parser.add_argument("--vol-filter-lookback-min", type=int, default=30 * 24 * 60)
    parser.add_argument("--high-only", action="store_true")

    args = parser.parse_args()

    defaults = DEFAULTS["spot"]
    spot_buckets_root = args.spot_buckets_root or defaults["spot_buckets_root"]
    fut_buckets_root = args.fut_buckets_root or defaults["fut_buckets_root"]
    pred_path = args.pred_path or defaults["pred_path"]

    print("=" * 80)
    print("RUN BASIS RESEARCH")
    print(f"spot_buckets_root : {spot_buckets_root}")
    print(f"fut_buckets_root  : {fut_buckets_root}")
    print(f"pred_path         : {pred_path}")
    print(f"date range        : {args.start_date} -> {args.end_date}")
    print(f"basis_lookback    : {args.basis_lookback_min}")
    print(f"entry/exit z      : {args.entry_z} / {args.exit_z}")
    print(f"cost_bps          : {args.cost_bps}")
    print(f"vol_filter_col    : {args.vol_filter_col}")
    print(f"vol_filter_q      : {args.vol_filter_quantile}")
    print(f"high_only         : {args.high_only}")
    print("=" * 80)

    df = prepare_basis_dataframe(
        spot_buckets_root=spot_buckets_root,
        fut_buckets_root=fut_buckets_root,
        freq_min=args.freq_min,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    df = add_historical_vol_signal(df, hist_horizon_min=30, freq_min=args.freq_min)
    df = add_basis_specific_signals(df, freq_min=args.freq_min)

    if os.path.exists(pred_path):
        pred_df = pd.read_parquet(pred_path)
        df = merge_predicted_vol_signal(df, pred_df, pred_col="pred_y_30m", out_col="pred_vol_spot_30m")

    bt = run_basis_zscore_strategy(
        df=df,
        basis_lookback_min=args.basis_lookback_min,
        freq_min=args.freq_min,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        cost_bps=args.cost_bps,
        vol_filter_col=args.vol_filter_col,
        vol_filter_quantile=args.vol_filter_quantile,
        vol_filter_lookback_min=args.vol_filter_lookback_min,
        high_vol_only=args.high_only,
    )

    print(summarize_backtest(bt, freq_min=args.freq_min))
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
