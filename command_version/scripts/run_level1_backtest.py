#!/usr/bin/env python3
#python scripts/run_level1_backtest.py
#python scripts/run_level1_backtest.py \
#  --initial-capital 10000 \
#  --max-position-fraction 0.10
#python scripts/run_level1_backtest.py \
#  --regime-signal-col basis_rv_30m \
#  --regime-quantile 0.80 \
#  --entry-z 1.5 \
#  --exit-z 1.0 \
#  --basis-lookback-min 240 \
#  --fee-bps 2.5 \
#  --slippage-bps 2.5
#python scripts/run_level1_backtest.py \
#  --regime-signal-col basis_vol_30m \
#  --regime-quantile 0.95
import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.basis_strategy import (  # noqa: E402
    prepare_basis_dataframe,
    add_basis_specific_signals,
)
from src.level1_backtest import (  # noqa: E402
    run_level1_backtest,
    summarize_level1,
)


DEFAULTS = {
    "default": {
        "spot_buckets_root": "/Volumes/profit/feature_store/buckets",
        "fut_buckets_root": "/Volumes/profit/feature_store_futures/buckets",
    }
}


def main():
    parser = argparse.ArgumentParser(
        description="Run Level 1 execution-aware basis backtest."
    )

    parser.add_argument("--spot-buckets-root", type=str, default=None)
    parser.add_argument("--fut-buckets-root", type=str, default=None)

    parser.add_argument("--freq-min", type=int, default=1)
    parser.add_argument("--start-date", type=str, default="2025-11-01")
    parser.add_argument("--end-date", type=str, default="2026-03-10")

    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--max-position-fraction", type=float, default=0.05)

    parser.add_argument("--basis-lookback-min", type=int, default=120)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=1.0)
    parser.add_argument("--max-holding-min", type=int, default=120)

    parser.add_argument("--regime-signal-col", type=str, default="basis_rv_30m")
    parser.add_argument("--regime-quantile", type=float, default=0.95)
    parser.add_argument("--regime-lookback-min", type=int, default=30 * 24 * 60)
    parser.add_argument("--low-only", action="store_true", help="Use low-only instead of default high-only regime.")

    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.5)

    args = parser.parse_args()

    defaults = DEFAULTS["default"]
    spot_buckets_root = args.spot_buckets_root or defaults["spot_buckets_root"]
    fut_buckets_root = args.fut_buckets_root or defaults["fut_buckets_root"]

    print("=" * 80)
    print("RUN LEVEL1 BACKTEST")
    print(f"spot_buckets_root    : {spot_buckets_root}")
    print(f"fut_buckets_root     : {fut_buckets_root}")
    print(f"date range           : {args.start_date} -> {args.end_date}")
    print(f"initial_capital      : {args.initial_capital}")
    print(f"max_position_fraction: {args.max_position_fraction}")
    print(f"basis_lookback_min   : {args.basis_lookback_min}")
    print(f"entry/exit z         : {args.entry_z} / {args.exit_z}")
    print(f"max_holding_min      : {args.max_holding_min}")
    print(f"regime_signal_col    : {args.regime_signal_col}")
    print(f"regime_quantile      : {args.regime_quantile}")
    print(f"fee/slippage bps     : {args.fee_bps} / {args.slippage_bps}")
    print("=" * 80)

    df = prepare_basis_dataframe(
        spot_buckets_root=spot_buckets_root,
        fut_buckets_root=fut_buckets_root,
        freq_min=args.freq_min,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    df = add_basis_specific_signals(df, freq_min=args.freq_min)

    bt = run_level1_backtest(
        df=df,
        initial_capital=args.initial_capital,
        max_position_fraction=args.max_position_fraction,
        basis_lookback_min=args.basis_lookback_min,
        freq_min=args.freq_min,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        max_holding_min=args.max_holding_min,
        regime_signal_col=args.regime_signal_col,
        regime_quantile=args.regime_quantile,
        regime_lookback_min=args.regime_lookback_min,
        high_only=(not args.low_only),
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
    )

    print(summarize_level1(bt, freq_min=args.freq_min))
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
