#!/usr/bin/env python3
#python scripts/run_vol_model.py --market spot
#python scripts/run_vol_model.py --market spot
#python scripts/run_vol_model.py --market spot --start 20251001 --end 20251231
#python scripts/run_vol_model.py --market spot --horizons 30
#python scripts/run_vol_model.py --market futures --out-dir /tmp/wf_out

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.vol_model import run_pipeline_v3  # noqa: E402


DEFAULTS = {
    "spot": {
        "features_root": "/Volumes/profit/feature_store/features",
        "out_dir": "/Volumes/profit/feature_store/wf_out_v3",
    },
    "futures": {
        "features_root": "/Volumes/profit/feature_store_futures/features",
        "out_dir": "/Volumes/profit/feature_store_futures/wf_out_v3",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Run walk-forward OOS volatility model pipeline."
    )

    parser.add_argument(
        "--market",
        choices=["spot", "futures"],
        default="spot",
        help="Which market config to use. Default: spot",
    )
    parser.add_argument(
        "--features-root",
        type=str,
        default=None,
        help="Override feature root directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )

    parser.add_argument("--freq-min", type=float, default=1.0)
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=1)
    parser.add_argument(
        "--horizons",
        type=str,
        default="30,90",
        help='Comma-separated horizons, e.g. "30,90"',
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)

    parser.add_argument("--w-combo", type=float, default=0.7)
    parser.add_argument("--combo-short", type=int, default=30)
    parser.add_argument("--combo-long", type=int, default=90)
    parser.add_argument("--alpha", type=float, default=1.0)

    args = parser.parse_args()

    defaults = DEFAULTS[args.market]
    features_root = args.features_root or defaults["features_root"]
    out_dir = args.out_dir or defaults["out_dir"]
    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())

    print("=" * 80)
    print("RUN VOL MODEL")
    print(f"market       : {args.market}")
    print(f"features_root: {features_root}")
    print(f"out_dir      : {out_dir}")
    print(f"freq_min     : {args.freq_min}")
    print(f"train_days   : {args.train_days}")
    print(f"test_days    : {args.test_days}")
    print(f"horizons     : {horizons}")
    print(f"start/end    : {args.start} -> {args.end}")
    print("=" * 80)

    summary, folds, preds, combo = run_pipeline_v3(
        buckets_root=features_root,
        out_dir=out_dir,
        freq_min=args.freq_min,
        train_days=args.train_days,
        test_days=args.test_days,
        horizons=horizons,
        start=args.start,
        end=args.end,
        w_combo=args.w_combo,
        combo_short=args.combo_short,
        combo_long=args.combo_long,
        alpha=args.alpha,
    )

    print(summary)
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
