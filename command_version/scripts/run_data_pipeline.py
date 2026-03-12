#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Optional

# 假设的 repo 结构是:
# repo/
#   src/
#     data_clean.py
#   scripts/
#     run_data_pipeline.py
#
# 这样写可以让脚本从 scripts/ 里正常 import src/

#python scripts/run_data_pipeline.py --market spot       #spot only, default path
#python scripts/run_data_pipeline.py --market futures    #future only, default path
#python scripts/run_data_pipeline.py --market futures --dates 20260118,20260119     #--dates (run only these dates)
#python scripts/run_data_pipeline.py --market spot --skip-stage2    #skip stage2 features
#python scripts/run_data_pipeline.py --market futures --skip-stage1 #skip stage1 
#python scripts/run_data_pipeline.py \   #customise output path
#  --market spot \
#  --buckets-out /tmp/my_buckets \
#  --features-out /tmp/my_features
#python scripts/run_data_pipeline.py --market futures --verbose #add verbose

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_clean import (  # noqa: E402
    compute_missing_days,
    stage1_build_buckets,
    stage2_build_features,
)


DEFAULTS = {
    "spot": {
        "base_dir": "/Volumes/profit/bitcoin_ticks/spot/btcusdt",
        "buckets_out": "/Volumes/profit/feature_store/buckets",
        "features_out": "/Volumes/profit/feature_store/features",
    },
    "futures": {
        "base_dir": "/Volumes/profit/bitcoin_ticks/futures/btcusdt",
        "buckets_out": "/Volumes/profit/feature_store_futures/buckets",
        "features_out": "/Volumes/profit/feature_store_futures/features",
    },
}


def parse_dates_arg(dates_arg: Optional[str]) -> Optional[List[str]]:
    """
    Parse comma-separated dates like:
      20251101,20251102,20251103
    """
    if dates_arg is None:
        return None
    dates = [x.strip() for x in dates_arg.split(",") if x.strip()]
    return dates or None


def resolve_paths(market: str, base_dir: Optional[str], buckets_out: Optional[str], features_out: Optional[str]):
    defaults = DEFAULTS[market]
    return {
        "base_dir": base_dir if base_dir else defaults["base_dir"],
        "buckets_out": buckets_out if buckets_out else defaults["buckets_out"],
        "features_out": features_out if features_out else defaults["features_out"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run raw tick -> bucket -> feature pipeline for BTC spot/futures."
    )

    # market toggle
    parser.add_argument(
        "--market",
        choices=["spot", "futures"],
        default="spot",
        help="Which market config to use. Default: spot",
    )

    # path overrides
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Override raw input base directory.",
    )
    parser.add_argument(
        "--buckets-out",
        type=str,
        default=None,
        help="Override bucket output directory.",
    )
    parser.add_argument(
        "--features-out",
        type=str,
        default=None,
        help="Override feature output directory.",
    )

    # pipeline config
    parser.add_argument(
        "--freq-min",
        type=int,
        default=1,
        help="Bucket frequency in minutes. Default: 1",
    )
    parser.add_argument(
        "--horizon-min",
        type=int,
        default=60,
        help="Forward RV horizon in minutes for stage2. Default: 60",
    )

    # date controls
    parser.add_argument(
        "--dates",
        type=str,
        default=None,
        help='Comma-separated explicit dates, e.g. "20251101,20251102". If not provided, compute missing days for stage1.',
    )
    parser.add_argument(
        "--stage2-dates",
        type=str,
        default=None,
        help='Optional comma-separated dates for stage2 only. If omitted, stage2 uses built stage1 dates.',
    )

    # stage toggle
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="Skip stage1 raw->buckets.",
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Skip stage2 buckets->features.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose mode for bad lines / diagnostics.",
    )

    args = parser.parse_args()

    paths = resolve_paths(
        market=args.market,
        base_dir=args.base_dir,
        buckets_out=args.buckets_out,
        features_out=args.features_out,
    )

    base_dir = paths["base_dir"]
    buckets_out = paths["buckets_out"]
    features_out = paths["features_out"]

    print("=" * 80)
    print("RUN DATA PIPELINE")
    print(f"market       : {args.market}")
    print(f"base_dir     : {base_dir}")
    print(f"buckets_out  : {buckets_out}")
    print(f"features_out : {features_out}")
    print(f"freq_min     : {args.freq_min}")
    print(f"horizon_min  : {args.horizon_min}")
    print("=" * 80)

    # stage1 dates
    explicit_dates = parse_dates_arg(args.dates)

    if explicit_dates is not None:
        dates_stage1 = explicit_dates
        print(f"[INFO] Using explicit stage1 dates: {dates_stage1}")
    else:
        dates_stage1 = compute_missing_days(base_dir, buckets_out, args.freq_min)
        print(f"[INFO] Computed missing stage1 dates: {len(dates_stage1)} day(s)")
        if len(dates_stage1) <= 20:
            print(f"[INFO] Missing dates: {dates_stage1}")

    # run stage1
    if not args.skip_stage1:
        if not dates_stage1:
            print("[Stage1] No dates to process.")
        else:
            stage1_build_buckets(
                base_dir=base_dir,
                out_dir=buckets_out,
                freq_min=args.freq_min,
                dates=dates_stage1,
                verbose=args.verbose,
            )
    else:
        print("[Stage1] Skipped by flag.")

    # stage2 dates
    explicit_stage2_dates = parse_dates_arg(args.stage2_dates)
    if explicit_stage2_dates is not None:
        dates_stage2 = explicit_stage2_dates
        print(f"[INFO] Using explicit stage2 dates: {dates_stage2}")
    else:
        # 如果给了 --dates，就优先用这些日期做 stage2
        candidate_dates = explicit_dates if explicit_dates is not None else dates_stage1

        # 只保留 bucket 已经存在的日期，避免 FileNotFoundError
        dates_stage2 = [
            d for d in candidate_dates
            if os.path.exists(
                os.path.join(
                    buckets_out,
                    f"freq={args.freq_min}min",
                    f"date={d}",
                    "buckets.parquet",
                )
            )
        ]
        print(f"[INFO] Stage2 dates with existing buckets: {len(dates_stage2)} day(s)")
        if len(dates_stage2) <= 20:
            print(f"[INFO] Stage2 dates: {dates_stage2}")

    # run stage2
    if not args.skip_stage2:
        if not dates_stage2:
            print("[Stage2] No bucket files available to process.")
        else:
            stage2_build_features(
                buckets_dir=buckets_out,
                out_dir=features_out,
                freq_min=args.freq_min,
                horizon_min=args.horizon_min,
                dates=dates_stage2,
            )
    else:
        print("[Stage2] Skipped by flag.")

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
