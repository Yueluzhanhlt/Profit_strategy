import os
import io
import json
import glob
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List

import numpy as np
import pandas as pd
import zstandard as zstd


# =========================
# I/O
# =========================
def iter_jsonl_zst(path: str, verbose: bool = True) -> Iterable[Dict[str, Any]]:
    """
    Stream-read a .jsonl.zst file and yield JSON objects line by line.

    Bad / malformed lines are skipped instead of crashing the full job.
    """
    bad_lines = 0

    with open(path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

            for line_no, line in enumerate(text, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    bad_lines += 1
                    if verbose:
                        preview = line[:300].replace("\n", " ")
                        print(
                            f"[BAD JSON] file={path} line={line_no} "
                            f"err={e} preview={preview}"
                        )
                except Exception as e:
                    bad_lines += 1
                    if verbose:
                        preview = line[:300].replace("\n", " ")
                        print(
                            f"[BAD LINE] file={path} line={line_no} "
                            f"err={type(e).__name__}: {e} preview={preview}"
                        )

    if bad_lines > 0 and verbose:
        print(f"[WARN] file={path} skipped_bad_lines={bad_lines}")


# =========================
# Helpers
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_freq_ms(freq_min: int) -> int:
    return int(freq_min) * 60_000


def floor_bucket_ms(ts_ms: int, freq_ms: int) -> int:
    return (ts_ms // freq_ms) * freq_ms


def ms_to_utc_datetime(ms: int) -> pd.Timestamp:
    return pd.to_datetime(ms, unit="ms", utc=True)


# =========================
# Stage 1: Tick -> Buckets
# =========================
@dataclass
class AggTradeBucket:
    last_price: float = np.nan
    sum_qty: float = 0.0
    sum_notional: float = 0.0
    n_trades: int = 0
    buy_qty: float = 0.0
    sell_qty: float = 0.0

    def update(self, price: float, qty: float, is_seller_maker: bool) -> None:
        self.last_price = price
        self.sum_qty += qty
        self.sum_notional += price * qty
        self.n_trades += 1

        if is_seller_maker:
            self.sell_qty += qty
        else:
            self.buy_qty += qty

    def to_row(self) -> Dict[str, Any]:
        vwap = self.sum_notional / self.sum_qty if self.sum_qty > 0 else np.nan
        buy_ratio = self.buy_qty / self.sum_qty if self.sum_qty > 0 else np.nan
        return {
            "px_last": self.last_price,
            "px_vwap": vwap,
            "vol_qty": self.sum_qty,
            "vol_notional": self.sum_notional,
            "n_trades": self.n_trades,
            "buy_qty": self.buy_qty,
            "sell_qty": self.sell_qty,
            "buy_ratio": buy_ratio,
        }


@dataclass
class BookTickerBucket:
    sum_spread: float = 0.0
    sum_rel_spread: float = 0.0
    sum_imb: float = 0.0
    n: int = 0

    last_mid: float = np.nan
    last_imb: float = np.nan

    def update(self, bid: float, ask: float, bidq: float, askq: float) -> None:
        mid = (bid + ask) / 2.0
        spread = ask - bid
        rel_spread = spread / mid if mid != 0 else np.nan

        denom = bidq + askq
        imb = (bidq - askq) / denom if denom != 0 else np.nan

        if not np.isnan(spread):
            self.sum_spread += spread
        if not np.isnan(rel_spread):
            self.sum_rel_spread += rel_spread
        if not np.isnan(imb):
            self.sum_imb += imb

        self.n += 1
        self.last_mid = mid
        self.last_imb = imb

    def to_row(self) -> Dict[str, Any]:
        if self.n <= 0:
            return {
                "mid_last": np.nan,
                "spread_mean": np.nan,
                "rel_spread_mean": np.nan,
                "imbalance_mean": np.nan,
                "imbalance_last": np.nan,
            }

        return {
            "mid_last": self.last_mid,
            "spread_mean": self.sum_spread / self.n,
            "rel_spread_mean": self.sum_rel_spread / self.n,
            "imbalance_mean": self.sum_imb / self.n,
            "imbalance_last": self.last_imb,
        }


def aggregate_aggtrade_file(path: str, freq_ms: int, verbose: bool = False) -> Dict[int, AggTradeBucket]:
    buckets: Dict[int, AggTradeBucket] = {}

    for obj in iter_jsonl_zst(path, verbose=verbose):
        t = obj.get("T")
        p = obj.get("p")
        q = obj.get("q")
        m = obj.get("m")

        if t is None or p is None or q is None or m is None:
            continue

        try:
            ts_ms = int(t)
            price = float(p)
            qty = float(q)
            is_seller_maker = bool(m)
        except Exception:
            continue

        bucket_ms = floor_bucket_ms(ts_ms, freq_ms)
        if bucket_ms not in buckets:
            buckets[bucket_ms] = AggTradeBucket()

        buckets[bucket_ms].update(price, qty, is_seller_maker)

    return buckets


def aggregate_bookticker_file(path: str, freq_ms: int, verbose: bool = False) -> Dict[int, BookTickerBucket]:
    buckets: Dict[int, BookTickerBucket] = {}

    for obj in iter_jsonl_zst(path, verbose=verbose):
        # Compatible across spot / futures naming
        t = obj.get("event_time", obj.get("T", obj.get("E")))
        if t is None:
            continue

        bid_s = obj.get("best_bid_price", obj.get("b"))
        ask_s = obj.get("best_ask_price", obj.get("a"))
        bidq_s = obj.get("best_bid_qty", obj.get("B"))
        askq_s = obj.get("best_ask_qty", obj.get("A"))

        if bid_s is None or ask_s is None or bidq_s is None or askq_s is None:
            continue

        try:
            ts_ms = int(t)
            bid = float(bid_s)
            ask = float(ask_s)
            bidq = float(bidq_s)
            askq = float(askq_s)
        except Exception:
            continue

        bucket_ms = floor_bucket_ms(ts_ms, freq_ms)
        if bucket_ms not in buckets:
            buckets[bucket_ms] = BookTickerBucket()

        buckets[bucket_ms].update(bid, ask, bidq, askq)

    return buckets


def build_buckets_for_day(day_dir: str, freq_min: int, verbose: bool = False) -> pd.DataFrame:
    """
    Expected directory structure:
      day_dir/HH/aggtrade/*.jsonl.zst
      day_dir/HH/bookticker/*.jsonl.zst
    """
    freq_ms = parse_freq_ms(freq_min)

    agg_all: Dict[int, AggTradeBucket] = {}
    book_all: Dict[int, BookTickerBucket] = {}

    hour_dirs = sorted([p for p in glob.glob(os.path.join(day_dir, "*")) if os.path.isdir(p)])
    n_agg_files = 0
    n_book_files = 0

    for hdir in hour_dirs:
        agg_files = sorted(glob.glob(os.path.join(hdir, "aggtrade", "*.jsonl.zst")))
        book_files = sorted(glob.glob(os.path.join(hdir, "bookticker", "*.jsonl.zst")))

        n_agg_files += len(agg_files)
        n_book_files += len(book_files)

        for f in agg_files:
            agg_buckets = aggregate_aggtrade_file(f, freq_ms, verbose=verbose)
            for k, v in agg_buckets.items():
                if k not in agg_all:
                    agg_all[k] = v
                else:
                    cur = agg_all[k]
                    cur.last_price = v.last_price if not np.isnan(v.last_price) else cur.last_price
                    cur.sum_qty += v.sum_qty
                    cur.sum_notional += v.sum_notional
                    cur.n_trades += v.n_trades
                    cur.buy_qty += v.buy_qty
                    cur.sell_qty += v.sell_qty

        for f in book_files:
            book_buckets = aggregate_bookticker_file(f, freq_ms, verbose=verbose)
            for k, v in book_buckets.items():
                if k not in book_all:
                    book_all[k] = v
                else:
                    cur = book_all[k]
                    cur.sum_spread += v.sum_spread
                    cur.sum_rel_spread += v.sum_rel_spread
                    cur.sum_imb += v.sum_imb
                    cur.n += v.n
                    cur.last_mid = v.last_mid if not np.isnan(v.last_mid) else cur.last_mid
                    cur.last_imb = v.last_imb if not np.isnan(v.last_imb) else cur.last_imb

    keys = sorted(set(agg_all.keys()) | set(book_all.keys()))

    cols = ["bucket_ms", "timestamp"]
    cols += list(AggTradeBucket().to_row().keys())
    cols += list(BookTickerBucket().to_row().keys())

    if len(keys) == 0:
        print(
            f"[Stage1] SKIP empty day={os.path.basename(day_dir)} "
            f"(hour_dirs={len(hour_dirs)} agg_files={n_agg_files} book_files={n_book_files})"
        )
        return pd.DataFrame(columns=cols)

    rows = []
    for k in keys:
        row = {"bucket_ms": k, "timestamp": ms_to_utc_datetime(k)}

        if k in agg_all:
            row.update(agg_all[k].to_row())
        else:
            row.update({c: np.nan for c in AggTradeBucket().to_row().keys()})

        if k in book_all:
            row.update(book_all[k].to_row())
        else:
            row.update({c: np.nan for c in BookTickerBucket().to_row().keys()})

        rows.append(row)

    return pd.DataFrame(rows, columns=cols).sort_values("timestamp").reset_index(drop=True)


def stage1_build_buckets(
    base_dir: str,
    out_dir: str,
    freq_min: int,
    dates: List[str] | None = None,
    verbose: bool = False,
) -> None:
    """
    Build per-day bucket parquet files.

    Output path:
      {out_dir}/freq={freq_min}min/date=YYYYMMDD/buckets.parquet
    """
    ensure_dir(out_dir)

    if dates is None:
        day_dirs = sorted([p for p in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(p)])
    else:
        day_dirs = [os.path.join(base_dir, d) for d in dates]

    for day_dir in day_dirs:
        date = os.path.basename(day_dir)

        if not os.path.isdir(day_dir):
            print(f"[Stage1] SKIP missing day_dir={day_dir}")
            continue
        if not date.isdigit():
            continue

        df = build_buckets_for_day(day_dir, freq_min=freq_min, verbose=verbose)
        if df.empty:
            continue

        out_path = os.path.join(out_dir, f"freq={freq_min}min", f"date={date}", "buckets.parquet")
        ensure_dir(os.path.dirname(out_path))
        df.to_parquet(out_path, index=False)
        print(f"[Stage1] wrote {out_path} rows={len(df)}")


# =========================
# Stage 2: Buckets -> Features
# =========================
def build_features_from_buckets(
    df: pd.DataFrame,
    freq_min: int,
    horizon_min: int,
    windows_min: List[int],
) -> pd.DataFrame:
    """
    Build minute-level features and forward realized variance label from bucket data.
    """
    df = (
        df.sort_values("timestamp")
          .drop_duplicates(subset=["timestamp"], keep="last")
          .reset_index(drop=True)
    )

    # Light forward-fill for book features across tiny gaps
    for c in ["mid_last", "spread_mean", "rel_spread_mean", "imbalance_mean", "imbalance_last"]:
        if c in df.columns:
            df[c] = df[c].ffill(limit=2)

    px = df["mid_last"].copy()
    px = px.fillna(df["px_last"]).ffill()

    df["ret"] = np.log(px).diff()
    df["ret2"] = df["ret"] ** 2

    step = int(freq_min)
    horizon_steps = int(round(horizon_min / step))
    if horizon_steps <= 0:
        raise ValueError("horizon_min must be >= freq_min")

    # Forward realized variance over next H steps
    df["rv_fwd"] = df["ret2"].rolling(horizon_steps).sum().shift(-horizon_steps)
    df["y"] = np.log(df["rv_fwd"] + 1e-12)

    # HAR-like past RV windows
    for wmin in windows_min:
        w = int(round(wmin / step))
        if w > 0:
            df[f"rv_{wmin}m_past"] = df["ret2"].rolling(w).sum()

    # Extra rolling microstructure features
    for wmin in [5, 15, 60, 240]:
        if wmin < step:
            continue
        w = int(round(wmin / step))

        if "rel_spread_mean" in df.columns:
            df[f"spread_{wmin}m"] = df["rel_spread_mean"].rolling(w).mean()
        if "imbalance_mean" in df.columns:
            df[f"imb_{wmin}m"] = df["imbalance_mean"].rolling(w).mean()
        if "vol_notional" in df.columns:
            df[f"vol_{wmin}m"] = df["vol_notional"].rolling(w).sum()
        if "n_trades" in df.columns:
            df[f"ntr_{wmin}m"] = df["n_trades"].rolling(w).sum()

    # UTC seasonality
    minute_of_day = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["sin_tod"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df["cos_tod"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

    return df


def stage2_build_features(
    buckets_dir: str,
    out_dir: str,
    freq_min: int,
    horizon_min: int,
    dates: List[str] | None = None,
) -> None:
    """
    Read bucket parquet files and write per-day feature parquet files.

    Input:
      {buckets_dir}/freq={freq_min}min/date=YYYYMMDD/buckets.parquet

    Output:
      {out_dir}/freq={freq_min}min/h={horizon_min}min/date=YYYYMMDD/features.parquet
    """
    ensure_dir(out_dir)

    if dates is None:
        pattern = os.path.join(buckets_dir, f"freq={freq_min}min", "date=*", "buckets.parquet")
        files = sorted(glob.glob(pattern))
    else:
        files = []
        for d in dates:
            f = os.path.join(buckets_dir, f"freq={freq_min}min", f"date={d}", "buckets.parquet")
            if os.path.exists(f):
                files.append(f)
        files = sorted(files)

    if not files:
        raise FileNotFoundError("No bucket parquet files found for given settings.")

    windows_min = [5, 15, 60, 240]

    all_df = [pd.read_parquet(f) for f in files]
    full = pd.concat(all_df, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    feat = build_features_from_buckets(
        full,
        freq_min=freq_min,
        horizon_min=horizon_min,
        windows_min=windows_min,
    )

    feat["date"] = feat["timestamp"].dt.strftime("%Y%m%d")
    for date, g in feat.groupby("date", sort=True):
        if dates is not None and date not in dates:
            continue

        out_path = os.path.join(
            out_dir,
            f"freq={freq_min}min",
            f"h={horizon_min}min",
            f"date={date}",
            "features.parquet",
        )
        ensure_dir(os.path.dirname(out_path))
        g = g.drop(columns=["date"])
        g.to_parquet(out_path, index=False)
        print(f"[Stage2] wrote {out_path} rows={len(g)}")


# =========================
# Utilities
# =========================
def list_available_days(base_dir: str) -> List[str]:
    return sorted([
        os.path.basename(p)
        for p in glob.glob(os.path.join(base_dir, "*"))
        if os.path.isdir(p)
    ])


def list_built_days(buckets_out: str, freq_min: int) -> List[str]:
    pattern = os.path.join(buckets_out, f"freq={freq_min}min", "date=*", "buckets.parquet")
    return sorted({
        p.split("date=")[-1].split(os.sep)[0]
        for p in glob.glob(pattern)
    })


def compute_missing_days(base_dir: str, buckets_out: str, freq_min: int) -> List[str]:
    have_raw = set(d for d in list_available_days(base_dir) if d.isdigit())
    have_buckets = set(list_built_days(buckets_out, freq_min))
    return sorted(have_raw - have_buckets)
