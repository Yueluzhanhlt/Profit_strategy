import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# =========================================================
# Feature loading
# =========================================================
def _date_range_yyyymmdd(start_date: str, end_date: str) -> List[str]:
    dates = pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq="D")
    return [d.strftime("%Y%m%d") for d in dates]


def load_feature_range(
    features_root: str,
    freq_min: int,
    horizon_min: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Load per-day feature parquet files from:
      {features_root}/freq={freq_min}min/h={horizon_min}min/date=YYYYMMDD/features.parquet
    """
    dates = _date_range_yyyymmdd(start_date, end_date)

    frames = []
    for d in dates:
        path = os.path.join(
            features_root,
            f"freq={freq_min}min",
            f"h={horizon_min}min",
            f"date={d}",
            "features.parquet",
        )
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))

    if not frames:
        raise FileNotFoundError(
            f"No feature parquet files found under {features_root} "
            f"for horizon={horizon_min}, range=[{start_date}, {end_date}]"
        )

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = (
        df.sort_values("timestamp")
          .drop_duplicates(subset=["timestamp"], keep="last")
          .reset_index(drop=True)
    )
    return df


def infer_feature_columns(df: pd.DataFrame, target_col: str = "y") -> List[str]:
    """
    Infer modeling features by excluding timestamp / labels / obvious helper columns.
    """
    exclude = {
        "timestamp",
        "bucket_ms",
        "date",
        "ret",
        "ret2",
        "rv_fwd",
        "y",
    }

    feature_cols = [c for c in df.columns if c not in exclude]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


# =========================================================
# Walk-forward calendar
# =========================================================
@dataclass
class FoldSpec:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    horizon_min: int


def make_walkforward_folds(
    available_days: List[str],
    train_days: int,
    test_days: int,
    horizon_min: int,
) -> List[FoldSpec]:
    """
    Build day-based walk-forward splits.

    Example:
      train_days=30, test_days=1
    """
    days = sorted(pd.to_datetime(available_days, format="%Y%m%d"))

    folds: List[FoldSpec] = []
    i = train_days
    while i + test_days <= len(days):
        train_slice = days[i - train_days : i]
        test_slice = days[i : i + test_days]

        folds.append(
            FoldSpec(
                train_start=train_slice[0].strftime("%Y%m%d"),
                train_end=train_slice[-1].strftime("%Y%m%d"),
                test_start=test_slice[0].strftime("%Y%m%d"),
                test_end=test_slice[-1].strftime("%Y%m%d"),
                horizon_min=horizon_min,
            )
        )
        i += test_days

    return folds


# =========================================================
# Model training / prediction
# =========================================================
def build_ridge_model(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def qlike_loss_from_logrv(y_true_logrv: np.ndarray, y_pred_logrv: np.ndarray) -> float:
    """
    QLIKE on RV scale.

    true_rv = exp(y_true), pred_rv = exp(y_pred)
    QLIKE = true/pred - log(true/pred) - 1
    """
    true_rv = np.exp(y_true_logrv)
    pred_rv = np.exp(y_pred_logrv)

    eps = 1e-12
    pred_rv = np.maximum(pred_rv, eps)
    ratio = true_rv / pred_rv
    qlike = ratio - np.log(ratio) - 1.0
    return float(np.mean(qlike))


def prepare_xy(
    df: pd.DataFrame,
    target_col: str = "y",
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if feature_cols is None:
        feature_cols = infer_feature_columns(df, target_col=target_col)

    keep_cols = ["timestamp", target_col] + feature_cols
    tmp = df[keep_cols].copy()

    tmp = tmp.replace([np.inf, -np.inf], np.nan)
    tmp = tmp.dropna(subset=[target_col])

    # For simple linear model, drop rows with missing features
    tmp = tmp.dropna(subset=feature_cols)

    X = tmp[feature_cols]
    y = tmp[target_col]
    return X, y, feature_cols


def run_single_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon_min: int,
    target_col: str = "y",
    feature_cols: Optional[List[str]] = None,
    alpha: float = 1.0,
) -> Tuple[pd.Series, pd.DataFrame]:
    X_train, y_train, feature_cols = prepare_xy(train_df, target_col=target_col, feature_cols=feature_cols)
    X_test, y_test, _ = prepare_xy(test_df, target_col=target_col, feature_cols=feature_cols)

    test_idx = test_df.loc[X_test.index, "timestamp"].reset_index(drop=True)

    model = build_ridge_model(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    fold_summary = pd.Series({
        "n_train": len(X_train),
        "n_test": len(X_test),
        "rmse": safe_rmse(y_test.to_numpy(), y_pred),
        "corr": safe_corr(y_test.to_numpy(), y_pred),
        "qlike": qlike_loss_from_logrv(y_test.to_numpy(), y_pred),
        "H_min": horizon_min,
    })

    preds = pd.DataFrame({
        "timestamp": test_idx,
        "y_true": y_test.reset_index(drop=True),
        "y_pred": y_pred,
        "y_col": f"y_{horizon_min}m",
        "H_min": horizon_min,
    })

    return fold_summary, preds


# =========================================================
# Pipeline
# =========================================================
def list_available_feature_days(
    features_root: str,
    freq_min: int,
    horizon_min: int,
) -> List[str]:
    pat = os.path.join(
        features_root,
        f"freq={freq_min}min",
        f"h={horizon_min}min",
        "date=*",
        "features.parquet",
    )
    return sorted({
        p.split("date=")[-1].split(os.sep)[0]
        for p in glob.glob(pat)
    })


def restrict_days(days: List[str], start: Optional[str], end: Optional[str]) -> List[str]:
    out = days
    if start is not None:
        out = [d for d in out if d >= start]
    if end is not None:
        out = [d for d in out if d <= end]
    return out


def run_pipeline_single_horizon(
    features_root: str,
    out_dir: Optional[str],
    freq_min: float,
    train_days: int,
    test_days: int,
    horizon_min: int,
    start: Optional[str] = None,
    end: Optional[str] = None,
    alpha: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run walk-forward OOS pipeline for one horizon.
    Returns:
      folds_df, preds_df
    """
    freq_min_int = int(freq_min)

    available_days = list_available_feature_days(features_root, freq_min_int, horizon_min)
    available_days = restrict_days(available_days, start, end)

    if len(available_days) < train_days + test_days:
        raise ValueError(
            f"Not enough days for horizon={horizon_min}. "
            f"Need at least {train_days + test_days}, got {len(available_days)}."
        )

    folds = make_walkforward_folds(
        available_days=available_days,
        train_days=train_days,
        test_days=test_days,
        horizon_min=horizon_min,
    )

    fold_rows = []
    pred_rows = []

    for fold_id, fold in enumerate(folds, start=1):
        train_df = load_feature_range(
            features_root=features_root,
            freq_min=freq_min_int,
            horizon_min=horizon_min,
            start_date=fold.train_start,
            end_date=fold.train_end,
        )
        test_df = load_feature_range(
            features_root=features_root,
            freq_min=freq_min_int,
            horizon_min=horizon_min,
            start_date=fold.test_start,
            end_date=fold.test_end,
        )

        fold_summary, preds = run_single_fold(
            train_df=train_df,
            test_df=test_df,
            horizon_min=horizon_min,
            target_col="y",
            feature_cols=None,
            alpha=alpha,
        )

        fold_summary["fold_id"] = fold_id
        fold_summary["train_start"] = fold.train_start
        fold_summary["train_end"] = fold.train_end
        fold_summary["test_start"] = fold.test_start
        fold_summary["test_end"] = fold.test_end

        preds["fold_id"] = fold_id

        fold_rows.append(fold_summary)
        pred_rows.append(preds)

    folds_df = pd.DataFrame(fold_rows)
    preds_df = pd.concat(pred_rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

        folds_path = os.path.join(out_dir, f"folds_h{horizon_min}m.parquet")
        preds_path = os.path.join(out_dir, f"preds_h{horizon_min}m.parquet")

        folds_df.to_parquet(folds_path, index=False)
        preds_df.to_parquet(preds_path, index=False)

    return folds_df, preds_df


def summarize_horizon_preds(preds_df: pd.DataFrame) -> pd.Series:
    y_true = preds_df["y_true"].to_numpy()
    y_pred = preds_df["y_pred"].to_numpy()

    return pd.Series({
        "n": len(preds_df),
        "rmse": safe_rmse(y_true, y_pred),
        "corr": safe_corr(y_true, y_pred),
        "qlike": qlike_loss_from_logrv(y_true, y_pred),
        "H_min": preds_df["H_min"].iloc[0],
    })


def run_pipeline_v3(
    buckets_root: str,
    out_dir: Optional[str],
    freq_min: float,
    train_days: int,
    test_days: int,
    horizons: Tuple[int, ...] = (30, 90),
    start: Optional[str] = None,
    end: Optional[str] = None,
    w_combo: float = 0.7,
    combo_short: int = 30,
    combo_long: int = 90,
    alpha: float = 1.0,
):
    """
    Main multi-horizon walk-forward OOS pipeline.

    Returns:
      summary_df, folds_df, preds_df, combo_df
    """
    all_fold_dfs = []
    all_pred_dfs = []
    summary_rows = []

    for h in horizons:
        folds_df, preds_df = run_pipeline_single_horizon(
            features_root=buckets_root,
            out_dir=out_dir,
            freq_min=freq_min,
            train_days=train_days,
            test_days=test_days,
            horizon_min=h,
            start=start,
            end=end,
            alpha=alpha,
        )

        horizon_summary = summarize_horizon_preds(preds_df)
        summary_rows.append(horizon_summary)

        all_fold_dfs.append(folds_df)
        all_pred_dfs.append(preds_df)

    folds_all = pd.concat(all_fold_dfs, ignore_index=True).sort_values(["H_min", "fold_id"]).reset_index(drop=True)
    preds_all = pd.concat(all_pred_dfs, ignore_index=True).sort_values(["timestamp", "H_min"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("H_min").reset_index(drop=True)

    # Simple combination of two horizons (if both available)
    combo_df = pd.DataFrame()
    if combo_short in horizons and combo_long in horizons:
        pred_short = preds_all.loc[preds_all["H_min"] == combo_short, ["timestamp", "y_true", "y_pred"]].copy()
        pred_long = preds_all.loc[preds_all["H_min"] == combo_long, ["timestamp", "y_pred"]].copy()

        pred_short = pred_short.rename(columns={"y_pred": f"pred_{combo_short}m"})
        pred_long = pred_long.rename(columns={"y_pred": f"pred_{combo_long}m"})

        combo_df = pred_short.merge(pred_long, on="timestamp", how="inner")
        combo_df["y_pred_combo"] = (
            w_combo * combo_df[f"pred_{combo_short}m"] +
            (1.0 - w_combo) * combo_df[f"pred_{combo_long}m"]
        )

        combo_df["rmse_combo"] = np.nan  # placeholder column for convenience

        if out_dir is not None:
            combo_path = os.path.join(out_dir, "preds_combo.parquet")
            combo_df.to_parquet(combo_path, index=False)

    if out_dir is not None:
        summary_path = os.path.join(out_dir, "summary.parquet")
        folds_path = os.path.join(out_dir, "folds_all.parquet")
        preds_path = os.path.join(out_dir, "preds_all.parquet")

        summary_df.to_parquet(summary_path, index=False)
        folds_all.to_parquet(folds_path, index=False)
        preds_all.to_parquet(preds_path, index=False)

    return summary_df, folds_all, preds_all, combo_df
