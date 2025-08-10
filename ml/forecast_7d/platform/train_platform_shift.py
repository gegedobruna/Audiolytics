# ml/forecast_7d/platform/train_platform_shift.py

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------------------------------------------------
# Paths (script lives in: .../Audiolytics/ml/forecast_7d/platform/)
# Data lives in:          .../Audiolytics/data/derived/sequences_with_clusters_platforms.parquet
# Outputs will be written in: same folder as this script, under "outputs/"
# ---------------------------------------------------------------------
HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parents[2]  # .../Audiolytics
ENRICHED_EVENTS = PROJECT_ROOT / "data" / "derived" / "sequences_with_clusters_platforms.parquet"
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Less noisy statsmodels logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _infer_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """Infer date and platform column names from the dataframe."""
    date_candidates = ["date", "day", "ts", "timestamp", "datetime"]
    plat_candidates = ["platform", "source", "service", "app", "provider"]

    date_col = next((c for c in date_candidates if c in df.columns), None)
    plat_col = next((c for c in plat_candidates if c in df.columns), None)

    if date_col is None:
        raise KeyError(
            f"Could not find a date column. Tried: {date_candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    if plat_col is None:
        raise KeyError(
            f"Could not find a platform column. Tried: {plat_candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return date_col, plat_col


def seasonal_naive(y, horizon: int = 7, season: int = 7) -> np.ndarray:
    """Seasonal naive forecast. Robust to empty/short inputs."""
    y = pd.Series(y)
    if len(y) == 0:
        return np.zeros(horizon, dtype=float)
    if season <= 0:
        season = 1
    if len(y) < season:
        return np.repeat(float(y.iloc[-1]), horizon)

    last_season = y.iloc[-season:].to_numpy(dtype=float)
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon]


def load_daily_platform_shares() -> pd.DataFrame:
    """Load parquet and compute daily platform shares."""
    print(">>> __file__:", HERE.joinpath(Path(__file__).name).resolve())
    print(">>> PROJECT_ROOT:", PROJECT_ROOT)
    print(">>> ENRICHED_EVENTS:", ENRICHED_EVENTS, "exists?", ENRICHED_EVENTS.exists())
    sys.stdout.flush()

    if not ENRICHED_EVENTS.exists():
        raise FileNotFoundError(
            f"Parquet not found at {ENRICHED_EVENTS}. "
            f"Fix ENRICHED_EVENTS if your path differs."
        )

    try:
        df = pd.read_parquet(ENRICHED_EVENTS)
    except Exception as e:
        print("!!! Failed to read parquet:", type(e).__name__, e)
        raise

    print("Loaded parquet shape:", df.shape)
    print(df.head(3))
    sys.stdout.flush()

    if len(df) == 0:
        # Return empty to signal caller
        return pd.DataFrame(columns=["date", "platform", "plays", "share"])

    date_col, plat_col = _infer_cols(df)

    # Normalize to daily date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.floor("D")
    df = df.dropna(subset=[date_col])

    # Aggregate: plays per day per platform (row count proxy)
    counts = df.groupby([date_col, plat_col]).size().reset_index(name="plays")
    # compute daily shares
    totals = counts.groupby(date_col)["plays"].transform("sum")
    counts["share"] = counts["plays"] / totals

    counts = counts.rename(columns={date_col: "date", plat_col: "platform"})
    return counts[["date", "platform", "plays", "share"]]


def ensure_daily_index(series: pd.Series) -> pd.Series:
    """Ensure the series has a complete daily DatetimeIndex and fill missing with 0.0."""
    if series.empty:
        return series

    idx = series.index
    start, end = idx.min(), idx.max()
    full_idx = pd.date_range(start, end, freq="D")
    s = series.reindex(full_idx)
    s = s.fillna(0.0).astype(float)
    # set freq if missing
    try:
        s.index.freq = s.index.freq or pd.infer_freq(s.index)
    except Exception:
        pass
    return s


def fit_and_forecast(y: pd.Series, horizon: int = 7, season: int = 7) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Backtest SARIMA vs Seasonal Naive on the last `horizon` points, pick winner,
    then produce forward `horizon` forecast + CI (lo/hi).
    Returns (mean, ci[lo,hi], winner_name).
    """
    # Safety: daily, float
    y = ensure_daily_index(y)
    y = y.astype(float)

    # If super short, just seasonal naive
    min_len = max(horizon + 1, season + 1)
    if len(y) < min_len:
        mean = seasonal_naive(y, horizon=horizon, season=season)
        recent_std = y.diff().dropna().rolling(14).std().iloc[-1] if len(y) > 15 else 0.01
        recent_std = float(recent_std) if pd.notna(recent_std) else 0.01
        ci = np.vstack([mean - recent_std, mean + recent_std]).T
        return mean, ci, "SeasonalNaive"

    # Train/test split for quick backtest
    test_h = horizon
    train_end = len(y) - test_h
    y_train = y.iloc[:train_end]
    y_test = y.iloc[train_end:]

    # Baseline
    sn_fore = seasonal_naive(y_train, horizon=test_h, season=season)
    sn_mae = float(np.mean(np.abs(y_test.values - sn_fore)))

    # SARIMAX (simple template)
    try:
        model = SARIMAX(
            y_train,
            order=(1, 0, 1),
            seasonal_order=(1, 1, 1, season),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        sar_pred = res.get_forecast(steps=test_h)
        sar_mean_bt = sar_pred.predicted_mean.values
        sar_mae = float(np.mean(np.abs(y_test.values - sar_mean_bt)))
    except Exception as e:
        # If SARIMA fails, fall back to naive
        sar_mae = float("inf")

    winner = "SARIMA" if sar_mae <= sn_mae else "SeasonalNaive"

    # Refit on full history and forecast forward
    if winner == "SARIMA":
        try:
            full = SARIMAX(
                y, order=(1, 0, 1), seasonal_order=(1, 1, 1, season),
                enforce_stationarity=False, enforce_invertibility=False
            ).fit(disp=False)
            fut = full.get_forecast(steps=horizon)
            mean = fut.predicted_mean.values
            ci_df = fut.conf_int(alpha=0.2)  # ~80% band
            ci = ci_df.to_numpy()  # shape (h, 2) -> [lo, hi]
        except Exception:
            # Fallback if full fit fails
            mean = seasonal_naive(y, horizon=horizon, season=season)
            recent_std = y.diff().dropna().rolling(14).std().iloc[-1] if len(y) > 15 else 0.01
            recent_std = float(recent_std) if pd.notna(recent_std) else 0.01
            ci = np.vstack([mean - recent_std, mean + recent_std]).T
            winner = "SeasonalNaive"
    else:
        mean = seasonal_naive(y, horizon=horizon, season=season)
        recent_std = y.diff().dropna().rolling(14).std().iloc[-1] if len(y) > 15 else 0.01
        recent_std = float(recent_std) if pd.notna(recent_std) else 0.01
        ci = np.vstack([mean - recent_std, mean + recent_std]).T

    return mean, ci, winner


def main():
    H = 7
    SEASON = 7

    counts = load_daily_platform_shares()
    if counts.empty:
        print("No rows found after loading; writing empty outputs.")
        # still write empty files so downstream doesn't break
        empty_fc = pd.DataFrame(columns=["date", "platform", "pred", "lo", "hi", "model"])
        empty_fc.to_parquet(OUTDIR / "platform_share_forecast.parquet", index=False)
        with open(OUTDIR / "metrics.json", "w") as f:
            json.dump({}, f, indent=2)
        print("✅ Saved empty outputs to:", OUTDIR)
        return

    platforms = counts["platform"].unique()
    print(f"Found {len(platforms)} unique platforms.")
    print(counts.groupby("platform")["date"].nunique().describe())
    sys.stdout.flush()

    all_forecasts = []
    metrics = {}

    for platform in platforms:
        dfp = (
            counts.loc[counts["platform"] == platform, ["date", "share"]]
            .sort_values("date")
            .set_index("date")
        )

        if dfp.empty:
            continue

        y = dfp["share"]
        y.index = pd.to_datetime(y.index)
        y = ensure_daily_index(y)

        try:
            mean, ci, winner = fit_and_forecast(y, horizon=H, season=SEASON)
        except Exception as e:
            print(f"!! Forecast failed for platform={platform}: {type(e).__name__} {e}")
            # last-resort fallback
            mean = seasonal_naive(y, horizon=H, season=SEASON)
            recent_std = y.diff().dropna().rolling(14).std().iloc[-1] if len(y) > 15 else 0.01
            recent_std = float(recent_std) if pd.notna(recent_std) else 0.01
            ci = np.vstack([mean - recent_std, mean + recent_std]).T
            winner = "SeasonalNaive"

        future_dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=H, freq="D")
        fc_df = pd.DataFrame({
            "date": future_dates,
            "platform": platform,
            "pred": mean.astype(float),
            "lo": ci[:, 0].astype(float),
            "hi": ci[:, 1].astype(float),
            "model": winner
        })
        all_forecasts.append(fc_df)

        metrics[platform] = {"winner": winner}

    final_forecast = (
        pd.concat(all_forecasts, ignore_index=True)
        if len(all_forecasts) > 0
        else pd.DataFrame(columns=["date", "platform", "pred", "lo", "hi", "model"])
    )

    final_path = OUTDIR / "platform_share_forecast.parquet"
    final_forecast.to_parquet(final_path, index=False)

    with open(OUTDIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Saved:")
    print(" -", final_path)
    print(" -", OUTDIR / "metrics.json")


if __name__ == "__main__":
    # run unbuffered: python -u -m ml.forecast_7d.platform.train_platform_shift
    main()
