# ml/forecast_7d/vibes/train_vibe_distribution.py
from pathlib import Path
import json
import numpy as np
import pandas as pd

from ml.forecast_7d.common_daily import load_events_for_extras, load_manifest

OUTDIR = Path("ml/forecast_7d/vibes"); OUTDIR.mkdir(parents=True, exist_ok=True)

def seasonal_naive_matrix(M: np.ndarray, horizon=7, season=7):
    """Seasonal naive per column; rows=time, cols=clusters; returns horizon x K."""
    if M.shape[0] < season:
        last = M[-1:]
        return np.repeat(last, horizon, axis=0)
    last_season = M[-season:]
    reps = int(np.ceil(horizon/season))
    out = np.vstack([last_season] * reps)[:horizon]
    return out

def row_normalize(M, eps=1e-9):
    s = M.sum(axis=1, keepdims=True)
    s = np.where(s <= eps, 1.0, s)
    return M / s

def main():
    mf = load_manifest()
    ts = mf["timestamp_col"]

    # Load event-level with clusters
    df = load_events_for_extras()  # has columns: ts/date/session_id/cluster_id/...
    # counts per day x cluster
    wide = (df.groupby(["date","cluster_id"])
              .size().unstack(fill_value=0).sort_index())
    # fill missing days with zeros for all clusters
    full_days = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full_days, fill_value=0)
    wide.index.name = "date"

    # Convert to shares per day
    shares = wide.div(wide.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    # Backtest window (last 7 days)
    H = 7
    train = shares.iloc[:-H]
    test  = shares.iloc[-H:]

    # Seasonal naive per cluster (weekly seasonality), then renormalize rows
    fc = seasonal_naive_matrix(train.values, horizon=H, season=7)
    fc = row_normalize(fc)

    # simple MAE across clusters (macro-average)
    mae = float(np.mean(np.abs(test.values - fc)))

    # For future forecast: use ALL history
    future_fc = seasonal_naive_matrix(shares.values, horizon=H, season=7)
    future_fc = row_normalize(future_fc)

    future_dates = pd.date_range(shares.index[-1] + pd.Timedelta(days=1), periods=H, freq="D")
    # Long format for plotting
    long = (pd.DataFrame(future_fc, index=future_dates, columns=shares.columns)
              .reset_index().melt(id_vars="index", var_name="cluster_id", value_name="pred_share")
              .rename(columns={"index":"date"}))
    # crude uncertainty band: +/- recent std of daily share per cluster (clipped)
    recent = shares.tail(28)
    stds = recent.std().to_dict()
    long["lo"] = (long.apply(lambda r: max(0.0, r["pred_share"] - stds.get(r["cluster_id"], 0.05)), axis=1))
    long["hi"] = (long.apply(lambda r: min(1.0, r["pred_share"] + stds.get(r["cluster_id"], 0.05)), axis=1))

    # Save
    out_parq = OUTDIR / "vibe_share_forecast.parquet"
    long.to_parquet(out_parq, index=False)
    (OUTDIR / "metrics.json").write_text(json.dumps({
        "horizon_days": H,
        "model": "SeasonalNaive (weekly) + renormalize",
        "backtest_mae_macro": mae
    }, indent=2))

    print("✅ Saved:")
    print(" -", out_parq)
    print(" -", OUTDIR / "metrics.json")
    print(f"Backtest macro-MAE: {mae:.4f}")

if __name__ == "__main__":
    main()
