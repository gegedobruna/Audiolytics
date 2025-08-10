import json
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from ml.forecast_7d.common_daily import load_daily

OUTDIR = Path("ml/forecast_7d/artists")
OUTDIR.mkdir(parents=True, exist_ok=True)

def seasonal_naive(y, horizon=7, season=7):
    if len(y) < season:
        return np.repeat(y.iloc[-1], horizon)
    last_season = y.iloc[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season.values, reps)[:horizon]

def main():
    # Load daily stats with columns: date, plays, new_artist_ratio
    df = load_daily(fill_missing=True)  # we’ll extend common_daily to also output new artist ratio
    if "new_artist_ratio" not in df.columns:
        raise ValueError("Expected 'new_artist_ratio' in daily data")

    y = df["new_artist_ratio"].astype(float)
    dates = df["date"]

    # Backtest split
    test_h = 7
    train_end = len(y) - test_h
    y_train = y.iloc[:train_end]

    # Seasonal naive
    sn_fore = seasonal_naive(y_train, horizon=test_h, season=7)
    sn_mae = float(np.mean(np.abs(y.iloc[train_end:] - sn_fore)))

    # SARIMA
    model = SARIMAX(y_train, order=(1,0,1), seasonal_order=(1,1,1,7),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    sar_fore = res.get_forecast(steps=test_h)
    sar_mean = sar_fore.predicted_mean.values
    sar_ci = sar_fore.conf_int(alpha=0.2).values
    sar_mae = float(np.mean(np.abs(y.iloc[train_end:] - sar_mean)))

    winner = "SARIMA" if sar_mae <= sn_mae else "SeasonalNaive"

    # Forecast future
    if winner == "SARIMA":
        full = SARIMAX(y, order=(1,0,1), seasonal_order=(1,1,1,7),
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fut = full.get_forecast(steps=test_h)
        mean = fut.predicted_mean.values
        ci = fut.conf_int(alpha=0.2).values
    else:
        mean = seasonal_naive(y, horizon=test_h, season=7)
        recent_std = float(y.diff().dropna().rolling(14).std().iloc[-1] or 0.05)
        ci = np.vstack([mean - recent_std, mean + recent_std]).T

    future_dates = pd.date_range(dates.iloc[-1] + pd.Timedelta(days=1),
                                 periods=test_h, freq="D")
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "pred": mean,
        "lo": ci[:, 0],
        "hi": ci[:, 1],
        "model": winner
    })

    OUTDIR.joinpath("new_vs_same_forecast.parquet").write_bytes(forecast_df.to_parquet(index=False))
    OUTDIR.joinpath("metrics.json").write_text(json.dumps({
        "backtest_horizon": test_h,
        "seasonal_naive_MAE": sn_mae,
        "sarima_MAE": sar_mae,
        "winner": winner
    }, indent=2))

    print("✅ Saved:")
    print(" - ml/forecast_7d/artists/new_vs_same_forecast.parquet")
    print(" - ml/forecast_7d/artists/metrics.json")
    print(f"Winner: {winner} | MAE SN={sn_mae:.4f} SARIMA={sar_mae:.4f}")

if __name__ == "__main__":
    main()
