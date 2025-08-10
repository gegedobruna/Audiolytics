# ml/forecast_7d/plays/train_total_plays.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
import ml.forecast_7d.common_daily as cd


OUTDIR = Path("ml/forecast_7d/vibes"); OUTDIR.mkdir(parents=True, exist_ok=True)

def seasonal_naive(y, horizon=7, season=7):
    if len(y) < season:  # fallback
        return np.repeat(y.iloc[-1], horizon)
    last_season = y.iloc[-season:]
    reps = int(np.ceil(horizon/season))
    return np.tile(last_season.values, reps)[:horizon]

def main():
    df = cd.load_daily(fill_missing=True)  # columns: date, plays, dow, week
    y = df["plays"].astype(float)
    # Keep last 28 days as backtest window
    test_h = 7
    train_end = len(y) - test_h
    y_train = y.iloc[:train_end]
    dates = df["date"]

    # --- Seasonal naive baseline ---
    sn_fore = seasonal_naive(y_train, horizon=test_h, season=7)
    sn_mae = float(np.mean(np.abs(y.iloc[train_end:] - sn_fore)))

    # --- SARIMA ---
    model = SARIMAX(y_train, order=(1,0,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    sar_fore = res.get_forecast(steps=test_h)
    sar_mean = sar_fore.predicted_mean.values
    sar_ci = sar_fore.conf_int(alpha=0.2).values  # ~80% band
    sar_mae = float(np.mean(np.abs(y.iloc[train_end:] - sar_mean)))

    # Choose winner by lower MAE
    winner = "SARIMA" if sar_mae <= sn_mae else "SeasonalNaive"

    # Now refit winner on ALL data and forecast next 7
    if winner == "SARIMA":
        full = SARIMAX(y, order=(1,0,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fut = full.get_forecast(steps=test_h)
        mean = fut.predicted_mean.values
        ci = fut.conf_int(alpha=0.2).values
    else:
        mean = seasonal_naive(y, horizon=test_h, season=7)
        # crude CI for naive: ±1 * recent std
        recent_std = float(y.diff().dropna().rolling(14).std().iloc[-1] or 1.0)
        ci = np.vstack([mean - recent_std, mean + recent_std]).T

    future_dates = pd.date_range(dates.iloc[-1] + pd.Timedelta(days=1), periods=test_h, freq="D")
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "pred": mean,
        "lo": ci[:,0],
        "hi": ci[:,1],
        "model": winner
    })

    OUTDIR.joinpath("total_plays_forecast.parquet").write_bytes(forecast_df.to_parquet(index=False))
    OUTDIR.joinpath("metrics.json").write_text(json.dumps({
        "backtest_horizon": test_h,
        "seasonal_naive_MAE": sn_mae,
        "sarima_MAE": sar_mae,
        "winner": winner
    }, indent=2))
    print("✅ Saved:")
    print(" - ml/forecast/plays/total_plays_forecast.parquet")
    print(" - ml/forecast/plays/metrics.json")
    print(f"Winner: {winner} | MAE SN={sn_mae:.2f} SARIMA={sar_mae:.2f}")

if __name__ == "__main__":
    main()
