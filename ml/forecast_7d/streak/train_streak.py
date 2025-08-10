# ml/forecast_7d/streak/train_streak.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from ml.forecast_7d.common_daily import load_daily

OUTDIR = Path("ml/forecast_7d/streak")
OUTDIR.mkdir(parents=True, exist_ok=True)

def build_features(df):
    # Streak length up to each date
    streaks = []
    streak = 0
    for plays in df["plays"]:
        if plays > 0:
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    df["streak_len"] = streaks

    # Yesterday's plays
    df["yesterday_plays"] = df["plays"].shift(1).fillna(0)

    # 7-day rolling mean of plays
    df["plays_7d_avg"] = df["plays"].rolling(7, min_periods=1).mean()

    # Day of week (one-hot)
    dow_dummies = pd.get_dummies(df["dow"], prefix="dow")
    df = pd.concat([df, dow_dummies], axis=1)

    return df

def prepare_dataset(df):
    df = build_features(df)

    # Target: listened next day (1 if plays > 0 tomorrow)
    df["target"] = (df["plays"].shift(-1) > 0).astype(int)

    # Drop last row (no target)
    df = df.iloc[:-1]

    # Features
    feature_cols = ["streak_len", "yesterday_plays", "plays_7d_avg"] + [c for c in df.columns if c.startswith("dow_")]
    X = df[feature_cols]
    y = df["target"]
    return X, y, feature_cols, df

def main():
    df = load_daily(fill_missing=True)
    X, y, feature_cols, df_full = prepare_dataset(df)

    # Train/test split (last 20% as test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Model pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500))
    ])

    pipe.fit(X_train, y_train)

    # Eval
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save metrics
    metrics = {"accuracy": acc}
    OUTDIR.joinpath("metrics.json").write_text(json.dumps(metrics, indent=2))

    # Forecast next 7 days using chained prediction
    future_preds = []
    df_forecast = df_full.copy()

    current_streak = int(df_forecast["streak_len"].iloc[-1])
    last_plays = float(df_forecast["plays"].iloc[-1])
    last_7d_avg = float(df_forecast["plays_7d_avg"].iloc[-1])
    last_dow = int(df_forecast["dow"].iloc[-1])

    for i in range(7):
        # Build one-hot for dow
        dow_oh = [0] * 7
        dow_oh[last_dow] = 1

        features = np.array([[current_streak, last_plays, last_7d_avg] + dow_oh])
        pred_prob = pipe.predict_proba(features)[0, 1]
        will_listen = pred_prob >= 0.5

        future_preds.append({
            "day_ahead": i+1,
            "prob_continue": pred_prob
        })

        # Update for next iteration
        if will_listen:
            current_streak += 1
            last_plays = last_7d_avg  # assume avg plays if continuing
        else:
            current_streak = 0
            last_plays = 0

        # Rolling avg crude update
        last_7d_avg = (last_7d_avg * 6 + last_plays) / 7

        last_dow = (last_dow + 1) % 7

    # Save forecast
    OUTDIR.joinpath("forecast.json").write_text(json.dumps(future_preds, indent=2))

    print("✅ Saved:")
    print(" - ml/forecast_7d/streak/metrics.json")
    print(" - ml/forecast_7d/streak/forecast.json")
    print(f"Test Accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
