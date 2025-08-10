# ml/forecast_7d/common_daily.py
from pathlib import Path
import json
import pandas as pd
import numpy as np

DERIVED = Path("data/derived")
DAILY_PARQ = DERIVED / "daily_counts.parquet"
SEQ_WITH_CLUSTERS = DERIVED / "sequences_with_clusters.parquet"
MANIFEST = DERIVED / "prep_manifest.json"

def load_manifest():
    return json.loads(MANIFEST.read_text())

def load_events_for_extras():
    mf = load_manifest()
    ts = mf["timestamp_col"]
    df = pd.read_parquet(SEQ_WITH_CLUSTERS)
    df[ts] = pd.to_datetime(df[ts], utc=True)
    df["date"] = df[ts].dt.tz_convert(None).dt.date
    df["date"] = pd.to_datetime(df["date"])
    return df

def load_daily(fill_missing=True):
    mf = load_manifest()
    df = pd.read_parquet(DAILY_PARQ)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if fill_missing:
        full = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")})
        df = full.merge(df, on="date", how="left").fillna({"plays": 0})

    # Add day-of-week and ISO week
    df["dow"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    # --- EXTRA: new_artist_ratio ---
    events = load_events_for_extras()

    # Drop rows with missing artist name
    events = events.dropna(subset=["master_metadata_album_artist_name"])

    # Ensure sorted by date for correct "first time" detection
    events = events.sort_values(["date"])

    # duplicated() returns bool, fill missing, ensure dtype
    dup_mask = events.groupby("master_metadata_album_artist_name")["date"] \
                     .transform(lambda x: x.duplicated()) \
                     .fillna(False) \
                     .astype(bool)

    # New artist = first time heard → not duplicated
    events["is_new_artist"] = ~dup_mask

    # Ratio of new artist plays per date
    daily_new = events.groupby("date")["is_new_artist"].mean().reset_index(name="new_artist_ratio")

    # Merge into daily table
    df = df.merge(daily_new, on="date", how="left").fillna({"new_artist_ratio": 0.0})

    return df
