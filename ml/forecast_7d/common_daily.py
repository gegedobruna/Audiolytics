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

def load_daily(fill_missing=True):
    mf = load_manifest()
    df = pd.read_parquet(DAILY_PARQ)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if fill_missing:
        full = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")})
        df = full.merge(df, on="date", how="left").fillna({"plays": 0})
    df["dow"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    return df

def load_events_for_extras():
    mf = load_manifest()
    ts = mf["timestamp_col"]
    df = pd.read_parquet(SEQ_WITH_CLUSTERS)
    df[ts] = pd.to_datetime(df[ts], utc=True)
    df["date"] = df[ts].dt.tz_convert(None).dt.date
    df["date"] = pd.to_datetime(df["date"])
    return df
