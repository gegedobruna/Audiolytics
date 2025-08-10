# ml/common_prep.py
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

AUDIO_FEATURE_CANDIDATES = [
    "danceability","energy","valence","acousticness","instrumentalness",
    "liveness","speechiness","tempo","loudness","key","mode"
]

# include 'ts' + common variants
TIMESTAMP_CANDIDATES = ["ts","timestamp","time","played_at","date","datetime","offline_timestamp"]

def find_timestamp_col(df: pd.DataFrame):
    cols = set(df.columns)
    # 1) direct name match
    for c in TIMESTAMP_CANDIDATES:
        if c in cols:
            return c
    # 2) auto-detect: pick the first column that parses to datetime for >=90% non-null rows
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
            ok_ratio = parsed.notna().mean()
            if ok_ratio >= 0.9:
                return c
        except Exception:
            pass
    raise ValueError(f"No timestamp column found. Tried names {TIMESTAMP_CANDIDATES} and auto-detect failed.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/Complete_Data.csv")
    ap.add_argument("--outdir", default="data/derived")
    ap.add_argument("--session_gap_min", type=int, default=30)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    # drop accidental index cols
    for junk in ["Unnamed: 0", "_unnamed", "index"]:
        if junk in df.columns:
            df = df.drop(columns=[junk])

    # pick timestamp col
    ts_col = find_timestamp_col(df)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    # map artist / track names (Spotify schema)
    artist_col = None
    for cand in ["artist","artist_name","master_metadata_album_artist_name"]:
        if cand in df.columns:
            artist_col = cand; break

    track_col = None
    for cand in ["track","track_name","master_metadata_track_name","spotify_track_uri","spotify_id","id"]:
        if cand in df.columns:
            track_col = cand; break

    # audio features present
    feature_cols = [c for c in AUDIO_FEATURE_CANDIDATES if c in df.columns]

    # helper time columns
    df["date"] = df[ts_col].dt.date
    df["hour"] = df[ts_col].dt.hour
    df["dow"]  = df[ts_col].dt.dayofweek  # 0=Mon

    # sessionization
    gap = pd.Timedelta(minutes=args.session_gap_min)
    df["prev_time"] = df[ts_col].shift(1)
    df["gap"] = df[ts_col] - df["prev_time"]
    df["new_session"] = (df["gap"].isna()) | (df["gap"] > gap)
    df["session_id"] = df["new_session"].cumsum()

    # event-level parquet
    event_cols = [ts_col, "date", "hour", "dow", "session_id", "gap"]
    if artist_col: event_cols.append(artist_col)
    if track_col:  event_cols.append(track_col)
    event_cols += feature_cols
    events = df[event_cols].copy()

    events_path = outdir / "sequences.parquet"
    events.to_parquet(events_path, index=False)

    # daily counts
    daily = df.groupby("date").size().reset_index(name="plays")
    daily_path = outdir / "daily_counts.parquet"
    daily.to_parquet(daily_path, index=False)

    manifest = {
        "input": str(Path(args.input).resolve()),
        "events_path": str(events_path.resolve()),
        "daily_path": str(daily_path.resolve()),
        "n_rows": int(len(df)),
        "n_sessions": int(events["session_id"].nunique()),
        "feature_cols": feature_cols,
        "timestamp_col": ts_col,
        "artist_col": artist_col,
        "track_col": track_col,
        "session_gap_min": args.session_gap_min,
    }
    (outdir / "prep_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("✅ Wrote:", events_path)
    print("✅ Wrote:", daily_path)
    print("📝 Manifest:", outdir / "prep_manifest.json")
    print("Detected feature cols:", feature_cols)
    print("Detected artist col:", artist_col, "| track col:", track_col)

if __name__ == "__main__":
    main()
