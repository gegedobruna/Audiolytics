# ml/prep/add_platforms.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def normalize_platform(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)): 
        return "Other"
    t = str(s).strip().lower()
    # common buckets
    if "windows" in t:                 return "Windows"
    if "mac" in t or "os x" in t:      return "MacOS"
    if "android" in t:                 return "Android"
    if "ios" in t or "iphone" in t or "ipad" in t:  return "iOS"
    if "xbox" in t:                    return "Xbox"
    if "playstation" in t or "ps4" in t or "ps5" in t: return "PlayStation"
    if "linux" in t or "ubuntu" in t:  return "Linux"
    if "web" in t or "browser" in t:   return "Web"
    return "Other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/Complete_Data.csv")
    ap.add_argument("--events_with_clusters", default="data/derived/sequences_with_clusters.parquet")
    ap.add_argument("--out", default="data/derived/sequences_with_clusters_platforms.parquet")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    ev_path  = Path(args.events_with_clusters)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load CSV: we only need ts + platform
    print(f"Loading CSV: {csv_path} …", flush=True)
    df_csv = pd.read_csv(csv_path)
    if "ts" not in df_csv.columns or "platform" not in df_csv.columns:
        raise SystemExit("Expected columns 'ts' and 'platform' in Complete_Data.csv")

    # Normalize timestamps
    df_csv["ts"] = pd.to_datetime(df_csv["ts"], utc=True, errors="coerce")
    # Normalize platform names
    df_csv["platform_norm"] = df_csv["platform"].map(normalize_platform)

    # Keep only what we need; if multiple rows share the same ts, keep the latest occurrence
    df_csv_small = df_csv[["ts", "platform_norm"]].dropna(subset=["ts"]).drop_duplicates(subset=["ts"], keep="last")

    # Load clustered events
    print(f"Loading clustered events: {ev_path} …", flush=True)
    df_ev = pd.read_parquet(ev_path)
    if "ts" not in df_ev.columns:
        raise SystemExit("Expected 'ts' column in sequences_with_clusters.parquet")
    df_ev["ts"] = pd.to_datetime(df_ev["ts"], utc=True, errors="coerce")

    n_before = len(df_ev)
    # Merge
    df_merge = df_ev.merge(df_csv_small, on="ts", how="left")
    coverage = 100.0 * df_merge["platform_norm"].notna().mean()

    # Fill missing with "Other"
    df_merge["platform"] = df_merge["platform_norm"].fillna("Other")
    df_merge.drop(columns=["platform_norm"], inplace=True)

    # Save new parquet
    df_merge.to_parquet(out_path, index=False)
    print("✅ Saved:", out_path)
    print(f"Rows: {n_before:,} | Platform coverage: {coverage:.1f}%")
    print("Sample platform counts (last 10 days):")
    # if a 'date' column exists, show a quick sanity table
    if "date" in df_merge.columns:
       tail_dates = df_merge["date"].sort_values().drop_duplicates().tail(10)
       sample = (df_merge[df_merge["date"].isin(tail_dates)]
                 .groupby(["date","platform"]).size().reset_index(name="plays"))
       print(sample.sort_values(["date","plays"], ascending=[True, False]).head(20).to_string(index=False))
    else:
       print(df_merge["platform"].value_counts().head(10).to_string())
    
if __name__ == "__main__":
    main()
