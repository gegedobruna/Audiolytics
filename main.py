import os
from fetch.lastfm_fetcher import fetch_recent_tracks

os.makedirs("data", exist_ok=True)

df = fetch_recent_tracks(limit=20)
print(df.head())

df.to_csv("data/audiolytics_raw.csv", index=False)
print("✅ Data saved to CSV")
