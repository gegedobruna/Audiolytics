import pandas as pd

files = [
    "data/audiolytics.parquet",
    "data/derived/daily_counts.parquet",
    "data/derived/sequences.parquet"
]

for file in files:
    try:
        df = pd.read_parquet(file)
        print(f"=== {file} ===")
        print("Columns:", df.columns.tolist())
        print("First few rows:")
        print(df.head(), "\n")
    except Exception as e:
        print(f"Error reading {file}: {e}\n")
