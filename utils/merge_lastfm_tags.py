import pandas as pd

def normalize(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()

def merge_tag_data(full_path, tag_path, output_path):
    # Load datasets
    full_df = pd.read_csv(full_path)
    tag_df = pd.read_csv(tag_path)

    # Normalize for merge
    full_df["artist_norm"] = full_df["artist"].apply(normalize)
    full_df["track_norm"] = full_df["track"].apply(normalize)

    tag_df["artist_norm"] = tag_df["artist"].apply(normalize)
    tag_df["track_norm"] = tag_df["track"].apply(normalize)

    # Merge tags into full dataset
    merged = full_df.merge(
        tag_df[["artist_norm", "track_norm", "tags"]],
        on=["artist_norm", "track_norm"],
        how="left"
    )

    # Save to new CSV
    merged.to_csv(output_path, index=False)
    print(f"✅ Merged dataset saved to: {output_path}")
    print(f"🔍 Tags matched: {merged['tags'].notna().sum()} / {len(merged)}")

if __name__ == "__main__":
    merge_tag_data(
        full_path="data/audiolytics_full.csv",
        tag_path="data/tag_cache_clean.csv",
        output_path="data/audiolytics_with_tags.csv"
    )
