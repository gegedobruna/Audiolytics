import requests
import pandas as pd
import time
import os
from tqdm import tqdm
from utils.env_loader import load_env

TAG_CACHE_PATH = "data/tag_cache.csv"
NOT_FOUND_PATH = "data/not_found_tags.txt"

def try_tag_request(method, api_key, artist=None, track=None, album=None, verbose=False):
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {
        "method": method,
        "api_key": api_key,
        "format": "json"
    }
    if artist: params["artist"] = artist
    if track: params["track"] = track
    if album: params["album"] = album

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if verbose:
            print(f"\n🔍 REQUEST: {url} | Method: {method}")
            print(f"📌 Params: {params}")
            print(f"📦 RAW RESPONSE:\n{data}")

        tags = data.get("toptags", {}).get("tag", [])
        if isinstance(tags, list) and tags:
            return [tag["name"] for tag in tags], method
        return None, None
    except Exception as e:
        if verbose:
            print(f"❌ Error during {method} for {artist} – {track or album}: {e}")
        return None, None

def get_best_tags(api_key, artist, track, album=None, verbose=False):
    # 1. Track-level
    tags, source = try_tag_request("track.getTopTags", api_key, artist=artist, track=track, verbose=verbose)
    if tags:
        return ", ".join(tags), source

    # 2. Album-level
    if album:
        tags, source = try_tag_request("album.getTopTags", api_key, artist=artist, album=album, verbose=verbose)
        if tags:
            return ", ".join(tags), source

    # 3. Artist-level
    tags, source = try_tag_request("artist.getTopTags", api_key, artist=artist, verbose=verbose)
    if tags:
        return ", ".join(tags), source

    return None, None

def enrich_tags_from_scrobbles(csv_path="data/audiolytics_full.csv", save_every=25, delay=0.3):
    env = load_env()
    api_key = env["LASTFM_API_KEY"]

    print("📂 Loading scrobbles...")
    df = pd.read_csv(csv_path)
    unique_tracks = df[["artist", "track", "album"]].drop_duplicates()

    # Load tag cache
    if os.path.exists(TAG_CACHE_PATH):
        tag_cache = pd.read_csv(TAG_CACHE_PATH)
        done_set = set(zip(tag_cache["artist"], tag_cache["track"]))
        print(f"🔁 Resuming from previous progress — {len(done_set)} tracks already cached")
    else:
        tag_cache = pd.DataFrame(columns=["artist", "track", "tags", "source"])
        done_set = set()

    # Filter only the unprocessed tracks
    mask = ~unique_tracks.apply(lambda row: (row["artist"], row["track"]) in done_set, axis=1)
    to_process = unique_tracks[mask].reset_index(drop=True)
    print(f"🎯 {len(to_process)} tracks remaining to tag")

    not_found = []
    new_entries = []

    DEBUG_MODE = False
    DEBUG_WHITELIST = [
        "Lana Del Rey – West Coast",
        "Eartheater – Solid Liquid Gas"
    ]

    for idx, row in tqdm(to_process.iterrows(), total=len(to_process)):
        artist = str(row["artist"]).strip()
        track = str(row["track"]).strip()
        album = str(row["album"]).strip() if "album" in row and pd.notna(row["album"]) else None
        full_key = f"{artist} – {track}"

        verbose = DEBUG_MODE or full_key in DEBUG_WHITELIST or idx < 3

        tags, source = get_best_tags(api_key, artist, track, album=album, verbose=verbose)

        if tags:
            new_entries.append({"artist": artist, "track": track, "tags": tags, "source": source})
        else:
            not_found.append(full_key)

        if len(new_entries) >= save_every:
            print(f"💾 Saving {len(new_entries)} new tag entries...")
            new_df = pd.DataFrame(new_entries)
            tag_cache = pd.concat([tag_cache, new_df], ignore_index=True)
            tag_cache.to_csv(TAG_CACHE_PATH, index=False)
            new_entries = []

        time.sleep(delay)

    # Final save if there are remaining entries
    if new_entries:
        print(f"💾 Final save of {len(new_entries)} entries")
        new_df = pd.DataFrame(new_entries)
        tag_cache = pd.concat([tag_cache, new_df], ignore_index=True)
        tag_cache.to_csv(TAG_CACHE_PATH, index=False)

    # Save not found list
    if not_found:
        with open(NOT_FOUND_PATH, "a", encoding="utf-8") as f:
            for line in not_found:
                f.write(line + "\n")

    print(f"✅ Enrichment complete — total cached: {len(tag_cache)}")
    print(f"📝 Tags saved to: {TAG_CACHE_PATH}")
    print(f"❌ Not found saved to: {NOT_FOUND_PATH}")
