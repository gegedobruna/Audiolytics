from fetch.lastfm_fetcher import fetch_all_scrobbles

fetch_all_scrobbles(start_page=491)
print("✅ Fetching complete. Check data/audiolytics_full.csv for results.")