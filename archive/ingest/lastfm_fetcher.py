# DEPRECATED: kept for provenance only. Not used by the Streamlit app.

# if __name__ == "__main__":
#     print("This script is deprecated. See MIGRATION.md for the current workflow.")
# 
# import requests
# import pandas as pd
# from utils.env_loader import load_env
# from time import sleep
# import os
# 
# # Fetch all scrobbles from Last.fm
# def fetch_all_scrobbles(limit_per_page=200, delay=0.5, max_pages=None, start_page=166, output_path="data/audiolytics_full.csv"):
#     env = load_env()
#     API_KEY = env["LASTFM_API_KEY"]
#     USERNAME = env["LASTFM_USER"]
# 
#     all_tracks = []
#     page = start_page
#     MAX_RETRIES = 3
# 
#     print(f"📡 Starting from page {start_page}...")
# 
#     while True:
#         params = {
#             "method": "user.getrecenttracks",
#             "user": USERNAME,
#             "api_key": API_KEY,
#             "format": "json",
#             "limit": limit_per_page,
#             "page": page
#         }
# 
#         for attempt in range(MAX_RETRIES):
#             try:
#                 response = requests.get("https://ws.audioscrobbler.com/2.0/", params=params, timeout=10)
#                 data = response.json()
#                 tracks = data["recenttracks"]["track"]
#                 break
#             except Exception as e:
#                 print(f"❌ Error on page {page}, attempt {attempt + 1}: {e}")
#                 if attempt < MAX_RETRIES - 1:
#                     print("⏳ Retrying in 5 seconds...")
#                     sleep(5)
#                 else:
#                     print("💾 Saving partial data before exit...")
#                     save_and_merge(all_tracks, output_path)
#                     print("🛑 Stopping due to repeated errors.")
#                     return
# 
#         if not tracks:
#             print("🏁 No more tracks — done!")
#             break
# 
#         for track in tracks:
#             if 'date' not in track:
#                 continue  # skip now-playing
#             all_tracks.append({
#                 "played_at": track['date']['#text'],
#                 "artist": track['artist']['#text'],
#                 "track": track['name'],
#                 "album": track['album']['#text'],
#                 "loved": track.get('loved', '0'),
#                 "mbid": track.get('mbid', None),
#                 "url": track.get('url', None)
#             })
# 
#         print(f"✅ Page {page} fetched ({len(tracks)} tracks)")
#         page += 1
# 
#         if max_pages and page > max_pages:
#             print("🔚 Reached max page limit")
#             break
# 
#         if len(tracks) < limit_per_page:
#             print("📉 Fewer than 200 tracks — assuming last page")
#             break
# 
#         sleep(delay)
# 
#     save_and_merge(all_tracks, output_path)
# 
# # Save and merge new data with existing CSV
# def save_and_merge(new_data, output_path):
#     df_new = pd.DataFrame(new_data)
#     if os.path.exists(output_path):
#         df_existing = pd.read_csv(output_path)
#         df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates()
#     else:
#         df_combined = df_new
# 
#     df_combined.to_csv(output_path, index=False)
#     print(f"💾 Total saved: {len(df_combined)} scrobbles → {output_path}")
# 