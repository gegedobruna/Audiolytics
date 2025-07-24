import requests
import pandas as pd
from utils.env_loader import load_env

def fetch_recent_tracks(limit=5):
    env = load_env()
    API_Key = env["LASTFM_API_KEY"]
    USERNAME = env["LASTFM_USER"]

    url = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "user.getrecenttracks",
        "user": USERNAME,
        "api_key": API_Key,
        "format": "json",
        "limit": limit
    }

    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    tracks = data['recenttracks']['track']

    parsed_tracks = []

    for track in tracks:
        if 'date' not in track:
            continue
        parsed_tracks.append({
            "played_at": track['date']['#text'],
            "artist": track['artist']['#text'],
            "track": track['name'],
            "album": track['album']['#text']
        })

    return pd.DataFrame(parsed_tracks)
