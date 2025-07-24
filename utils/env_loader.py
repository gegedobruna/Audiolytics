from dotenv import load_dotenv
import os

def load_env():
    load_dotenv()

    return {
        "LASTFM_API_KEY": os.getenv("LASTFM_API_KEY"),
        "LASTFM_USER": os.getenv("LASTFM_USER"),
        "SPOTIFY_CLIENT_ID": os.getenv("SPOTIFY_CLIENT_ID"),
        "SPOTIFY_CLIENT_SECRET": os.getenv("SPOTIFY_CLIENT_SECRET")
    }
