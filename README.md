# Audiolytics

An interactive Streamlit app that turns a single listening-history file into clear insights — genres over time, mood & energy, discovery habits, device mix, streaks, comebacks, and more.

## What it does
- **Timeline & habits:** daily minutes, prime hours, **streaks** (calendar), seasonality.
- **Genres:** monthly share, **variety (entropy)**, first/last appearance, rising vs. falling.
- **Artists:** **comeback tracker** (returns after X+ months), top artists.
- **Mood & energy:** **valence × energy** density map; rolling trends (y-domain tightened for contrast).
- **Platform:** regex-based **platform normalization** + **Top‑N + Other** chart.
- **Discovery & nostalgia:** new vs repeat shares, “forgotten” past favorites.
- **Extras:** feature fingerprint & change between eras, optional session clustering.

## How it works
- The app loads a prepared dataset and converts it to **Parquet** (if needed) for speed.
- Heavy transforms are cached; some visuals compute on demand.
- Charts use **Altair**; large-data views use sampling to stay responsive.

## What I built / used
**Stack:** Streamlit, Pandas, NumPy, Altair, scikit‑learn, PyArrow.  
**Data prep:** Spotify audio features + ReccoBeats tags/genres; merges/cleaning.  
**Notable touches:** platform regex mapper, Top‑N bucketing, 2D binned mood map, genre entropy & entry/exit, rising/falling genres, comeback tracker, fingerprint deltas, streaks calendar.

---

## Deployment
- Works on Streamlit Cloud with no secrets.
- **Main file:** `app/streamlit_app.py`

---

## Dataset (included)
This repo **includes** a dataset so the live app runs as‑is:

- Primary file: `data/audiolytics.parquet` (fastest to load).  
- Optional CSV: you *can* also include `data/Complete_Data_with_genres_filled.csv` (≈32 MB), but it’s redundant — the app reads the Parquet first. Keeping only the Parquet keeps the repo smaller and faster to clone.

**Source:** Kaggle base + my enrichment/merges (Spotify features, ReccoBeats, cleaning).  
**Note:** Included here strictly so reviewers can see the full app experience.

---

## Project structure
```
app/                 # Streamlit app (entrypoint: streamlit_app.py)
data/                # dataset lives here (parquet included)
ingest/ (optional)   # any scripts used during enrichment (no secrets)
archive/             # old prototypes kept for provenance
```

## Changes from the early plan
Moved from an early Last.fm concept to a **self‑contained CSV/Parquet workflow** with enrichment. Added platform normalization, mood density map, genre entropy/entry‑exit, rising/falling, comeback tracker, fingerprint deltas, streaks calendar, discovery score, nostalgia list — plus caching & Parquet for performance.

## Getting started (local)
```bash
# clone
git clone https://github.com/gegedobruna/Audiolytics.git
cd Audiolytics

# (optional) create a venv
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# install
pip install -r requirements.txt

# run
streamlit run app/streamlit_app.py
```

> First launch may take a minute while caches build. Next runs are snappy.

## License
Code: MIT. Dataset: included for demo/review; please do not redistribute outside this repo context.
