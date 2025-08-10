# Audiolytics

An interactive Streamlit app that turns a single listening-history file into clear insights — genres over time, mood & energy, discovery habits, device mix, streaks, comebacks, and more — plus **ML-powered 7-day forecasts**.

## What it does
- **Timeline & habits:** daily minutes, prime hours, **streaks** (calendar), seasonality.
- **Genres:** monthly share, **variety (entropy)**, first/last appearance, rising vs. falling.
- **Artists:** **comeback tracker** (returns after X+ months), top artists.
- **Mood & energy:** **valence × energy** density map; rolling trends.
- **Platform:** regex-based **platform normalization** + **Top-N + Other** chart.
- **Discovery & nostalgia:** new vs repeat shares, “forgotten” past favorites.
- **Extras:** feature fingerprint & change between eras, optional session clustering.
- **Forecasts (ML):** **7-day platform-share forecasts** with confidence bands and backtests.

## How it works
- The app loads a prepared dataset and converts it to **Parquet** (if needed) for speed.
- Heavy transforms are cached; large views use sampling to stay responsive.
- Charts use **Altair**.
- ML forecasts are trained offline and read from Parquet at runtime by the Forecasts page.

## ML at a glance
- **Target:** next-7-days platform share (normalized shares by day).
- **Models:** statsmodels SARIMAX / ETS baselines (per-platform series).
- **Evaluation:** rolling backtests (expanding origin), MAE/MAPE logged to disk.
- **Artifacts:** predictions with 80/95% intervals saved to Parquet for the app.

## Deployment
- Works on Streamlit Cloud with no secrets at [audiolytics.streamlit.app](audiolytics.streamlit.app)
- **Main file:** `app/streamlit_app.py`

## Dataset (included)
This repo **includes** a dataset so the live app runs as-is:

- **Primary:** `data/audiolytics.parquet` (fastest to load)  
- **Optional:** `data/Complete_Data_with_genres_filled.csv` (≈32 MB). It’s redundant for the app, but you can keep it for transparency. The app prefers the Parquet.

**Source/enrichment:** Kaggle base + cleaning + Spotify/Reccobeats audio features.

> Dataset is included strictly so reviewers can experience the app end-to-end.

## Getting started (local)
```bash
# clone
git clone https://github.com/gegedobruna/Audiolytics.git
cd Audiolytics

# (optional) create venv
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# install & run
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

---
## License
Code: MIT.  
Dataset: included for demo/review; please don’t redistribute outside this repo context.
