# Ingestion scripts (optional)

These rebuild local data used by the app. The app itself runs from `data/Complete_Data_with_genres_filled.csv`.

## Setup
pip install -r requirements.txt
copy .env.example .env   # fill in secrets if needed by your scripts

## Scripts
- `tag_enricher.py`: add tags/features to tracks, writes CSVs into `data/`.
