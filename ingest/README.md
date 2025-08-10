# Ingestion scripts
These are the scripts I used to enrich/build the dataset locally.  
The Streamlit app itself runs from `data/audiolytics.parquet`.

## Setup
pip install -r requirements.txt
cp .env.example .env  # fill any keys these scripts need

## Scripts
- `tag_enricher.py` — adds tags/features and writes CSV/Parquet into `data/`.
- `env_loader.py` — tiny helper to load environment variables.
