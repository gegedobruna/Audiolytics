
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import os
import re

st.set_page_config(page_title="Audiolytics (Fast+)", layout="wide")

# ---- Platform classification (robust + tidy) ----
PLATFORM_RULES = [
    ("Xbox", r"\bxbox\b|\bxbox\s?(one|series|360)\b"),
    ("PlayStation", r"playstation|ps[345]"),
    ("Web", r"\bweb[_\s-]*player\b|\bbrowser\b|chrome|firefox|edge|safari"),
    ("Android", r"android|huawei|xiaomi|samsung.*android|api\s?\d+"),
    ("iOS", r"\bios\b|iphone|ipad|ipod|watch ?os|watchos"),
    ("macOS", r"\bmac\b|osx|mac\s?os|macos|darwin"),
    ("Windows", r"windows|win32|win64|\bwin\b|windows\s?\d{1,2}|windows\s?10"),
    ("Linux", r"linux|ubuntu|debian|fedora|arch|manjaro"),
    ("TV", r"\btv\b|smart\s?tv|roku|chromecast|partner.*tv|samsung.*tv|apple\s?tv"),
]

def classify_platform(raw: str) -> str:
    if raw is None:
        return "Other"
    s = str(raw).strip().lower().replace("_", " ")
    s = re.sub(r"[()\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    for label, pat in PLATFORM_RULES:
        if re.search(pat, s):
            return label
    if s in {"not applicable", "n/a", "na", "unknown", ""}:
        return "Other"
    return "Other"


def render_streaks_calendar(dff, key_prefix="time_tab"):
    import pandas as pd, numpy as np, altair as alt
    st.subheader("Listening Streaks")
    st.markdown("Green squares mark days that met the threshold.")

    threshold = st.number_input(
        "Streak threshold (minutes/day)", 1, 180, 15, 5,
        key=f"{key_prefix}_threshold"
    )
    weeks_back = st.slider(
        "Show last N weeks", 8, 52, 26, 2,
        key=f"{key_prefix}_weeks"
    )

    if "date" not in dff.columns or dff.empty:
        st.info("No data in the current filter.")
        return

    df_day = (dff.groupby("date", as_index=False)["ms_played"].sum()
                .assign(minutes=lambda x: x["ms_played"]/60000))
    df_day["date"] = pd.to_datetime(df_day["date"])
    df_day = df_day.sort_values("date")
    df_day["meets"] = df_day["minutes"] >= threshold

    # metrics
    grp = (df_day["meets"] != df_day["meets"].shift()).cumsum()
    df_day["streak_id"] = grp.where(df_day["meets"])
    streak_sizes = df_day.groupby("streak_id", dropna=True).size()
    longest = int(streak_sizes.max()) if not streak_sizes.empty else 0
    current = 0
    if not df_day.empty and df_day.iloc[-1]["meets"]:
        sid = df_day.iloc[-1]["streak_id"]
        current = int((df_day["streak_id"] == sid).sum())
    c1, c2 = st.columns(2)
    with c1: st.metric("Current streak", f"{current} days")
    with c2: st.metric("Longest streak", f"{longest} days")

    # Build calendar grid (GitHub-style) for the last N weeks
    all_days = pd.DataFrame({"date": pd.date_range(df_day["date"].min(), df_day["date"].max(), freq="D")})
    all_days = all_days.merge(df_day[["date","meets","minutes"]], on="date", how="left").fillna({"meets": False, "minutes": 0})
    all_days["week"] = all_days["date"].dt.isocalendar().week.astype(int)
    all_days["year"] = all_days["date"].dt.isocalendar().year.astype(int)
    all_days["dow"] = all_days["date"].dt.dayofweek  # 0=Mon

    # Normalize to a running week index so we can slice last N weeks easily
    all_days["week_start"] = all_days["date"] - pd.to_timedelta(all_days["date"].dt.dayofweek, unit="D")
    all_days = all_days.sort_values("week_start")
    all_days["week_idx"] = (all_days["week_start"].astype("int64") // 10**9) // (7*24*3600)  # integer week counter

    max_week = all_days["week_idx"].max()
    keep = all_days[all_days["week_idx"] >= max_week - weeks_back + 1]

    heat = alt.Chart(keep).mark_rect().encode(
        x=alt.X("week_idx:O", title=""),
        y=alt.Y("dow:O", title="", sort=[0,1,2,3,4,5,6],
                axis=alt.Axis(labelExpr='["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][datum.value]')),
        color=alt.condition("datum.meets", alt.value("#34a853"), alt.value("#e0e0e0")),
        tooltip=[alt.Tooltip("date:T", title="Date"),
                 alt.Tooltip("minutes:Q", title="Minutes", format=".1f"),
                 alt.Tooltip("meets:N", title="Met threshold")]
    ).properties(height=140)

    st.altair_chart(heat, use_container_width=True)
    st.caption("Green = met threshold. Adjust weeks and threshold above.")


# ---------------------- I/O & CACHING ----------------------
CSV_CANDIDATES = [
    "Complete_Data.csv",
    "./data/Complete_Data.csv",
    "/mnt/data/Complete_Data.csv"
]
PARQUET_CANDIDATES = [
    "data/audiolytics.parquet",
    "./data/audiolytics.parquet",
    "/mnt/data/audiolytics.parquet",
]

@st.cache_data(show_spinner=False)
def to_parquet_once() -> str:
    # If a parquet already exists, just use it
    for p in PARQUET_CANDIDATES:
        if os.path.exists(p):
            return p
    # Else convert from first existing CSV
    for c in CSV_CANDIDATES:
        if os.path.exists(c):
            df = pd.read_csv(c)
            out = "data/audiolytics.parquet"
            df.to_parquet(out, index=False)
            return out
    st.error("No input file found. Place 'Complete_Data.csv' next to this app or under ./data/.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_df() -> pd.DataFrame:
    pq = to_parquet_once()
    df = pd.read_parquet(pq)
    return df

# ---------------------- UTIL & PREP ----------------------
WINDOWS_ALIASES = {"windows","windows desktop","windows 10","win32","windows store","microsoft store","windows phone","windows-10","windows10","win64"}

@st.cache_data(show_spinner=False)
def optimize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    # Coerce numerics
    num_cols = ["ms_played", "acousticness", "danceability", "energy",
                "instrumentalness", "liveness", "loudness", "speechiness",
                "tempo", "valence"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Downcast
    for c in df.select_dtypes(include=["float", "int"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")

    # Add stable row_id for fast joins
    df["row_id"] = np.arange(len(df), dtype="int64")

    # Time features
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df["date"] = df["ts"].dt.date
        df["year"] = df["ts"].dt.year
        df["month"] = df["ts"].dt.to_period("M").astype(str)
        df["hour"] = df["ts"].dt.hour
        df["dow"] = df["ts"].dt.day_name()

    # Categoricals
    for c in ["master_metadata_album_artist_name","master_metadata_track_name",
              "platform","conn_country","reason_start","reason_end"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    
    # Platform normalization (regex grouping → tidy buckets)
    if "platform" in df.columns:
        df["platform"] = df["platform"].astype(str).str.strip()
        df["platform_norm"] = df["platform"].map(classify_platform).astype("category")
    else:
        df["platform_norm"] = pd.Series(dtype="category")
    # Alias for plotting/grouping
    df["platform_group"] = df["platform_norm"]


    # Genres list
    if "artist_genres" in df.columns:
        def split_genres(x):
            if pd.isna(x): return []
            s = str(x)
            for d in ["|",";","||"," / ",","]:
                if d in s:
                    return [p.strip() for p in s.split(d) if p.strip()]
            return [s.strip()] if s.strip() else []
        df["artist_genres_list"] = df["artist_genres"].apply(split_genres)
    return df

@st.cache_data(show_spinner=False)
def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    if "artist_genres_list" not in df.columns:
        return df.head(0)
    temp = df.copy()
    temp["artist_genres_list"] = temp["artist_genres_list"].apply(tuple)
    return temp.explode("artist_genres_list")

@st.cache_data(show_spinner=False)
def build_exploded_all(df: pd.DataFrame) -> pd.DataFrame:
    """Explode FULL dataset once and cache. Returns: row_id, month, artist_genres_list, ms_played."""
    if "artist_genres_list" not in df.columns:
        return df.head(0)
    tmp = df[["row_id","month","artist_genres_list","ms_played"]].copy()
    tmp["artist_genres_list"] = tmp["artist_genres_list"].apply(tuple)
    exp = tmp.explode("artist_genres_list").dropna(subset=["artist_genres_list"])
    return exp

@st.cache_data(show_spinner=False)
def get_exploded_for_filter(dff: pd.DataFrame, exploded_all: pd.DataFrame) -> pd.DataFrame:
    """Subset pre-exploded ALL rows to the current filtered rows via row_id."""
    if "row_id" not in dff.columns or exploded_all.empty:
        return dff.head(0)
    ids = dff["row_id"].astype("int64").unique()
    sub = exploded_all[exploded_all["row_id"].isin(ids)].copy()
    return sub

@st.cache_data(show_spinner=False)
def first_last_seen(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Return per-id first_seen, last_seen dates. Assumes df has 'date'."""
    if "date" not in df.columns:
        return pd.DataFrame(columns=[id_col, "first_seen", "last_seen"]).astype({id_col:"category"})
    g = df.groupby(id_col)["date"].agg(["min","max"]).reset_index().rename(columns={"min":"first_seen","max":"last_seen"})
    return g

@st.cache_data(show_spinner=False)
def month_genre_pivot_minutes(exploded: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows=month, cols=genre, values=minutes (ms_played/60000)."""
    if not {"month","artist_genres_list","ms_played"}.issubset(exploded.columns):
        return pd.DataFrame()
    tmp = exploded.groupby(["month","artist_genres_list"], as_index=False)["ms_played"].sum()
    tmp["minutes"] = tmp["ms_played"]/60000
    pivot = tmp.pivot_table(index="month", columns="artist_genres_list", values="minutes", fill_value=0)
    pivot = pivot.sort_index()
    return pivot

def shannon_entropy(proportions: np.ndarray) -> float:
    p = proportions[proportions > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

@st.cache_data(show_spinner=False)
def monthly_genre_entropy(exploded: pd.DataFrame) -> pd.DataFrame:
    pivot = month_genre_pivot_minutes(exploded)
    if pivot.empty:
        return pd.DataFrame(columns=["month","entropy"])
    shares = pivot.div(pivot.sum(axis=1), axis=0).fillna(0)
    ent = shares.apply(lambda row: shannon_entropy(row.values), axis=1)
    out = ent.reset_index().rename(columns={0:"entropy"})
    return out

@st.cache_data(show_spinner=False)
def build_sessions(dfin: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """Reconstruct sessions per day with a time gap threshold."""
    if "ts" not in dfin.columns:
        return pd.DataFrame()
    df = dfin.sort_values("ts").copy()
    # A new session starts if time gap from previous play > threshold
    gap = pd.to_timedelta(gap_minutes, unit="m")
    df["prev_ts"] = df["ts"].shift(1)
    df["new_session"] = (df["prev_ts"].isna()) | ((df["ts"] - df["prev_ts"]) > gap)
    df["session_id"] = df["new_session"].cumsum()
    # Aggregate to session level
    feat_cols = [c for c in ["acousticness","danceability","energy","instrumentalness","liveness",
                             "loudness","speechiness","tempo","valence"] if c in df.columns]
    agg = {
        "ts": ["min","max","count"],
        "ms_played": "sum",
    }
    for c in feat_cols:
        agg[c] = "mean"
    sess = df.groupby("session_id").agg(agg)
    # flatten columns
    sess.columns = ["_".join([c for c in col if c]) for col in sess.columns.values]
    sess = sess.reset_index()
    if "ts_min" in sess.columns:
        sess["date"] = pd.to_datetime(sess["ts_min"]).dt.date
        sess["duration_min"] = sess["ms_played_sum"] / 60000.0
        sess["plays"] = sess["ts_count"]
    return sess

@st.cache_data(show_spinner=False)
def cluster_sessions(sess: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Cluster sessions by mood features; return dataframe with cluster labels."""
    feats = [c for c in ["energy_mean","valence_mean","danceability_mean","loudness_mean","tempo_mean"] if c in sess.columns]
    if len(feats) < 2 or len(sess) < k:
        return pd.DataFrame()
    X = sess[feats].fillna(sess[feats].mean())
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=512, n_init="auto")
    labels = km.fit_predict(X)
    out = sess.copy()
    out["cluster"] = labels
    # Human-friendly labels by centroid ordering
    centers = pd.DataFrame(km.cluster_centers_, columns=feats)
    # Heuristic label based on energy/valence/danceability
    centers["score"] = centers.get("energy_mean",0) + centers.get("valence_mean",0) + centers.get("danceability_mean",0)
    order = centers["score"].rank(method="dense").astype(int)
    label_map = {}
    names = ["Chill", "Focus Mode", "Melancholy", "Groovy", "Party", "Hype"]
    # Map lowest score -> "Chill", highest -> "Hype"/"Party"
    for idx in range(len(centers)):
        r = order.iloc[idx] - 1
        r = max(0, min(r, len(names)-1))
        label_map[idx] = names[r]
    out["session_label"] = out["cluster"].map(label_map)
    return out

@st.cache_data(show_spinner=False)
def discovery_by_month(df: pd.DataFrame, id_col_track: str, id_col_artist: str):
    """Return monthly new vs repeat shares plus a Discovery Score."""
    if "date" not in df.columns or "month" not in df.columns:
        return pd.DataFrame()
    # First seen maps
    ft = df.groupby(id_col_track)["date"].min().rename("track_first_seen")
    fa = df.groupby(id_col_artist)["date"].min().rename("artist_first_seen")
    d = df.copy()
    d = d.join(ft, on=id_col_track)
    d = d.join(fa, on=id_col_artist)
    d["is_new_track"] = d["date"] == d["track_first_seen"]
    d["is_new_artist"] = d["date"] == d["artist_first_seen"]
    m = d.groupby("month").agg(
        plays=("date","count"),
        new_track=("is_new_track","sum"),
        new_artist=("is_new_artist","sum"),
    ).reset_index()
    m["new_track_share"] = m["new_track"] / m["plays"]
    m["new_artist_share"] = m["new_artist"] / m["plays"]
    m["discovery_score"] = 0.7*m["new_track_share"] + 0.3*m["new_artist_share"]
    return m

@st.cache_data(show_spinner=False)
def forgotten_tracks(df: pd.DataFrame, top_n_per_month: int = 20, stale_months: int = 12):
    """Tracks that were once top-N in a month but haven't been played in `stale_months`."""
    if "month" not in df.columns or "date" not in df.columns:
        return pd.DataFrame()
    if "master_metadata_track_name" not in df.columns:
        return pd.DataFrame()
    tcol = "master_metadata_track_name"
    # rank by plays per month
    m = df.groupby(["month", tcol], as_index=False)["ms_played"].sum()
    m["rank"] = m.groupby("month")["ms_played"].rank(method="first", ascending=False)
    big = m[m["rank"] <= top_n_per_month].copy()
    # last seen date per track
    last_play = df.groupby(tcol)["date"].max().rename("last_seen")
    first_play = df.groupby(tcol)["date"].min().rename("first_seen")
    big = big.join(last_play, on=tcol).join(first_play, on=tcol)
    cutoff = pd.to_datetime(df["date"]).max() - pd.DateOffset(months=stale_months)
    big["stale"] = pd.to_datetime(big["last_seen"]) < cutoff
    out = big[big["stale"]].sort_values(["month","rank"]).copy()
    return out[["month", tcol, "ms_played", "rank", "first_seen", "last_seen"]].reset_index(drop=True)

# ---------------------- LOAD & PREP ----------------------
df_raw = load_df()
df = optimize(df_raw)
exp_all = build_exploded_all(df)

artist_col = "master_metadata_album_artist_name" if "master_metadata_album_artist_name" in df.columns else None
track_col = "master_metadata_track_name" if "master_metadata_track_name" in df.columns else None

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("Filters & Performance")
fast_mode = st.sidebar.checkbox("⚡ Fast mode (sample for heavy charts)", value=True)
MAX_ROWS = st.sidebar.number_input("Max rows in Fast mode", min_value=5_000, max_value=200_000, value=30_000, step=5_000)
run_pca = st.sidebar.checkbox("Enable PCA (can be slow)", value=False)
enable_sessions = st.sidebar.checkbox("Reconstruct & cluster sessions (slow)", value=False)
k_clusters = st.sidebar.slider("Session clusters (k)", 3, 8, 5)

lazy_heavy = st.sidebar.checkbox("Lazy compute heavy sections", value=True, help="Heavy charts compute only when you click a button")

gap_mins = st.sidebar.slider("Session gap threshold (minutes)", 10, 60, 30, 5)

# Date range
if "date" in df.columns:
    min_date = pd.to_datetime(df["date"]).min()
    max_date = pd.to_datetime(df["date"]).max()
    dr = st.sidebar.date_input("Date range", (min_date, max_date))
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        start, end = dr
    else:
        start, end = min_date, max_date
else:
    start = end = None

# Genres
if "artist_genres_list" in df.columns:
    all_genres = sorted({g for lst in df["artist_genres_list"].dropna() for g in lst})
    selected_genres = st.sidebar.multiselect("Genres", options=all_genres, default=[])
else:
    selected_genres = []

# Artists
if artist_col is not None:
    top_artists = df[artist_col].value_counts().head(300).index.tolist()
    selected_artists = st.sidebar.multiselect("Artists (top 300)", options=top_artists, default=[])
else:
    selected_artists = []

platforms = sorted(df["platform_norm"].dropna().unique()) if "platform_norm" in df.columns else []
selected_platforms = st.sidebar.multiselect("Platforms", options=platforms, default=[])

countries = sorted(df["conn_country"].dropna().unique()) if "conn_country" in df.columns else []
selected_countries = st.sidebar.multiselect("Countries", options=countries, default=[])

# ---------------------- FILTER ----------------------
mask = pd.Series(True, index=df.index)
if start and end and "date" in df.columns:
    mask &= (pd.to_datetime(df["date"]) >= pd.to_datetime(start)) & (pd.to_datetime(df["date"]) <= pd.to_datetime(end))
if selected_genres and "artist_genres_list" in df.columns:
    mask &= df["artist_genres_list"].apply(lambda lst: any(g in selected_genres for g in lst) if isinstance(lst, list) else False)
if selected_artists and artist_col is not None:
    mask &= df[artist_col].isin(selected_artists)
if selected_platforms and "platform_norm" in df.columns:
    mask &= df["platform_norm"].isin(selected_platforms)
if selected_countries and "conn_country" in df.columns:
    mask &= df["conn_country"].isin(selected_countries)

dff = df[mask].copy()

# Decide sampled frame for heavy charts
if fast_mode and len(dff) > MAX_ROWS:
    dff_plot = dff.sample(int(MAX_ROWS), random_state=42)
else:
    dff_plot = dff

st.title("🎧 Audiolytics — Fast+")
st.caption("Optimized loading, cached transforms, sampling for heavy charts, and richer insights with explanations.")

# ---------------------- KPIs ----------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Plays", f"{len(dff):,}")
with c2:
    minutes = (dff["ms_played"].fillna(0).sum()/60000) if "ms_played" in dff.columns else 0
    st.metric("Total Minutes", f"{minutes:,.1f}")
with c3:
    ua = dff[artist_col].nunique() if artist_col else 0
    st.metric("Unique Artists", f"{ua:,}")
with c4:
    ug = len({g for lst in dff.get("artist_genres_list", pd.Series([])).dropna() for g in lst}) if "artist_genres_list" in dff.columns else 0
    st.metric("Unique Genres", f"{ug:,}")

st.divider()

# Tabs — keep originals, then add new ones
tabs = st.tabs([
    "📈 Overview","🎭 Genres","👩‍🎤 Artists","🔊 Features","🕒 Time","🌍 Platform/Country","🧮 Corr","🧪 PCA",
    "🧠 Mood & Energy","🧭 Discovery","🕰️ Sessions","🧩 Personality","🎞️ Nostalgia"
])

# ---------------------- OVERVIEW ----------------------
with tabs[0]:
    st.subheader("Daily Minutes")
    st.markdown("This line shows **total listening minutes per day** in your current filter. Use it to spot streaks, dips, and binge days.")
    if "date" in dff_plot.columns:
        daily = dff_plot.groupby("date", as_index=False)["ms_played"].sum()
        daily["minutes"] = daily["ms_played"]/60000
        line = alt.Chart(daily).mark_line(point=True).encode(
            x="date:T", y="minutes:Q", tooltip=["date:T","minutes:Q"]
        ).properties(height=300)
        st.altair_chart(line, use_container_width=True)

# ---------------------- GENRES ----------------------
with tabs[1]:
    st.subheader("Genre Share Over Time (stacked)")
    st.markdown("Shows how **genre shares** shift over months. Each band's thickness is that genre's share of total listening.")

    if not lazy_heavy or st.button("Compute Genre Stack / Entropy", key="compute_genres"):
        exploded = get_exploded_for_filter(dff_plot, exp_all)
        if exploded.empty:
            st.info("No genre data in the current filter.")
        else:
            g = exploded.groupby(["artist_genres_list","month"], as_index=False)["ms_played"].sum()
            g["minutes"] = g["ms_played"] / 60000

            area = alt.Chart(g).mark_area(opacity=0.8).encode(
                x="month:T",
                y=alt.Y("minutes:Q", stack="normalize", title="Share"),
                color="artist_genres_list:N",
                tooltip=["month:T","artist_genres_list:N","minutes:Q"]
            ).properties(height=320)
            st.altair_chart(area, use_container_width=True)

            st.subheader("Genre Entropy (Variety)")
            ent = monthly_genre_entropy(exploded)
            if not ent.empty:
                ent_line = alt.Chart(ent).mark_line(point=True).encode(
                    x="month:T", y=alt.Y("entropy:Q", title="Shannon Entropy"),
                    tooltip=["month:T","entropy:Q"]
                ).properties(height=260)
                st.altair_chart(ent_line, use_container_width=True)
            else:
                st.info("Not enough data to compute entropy.")

            # --- Entry/Exit + Rising/Falling ---
            st.subheader("Genre Entry & Exit")
            first_last = (exploded.groupby("artist_genres_list")["month"]
                          .agg(["min","max","nunique"])
                          .reset_index()
                          .rename(columns={"min":"first_month","max":"last_month","nunique":"active_months"}))
            st.dataframe(first_last.sort_values("first_month").head(200), use_container_width=True)

            st.subheader("Rising vs Falling Genres")
            gm = (exploded.groupby(["month","artist_genres_list"], as_index=False)["ms_played"].sum()
                  .assign(minutes=lambda x: x["ms_played"]/60000)
                  .sort_values(["artist_genres_list","month"]))
            gm["mom"] = gm.groupby("artist_genres_list")["minutes"].pct_change()
            latest_month = gm["month"].max()
            last_slice = gm[gm["month"] == latest_month].dropna(subset=["mom"])
            top_rising = last_slice.sort_values("mom", ascending=False).head(15)
            top_falling = last_slice.sort_values("mom", ascending=True).head(15)

            rbar = alt.Chart(top_rising).mark_bar().encode(
                x=alt.X("mom:Q", title="MoM Growth"),
                y=alt.Y("artist_genres_list:N", sort="-x"),
                tooltip=["artist_genres_list","mom:Q"]
            ).properties(height=280, title=f"Top Rising Genres — {latest_month}")
            fbar = alt.Chart(top_falling).mark_bar().encode(
                x=alt.X("mom:Q", title="MoM Growth"),
                y=alt.Y("artist_genres_list:N", sort="x"),
                tooltip=["artist_genres_list","mom:Q"]
            ).properties(height=280, title=f"Top Falling Genres — {latest_month}")
            st.altair_chart(rbar, use_container_width=True)
            st.altair_chart(fbar, use_container_width=True)
    else:
        st.caption("Lazy compute is ON — click the button above to render heavy genre charts.")
# ---------------------- ARTISTS ----------------------
with tabs[2]:
    if artist_col:
        st.subheader("Artist Comeback Tracker")
        if "date" in dff.columns:
            min_gap = st.number_input(
                "Minimum absence (months) to count as a comeback",
                1, 36, 6, 1,
                key="artist_comeback_min_gap"
            )
            top_k = st.slider("Show top N comebacks", 5, 200, 50, 5, key="artist_comeback_topk")

            dd = dff[[artist_col, "date"]].copy()
            dd["date"] = pd.to_datetime(dd["date"])

            # monthly presence per artist
            monthly = dd.groupby(
                [artist_col, dd["date"].dt.to_period("M").astype(str)],
                as_index=False
            ).size()
            monthly.rename(columns={"size": "plays"}, inplace=True)

            def comeback_events(gdf):
                # gdf columns: [artist, month_period_str, plays]
                gdf = gdf.sort_values(by=gdf.columns[1])
                months = pd.to_datetime(gdf.iloc[:, 1]).dt.to_period("M")
                out = []
                if len(months) <= 1:
                    return out
                diffs = months[1:].astype(int).values - months[:-1].astype(int).values
                for i, gap in enumerate(diffs):
                    if gap >= min_gap:
                        out.append({
                            "gap_months": int(gap),
                            "last_before": months.iloc[i].strftime("%Y-%m"),
                            "return_month": months.iloc[i+1].strftime("%Y-%m"),
                        })
                return out

            rows = []
            for a, g in monthly.groupby(artist_col):
                for e in comeback_events(g):
                    rows.append({"artist": a, **e})

            cmb = pd.DataFrame(rows)
            if not cmb.empty:
                cmb = cmb.sort_values(["gap_months", "artist"], ascending=[False, True]).head(top_k)
                st.dataframe(cmb, use_container_width=True, hide_index=True)
            else:
                st.info("No comebacks found at this threshold.")
        else:
            st.info("No date column found, can’t compute comebacks.")

        st.subheader("Most Played Artists (by plays)")
        st.markdown("Simple **play counts** by artist within your filters.")
        a = dff.groupby(artist_col, as_index=False).size().rename(columns={"size": "plays"})
        a = a.sort_values("plays", ascending=False).head(50)
        chart = alt.Chart(a).mark_bar().encode(
            x="plays:Q",
            y=alt.Y(f"{artist_col}:N", sort="-x"),
            tooltip=[artist_col, "plays:Q"]
        ).properties(height=600)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Artist column not found in this dataset.")
# ---------------------- FEATURES ----------------------
# ---------------------- FEATURES ----------------------
with tabs[3]:
    st.subheader("Feature Distributions")
    st.markdown("Distributions for Spotify-like audio features. Use filters to see how your tastes shift.")
    feats = [c for c in ["acousticness","danceability","energy","instrumentalness",
                         "liveness","loudness","speechiness","tempo","valence"]
             if c in dff_plot.columns]
    cols = st.columns(3)
    for i, f in enumerate(feats[:9]):
        with cols[i % 3]:
            hist = alt.Chart(dff_plot).mark_bar().encode(
                x=alt.X(f"{f}:Q", bin=alt.Bin(maxbins=40)),
                y="count()", tooltip=[f"{f}:Q","count()"]
            ).properties(height=220, title=f)
            st.altair_chart(hist, use_container_width=True)

    if len(feats) >= 2:
        st.subheader("2D Feature Scatter (sampled)")
        st.markdown("Plot any two features to see **clusters of sound**. In Fast mode we sample to keep things snappy.")
        x = st.selectbox("X", feats, index=1 if len(feats)>1 else 0)
        y = st.selectbox("Y", feats, index=2 if len(feats)>2 else 0)
        color = "artist_genres_list:N" if "artist_genres_list" in dff_plot.columns else None
        dfp = explode_genres(dff_plot) if color else dff_plot
        scatter = alt.Chart(dfp.dropna(subset=[x,y])).mark_circle(opacity=0.4).encode(
            x=f"{x}:Q", y=f"{y}:Q", color=color, tooltip=[x,y]
        ).properties(height=420)
        st.altair_chart(scatter, use_container_width=True)

# ---------------------- TIME ----------------------
with tabs[4]:
    st.subheader("Hour × Day Heatmap (minutes)")
    st.markdown("Dark squares = **heavier listening**. Use it to find your **prime times**.")
    if "hour" in dff.columns and "dow" in dff.columns:
        heat = dff.groupby(["dow","hour"], as_index=False)["ms_played"].sum()
        heat["minutes"] = heat["ms_played"]/60000
        heat["dow"] = pd.Categorical(heat["dow"], categories=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], ordered=True)
        h = alt.Chart(heat).mark_rect().encode(
            x="hour:O", y=alt.Y("dow:N", sort=list(heat["dow"].cat.categories)), color="minutes:Q",
            tooltip=["dow","hour","minutes:Q"]
        ).properties(height=300)
        st.altair_chart(h, use_container_width=True)

    st.subheader("Seasonality (Monthly Minutes by Year)")
    st.markdown("Compares **monthly listening** across years. Look for **summer spikes** or **exam-season dips**.")
    if "month" in dff.columns and "year" in dff.columns:
        monthly = dff.groupby(["year","month"], as_index=False)["ms_played"].sum()
        monthly["minutes"] = monthly["ms_played"]/60000
        line = alt.Chart(monthly).mark_line(point=True).encode(
            x="month:T", y="minutes:Q", color="year:N", tooltip=["year:N","month:T","minutes:Q"]
        ).properties(height=320)
        st.altair_chart(line, use_container_width=True)
    render_streaks_calendar(dff, key_prefix="time_tab")
# ---------------------- PLATFORM / COUNTRY ----------------------
with tabs[5]:
    if "platform_group" in dff_plot.columns and "month" in dff_plot.columns:
        st.subheader("Platform Mix Over Time")
        st.markdown("Platforms are **normalized** (e.g., Windows variants → Windows). Use *Top N + Other* to keep it tidy.")

        topn = st.slider("Show top N platforms (rest as Other)", 3, 12, 6, 1, key="platform_topn")
        plat0 = dff_plot[["month","platform_group","ms_played"]].copy()

        # Global top platforms by minutes
        totals = plat0.groupby("platform_group")["ms_played"].sum().sort_values(ascending=False)
        tops = list(totals.head(int(topn)).index)
        plat0["platform_group_top"] = np.where(plat0["platform_group"].isin(tops), plat0["platform_group"], "Other")

        plat = plat0.groupby(["month","platform_group_top"], as_index=False)["ms_played"].sum()
        plat["minutes"] = plat["ms_played"]/60000

        # Legend order: Top groups then Other
        order_tot = plat.groupby("platform_group_top")["minutes"].sum().sort_values(ascending=False).index.tolist()
        if "Other" in order_tot:
            order_tot = [x for x in order_tot if x != "Other"] + ["Other"]

        area = alt.Chart(plat).mark_area(opacity=0.85).encode(
            x="month:T",
            y=alt.Y("minutes:Q", stack="normalize", title="Share"),
            color=alt.Color("platform_group_top:N", sort=order_tot, title="Platform"),
            tooltip=["month:T","platform_group_top:N","minutes:Q"]
        ).properties(height=300)
        st.altair_chart(area, use_container_width=True)

        with st.expander("Audit platform mapping (top raw labels → group)"):
            if "platform" in dff.columns:
                raw_counts = (dff["platform"].astype(str).value_counts().head(40).rename_axis("raw").reset_index(name="rows"))
                raw_counts["mapped_group"] = raw_counts["raw"].map(classify_platform)
                st.dataframe(raw_counts, use_container_width=True, hide_index=True)
            else:
                st.caption("No raw platform column found.")
        

    if "conn_country" in dff.columns:
        st.subheader("Listening by Country")
        st.markdown("Which countries do you listen from the most?")
        ctry = dff.groupby("conn_country", as_index=False)["ms_played"].sum()
        ctry["minutes"] = ctry["ms_played"]/60000
        bar = alt.Chart(ctry.sort_values("minutes", ascending=False).head(30)).mark_bar().encode(
            x="minutes:Q", y=alt.Y("conn_country:N", sort="-x"), tooltip=["conn_country","minutes:Q"]
        ).properties(height=500)
        st.altair_chart(bar, use_container_width=True)

# ---------------------- CORR ----------------------
with tabs[6]:
    st.subheader("Feature Correlation")
    st.markdown("Blue/orange shows **positive/negative** correlation between audio features.")
    feats = [c for c in ["acousticness","danceability","energy","instrumentalness",
                         "liveness","loudness","speechiness","tempo","valence"]
             if c in dff.columns]
    if len(feats) >= 2:
        corr = dff[feats].corr().reset_index().melt(id_vars="index", var_name="feature", value_name="corr")
        heat = alt.Chart(corr).mark_rect().encode(
            x=alt.X("feature:N", title=""),
            y=alt.Y("index:N", title=""),
            color=alt.Color("corr:Q", scale=alt.Scale(scheme="blueorange", domain=[-1,1])),
            tooltip=["index","feature","corr"]
        ).properties(height=380)
        st.altair_chart(heat, use_container_width=True)

    st.subheader("Correlation Over Time (Compare Eras)")
    feats = [c for c in ["acousticness","danceability","energy","instrumentalness",
                         "liveness","loudness","speechiness","tempo","valence"] if c in dff.columns]
    if len(feats) >= 2 and "date" in dff.columns:
        left, right = st.columns(2)
        with left:
            st.caption("Era A")
            era_a = st.date_input("Date range A", (pd.to_datetime(dff["date"]).min(), pd.to_datetime(dff["date"]).quantile(0.5)))
        with right:
            st.caption("Era B")
            era_b = st.date_input("Date range B", (pd.to_datetime(dff["date"]).quantile(0.5), pd.to_datetime(dff["date"]).max()))
        def _corr_range(r):
            if isinstance(r, (list, tuple)) and len(r)==2:
                m = (pd.to_datetime(dff["date"]) >= pd.to_datetime(r[0])) & (pd.to_datetime(dff["date"]) <= pd.to_datetime(r[1]))
                sub = dff.loc[m, feats].dropna()
                return sub.corr()
            return None
        ca = _corr_range(era_a)
        cb = _corr_range(era_b)
        if ca is not None and cb is not None and not ca.empty and not cb.empty:
            a_long = ca.reset_index().melt(id_vars="index", var_name="feature", value_name="corr")
            b_long = cb.reset_index().melt(id_vars="index", var_name="feature", value_name="corr")
            a_chart = alt.Chart(a_long).mark_rect().encode(x="feature:N", y="index:N", color=alt.Color("corr:Q", scale=alt.Scale(scheme="blueorange", domain=[-1,1]))).properties(height=260, title="Era A")
            b_chart = alt.Chart(b_long).mark_rect().encode(x="feature:N", y="index:N", color=alt.Color("corr:Q", scale=alt.Scale(scheme="blueorange", domain=[-1,1]))).properties(height=260, title="Era B")
            st.altair_chart(a_chart, use_container_width=True)
            st.altair_chart(b_chart, use_container_width=True)
            delta = (cb - ca).fillna(0)
            d_long = delta.reset_index().melt(id_vars="index", var_name="feature", value_name="delta")
            d_chart = alt.Chart(d_long).mark_rect().encode(x="feature:N", y="index:N", color=alt.Color("delta:Q", scale=alt.Scale(scheme="redblue", domain=[-1,1]))).properties(height=280, title="Δ Correlation (Era B - Era A)")
            st.altair_chart(d_chart, use_container_width=True)
        else:
            st.info("Pick valid date ranges with enough data to compute correlations.")
    
# ---------------------- PCA ----------------------
with tabs[7]:
    st.subheader("PCA (2D)")
    st.markdown("Reduces many features into **two components** so you can see **clusters**. Toggle in the sidebar to run.")
    if run_pca:
        feats = [c for c in ["acousticness","danceability","energy","instrumentalness",
                             "liveness","loudness","speechiness","tempo","valence"]
                 if c in dff_plot.columns]
        if len(feats) >= 2:
            X = dff_plot[feats].dropna()
            if len(X) > 10:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X.values)
                pca = PCA(n_components=2)
                comps = pca.fit_transform(Xs)
                pca_df = pd.DataFrame(comps, columns=["PC1","PC2"])
                if "artist_genres_list" in dff_plot.columns:
                    ex = explode_genres(dff_plot.loc[X.index])
                    pca_df = pca_df.iloc[:len(ex)]
                    pca_df["genre"] = ex["artist_genres_list"].values[:len(pca_df)]
                    color = "genre:N"
                else:
                    color = alt.value("steelblue")
                expl = (pca.explained_variance_ratio_ * 100).round(1)
                scatter = alt.Chart(pca_df).mark_circle(opacity=0.5).encode(
                    x=alt.X("PC1:Q", title=f"PC1 ({expl[0]}%)"),
                    y=alt.Y("PC2:Q", title=f"PC2 ({expl[1]}%)"),
                    color=color
                ).properties(height=420)
                st.altair_chart(scatter, use_container_width=True)
            else:
                st.info("Not enough complete rows for PCA.")
        else:
            st.info("Need at least two feature columns.")
    else:
        st.caption("Enable PCA from the sidebar to compute (on sampled data if Fast mode is on).")

# ---------------------- MOOD & ENERGY ----------------------
with tabs[8]:
    st.subheader("Mood Landscape (Valence × Energy)")
    st.markdown("Instead of dots, we **bin** plays into a grid so dense areas stand out. Color shows **play density**; optional overlay shows **avg danceability**.")

    if {"valence","energy"}.issubset(dff_plot.columns):
        dfp = dff_plot.copy()
        # Optional: facet by top genre to break ties visually
        facet = st.checkbox("Facet by top genre (optional)", value=False)
        if "artist_genres_list" in dfp.columns:
            dfp_ex = explode_genres(dfp)
        else:
            dfp_ex = dfp
            dfp_ex["artist_genres_list"] = None

        # Build 2D binned heatmap
        base = alt.Chart(dfp_ex.dropna(subset=["valence","energy"])).transform_bin(
            ["val_bin", "valence"], field="valence", bin=alt.Bin(maxbins=24)
        ).transform_bin(
            ["eng_bin", "energy"], field="energy", bin=alt.Bin(maxbins=24)
        ).transform_aggregate(
            count="count()", avg_danceability="mean(danceability)",
            groupby=["val_bin","eng_bin"] + (["artist_genres_list"] if facet else [])
        )

        heat = base.mark_rect().encode(
            x=alt.X("val_bin:Q", title="Valence →"),
            y=alt.Y("eng_bin:Q", title="Energy ↑"),
            color=alt.Color("count:Q", title="Plays", scale=alt.Scale(scheme="greens")),
            tooltip=[alt.Tooltip("count:Q", title="Plays"), alt.Tooltip("avg_danceability:Q", title="Avg danceability", format=".2f")]
        ).properties(height=420)

        # Optional overlay: avg danceability bubbles
        if st.checkbox("Overlay avg danceability bubbles", value=True):
            bubbles = base.mark_circle(opacity=0.5).encode(
                x="val_bin:Q", y="eng_bin:Q",
                size=alt.Size("avg_danceability:Q", title="Avg danceability"),
                tooltip=[alt.Tooltip("avg_danceability:Q", format=".2f")]
            )
            heat = heat + bubbles

        if facet:
            heat = heat.facet(column=alt.Column("artist_genres_list:N", title=None, header=alt.Header(labelAngle=0)))

        st.altair_chart(heat.resolve_scale(color="independent"), use_container_width=True)
    else:
        st.info("Missing valence/energy columns.")

    st.subheader("Mood Swings Over Time")
    st.markdown("Rolling averages of **valence** and **energy** reveal shifts from **chill** to **hype** periods. Try filtering to summer months.")
    if "month" in dff.columns and {"valence","energy"}.issubset(dff.columns):
        m = dff.groupby("month", as_index=False)[["valence","energy"]].mean().sort_values("month")
        m["valence_rolling"] = m["valence"].rolling(3, min_periods=1).mean()
        m["energy_rolling"] = m["energy"].rolling(3, min_periods=1).mean()
        base = alt.Chart(m).encode(x="month:T")
        vline = base.mark_line(point=True).encode(y=alt.Y("valence_rolling:Q", title="Rolling (3)", scale=alt.Scale(domain=[0.4, 0.7])))
    eline = base.mark_line(point=True).encode(y=alt.Y("energy_rolling:Q", scale=alt.Scale(domain=[0.4, 0.7])), color=alt.value("orange"))
    st.altair_chart(vline.properties(height=280) + eline.properties(height=280), use_container_width=True)

    st.subheader("Are You More Energetic in Summer?")
    st.markdown("Compares **mean energy** by **month**. Look for peaks in June–August.")
    if "month" in dff.columns and "energy" in dff.columns:
        m2 = dff.groupby("month", as_index=False)["energy"].mean()
        bar = alt.Chart(m2).mark_bar().encode(x="month:T", y="energy:Q", tooltip=["month:T","energy:Q"]).properties(height=260)
        st.altair_chart(bar, use_container_width=True)
# ---------------------- DISCOVERY ----------------------
with tabs[9]:
    st.subheader("Discovery vs Repeat")
    st.markdown("**New track/artist shares** per month measure how adventurous you were. The **Discovery Score** blends both.")
    if track_col and artist_col:
        disc = discovery_by_month(dff, track_col, artist_col)
        if not disc.empty:
            area = alt.Chart(disc).transform_fold(
                ["new_track_share","new_artist_share"]
            ).mark_area(opacity=0.6).encode(
                x="month:T",
                y=alt.Y("value:Q", title="Share"),
                color=alt.Color("key:N", title=""),
                tooltip=["month:T","key:N","value:Q"]
            ).properties(height=300, title="New vs Repeat Shares")
            line = alt.Chart(disc).mark_line(point=True).encode(
                x="month:T", y=alt.Y("discovery_score:Q", title="Discovery Score"),
                tooltip=["month:T","discovery_score:Q"]
            ).properties(height=260)
            st.altair_chart(area, use_container_width=True)
            st.altair_chart(line, use_container_width=True)
        else:
            st.info("Not enough data to compute discovery metrics.")

# ---------------------- SESSIONS ----------------------
with tabs[10]:
    st.subheader("Listening Sessions")
    st.markdown("Reconstruct sessions using a **time gap** (default 30 minutes). Toggle clustering to label sessions by **mood**.")
    if enable_sessions:
        sess = build_sessions(dff, gap_minutes=gap_mins)
        if not sess.empty:
            st.markdown("**Session sizes and durations:**")
            sdist = sess[["plays","duration_min"]].describe().round(2)
            st.dataframe(sdist)

            st.markdown("**Sessions over time (count):**")
            if "date" in sess.columns:
                scount = sess.groupby("date", as_index=False).size().rename(columns={"size":"sessions"})
                line = alt.Chart(scount).mark_line(point=True).encode(x="date:T", y="sessions:Q").properties(height=260)
                st.altair_chart(line, use_container_width=True)

            if len(sess) > 10:
                st.markdown("**Average session mood (energy vs valence):**")
                scatter = alt.Chart(sess.dropna(subset=["energy_mean","valence_mean"])).mark_circle(opacity=0.5).encode(
                    x="valence_mean:Q", y="energy_mean:Q", size="plays:Q", tooltip=["plays","duration_min","valence_mean","energy_mean"]
                ).properties(height=340)
                st.altair_chart(scatter, use_container_width=True)

            st.markdown("**Cluster Sessions (MiniBatchKMeans):**")
            cs = cluster_sessions(sess, k=k_clusters)
            if not cs.empty:
                cbar = alt.Chart(cs).mark_bar().encode(
                    x="count():Q", y=alt.Y("session_label:N", sort="-x"), tooltip=["count()"]
                ).properties(height=260, title="Session clusters")
                st.altair_chart(cbar, use_container_width=True)
                st.dataframe(cs[["session_id","session_label","plays","duration_min","energy_mean","valence_mean","danceability_mean"]].head(50))
            else:
                st.info("Not enough data to cluster sessions (need mood features).")
    else:
        st.caption("Toggle 'Reconstruct & cluster sessions' in the sidebar to compute.")

# ---------------------- PERSONALITY ----------------------
with tabs[11]:
    st.subheader("Music Personality (Fingerprint)")
    st.markdown("Averages of audio features create your **listening fingerprint**. Use filters to compare eras.")
    feats = [c for c in ["acousticness","danceability","energy","instrumentalness",
                         "liveness","loudness","speechiness","tempo","valence"] if c in dff.columns]
    if feats:
        fp = dff[feats].mean().reset_index()
        fp.columns = ["feature","value"]
        bar = alt.Chart(fp).mark_bar().encode(x="value:Q", y=alt.Y("feature:N", sort="-x"), tooltip=["feature","value:Q"]).properties(height=340)
        st.altair_chart(bar, use_container_width=True)
        st.caption("Tip: Change the date filter to compare different periods (e.g., semesters, summers).")
        st.subheader("Fingerprint Change Map")
        feats_fp = [c for c in ["acousticness","danceability","energy","instrumentalness",
                                "liveness","loudness","speechiness","tempo","valence"] if c in dff.columns]
        if feats_fp and "date" in dff.columns:
            left_, right_ = st.columns(2)
            with left_:
                era_a_fp = st.date_input("Select Era A (fingerprint)", (pd.to_datetime(dff["date"]).min(), pd.to_datetime(dff["date"]).quantile(0.5)))
            with right_:
                era_b_fp = st.date_input("Select Era B (fingerprint)", (pd.to_datetime(dff["date"]).quantile(0.5), pd.to_datetime(dff["date"]).max()))
            def avg_feats_rng(rng):
                if isinstance(rng, (list, tuple)) and len(rng)==2:
                    m = (pd.to_datetime(dff["date"]) >= pd.to_datetime(rng[0])) & (pd.to_datetime(dff["date"]) <= pd.to_datetime(rng[1]))
                    return dff.loc[m, feats_fp].mean()
                return None
            a_fp = avg_feats_rng(era_a_fp)
            b_fp = avg_feats_rng(era_b_fp)
            if a_fp is not None and b_fp is not None:
                delta_fp = (b_fp - a_fp).reset_index()
                delta_fp.columns = ["feature","delta"]
                bar_fp = alt.Chart(delta_fp).mark_bar().encode(
                    x=alt.X("delta:Q", title="Δ (Era B - Era A)"),
                    y=alt.Y("feature:N", sort="-x"),
                    tooltip=["feature","delta:Q"]
                ).properties(height=320)
                st.altair_chart(bar_fp, use_container_width=True)
                st.caption("Positive values mean the feature average is **higher** in Era B.")
            else:
                st.info("Pick valid date ranges to compare fingerprints.")
    
    else:
        st.info("No audio features found.")

# ---------------------- NOSTALGIA ----------------------
with tabs[12]:
    st.subheader("Music You Forgot")
    st.markdown("Tracks that were once **Top N** in a month but haven't been played in a while. Great for a **Nostalgia Playlist**.")
    top_n = st.number_input("Top N threshold per month", 5, 100, 20, 5)
    stale = st.number_input("Consider 'forgotten' if not played in the last N months", 3, 60, 18, 1)
    ft = forgotten_tracks(dff, top_n_per_month=int(top_n), stale_months=int(stale))
    if not ft.empty:
        st.dataframe(ft.head(200))
        # Export CSV
        csv = ft.to_csv(index=False).encode("utf-8")
        st.download_button("Download Nostalgia CSV", data=csv, file_name="nostalgia_playlist_candidates.csv", mime="text/csv")
    else:
        st.info("No forgotten tracks found under current settings/filters.")

st.caption("This Fast+ build adds genre entropy, mood/energy maps, discovery metrics, sessions, personality, and nostalgia — all optimized with caching and sampling.")
