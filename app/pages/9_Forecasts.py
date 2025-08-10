# streamlit_app/pages/07_Forecasts.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Forecasts", page_icon="🗓️", layout="wide")
st.title("🗓️ 7-Day Forecasts")

# ------------------------------
# Repo root + file locations
# ------------------------------
HERE = Path(__file__).resolve().parent

def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "ml").exists() and (p / "data").exists():
            return p
        if (p / ".git").exists():
            return p
    return start.parents[1]

ROOT = find_root(HERE)
P_PLATFORM = ROOT / "ml" / "forecast_7d" / "platform" / "outputs" / "platform_share_forecast.parquet"
P_PLAYS    = ROOT / "ml" / "forecast_7d" / "plays"    / "total_plays_forecast.parquet"
P_VIBES    = ROOT / "ml" / "forecast_7d" / "vibes"    / "vibe_share_forecast.parquet"
P_ARTISTS  = ROOT / "ml" / "forecast_7d" / "artists"  / "new_vs_same_forecast.parquet"
P_STREAK   = ROOT / "ml" / "forecast_7d" / "streak"   / "forecast.json"

with st.expander("🔎 Diagnostics (paths)", expanded=False):
    st.write("ROOT:", str(ROOT))
    st.write("Platform:", str(P_PLATFORM), "exists:", P_PLATFORM.exists())
    st.write("Plays:",    str(P_PLAYS),    "exists:", P_PLAYS.exists())
    st.write("Vibes:",    str(P_VIBES),    "exists:", P_VIBES.exists())
    st.write("Artists:",  str(P_ARTISTS),  "exists:", P_ARTISTS.exists())
    st.write("Streak:",   str(P_STREAK),   "exists:", P_STREAK.exists())

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data(show_spinner=False)
def load_parquet(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_json(p: Path) -> Optional[list[dict]]:
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # normalize to list[dict]
        if isinstance(data, dict):
            data = data.get("forecast", data.get("data", []))
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []

def clamp01(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(0.0, 1.0)

def to_day_ahead(df: pd.DataFrame, group_cols: list[str] | None) -> pd.DataFrame:
    """Add day_ahead = 1..n. If no group keys, enumerate globally."""
    if df.empty:
        return df
    x = df.copy()
    group_cols = group_cols or []  # allow None/[]
    # stable sort
    sort_cols = []
    if group_cols:
        sort_cols.extend([c for c in group_cols if c in x.columns])
    if "date" in x.columns:
        sort_cols.append("date")
    if sort_cols:
        x = x.sort_values(sort_cols)
    # assign day_ahead
    if group_cols:
        x["day_ahead"] = x.groupby(group_cols, dropna=False).cumcount() + 1
    else:
        x["day_ahead"] = np.arange(1, len(x) + 1)
    return x

def sharey(df: pd.DataFrame, cols=("pred","lo","hi")) -> pd.DataFrame:
    """If series looks like a share, clamp to [0,1]."""
    out = df.copy()
    if "pred" in out.columns:
        # If most preds are between 0 and 1, treat as share
        if pd.to_numeric(out["pred"], errors="coerce").between(0, 1).mean() > 0.6:
            for c in cols:
                if c in out.columns:
                    out[c] = clamp01(out[c])
    return out

def safe_area_layer(base: alt.Chart, df: pd.DataFrame) -> Optional[alt.LayerChart]:
    if {"lo", "hi"}.issubset(df.columns) and not df["lo"].isna().all() and not df["hi"].isna().all():
        return base.mark_area(opacity=0.2).encode(y="lo:Q", y2="hi:Q")
    return None

def standardize_forecast_columns(
    df: pd.DataFrame,
    pred_names=("pred","yhat","mean","median","forecast","point","estimate","value",
                "share","prob","p","mu","p50","avg","expected"),
    lo_names=("lo","lower","lo_80","lo_20","yhat_lower","lower_ci","p10","p25"),
    hi_names=("hi","upper","hi_80","hi_20","yhat_upper","upper_ci","p90","p75"),
) -> pd.DataFrame:
    x = df.copy()
    if x.empty:
        return x

    # case-insensitive lookup
    lower = {c.lower(): c for c in x.columns}
    def pick(cands):
        for n in cands:
            if n in lower:
                return lower[n]
        return None

    pred_col = pick(pred_names)
    lo_col   = pick(lo_names)
    hi_col   = pick(hi_names)

    # if we found bands but not pred -> synthesize pred = (lo+hi)/2
    if not pred_col and lo_col and hi_col:
        x["pred"] = pd.to_numeric(x[lo_col], errors="coerce").add(
            pd.to_numeric(x[hi_col], errors="coerce"), fill_value=np.nan
        ) / 2.0
    elif pred_col and pred_col != "pred":
        x = x.rename(columns={pred_col: "pred"})

    # rename bands if present
    if lo_col and lo_col != "lo":
        x = x.rename(columns={lo_col: "lo"})
    if hi_col and hi_col != "hi":
        x = x.rename(columns={hi_col: "hi"})

    # If still no 'pred', try to infer from numeric columns
    if "pred" not in x.columns:
        ignore = {"date","platform","vibe_name","vibe","cluster","cluster_id",
                  "label","category","model","day_ahead","step"}
        num_cols = [c for c in x.columns if c not in ignore and pd.api.types.is_numeric_dtype(x[c])]
        if len(num_cols) == 1:
            x = x.rename(columns={num_cols[0]: "pred"})
        elif len(num_cols) > 1:
            # choose the numeric column with the highest variance (most informative)
            var = x[num_cols].var(numeric_only=True).sort_values(ascending=False)
            best = var.index[0]
            x = x.rename(columns={best: "pred"})

    return x

def compute_daily_unique_artists(root: Path) -> tuple[int, int]:
    """
    Returns (daily_unique_avg, num_days_used).
    Looks at sequences.parquet first, then audiolytics.parquet.
    """
    seq_p = root / "data" / "derived" / "sequences.parquet"
    aud_p = root / "data" / "audiolytics.parquet"

    def from_sequences(p: Path) -> tuple[int, int] | None:
        if not p.exists():
            return None
        df = pd.read_parquet(p, columns=["date","master_metadata_album_artist_name"])
        df = df.dropna(subset=["date","master_metadata_album_artist_name"])
        # ensure date is date-like
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        daily = (df.groupby("date")["master_metadata_album_artist_name"]
                   .nunique()
                   .rename("unique_artists"))
        if daily.empty:
            return None
        return int(round(daily.mean())), int(daily.shape[0])

    def from_audiolytics(p: Path) -> tuple[int, int] | None:
        if not p.exists():
            return None
        cols = ["ts","master_metadata_album_artist_name"]
        usecols = [c for c in cols if c in pd.read_parquet(p, nrows=0).columns]
        df = pd.read_parquet(p, columns=usecols)
        if "ts" not in df.columns or "master_metadata_album_artist_name" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["ts"], errors="coerce").dt.tz_localize(None).dt.date
        df = df.dropna(subset=["date","master_metadata_album_artist_name"])
        daily = (df.groupby("date")["master_metadata_album_artist_name"]
                   .nunique()
                   .rename("unique_artists"))
        if daily.empty:
            return None
        return int(round(daily.mean())), int(daily.shape[0])

    out = from_sequences(seq_p) or from_audiolytics(aud_p)
    if out is None:
        return (12, 0)  # safe default
    return out



# ------------------------------
# Vibe names (baked-in from you)
# ------------------------------
VIBE_LABELS = {
    0: "High Energy, Groovy, Neutral (Loud, Fast)",
    1: "Low Energy, Dancy, Neutral (Acoustic)",
    2: "High Energy, Dancy, Upbeat (Loud)",
    3: "Low Energy, Undancy, Moody (Acoustic, Instrumental)",
    4: "Mid Energy, Undancy, Moody (Loud, Live)",
    5: "Mid Energy, Groovy, Upbeat (Slow)",
    6: "Low Energy, Undancy, Moody (Acoustic, Instrumental)",
    7: "Mid Energy, Groovy, Neutral (Instrumental, Live)",
    8: "High Energy, Groovy, Upbeat (Live, Fast)",
    9: "Mid Energy, Dancy, Neutral (Fast)",
}

def apply_vibe_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cluster_col = next((c for c in ["cluster_id","cluster","vibe","label_id"] if c in df.columns), None)
    if cluster_col is None:
        return df.assign(vibe_name="(unknown)")
    return df.assign(
        vibe_name=df[cluster_col].astype("Int64").map(VIBE_LABELS).astype("string").fillna(
            "Vibe " + df[cluster_col].astype("string")
        )
    )

# ------------------------------
# Load data
# ------------------------------
df_platform = load_parquet(P_PLATFORM)  # date, platform, pred, lo, hi, model
df_plays    = load_parquet(P_PLAYS)     # date, pred, lo, hi, model
df_vibes    = apply_vibe_labels(load_parquet(P_VIBES))  # + vibe_name
df_artists  = load_parquet(P_ARTISTS)   # long or wide
js_streak   = load_json(P_STREAK)       # list of dicts

# ------------------------------
# PLATFORM tab
# ------------------------------
def _ensure_col(df: pd.DataFrame, candidates: tuple[str, ...], canonical: str) -> pd.DataFrame:
    """Rename the first present candidate column to `canonical`."""
    for c in candidates:
        if c in df.columns and c != canonical:
            return df.rename(columns={c: canonical})
        if c == canonical and c in df.columns:
            return df
    return df  # none found; caller should handle

def tab_platform(df: pd.DataFrame):
    st.subheader("Platforms — most likely this week")
    st.info(
    "We collapse the next 7 daily platform predictions into a single **weekly probability per platform**. "
    "The sentence highlights the top platform; the bars show the full ranking. "
    "_Numbers are normalized so they sum to 100% across platforms._"
)


    if df.empty:
        st.info("No platform forecast found. Run: `python -u -m ml.forecast_7d.platform.train_platform_shift`")
        return

    # 1) Make sure we have a forecast column named 'pred'
    df = standardize_forecast_columns(df)
    if "pred" not in df.columns:
        st.error("Couldn't find a forecast value column (pred/yhat/mean/forecast/value/share/prob/p).")
        with st.expander("Peek at columns"):
            st.dataframe(df.head(20), use_container_width=True)
        return

    # 2) Ensure the platform label column is called 'platform'
    df = _ensure_col(df, ("platform","device","source","app","service"), "platform")
    if "platform" not in df.columns:
        st.error("Couldn't find a platform/device column.")
        with st.expander("Peek at columns"):
            st.dataframe(df.head(20), use_container_width=True)
        return

    # 3) If it looks like shares, clamp to [0,1]
    df = sharey(df)

    # 4) Keep the first 7 steps per platform (ordered by date if present)
    x = df.copy()
    if "date" in x.columns:
        x = x.sort_values(["platform", "date"])
    else:
        x = x.sort_values(["platform"])
    x["step"] = x.groupby("platform", dropna=False).cumcount() + 1
    x = x[x["step"] <= 7].dropna(subset=["pred"])

    if x.empty:
        st.caption("No forecast rows for the upcoming week.")
        return

    # 5) Convert to weekly probabilities:
    #    sum each platform's 7-day preds, then normalize across platforms
    weekly = (x.groupby("platform", dropna=False)["pred"]
                .sum()
                .reset_index(name="score"))
    total = weekly["score"].sum()
    weekly["share"] = np.where(total > 0, weekly["score"] / total, 0.0)
    weekly = weekly.sort_values("share", ascending=False).reset_index(drop=True)

    # 6) Friendly summary
    top = weekly.iloc[0]
    st.markdown(
        f"**Most likely:** **{top['platform']}** "
        f"(~{top['share']*100:.0f}% of your listening this week)."
    )

    # 7) Horizontal bar chart of all platforms
    chart = (
        alt.Chart(weekly)
        .mark_bar()
        .encode(
            y=alt.Y("platform:N", sort="-x", title=None),
            x=alt.X("share:Q", axis=alt.Axis(format=".0%"), title="Probability this week"),
            tooltip=[
                alt.Tooltip("platform:N", title="Platform"),
                alt.Tooltip("share:Q", title="Probability", format=".0%")
            ],
        )
        .properties(height=max(220, 28 * len(weekly)))
    )
    labels = chart.mark_text(align="left", dx=6).encode(text=alt.Text("share:Q", format=".0%"))
    st.altair_chart(chart + labels, use_container_width=True)

    st.caption("Computed by summing each platform’s next 7 forecasted shares and normalizing across platforms.")


# ------------------------------
# TOTAL PLAYS tab
# ------------------------------
def tab_plays(df: pd.DataFrame):
    st.subheader("Total plays (next 7 days)")
    st.info(
    "Forecast of your **total plays** for each of the next 7 days (Day 1–7). "
    "Dashed line = predicted plays; shaded area (if shown) = model uncertainty; dots = day-by-day point estimates."
)

    if df.empty:
        st.info("No total-plays forecast found. Run: `python -u -m ml.forecast_7d.plays.train_total_plays`")
        return

    df = to_day_ahead(df, [])  # no group
    d = df.sort_values("day_ahead")

    base = alt.Chart(d).encode(x=alt.X("day_ahead:O", title="Day ahead (1–7)"))
    band = safe_area_layer(base, d)
    line = base.mark_line(strokeDash=[4, 4]).encode(y=alt.Y("pred:Q", title="Plays/day"))
    pts  = base.mark_point().encode(y="pred:Q", tooltip=["day_ahead:O","pred:Q","lo:Q","hi:Q","model:N"])

    layers = [l for l in [band, line, pts] if l is not None]
    st.altair_chart(alt.layer(*layers).properties(height=420), use_container_width=True)

# ------------------------------
# VIBES tab
# ------------------------------
def tab_vibes(df: pd.DataFrame):
    st.subheader("Vibe forecast — most probable over next week")
    st.info(
    "Not day-by-day. We average each vibe’s next 7 predictions, then **normalize** so you see the overall **weekly vibe mix**. "
    "The list shows the top vibes; the bars show the full distribution across vibes (sums to 100%)."
)


    if df.empty:
        st.info("No vibe forecast found. Run: `python -u -m ml.forecast_7d.vibes.train_vibe_distribution`")
        return

    if "vibe_name" not in df.columns:
        st.error("Missing 'vibe_name' column. (Make sure the mapping step ran.)")
        st.dataframe(df.head(20))
        return

    # 🔁 standardize / infer the forecast column as 'pred'
    df = standardize_forecast_columns(df)

    if "pred" not in df.columns:
        st.error("Still couldn't infer the forecast value column automatically.")
        with st.expander("Peek at columns to help me auto-map"):
            st.write(list(df.columns))
            st.dataframe(df.head(20), use_container_width=True)
        return

    # If it looks like shares, clamp to [0,1]
    df = sharey(df)

    # Keep the first 7 forecast steps per vibe (ordered by date if present)
    x = df.copy()
    if "date" in x.columns:
        x = x.sort_values(["vibe_name", "date"])
    else:
        x = x.sort_values(["vibe_name"])
    x["step"] = x.groupby("vibe_name", dropna=False).cumcount() + 1
    x = x[x["step"] <= 7]

    # Aggregate to a single score per vibe (mean over the week)
    week = (x.groupby("vibe_name", dropna=False)["pred"]
              .mean()
              .reset_index(name="score"))

    # Normalize so it reads like probabilities (sums to 1)
    total = week["score"].sum()
    if total > 0:
        week["prob"] = week["score"] / total
    else:
        week["prob"] = week["score"]  # all zeros is fine

    week = week.sort_values("prob", ascending=False).reset_index(drop=True)

    # Friendly top list
    st.markdown("**Top vibes for the upcoming week:**")
    topn = week.head(5)
    for i, r in topn.iterrows():
        st.write(f"{i+1}. **{r['vibe_name']}** — {(r['prob']*100):.0f}%")

    # Bar chart (all vibes)
    chart = (
        alt.Chart(week)
        .mark_bar()
        .encode(
            y=alt.Y("vibe_name:N", sort="-x", title=None),
            x=alt.X("prob:Q", axis=alt.Axis(format=".0%"), title="Probability this week"),
            tooltip=[
                alt.Tooltip("vibe_name:N", title="Vibe"),
                alt.Tooltip("prob:Q", title="Probability", format=".0%")
            ],
        )
        .properties(height=max(280, 22 * len(week)))
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption("Computed by averaging the first 7 forecast steps per vibe and normalizing across vibes.")


# ------------------------------
# ARTISTS (new vs same) tab
# ------------------------------
def normalize_artists(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Long form?
    if {"category","pred"}.issubset(df.columns):
        out = df.copy()
    # Wide form?
    elif {"new","same"}.issubset(df.columns):
        out = df.melt(id_vars=[c for c in ["date"] if c in df.columns],
                      value_vars=["new","same"],
                      var_name="category", value_name="pred")
        # optional bands
        if {"lo_new","hi_new","lo_same","hi_same"}.issubset(df.columns):
            lo = df.melt(id_vars=[c for c in ["date"] if c in df.columns],
                         value_vars=["lo_new","lo_same"], var_name="c", value_name="lo")
            hi = df.melt(id_vars=[c for c in ["date"] if c in df.columns],
                         value_vars=["hi_new","hi_same"], var_name="c", value_name="hi")
            # match categories
            lo["category"] = lo["c"].str.replace("lo_", "", regex=False)
            hi["category"] = hi["c"].str.replace("hi_", "", regex=False)
            out = out.merge(lo[["date","category","lo"]], on=["date","category"], how="left")
            out = out.merge(hi[["date","category","hi"]], on=["date","category"], how="left")
    else:
        # Try to guess a label col
        label_col = next((c for c in ["label","type","kind"] if c in df.columns), None)
        if label_col and "pred" in df.columns:
            out = df.rename(columns={label_col:"category"}).copy()
        else:
            # give up gracefully
            out = pd.DataFrame(columns=["date","category","pred","lo","hi"])

    # keep only 'new'/'same' if present
    if "category" in out.columns:
        mask = out["category"].isin(["new","same"])
        if mask.any():
            out = out[mask]

    return out

def tab_artists(df: pd.DataFrame):
    st.subheader("Artist churn: new vs same (next 7 days)")
    st.info(
    "We average the next 7 days of your **new-artist ratio**, then split the week into **New vs Familiar**. "
    "To make it concrete, we estimate the **number of new artists** by multiplying that ratio by your "
    "auto-calculated average **unique artists/day** from your listening history."
)

    if df.empty:
        st.info("No new-vs-same forecast found. Run: `python -u -m ml.forecast_7d.artists.train_new_vs_same`")
        return

    df = normalize_artists(df)
    if df.empty:
        st.warning("Couldn’t detect 'new'/'same' structure in your file.")
        st.dataframe(load_parquet(P_ARTISTS).head(20))
        return

    df = sharey(df)
    df = to_day_ahead(df, ["category"]).sort_values(["category","day_ahead"])

    cats = [c for c in ["new","same"] if c in df["category"].unique()]
    if not cats:
        cats = sorted(df["category"].unique().tolist())

    sel = st.multiselect("Show categories", cats, default=cats)
    d = df[df["category"].isin(sel)]

    base = alt.Chart(d).encode(
        x=alt.X("day_ahead:O", title="Day ahead (1–7)"),
        color=alt.Color("category:N", title="Category")
    )
    band = None
    if {"lo","hi"}.issubset(d.columns):
        band = base.mark_area(opacity=0.15).encode(y="lo:Q", y2="hi:Q")
    line = base.mark_line(strokeDash=[4, 4]).encode(y=alt.Y("pred:Q", title="Share / probability"))
    pts  = base.mark_point().encode(y="pred:Q", tooltip=["category:N","day_ahead:O","pred:Q","lo:Q","hi:Q"])

    layers = [l for l in [band, line, pts] if l is not None]
    st.altair_chart(alt.layer(*layers).properties(height=420), use_container_width=True)
    st.caption("‘New’ = artists not heard recently; ‘Same’ = artists you’ve been listening to.")

# --- Simple "New artists next week" tab (drop-in) ---
def tab_artists_simple(df: pd.DataFrame):
    st.subheader("New artists forecast — next 7 days")
    st.info(
    "We average the next 7 days of your **new-artist ratio**, then split the week into **New vs Familiar**. "
    "To make it concrete, we estimate the **number of new artists** by multiplying that ratio by your "
    "auto-calculated average **unique artists/day** from your listening history."
)
    if df.empty:
        st.info("No artist churn forecast found. Run: `python -u -m ml.forecast_7d.artists.train_new_vs_same`")
        return

    # standardize / infer forecast column as 'pred'
    df = standardize_forecast_columns(df)
    if "pred" not in df.columns:
        st.error("Couldn't find a forecast value column in artists forecast.")
        st.dataframe(df.head(20))
        return

    # take the next 7 steps
    d = df.sort_values("date").head(7).copy()
    # treat as a ratio (0..1) and clamp just in case
    d["pred"] = pd.to_numeric(d["pred"], errors="coerce").clip(0, 1)
    avg_new_ratio = float(d["pred"].mean())
    avg_same_ratio = 1.0 - avg_new_ratio

    # auto-compute your typical unique artists/day from history
    daily_unique, n_days = compute_daily_unique_artists(ROOT)
    est_new_artists_week = int(round(avg_new_ratio * daily_unique * 7))

    # summary sentence
    st.markdown(
        f"**This week:** about **{avg_new_ratio*100:.0f}% new artists** "
        f"and **{avg_same_ratio*100:.0f}% familiar artists**.\n\n"
        f"Based on your history (~**{daily_unique}** unique artists/day, from {n_days} days), "
        f"that’s roughly **{est_new_artists_week} new artists** this week."
    )

    # clean two-bar chart
    summary = pd.DataFrame({
        "category": ["New", "Familiar"],
        "share": [avg_new_ratio, avg_same_ratio]
    })
    chart = (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X("share:Q", axis=alt.Axis(format=".0%"), title="Share of listening this week"),
            y=alt.Y("category:N", sort=["New","Familiar"], title=None),
            tooltip=[alt.Tooltip("category:N"), alt.Tooltip("share:Q", format=".0%")]
        )
        .properties(height=120)
    )
    labels = chart.mark_text(align="left", dx=6).encode(text=alt.Text("share:Q", format=".0%"))
    st.altair_chart(chart + labels, use_container_width=True)


# ------------------------------
# STREAK tab
# ------------------------------
def tab_streak(js: Optional[list[dict]]):
    st.subheader("Listening streak (next 7 days)")
    st.info(
    "For each of the next 7 days, this shows the **probability you keep your listening streak**. "
    "Bars give a quick visual; the line/dots trace the same probabilities."
)

    if not js:
        st.info("No streak forecast found. Run: `python -u -m ml.forecast_7d.streak.train_streak`")
        return
    df = pd.DataFrame(js)
    # normalize columns
    if "day_ahead" not in df.columns:
        # try to infer: take order as day_ahead
        df["day_ahead"] = np.arange(1, len(df) + 1)
    prob_col = next((c for c in ["prob_continue","probability","p_continue","pred"] if c in df.columns), None)
    if not prob_col:
        st.json(js, expanded=False)
        return
    df = df[["day_ahead", prob_col]].rename(columns={prob_col: "prob_continue"})
    df["prob_continue"] = clamp01(df["prob_continue"])
    df = df.sort_values("day_ahead")

    base = alt.Chart(df).encode(x=alt.X("day_ahead:O", title="Day ahead (1–7)"))
    bars = base.mark_bar(opacity=0.3).encode(y=alt.Y("prob_continue:Q", title="Probability"))
    line = base.mark_line().encode(y="prob_continue:Q")
    pts  = base.mark_point().encode(y="prob_continue:Q", tooltip=["day_ahead:O","prob_continue:Q"])
    st.altair_chart(alt.layer(bars, line, pts).properties(height=360), use_container_width=True)
    st.caption("Probability of continuing your listening streak on each of the next 7 days.")

# ------------------------------
# Tabs
# ------------------------------
tabs = st.tabs(["Platform", "Total plays", "Vibes", "Artists (new vs same)", "Streak"])
with tabs[0]: tab_platform(df_platform)
with tabs[1]: tab_plays(df_plays)
with tabs[2]: tab_vibes(df_vibes)
with tabs[3]: tab_artists_simple(df_artists)
with tabs[4]: tab_streak(js_streak)
