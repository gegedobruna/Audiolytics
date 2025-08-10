import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ---------- Config ----------
DERIVED_DIR = Path("data/derived")
ML_CLUSTERING_DIR = Path("ml/clustering")
ML_MARKOV_DIR = Path("ml/markov")

EVENTS_WITH_CLUSTERS = DERIVED_DIR / "sequences_with_clusters.parquet"
CLUSTERS_META_JSON = ML_CLUSTERING_DIR / "clusters_meta.json"
CENTROIDS_CSV = ML_CLUSTERING_DIR / "cluster_centroids.csv"
TRANSITION_MATRIX_NPY = ML_MARKOV_DIR / "transition_matrix.npy"
START_PROBS_NPY = ML_MARKOV_DIR / "start_probs.npy"
MARKOV_META_JSON = ML_MARKOV_DIR / "markov_meta.json"
CLUSTER_LABELS_JSON = ML_CLUSTERING_DIR / "cluster_labels.json"



# ---------- Cached loaders ----------
@st.cache_data(show_spinner=False)
def load_labels():
    if CLUSTER_LABELS_JSON.exists():
        return json.loads(CLUSTER_LABELS_JSON.read_text())
    return {}

@st.cache_data(show_spinner=False)
def load_sequences():
    df = pd.read_parquet(EVENTS_WITH_CLUSTERS)
    return df

@st.cache_data(show_spinner=False)
def load_markov():
    P = np.load(TRANSITION_MATRIX_NPY)
    start = np.load(START_PROBS_NPY)
    meta = json.loads(MARKOV_META_JSON.read_text())
    return P, start, meta

@st.cache_data(show_spinner=False)
def load_cluster_info():
    meta = json.loads(CLUSTERS_META_JSON.read_text())
    K = int(meta["chosen_k"])
    cents = pd.read_csv(CENTROIDS_CSV, index_col="cluster_id")
    # Friendly sort by energy/valence if available
    sort_cols = [c for c in ["energy", "valence", "danceability"] if c in cents.columns]
    if sort_cols:
        cents = cents.sort_values(sort_cols, ascending=[False]*len(sort_cols))
        # Reindex clusters to new order for display only
    return K, cents, meta

def top_k_from_row(row_probs: np.ndarray, k: int = 3):
    idx = np.argpartition(-row_probs, k-1)[:k]
    # sort by probability desc
    idx = idx[np.argsort(-row_probs[idx])]
    return idx, row_probs[idx]

def name_for(c_id: int) -> str:
    if not isinstance(c_id, (int, np.integer)): return str(c_id)
    return labels.get(str(c_id)) or labels.get(int(c_id)) or f"Cluster {c_id}"

def compute_empirical_counts(df_clusters: pd.Series) -> pd.DataFrame:
    """Empirical transition counts from your history."""
    a = df_clusters.to_numpy()
    pairs = list(zip(a[:-1], a[1:]))
    vc = pd.Series(pairs).value_counts()
    out = vc.reset_index()
    out.columns = ["pair", "count"]
    out["src"] = out["pair"].apply(lambda x: int(x[0]))
    out["dst"] = out["pair"].apply(lambda x: int(x[1]))
    out.drop(columns="pair", inplace=True)
    return out

def top_jumps_for_src(P: np.ndarray, src: int, k: int = 5):
    probs = P[src]
    idx = np.argpartition(-probs, k-1)[:k]
    idx = idx[np.argsort(-probs[idx])]
    return [(int(i), float(probs[i])) for i in idx]


# ---------- UI ----------
st.set_page_config(page_title="Predictions — Vibes (Markov)", page_icon="🎛️", layout="wide")
st.title("🎛️ Vibe Predictions — Markov Baseline")
st.info(
    "How to read this page:  \n"
    "• We grouped your songs into vibe clusters (by energy, danceability, valence, etc.).  \n"
    "• The bar chart shows the 3 vibes you most often switch to from your current vibe.  \n"
    "• The heatmap shows the full map: rows = current vibe, columns = next vibe, color = probability."
)



st.caption("Shows what vibe cluster is most likely **next**, based on your historical transitions. \
We’ll compare this to the GRU model later.")

# Load data/artifacts
with st.spinner("Loading artifacts…"):
    df = load_sequences()
    P, start_probs, markov_meta = load_markov()
    K, centroids, clusters_meta = load_cluster_info()
    labels = load_labels()

ts_col = "ts" if "ts" in df.columns else (clusters_meta.get("timestamp_col") or "ts")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.write(f"**Clusters (K)**: {K}")
    st.write("**Artifacts:**")
    st.code(str(TRANSITION_MATRIX_NPY), language="text")
    st.code(str(EVENTS_WITH_CLUSTERS), language="text")

    m = markov_meta.get("metrics", {})
    st.markdown("**Baseline accuracy (Markov)**")
    st.write(f"Val top-1/top-3: {m.get('val_top1', None)} / {m.get('val_top3', None)}")
    st.write(f"Test top-1/top-3: {m.get('test_top1', None)} / {m.get('test_top3', None)}")

# ---- Current / chosen cluster ----
colA, colB = st.columns([1,1])
with colA:
    st.subheader("Latest vibe → Next-up probabilities")
    latest_cluster = int(df["cluster_id"].iloc[-1])
    override = st.checkbox("Pick cluster manually", value=False)
    chosen_cluster = latest_cluster
    if override:
        chosen_cluster = st.number_input("Cluster ID", min_value=0, max_value=K-1, value=latest_cluster, step=1)

    next_ids, next_probs = top_k_from_row(P[chosen_cluster], k=3)
    bar_df = pd.DataFrame({
        "cluster": [name_for(int(i)) for i in next_ids],
        "probability": next_probs
    })

    st.write(f"**Current cluster:** {name_for(chosen_cluster)} (#{chosen_cluster})")
    st.bar_chart(bar_df.set_index("cluster"))

    # Text explanation of the top-3 from the current vibe
names = [name_for(int(i)) for i in next_ids]
st.markdown(
    f"From **{name_for(chosen_cluster)}**, you usually go to:\n"
    f"1) **{names[0]}** ({next_probs[0]:.1%})  \n"
    f"2) **{names[1]}** ({next_probs[1]:.1%})  \n"
    f"3) **{names[2]}** ({next_probs[2]:.1%})"
)


with colB:
    st.subheader("Cluster centroids (summary)")
    if not centroids.empty:
        show_cols = [c for c in ["energy","danceability","valence","tempo","acousticness","instrumentalness","liveness","loudness"] if c in centroids.columns]
        st.dataframe(centroids[show_cols].style.format(precision=2), use_container_width=True)
    else:
        st.info("Centroid table not available.")

with st.expander("✏️ Edit cluster names"):
    edited = {}
    for i in range(K):
        edited[i] = st.text_input(f"Cluster {i}", value=name_for(i))
    if st.button("Save labels"):
        save_dict = {int(i): edited[i] for i in range(K)}
        CLUSTER_LABELS_JSON.write_text(json.dumps(save_dict, indent=2, ensure_ascii=False))
        st.success("Saved. Refresh the page to see updated names.")
        st.cache_data.clear()  # refresh loaders

with st.expander("Legend: Cluster IDs → Names"):
    legend_df = pd.DataFrame({"cluster_id": range(K), "name": [name_for(i) for i in range(K)]})
    st.dataframe(legend_df, use_container_width=True)


# ---- Heatmap ----
st.subheader("Transition matrix heatmap (P[next | current])")
fig = plt.figure(figsize=(7, 6))
ax = plt.gca()
im = ax.imshow(P, aspect="auto")
ax.set_xticks(range(K))
ax.set_yticks(range(K))
ax.set_xticklabels([name_for(i) for i in range(K)], rotation=90, fontsize=8)
ax.set_yticklabels([name_for(i) for i in range(K)], fontsize=8)
ax.set_xlabel("Next cluster")
ax.set_ylabel("Current cluster")
ax.set_title("Cluster-to-Cluster Transition Probabilities")
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Probability", rotation=90)
st.pyplot(fig, clear_figure=True)

with st.expander("What does this heatmap mean?"):
    st.markdown(
        "- **Rows** = the vibe you're in now.  \n"
        "- **Columns** = the vibe you tend to play **next**.  \n"
        "- **Brighter color** = higher chance of that jump.  \n"
        "- **Bright near the diagonal** = you often stay in a similar vibe.  \n"
        "- **Hot spots** = your most common vibe changes."
    )

    # Empirical top jumps overall (by count, not probability)
counts_df = compute_empirical_counts(df["cluster_id"])
overall = (
    counts_df.assign(
        src_name=counts_df["src"].apply(name_for),
        dst_name=counts_df["dst"].apply(name_for)
    )
    .sort_values("count", ascending=False)
    .head(10)[["src_name","dst_name","count"]]
)
st.subheader("Most common vibe-to-vibe jumps overall")
st.dataframe(overall.rename(columns={
    "src_name":"From vibe", "dst_name":"To vibe", "count":"Times"
}), use_container_width=True)

# Top-5 jumps from the current vibe (by Markov probability)
st.subheader("Top next-vibes from your current vibe")
top5 = top_jumps_for_src(P, chosen_cluster, k=5)
friendly = pd.DataFrame({
    "Next vibe": [name_for(i) for (i,_) in top5],
    "Probability": [p for (_,p) in top5]
})
st.dataframe(friendly.style.format({"Probability":"{:.3%}"}), use_container_width=True)

# ---- Sequence preview + quick simulate ----
st.subheader("Quick simulate (1–3 steps ahead)")
steps = st.slider("Steps ahead", 1, 3, 1)
seed_cluster = chosen_cluster
dist = np.zeros(K); dist[seed_cluster] = 1.0
for _ in range(steps):
    dist = dist @ P
top_ids, top_ps = top_k_from_row(dist, k=5)
sim_df = pd.DataFrame({
    "cluster": [name_for(int(i)) for i in top_ids],
    "probability": top_ps
})
st.dataframe(sim_df.style.format({"probability": "{:.3f}"}), use_container_width=True)
st.write(f"From cluster **{seed_cluster}** in **{steps}** step(s):")

# ---- Recent cluster timeline (context) ----
st.subheader("Recent cluster timeline")
n_show = st.slider("Show last N events", 200, 2000, 500, step=100)
tail = df[[ts_col, "cluster_id"]].tail(n_show).reset_index(drop=True)
tail["idx"] = np.arange(len(tail))
line_fig = plt.figure(figsize=(10, 2.8))
ax2 = plt.gca()
ax2.plot(tail["idx"], tail["cluster_id"], linewidth=1.0)
ax2.set_yticks(range(min(tail["cluster_id"]), max(tail["cluster_id"])+1))
ax2.set_xlabel("Event index (recent →)")
ax2.set_ylabel("Cluster")
ax2.set_title("Recent Vibe Sequence")
st.pyplot(line_fig, clear_figure=True)

st.caption("Tip: use the sidebar to check baseline accuracy. The GRU page will show a side-by-side comparison soon.")
