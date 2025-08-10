import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import torch
from torch import nn
import matplotlib.pyplot as plt

# ---------- Paths ----------
DERIVED_DIR = Path("data/derived")
EVENTS = DERIVED_DIR / "sequences.parquet"
EVENTS_WITH_CLUSTERS = DERIVED_DIR / "sequences_with_clusters.parquet"

CLUSTER_DIR = Path("ml/clustering")
IMPUTER_PKL = CLUSTER_DIR / "imputer.pkl"
SCALER_PKL = CLUSTER_DIR / "scaler.pkl"
LABELS_JSON = CLUSTER_DIR / "cluster_labels.json"

MARKOV_DIR = Path("ml/markov")
P_NPY = MARKOV_DIR / "transition_matrix.npy"
MARKOV_META_JSON = MARKOV_DIR / "markov_meta.json"

GRU_DIR = Path("ml/gru_next_cluster")
GRU_PT = GRU_DIR / "gru_model.pt"
GRU_CFG = GRU_DIR / "config.json"
GRU_METRICS = GRU_DIR / "metrics.json"


# ---------- GRU model ----------
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=1, dropout=0.2, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden, num_layers=layers,
                          batch_first=True, dropout=(dropout if layers > 1 else 0.0))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, num_classes)
    def forward(self, x):
        _, h_n = self.gru(x)
        h = h_n[-1]
        h = self.dropout(h)
        return self.out(h)

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_labels():
    if LABELS_JSON.exists():
        obj = json.loads(LABELS_JSON.read_text())
        # keys might be int or str; normalize to int->str
        return {int(k): v for k, v in obj.items()}
    return {}

def name_for(labels, i):
    return labels.get(int(i), f"Cluster {i}")

@st.cache_data(show_spinner=False)
def load_markov():
    P = np.load(P_NPY)
    meta = json.loads(MARKOV_META_JSON.read_text())
    return P, meta

@st.cache_resource(show_spinner=False)
def load_gru():
    cfg = json.loads(GRU_CFG.read_text())
    model = GRUClassifier(
        input_dim=len(cfg["feature_cols"]),
        hidden=cfg["hidden"],
        layers=cfg["layers"],
        dropout=cfg["dropout"],
        num_classes=cfg["K"],
    )
    state = torch.load(GRU_PT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    imputer = joblib.load(IMPUTER_PKL)
    scaler = joblib.load(SCALER_PKL)
    return cfg, model, imputer, scaler

@st.cache_data(show_spinner=False)
def load_events():
    df = pd.read_parquet(EVENTS)
    dfc = pd.read_parquet(EVENTS_WITH_CLUSTERS)
    # ensure same order/length
    if len(df) != len(dfc):
        raise RuntimeError("sequences.parquet and sequences_with_clusters.parquet length mismatch")
    return df, dfc

def topk(arr, k=3):
    idx = np.argpartition(-arr, k-1)[:k]
    idx = idx[np.argsort(-arr[idx])]
    return idx, arr[idx]

# ---------- UI ----------
st.set_page_config(page_title="Predictions — GRU vs Markov", page_icon="🧠", layout="wide")
st.title("🧠 Next Vibe — GRU vs Markov")
st.markdown(
    """
**What this page does**
- **GRU** looks at your last *L* tracks (features) and predicts the next vibe.
- **Markov** looks only at your current vibe → next vibe probabilities.
- Compare their **top-3 guesses** and see which one feels right.
    """
)

# Load artifacts
with st.spinner("Loading data and models…"):
    labels = load_labels()
    P, markov_meta = load_markov()
    df_feat, df_clusters = load_events()

# Block if model isn't trained yet
if not (GRU_PT.exists() and GRU_CFG.exists()):
    st.warning("GRU model not found yet. Train it first (see ml/gru_next_cluster/train_gru.py).")
    st.stop()

cfg, model, imputer, scaler = load_gru()
feature_cols = cfg["feature_cols"]
K = cfg["K"]
L = cfg["seq_len"]

# Controls
st.sidebar.header("Controls")
L = st.sidebar.number_input("Sequence length (L)", min_value=5, max_value=100, value=int(L), step=1)
n_context = st.sidebar.slider("Show last N events in timeline", 200, 2000, 500, step=100)

# Build last-L window for GRU
if len(df_feat) < L + 1:
    st.error("Not enough events to build a window.")
    st.stop()

last_L = df_feat.tail(L).copy()
X = last_L[feature_cols].to_numpy(dtype=np.float32)
X = scaler.transform(imputer.transform(X)).astype(np.float32)
x = torch.from_numpy(X[None, ...])  # (1,L,F)

with torch.no_grad():
    logits = model(x)
    probs_gru = torch.softmax(logits, dim=1).numpy()[0]

# Markov based on current cluster
current_cluster = int(df_clusters["cluster_id"].iloc[-1])
probs_mkv = P[current_cluster]

# Top-3
ids_g, p_g = topk(probs_gru, k=3)
ids_m, p_m = topk(probs_mkv, k=3)

col1, col2 = st.columns(2)
with col1:
    st.subheader("GRU — top-3 next vibes")
    df_g = pd.DataFrame({
        "vibe": [name_for(labels, i) for i in ids_g],
        "probability": p_g
    })
    st.bar_chart(df_g.set_index("vibe"))
    st.markdown(
        f"**GRU says:** 1) {name_for(labels, ids_g[0])} ({p_g[0]:.1%}), "
        f"2) {name_for(labels, ids_g[1])} ({p_g[1]:.1%}), "
        f"3) {name_for(labels, ids_g[2])} ({p_g[2]:.1%})"
    )

with col2:
    st.subheader("Markov — top-3 next vibes")
    df_m = pd.DataFrame({
        "vibe": [name_for(labels, i) for i in ids_m],
        "probability": p_m
    })
    st.bar_chart(df_m.set_index("vibe"))
    st.markdown(
        f"**Markov says:** 1) {name_for(labels, ids_m[0])} ({p_m[0]:.1%}), "
        f"2) {name_for(labels, ids_m[1])} ({p_m[1]:.1%}), "
        f"3) {name_for(labels, ids_m[2])} ({p_m[2]:.1%})"
    )

st.markdown("---")
st.subheader("Recent vibe timeline")
ts_col = "ts" if "ts" in df_clusters.columns else "date"
tail = df_clusters[[ts_col, "cluster_id"]].tail(n_context).reset_index(drop=True)
tail["idx"] = np.arange(len(tail))
fig = plt.figure(figsize=(10, 2.8))
ax = plt.gca()
ax.plot(tail["idx"], tail["cluster_id"], linewidth=1.0)
ax.set_yticks(range(min(tail["cluster_id"]), max(tail["cluster_id"])+1))
ax.set_xlabel("Event index (recent →)")
ax.set_ylabel("Cluster")
ax.set_title("Recent Vibe Sequence")
st.pyplot(fig, clear_figure=True)

# Metrics
if GRU_METRICS.exists():
    mets = json.loads(GRU_METRICS.read_text())
    vt = mets.get("test", {})
    st.caption(
        f"**Test metrics (GRU)** — top-1: {vt.get('top1', None):.3f}, top-3: {vt.get('top3', None):.3f}"
        if vt else "Test metrics (GRU) unavailable."
    )

with st.expander("What am I looking at?"):
    st.markdown(
        "- **GRU** uses your last *L* songs (their features) to guess the next vibe.\n"
        "- **Markov** only uses your **current** vibe to guess the next.\n"
        "- Compare which one matches your behavior better right now."
    )
