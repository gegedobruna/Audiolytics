import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def load_json(p): return json.loads(Path(p).read_text())

def compute_P(train_seq, K, alpha=0.5):
    counts = np.zeros((K, K), dtype=np.float64)
    for a, b in zip(train_seq[:-1], train_seq[1:]):
        if 0 <= a < K and 0 <= b < K:
            counts[a, b] += 1
    counts += alpha  # Laplace smoothing
    row_sums = counts.sum(axis=1, keepdims=True)
    P = counts / np.clip(row_sums, 1e-12, None)
    return P, counts

def topk_accuracy(seq, P, k=3):
    if len(seq) < 2: return np.nan
    hits = 0; total = 0
    for a, b in zip(seq[:-1], seq[1:]):
        probs = P[a]
        topk = np.argpartition(-probs, k-1)[:k]
        hits += int(b in topk); total += 1
    return hits / total if total else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_with_clusters", default="data/derived/sequences_with_clusters.parquet")
    ap.add_argument("--clusters_meta", default="ml/clustering/clusters_meta.json")
    ap.add_argument("--manifest", default="data/derived/prep_manifest.json")
    ap.add_argument("--outdir", default="ml/markov")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.5, help="Laplace smoothing")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    meta = load_json(args.clusters_meta)
    K = int(meta["chosen_k"])
    ts_col = load_json(args.manifest)["timestamp_col"]

    print("Loading clustered sequences…", flush=True)
    df = pd.read_parquet(args.events_with_clusters).sort_values(ts_col).reset_index(drop=True)
    seq = df["cluster_id"].to_numpy()
    n = len(seq)
    i_train = int(n*args.train_frac)
    i_val = i_train + int(n*args.val_frac)

    train, val, test = seq[:i_train], seq[i_train:i_val], seq[i_val:]
    print(f"Rows: {n:,} | train={len(train):,} val={len(val):,} test={len(test):,}", flush=True)

    P, counts = compute_P(train, K, alpha=args.alpha)

    # simple metrics
    t1v = topk_accuracy(val, P, k=1); t3v = topk_accuracy(val, P, k=3)
    t1t = topk_accuracy(test, P, k=1); t3t = topk_accuracy(test, P, k=3)
    print(f"Val top-1/top-3: {t1v:.3f} / {t3v:.3f}", flush=True)
    print(f"Test top-1/top-3: {t1t:.3f} / {t3t:.3f}", flush=True)

    # start probs (tiny prior)
    start = np.zeros(K, dtype=np.float64) + args.alpha
    if len(train): start[train[0]] += 1
    start = start / start.sum()

    np.save(outdir / "transition_matrix.npy", P)
    np.save(outdir / "start_probs.npy", start)
    (outdir / "markov_meta.json").write_text(json.dumps({
        "K": K,
        "alpha": args.alpha,
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "metrics": {"val_top1": float(t1v), "val_top3": float(t3v),
                    "test_top1": float(t1t), "test_top3": float(t3t)}
    }, indent=2))

    print("✅ Saved transition_matrix.npy, start_probs.npy, markov_meta.json", flush=True)

if __name__ == "__main__":
    main()
