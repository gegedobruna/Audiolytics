# ml/clustering/train_clusters.py
import argparse, json, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_manifest(manifest_path):
    return json.loads(Path(manifest_path).read_text())


def timed(msg):
    print(msg, flush=True)
    return time.time()


def pick_k(X_std, k_min, k_max, sil_sample_max=50000, random_state=42, n_init=10):
    """
    Try K in [k_min, k_max]. Compute silhouette on a sample (for speed).
    Returns: best_model, metadata(dict)
    """
    results = []
    # sample indices for silhouette
    if X_std.shape[0] > sil_sample_max:
        rng = np.random.RandomState(random_state)
        sil_idx = rng.choice(X_std.shape[0], sil_sample_max, replace=False)
        X_sil = X_std[sil_idx]
        print(f"Silhouette sample: {sil_sample_max:,} rows", flush=True)
    else:
        X_sil = X_std
        print(f"Silhouette on full data: {X_std.shape[0]:,} rows", flush=True)

    for k in range(k_min, k_max + 1):
        t0 = timed(f"[KMeans] Fitting k={k} …")
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X_std)
        fit_sec = time.time() - t0
        inertia = float(km.inertia_)

        # silhouette on the sample (reassign sample labels)
        t1 = time.time()
        sil_labels = km.predict(X_sil)
        sil = float(silhouette_score(X_sil, sil_labels)) if k > 1 else -1.0
        sil_sec = time.time() - t1

        print(f"[KMeans] k={k} done | fit {fit_sec:.1f}s | sil {sil_sec:.1f}s | "
              f"silhouette={sil:.4f} | inertia={inertia:.0f}", flush=True)

        results.append((k, sil, inertia, km))

    # choose best by silhouette, tie-break by lower inertia
    results.sort(key=lambda t: (t[1], -t[2]), reverse=True)
    best_k, best_sil, best_inertia, best_model = results[0]

    table = [{"k": int(k), "silhouette": float(s), "inertia": float(i)} for k, s, i, _ in results]
    meta = {"chosen_k": int(best_k), "chosen_silhouette": float(best_sil), "grid": table}
    print(f"✅ Picked K={best_k} (silhouette={best_sil:.4f})", flush=True)
    return best_model, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/derived/sequences.parquet")
    ap.add_argument("--manifest", default="data/derived/prep_manifest.json")
    ap.add_argument("--outdir", default="ml/clustering")
    ap.add_argument("--kmin", type=int, default=10)
    ap.add_argument("--kmax", type=int, default=14)
    ap.add_argument("--sample_max", type=int, default=0,
                    help="If >0, use at most this many rows for K selection (speeds up).")
    ap.add_argument("--sil_sample_max", type=int, default=50000,
                    help="Sample size for silhouette score.")
    ap.add_argument("--final_refit_on_full", action="store_true",
                    help="Refit best K on FULL data before saving (slower but best).")
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load schema info
    mf = load_manifest(args.manifest)
    feature_cols = mf["feature_cols"]
    ts_col = mf["timestamp_col"]

    if not feature_cols:
        raise SystemExit("No audio feature columns found in manifest. Aborting.")

    timed("Loading events parquet …")
    df = pd.read_parquet(args.events).sort_values(ts_col).reset_index(drop=True)
    n_rows = df.shape[0]
    print(f"Rows: {n_rows:,} | Feature cols: {feature_cols}", flush=True)

    # Prepare feature matrix
    Xraw = df[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    Ximp = imputer.fit_transform(Xraw)
    scaler = StandardScaler()
    X = scaler.fit_transform(Ximp)

    # Optional subsample for fast K selection
    X_for_k = X
    if args.sample_max and X.shape[0] > args.sample_max:
        rng = np.random.RandomState(args.random_state)
        idx = rng.choice(X.shape[0], args.sample_max, replace=False)
        X_for_k = X[idx]
        print(f"Subsampled to {args.sample_max:,} rows for K selection.", flush=True)

    print(f"Trying K ∈ [{args.kmin}, {args.kmax}] …", flush=True)
    best_model, meta = pick_k(
        X_for_k,
        k_min=args.kmin,
        k_max=args.kmax,
        sil_sample_max=args.sil_sample_max,
        random_state=args.random_state,
        n_init=10
    )

    # Optionally refit on full data for highest quality centroids
    if args.final_refit_on_full:
        t = timed("Refitting best K on FULL data …")
        best_model = KMeans(
            n_clusters=meta["chosen_k"],
            random_state=args.random_state,
            n_init=10
        ).fit(X)
        print(f"Refit done in {time.time() - t:.1f}s", flush=True)

    # Save artifacts
    print("Saving artifacts …", flush=True)
    joblib.dump(imputer, outdir / "imputer.pkl")
    joblib.dump(scaler, outdir / "scaler.pkl")
    joblib.dump(best_model, outdir / "kmeans.pkl")
    (outdir / "clusters_meta.json").write_text(json.dumps({
        **meta,
        "feature_cols": feature_cols
    }, indent=2))

    # Centroids (original scale for readability)
    centroids_std = best_model.cluster_centers_
    centroids_orig = scaler.inverse_transform(centroids_std)
    cent_df = pd.DataFrame(centroids_orig, columns=feature_cols)
    cent_df.index.name = "cluster_id"
    cent_df.to_csv(outdir / "cluster_centroids.csv", index=True)

    # Assign labels to ALL events
    t = timed("Predicting cluster labels for all events …")
    labels = best_model.predict(X)
    print(f"Predicted {labels.shape[0]:,} labels in {time.time() - t:.1f}s", flush=True)

    # Write sequences with clusters (keep ts + session_id if present)
    keep_cols = [ts_col]
    if "session_id" in df.columns:
        keep_cols.append("session_id")
    # Keep everything except the raw feature columns to avoid duplication bloat
    meta_cols = [c for c in df.columns if c not in feature_cols or c in keep_cols]
    df_out = df[meta_cols].copy()
    df_out["cluster_id"] = labels
    Path("data/derived").mkdir(parents=True, exist_ok=True)
    df_out.to_parquet("data/derived/sequences_with_clusters.parquet", index=False)

    print("✅ Saved:")
    print("  -", outdir / "imputer.pkl")
    print("  -", outdir / "scaler.pkl")
    print("  -", outdir / "kmeans.pkl")
    print("  -", outdir / "clusters_meta.json")
    print("  -", outdir / "cluster_centroids.csv")
    print("  - data/derived/sequences_with_clusters.parquet")
    print(f"🎯 Chosen K = {meta['chosen_k']}  | silhouette = {meta['chosen_silhouette']:.4f}", flush=True)


if __name__ == "__main__":
    main()
