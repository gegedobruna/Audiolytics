# ml/gru_next_cluster/train_gru.py
import argparse, json, math, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Data prep utilities


def chronological_split_idx(n, train_frac=0.7, val_frac=0.15):
    i_train = int(n * train_frac)
    i_val = i_train + int(n * val_frac)
    return i_train, i_val

def make_windows(cluster_ids: np.ndarray, X_feat: np.ndarray, L: int):
    """
    Build sliding windows:
      inputs:  [t-L, ..., t-1]  (shape L x F)
      target:  cluster at time t (scalar)
    Returns arrays of shapes:
      X_windows: (Nwin, L, F)
      y: (Nwin,)
    """
    n = len(cluster_ids)
    if n <= L: return np.empty((0, L, X_feat.shape[1])), np.empty((0,), dtype=np.int64)
    # build indices for windows ending at t in [L..n-1]
    Nwin = n - L
    Xw = np.empty((Nwin, L, X_feat.shape[1]), dtype=np.float32)
    y  = np.empty((Nwin,), dtype=np.int64)
    for i in range(Nwin):
        Xw[i] = X_feat[i:i+L]
        y[i]  = int(cluster_ids[i+L])
    return Xw, y

class SeqDataset(Dataset):
    def __init__(self, Xw, y):
        self.Xw = torch.from_numpy(Xw)  # (N, L, F)
        self.y  = torch.from_numpy(y)   # (N,)
    def __len__(self): return self.Xw.shape[0]
    def __getitem__(self, idx): return self.Xw[idx], self.y[idx]

# GRU classifier

class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=1, dropout=0.2, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden, num_layers=layers,
                          batch_first=True, dropout=(dropout if layers > 1 else 0.0))
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (B, L, F)
        _, h_n = self.gru(x)  # h_n: (layers, B, H)
        h = h_n[-1]           # (B, H)
        h = self.dropout(h)
        return self.out(h)    # (B, C)

# Metrics

@torch.no_grad()
def eval_epoch(model, dl, device):
    model.eval()
    total, correct1, correct3, loss_sum = 0, 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    for X, y in dl:
        X = X.to(device)
        y = y.to(device)
        logits = model(X)
        loss = ce(logits, y)
        loss_sum += float(loss.item()) * y.size(0)
        total += y.size(0)
        # top-k
        top1 = logits.argmax(dim=1)
        correct1 += int((top1 == y).sum().item())
        top3 = torch.topk(logits, k=min(3, logits.size(1)), dim=1).indices
        correct3 += int((top3 == y.unsqueeze(1)).any(dim=1).sum().item())
    return {
        "loss": loss_sum / max(total, 1),
        "top1": correct1 / max(total, 1),
        "top3": correct3 / max(total, 1),
        "n": total
    }

# Main train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events_with_clusters", default="data/derived/sequences_with_clusters.parquet")
    ap.add_argument("--clusters_meta", default="ml/clustering/clusters_meta.json")
    ap.add_argument("--imputer", default="ml/clustering/imputer.pkl")
    ap.add_argument("--scaler", default="ml/clustering/scaler.pkl")
    ap.add_argument("--manifest", default="data/derived/prep_manifest.json")
    ap.add_argument("--outdir", default="ml/gru_next_cluster")
    # model/data params
    ap.add_argument("--seq_len", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    # training
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    clusters_meta = json.loads(Path(args.clusters_meta).read_text())
    K = int(clusters_meta["chosen_k"])
    manifest = json.loads(Path(args.manifest).read_text())
    ts_col = manifest["timestamp_col"]
    feature_cols = manifest["feature_cols"]

    # Load artifacts
    imputer = joblib.load(args.imputer)
    scaler  = joblib.load(args.scaler)

    print("Loading events with clusters…", flush=True)
    df = pd.read_parquet(args.events_with_clusters).sort_values(ts_col).reset_index(drop=True)

    # Check if sequences are aligned with features
    seq_path = Path("data/derived/sequences.parquet")
    if not seq_path.exists():
        raise SystemExit("data/derived/sequences.parquet not found. Re-run common_prep.py or keep it in repo.")
    df_feat = pd.read_parquet(seq_path).sort_values(ts_col).reset_index(drop=True)

    if len(df) != len(df_feat):
        raise SystemExit("Row count mismatch between sequences_with_clusters and sequences. They must align chronologically.")

    Xraw = df_feat[feature_cols].to_numpy(dtype=np.float32)
    Ximp = imputer.transform(Xraw)
    Xstd = scaler.transform(Ximp).astype(np.float32)
    y_clusters = df["cluster_id"].to_numpy(dtype=np.int64)

    n = len(y_clusters)
    i_train, i_val = chronological_split_idx(n, args.train_frac, args.val_frac)
    train_slice = slice(0, i_train)
    val_slice   = slice(i_train, i_val)
    test_slice  = slice(i_val, n)

    # Build windows per split
    def build(split_slice, name):
        Xw, y = make_windows(y_clusters[split_slice], Xstd[split_slice], args.seq_len)
        print(f"{name} windows: {len(y):,}")
        return Xw, y

    Xw_tr, y_tr = build(train_slice, "train")
    Xw_va, y_va = build(val_slice,   "val")
    Xw_te, y_te = build(test_slice,  "test")

    # Drop if a split got too small
    if len(y_tr) == 0 or len(y_va) == 0 or len(y_te) == 0:
        raise SystemExit("Not enough windows in one of the splits. Try reducing --seq_len.")

    # Dataloaders
    dl_tr = DataLoader(SeqDataset(Xw_tr, y_tr), batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(SeqDataset(Xw_va, y_va), batch_size=args.batch_size, shuffle=False, drop_last=False)
    dl_te = DataLoader(SeqDataset(Xw_te, y_te), batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device(args.device)
    model = GRUClassifier(input_dim=Xw_tr.shape[2], hidden=args.hidden, layers=args.layers,
                          dropout=args.dropout, num_classes=K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    best_val = math.inf
    best_path = outdir / "gru_model.pt"
    history = []

    print("Training…", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_seen = 0.0, 0
        t0 = time.time()
        for X, y in dl_tr:
            X = X.to(device); y = y.to(device)
            logits = model(X)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * y.size(0)
            n_seen += y.size(0)
        tr_loss = loss_sum / max(n_seen, 1)
        val_metrics = eval_epoch(model, dl_va, device)
        sec = time.time() - t0
        print(f"[{epoch:02d}] train_loss={tr_loss:.4f} | val_loss={val_metrics['loss']:.4f} "
              f"| val_top1={val_metrics['top1']:.3f} | val_top3={val_metrics['top3']:.3f} | {sec:.1f}s", flush=True)
        history.append({"epoch": epoch, "train_loss": tr_loss, **{f"val_{k}":v for k,v in val_metrics.items()}})

        # Early stop on val_loss
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), best_path)

    # Load best and test
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = eval_epoch(model, dl_te, device)
    print(f"Test: top1={test_metrics['top1']:.3f} | top3={test_metrics['top3']:.3f} | loss={test_metrics['loss']:.4f}")

    # Save config + metrics
    cfg = {
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "layers": args.layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "K": K,
        "feature_cols": feature_cols,
        "splits": {
            "n_rows": int(n),
            "train_frac": args.train_frac,
            "val_frac": args.val_frac
        }
    }
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))
    (outdir / "metrics.json").write_text(json.dumps({
        "val": history[-1] if history else None,
        "test": test_metrics
    }, indent=2))

    print("✅ Saved:")
    print("  -", best_path)
    print("  -", outdir / "config.json")
    print("  -", outdir / "metrics.json")

if __name__ == "__main__":
    main()
