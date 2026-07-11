"""Train non-SVD item embeddings for the backbone sensitivity study.

Reviewer concern addressed: "SVD-only embeddings". These backbones exist to
test whether the ANN *method ranking* is stable under different embedding
geometries — they are sensitivity checks, not the main pipeline, and make no
state-of-the-art claim.

Backbones:
    bpr_matrix_factorization  BPR-MF trained with pure NumPy SGD (no extra deps)
    two_tower_mlp             ID-embedding two-tower with one hidden layer;
                              REQUIRES PyTorch (optional dependency, not in
                              requirements.txt). Fails with a clear message
                              when torch is absent.

Outputs (same layout as train_embeddings.py, so build_index/eval work as-is):
    item_vecs.npy, item_ids.npy, embedding_meta.json (embedding_backend field)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.common import set_global_seed

SCRIPT = "train_neural_embeddings"
BACKBONES = ("bpr_matrix_factorization", "two_tower_mlp")


def _encode(df):
    users = df["user_id"].astype("category")
    items = df["item_id"].astype("category")
    return (users.cat.codes.values.astype(np.int64),
            items.cat.codes.values.astype(np.int64),
            users.cat.categories.values, items.cat.categories.values)


def train_bpr(u, i, n_users, n_items, dim, epochs, lr, reg, n_negatives, seed):
    """BPR matrix factorization with SGD, NumPy only. Deterministic per seed."""
    rng = np.random.default_rng(seed)
    P = 0.01 * rng.standard_normal((n_users, dim)).astype(np.float64)
    Q = 0.01 * rng.standard_normal((n_items, dim)).astype(np.float64)
    n = len(u)
    for epoch in range(epochs):
        order = rng.permutation(n)
        loss_acc = 0.0
        for t in order:
            uu, ii = u[t], i[t]
            for _ in range(n_negatives):
                jj = int(rng.integers(0, n_items))
                x = P[uu] @ (Q[ii] - Q[jj])
                sig = 1.0 / (1.0 + np.exp(-x))
                g = (1.0 - sig)
                pu = P[uu].copy()
                P[uu] += lr * (g * (Q[ii] - Q[jj]) - reg * P[uu])
                Q[ii] += lr * (g * pu - reg * Q[ii])
                Q[jj] += lr * (-g * pu - reg * Q[jj])
                loss_acc += -np.log(max(sig, 1e-10))
        print(f"[{SCRIPT}] bpr epoch {epoch + 1}/{epochs} "
              f"mean_loss={loss_acc / (n * n_negatives):.4f}")
    return Q.astype(np.float32)


def train_two_tower(u, i, n_users, n_items, dim, epochs, lr, hidden_dim,
                    batch_size, seed):
    """Small ID-embedding two-tower model with in-batch negatives.

    PyTorch is an OPTIONAL dependency: install manually (pip install torch)
    only if you want this backbone.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise SystemExit(
            f"[{SCRIPT}] ERROR: backbone 'two_tower_mlp' requires PyTorch, "
            f"which is optional and not in requirements.txt. Install it with "
            f"'pip install torch' (CPU build) or choose another backbone.")

    torch.manual_seed(seed)
    device = "cpu"  # CPU-only benchmark

    class Tower(nn.Module):
        def __init__(self, n, dim, hidden):
            super().__init__()
            self.emb = nn.Embedding(n, dim)
            nn.init.normal_(self.emb.weight, std=0.01)
            self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, dim))

        def forward(self, ids):
            return self.mlp(self.emb(ids))

    user_tower = Tower(n_users, dim, hidden_dim).to(device)
    item_tower = Tower(n_items, dim, hidden_dim).to(device)
    opt = torch.optim.Adam(list(user_tower.parameters())
                           + list(item_tower.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    u_t = torch.from_numpy(np.asarray(u))
    i_t = torch.from_numpy(np.asarray(i))
    n = len(u)
    g = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            uu, ii = u_t[idx], i_t[idx]
            zu = user_tower(uu)
            zi = item_tower(ii)
            logits = zu @ zi.T  # in-batch negatives
            labels = torch.arange(len(idx))
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        print(f"[{SCRIPT}] two_tower epoch {epoch + 1}/{epochs} "
              f"mean_loss={total / n:.4f}")

    with torch.no_grad():
        all_items = torch.arange(n_items)
        item_vecs = item_tower(all_items).cpu().numpy().astype(np.float32)
    return item_vecs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--backbone", choices=list(BACKBONES), required=True)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--reg", type=float, default=0.002)
    ap.add_argument("--n_negatives", type=int, default=1)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--normalize", choices=["none", "l2"], default="l2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config_hash", default="unknown",
                    help="resolved experiment configuration hash")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.interactions}")
    print(f"[{SCRIPT}] output path: {args.out_dir}")

    set_global_seed(args.seed)

    df = pd.read_csv(args.interactions)
    u, i, user_cats, item_cats = _encode(df)
    n_users, n_items = len(user_cats), len(item_cats)
    print(f"[{SCRIPT}] backbone={args.backbone} users={n_users} items={n_items} "
          f"interactions={len(df)}")

    if args.backbone == "bpr_matrix_factorization":
        epochs = args.epochs if args.epochs is not None else 10
        lr = args.lr if args.lr is not None else 0.05
        item_vecs = train_bpr(u, i, n_users, n_items, args.dim, epochs, lr,
                              args.reg, args.n_negatives, args.seed)
    else:
        epochs = args.epochs if args.epochs is not None else 5
        lr = args.lr if args.lr is not None else 0.001
        item_vecs = train_two_tower(u, i, n_users, n_items, args.dim, epochs,
                                    lr, args.hidden_dim, args.batch_size,
                                    args.seed)

    if args.normalize == "l2":
        norms = np.linalg.norm(item_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        item_vecs = (item_vecs / norms).astype(np.float32)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "item_vecs.npy", item_vecs)
    np.save(out / "item_ids.npy", item_cats)
    meta = {
        "interactions": str(args.interactions),
        "embedding_backend": args.backbone,
        "n_users": int(n_users), "n_items": int(n_items),
        "n_interactions": int(len(df)),
        "dim": int(args.dim), "epochs": int(epochs), "lr": float(lr),
        "reg": float(args.reg), "n_negatives": int(args.n_negatives),
        "hidden_dim": int(args.hidden_dim),
        "batch_size": int(args.batch_size),
        "normalize": args.normalize, "seed": int(args.seed),
        "config_hash": str(args.config_hash),
    }
    with open(out / "embedding_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[{SCRIPT}] saved vectors to {out} shape={item_vecs.shape}")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
