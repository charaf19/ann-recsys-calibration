"""Train item embeddings with TruncatedSVD over a (optionally weighted)
user-item interaction matrix.

New in the revision: --weighting {none,tfidf,bm25} (with --bm25_k1/--bm25_b),
--normalize {none,l2}, and an explicit --seed. Defaults reproduce the
original behavior (raw binary matrix, no normalization, seed 42).

Outputs to --out_dir:
    item_vecs.npy        float32 (n_items, dim)
    item_ids.npy         original item ids aligned with rows of item_vecs
    embedding_meta.json  provenance (weighting, dim, seed, matrix stats)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from utils.weighting import apply_weighting, WEIGHTING_CHOICES
from utils.common import set_global_seed
from utils.preprocessing import (filter_min_user_interactions,
                                 DEFAULT_MIN_USER_INTERACTIONS)

SCRIPT = "train_embeddings"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", type=str, required=True, help="CSV user_id,item_id,timestamp")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="data/emb")
    ap.add_argument("--weighting", choices=list(WEIGHTING_CHOICES), default="none",
                    help="interaction weighting applied before SVD (default: none)")
    ap.add_argument("--bm25_k1", type=float, default=1.2)
    ap.add_argument("--bm25_b", type=float, default=0.75)
    ap.add_argument("--normalize", choices=["none", "l2"], default="none",
                    help="L2-normalize item vectors after SVD (default: none)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_user_interactions", type=int,
                    default=DEFAULT_MIN_USER_INTERACTIONS,
                    help="k-core filter: drop users with fewer interactions "
                         "(canonical value comes from data.min_user_interactions)")
    ap.add_argument("--config_hash", default="unknown",
                    help="resolved experiment configuration hash")
    args = ap.parse_args()

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.interactions}")
    print(f"[{SCRIPT}] output path: {args.out_dir}")

    set_global_seed(args.seed)

    df = pd.read_csv(args.interactions)
    n_before = len(df)
    df = filter_min_user_interactions(df, args.min_user_interactions)
    print(f"[{SCRIPT}] k-core filter min_user_interactions="
          f"{args.min_user_interactions}: interactions {n_before} -> {len(df)}")
    users = df["user_id"].astype("category")
    items = df["item_id"].astype("category")
    user_codes = users.cat.codes.values
    item_codes = items.cat.codes.values
    n_users = users.cat.categories.size
    n_items = items.cat.categories.size
    print(f"[{SCRIPT}] users={n_users} items={n_items} interactions={len(df)}")

    R = sp.coo_matrix((np.ones(len(df)), (user_codes, item_codes)),
                      shape=(n_users, n_items)).tocsr()
    # collapse duplicate (user,item) pairs to counts, then weight
    R.sum_duplicates()
    print(f"[{SCRIPT}] applying weighting: {args.weighting}")
    W = apply_weighting(R, args.weighting, bm25_k1=args.bm25_k1, bm25_b=args.bm25_b)

    dim = min(args.dim, max(2, min(W.shape) - 1))
    if dim != args.dim:
        print(f"[{SCRIPT}] WARN: dim reduced {args.dim} -> {dim} (matrix rank limit)")

    svd = TruncatedSVD(n_components=dim, random_state=args.seed)
    item_vecs = svd.fit_transform(W.transpose().tocsr()).astype("float32")

    if args.normalize == "l2":
        norms = np.linalg.norm(item_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        item_vecs = (item_vecs / norms).astype("float32")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "item_vecs.npy", item_vecs)
    np.save(out / "item_ids.npy", items.cat.categories.values)

    meta = {
        "interactions": str(args.interactions),
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_interactions": int(len(df)),
        "dim_requested": int(args.dim),
        "dim_effective": int(dim),
        "weighting": args.weighting,
        "bm25_k1": float(args.bm25_k1),
        "bm25_b": float(args.bm25_b),
        "normalize": args.normalize,
        "seed": int(args.seed),
        "min_user_interactions": int(args.min_user_interactions),
        "config_hash": str(args.config_hash),
        "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
    }
    with open(out / "embedding_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[{SCRIPT}] saved vectors to {out} shape={item_vecs.shape}")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
