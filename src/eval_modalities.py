"""Modality-separated evaluation: user-to-item (U2I) and item-to-item (I2I).

Protocol (deterministic, temporal leave-one-out; see utils/splits.py):
  - For each user with >= 2 interactions, the chronologically last interaction
    is the held-out positive; the rest is the training history.
  - U2I: the query vector is the mean of the user's training item vectors;
    training items are excluded from the ranked list.
  - I2I: the query vector is the user's chronologically last *training* item;
    the anchor item itself is excluded from the ranked list.

Outputs:
  - Aggregate JSON:      {out_dir}/{dataset}__{weighting}__{modality}__{method}.json
  - Per-query metrics:   {out_dir}/perquery/{dataset}__{weighting}__{modality}__{method}.npz
    (consumed by bootstrap_significance.py and effect_size_tables.py; per-query
    arrays align across methods because query construction is method-independent.)

Long-tail terminology: long_tail_exposure, long_tail_uplift, exposure_proxy
(see utils/metrics.py).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import metrics as M
from utils.ann_io import load_ann_index, build_exact_index
from utils.splits import temporal_leave_one_out, build_eval_cases
from utils.common import set_global_seed

SCRIPT = "eval_modalities"


def _load_id_map(item_vecs_path: Path, N: int):
    ids_path = item_vecs_path.with_name("item_ids.npy")
    if ids_path.is_file():
        arr = np.load(ids_path, allow_pickle=True)
        return {str(arr[i]): i for i in range(len(arr))}
    return {str(i): i for i in range(N)}


def _load_interactions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["user_id", "item_id", "timestamp"])
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
    return df


def _popularity(train_df: pd.DataFrame, id2idx: dict, N: int) -> np.ndarray:
    pop = np.zeros(N, dtype=np.int64)
    codes = train_df["item_id"].map(lambda x: id2idx.get(str(x), -1)).to_numpy()
    codes = codes[codes >= 0].astype(np.int64)
    np.add.at(pop, codes, 1)
    return pop


def _build_queries(modality: str, item_vecs: np.ndarray, train_idx_list, test_idx):
    """Return (Q, exclusions, positives) for a modality.

    exclusions[i]: set of item indices to strip from the ranked list.
    positives[i]: set containing the held-out item index.
    """
    n = len(train_idx_list)
    D = item_vecs.shape[1]
    Q = np.zeros((n, D), dtype=np.float32)
    exclusions = []
    for i, hist in enumerate(train_idx_list):
        if modality == "u2i":
            Q[i] = item_vecs[hist].mean(axis=0)
            exclusions.append(set(int(h) for h in hist))
        elif modality == "i2i":
            anchor = int(hist[-1])  # chronologically last training item
            Q[i] = item_vecs[anchor]
            exclusions.append({anchor})
        else:
            raise ValueError(f"unknown modality {modality}")
    positives = [{int(t)} for t in test_idx]
    return Q, exclusions, positives


def _ranked_lists(index, Q, exclusions, topk, N, batch=1024):
    """Search in batches with extra depth, then strip excluded items."""
    max_excl = max((len(e) for e in exclusions), default=0)
    depth = min(N, topk + min(max_excl, 300) + 1)
    ranked = []
    for s in range(0, Q.shape[0], batch):
        I = index.search(Q[s:s + batch], depth)
        for r, row in enumerate(I):
            ex = exclusions[s + r]
            ranked.append([int(x) for x in row if int(x) >= 0 and int(x) not in ex][:topk])
    return ranked


def evaluate_modality(modality, item_vecs, ann, exact, train_idx_list, test_idx,
                      pop_counts, topk, metric_topk, tail_frac):
    N = item_vecs.shape[0]
    Q, exclusions, positives = _build_queries(modality, item_vecs, train_idx_list, test_idx)

    ann_ranked = _ranked_lists(ann, Q, exclusions, topk, N)
    exact_ranked = _ranked_lists(exact, Q, exclusions, topk, N)

    k = int(metric_topk)
    per_query = {name: np.zeros(len(ann_ranked), dtype=np.float64)
                 for name in M.PER_QUERY_METRICS}
    ann_agreement = np.zeros(len(ann_ranked), dtype=np.float64)
    exposure_counts = np.zeros(N, dtype=np.int64)

    for i, (ranked, gt_ranked, pos) in enumerate(zip(ann_ranked, exact_ranked, positives)):
        for name, fn in M.PER_QUERY_METRICS.items():
            per_query[name][i] = fn(ranked, pos, k)
        ann_agreement[i] = M.recall_at_k(ranked, set(gt_ranked[:k]), k)
        for iid in ranked[:k]:
            exposure_counts[iid] += 1

    tail_mask = M.tail_mask_from_popularity(pop_counts, tail_frac)
    aggregate = {
        "modality": modality,
        "queries": int(len(ann_ranked)),
        "topk": int(topk),
        "metric_topk": k,
        **{f"{name}_at_k_mean": float(v.mean()) for name, v in per_query.items()},
        "ann_recall_vs_exact_at_k_mean": float(ann_agreement.mean()),
        "coverage_at_k": M.coverage_at_k(exposure_counts),
        "gini_exposure": M.gini_exposure(exposure_counts),
        "long_tail_exposure": M.long_tail_exposure(exposure_counts, tail_mask),
        "long_tail_uplift": M.long_tail_uplift(exposure_counts, pop_counts, tail_frac),
        "tail_frac": float(tail_frac),
    }
    per_query["ann_recall_vs_exact"] = ann_agreement
    return aggregate, per_query, exposure_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--item_vecs", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--ann_method", required=True,
                    help="label recorded in outputs (flat/hnsw/ivfflat/ivfpq/flatpq)")
    ap.add_argument("--modality", choices=["u2i", "i2i", "both"], default="both")
    ap.add_argument("--queries", default="2000",
                    help="int, or 'full' to evaluate every eligible user")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--metric_topk", type=int, default=10)
    ap.add_argument("--ef", type=int, default=128)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--tail_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default=None, help="dataset tag; defaults to interactions stem")
    ap.add_argument("--weighting", default="none", help="embedding weighting tag (metadata only)")
    ap.add_argument("--out_dir", default="results/main")
    args = ap.parse_args()

    dataset = args.dataset or Path(args.interactions).stem
    out_dir = Path(args.out_dir)
    perquery_dir = out_dir / "perquery"

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.interactions}")
    print(f"[{SCRIPT}] input path: {args.item_vecs}")
    print(f"[{SCRIPT}] input path: {args.index}")
    print(f"[{SCRIPT}] output path: {out_dir}")

    set_global_seed(args.seed)

    item_vecs_path = Path(args.item_vecs)
    item_vecs = np.load(item_vecs_path).astype("float32")
    N, D = item_vecs.shape
    id2idx = _load_id_map(item_vecs_path, N)

    inter = _load_interactions(args.interactions)
    train_df, test_df = temporal_leave_one_out(inter)
    max_q = "full" if str(args.queries).lower() == "full" else int(args.queries)
    users, train_idx_list, test_idx = build_eval_cases(
        train_df, test_df, id2idx, max_queries=max_q, seed=args.seed)
    pop_counts = _popularity(train_df, id2idx, N)
    print(f"[{SCRIPT}] split: train={len(train_df)} test_users={len(users)} "
          f"modality={args.modality} method={args.ann_method}")

    ann = load_ann_index(args.index, D, N, ef=args.ef, nprobe=args.nprobe)
    exact = build_exact_index(item_vecs)

    modalities = ["u2i", "i2i"] if args.modality == "both" else [args.modality]
    out_dir.mkdir(parents=True, exist_ok=True)
    perquery_dir.mkdir(parents=True, exist_ok=True)

    for mod in modalities:
        aggregate, per_query, exposure_counts = evaluate_modality(
            mod, item_vecs, ann, exact, train_idx_list, test_idx,
            pop_counts, args.topk, args.metric_topk, args.tail_frac)
        aggregate.update({
            "dataset": dataset,
            "weighting": args.weighting,
            "method": args.ann_method,
            "detected_method": ann.method,
            "N": int(N), "D": int(D),
            "ef": int(args.ef), "nprobe": int(args.nprobe),
            "seed": int(args.seed),
        })

        stem = f"{dataset}__{args.weighting}__{mod}__{args.ann_method}"
        agg_path = out_dir / f"{stem}.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)

        npz_path = perquery_dir / f"{stem}.npz"
        np.savez_compressed(
            npz_path,
            meta=json.dumps({"dataset": dataset, "weighting": args.weighting,
                             "modality": mod, "method": args.ann_method,
                             "metric_topk": int(args.metric_topk),
                             "seed": int(args.seed)}),
            exposure_proxy=M.exposure_proxy(exposure_counts),
            **per_query,
        )
        print(f"[{SCRIPT}] output path: {agg_path}")
        print(f"[{SCRIPT}] output path: {npz_path}")
        print(json.dumps(aggregate, indent=2))

    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
