import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import faiss


# ----------------------------
# Index resolution & ANN setup
# ----------------------------
def _resolve_index_path(p: str) -> Path:
    pth = Path(p)
    if pth.is_file():
        return pth
    if not pth.is_dir():
        raise ValueError(f"Index path not found: {p}")
    known = [
        "index.faiss",
        "faiss_hnsw.index",
        "hnsw.bin",
        "hnsw_index.bin",
        "faiss_ivfpq.index",
        "faiss_ivfflat.index",
        "faiss_flatpq.index",
        "faiss_flat.index",
    ]
    for name in known:
        q = pth / name
        if q.is_file():
            return q
    faiss_files = [f for f in pth.iterdir() if f.is_file() and f.suffix.lower() == ".faiss"]
    if faiss_files:
        faiss_files.sort(key=lambda f: (0 if f.name == "index.faiss" else 1, -f.stat().st_size))
        return faiss_files[0]
    raise ValueError("No known index files in the directory.")


def _prepare_ann(fpath: Path, dim: int, N: int, nprobe: int = None, ef: int = None):
    index = faiss.read_index(str(fpath))
    index_dc = faiss.downcast_index(index)
    ivf = None
    try:
        ivf = faiss.extract_index_ivf(index)
    except Exception:
        pass
    if ivf is not None:
        ivf_dc = faiss.downcast_index(ivf)
        if isinstance(ivf_dc, faiss.IndexIVFPQ):
            method = "ivfpq"
        else:
            method = "ivfflat"
        if nprobe is not None:
            ivf.nprobe = int(nprobe)
    else:
        if isinstance(index_dc, faiss.IndexHNSWFlat):
            method = "hnsw"
            if ef is not None:
                index_dc.hnsw.efSearch = int(ef)
        else:
            method = "flatpq" if isinstance(index_dc, faiss.IndexPQ) else "flat"

    def _search(Q, topk: int):
        _, I = index.search(Q, topk)
        return I

    return method, _search


# ----------------------------
# Metrics
# ----------------------------
def _dcg(rels):
    gains = (2 ** rels - 1)
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(gains * discounts))


def _ndcg_at_k(ranked, positives, k):
    if k <= 0:
        return 0.0
    pos = set(positives)
    rels = np.array([1 if i in pos else 0 for i in ranked[:k]], dtype=np.float32)
    dcg = _dcg(rels)
    ideal = _dcg(np.ones(min(k, len(pos)), dtype=np.float32))
    return 0.0 if ideal == 0.0 else dcg / ideal


def _precision_at_k(ranked, positives, k):
    if k <= 0:
        return 0.0
    return float(len(set(ranked[:k]).intersection(positives))) / float(k)


def _recall_at_k(ranked, positives, k):
    if k <= 0 or len(positives) == 0:
        return 0.0
    return float(len(set(ranked[:k]).intersection(positives))) / float(min(k, len(positives)))


def _hit_rate_at_k(ranked, positives, k):
    return 1.0 if len(set(ranked[:k]).intersection(positives)) > 0 else 0.0


def _average_precision_at_k(ranked, positives, k):
    if k <= 0:
        return 0.0
    hits = 0
    sum_prec = 0.0
    pos = set(positives)
    K = min(k, len(ranked))
    for i in range(K):
        if ranked[i] in pos:
            hits += 1
            sum_prec += hits / float(i + 1)
    return 0.0 if k == 0 else sum_prec / float(k)


def _mrr_at_k(ranked, positives, k):
    pos = set(positives)
    K = min(k, len(ranked))
    for i in range(K):
        if ranked[i] in pos:
            return 1.0 / float(i + 1)
    return 0.0


def _gini_exposure(counts):
    x = np.sort(counts.astype(np.float64))
    n = x.size
    if n == 0:
        return 0.0
    cumx = np.cumsum(x)
    if cumx[-1] == 0:
        return 0.0
    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(g)


# ----------------------------
# Interaction utilities
# ----------------------------
def _load_id_map(item_vecs_path: Path, N: int):
    ids_path = item_vecs_path.with_name("item_ids.npy")
    if ids_path.is_file():
        arr = np.load(ids_path, allow_pickle=True)
        return {str(arr[i]): i for i in range(len(arr))}
    # fallback: assume 0..N-1 ids
    return {str(i): i for i in range(N)}


def _load_interactions(csv_path: str):
    usecols = ["user_id", "item_id", "timestamp"]
    df = pd.read_csv(csv_path, usecols=usecols)
    # normalize dtypes to strings for robust joining
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    # timestamp may be missing or non-numeric in some sources; coerce
    if "timestamp" not in df.columns:
        df["timestamp"] = 0
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(np.int64)
    return df


def _make_loo_queries(df: pd.DataFrame, id2idx: dict, max_q: int, seed: int = 42):
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    groups = df.groupby("user_id", sort=False)

    anchors = []
    positives = []

    for _, g in groups:
        if len(g) < 2:
            continue
        last = g.iloc[-1]
        train = g.iloc[:-1]
        a = id2idx.get(str(last["item_id"]))
        if a is None:
            continue
        pos_idx = [id2idx.get(str(x), None) for x in train["item_id"].tolist()]
        pos_idx = [int(p) for p in pos_idx if p is not None]
        if not pos_idx:
            continue
        anchors.append(int(a))
        positives.append(np.array(pos_idx, dtype=np.int32))

    if not anchors:
        raise ValueError("No valid leave-one-out queries could be built from interactions.")

    rng = np.random.default_rng(seed)
    if len(anchors) > max_q:
        sel = rng.choice(len(anchors), size=max_q, replace=False)
        anchors = [anchors[i] for i in sel]
        positives = [positives[i] for i in sel]

    return np.array(anchors, dtype=np.int32), positives


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", required=True)          # NEW: interactions CSV path
    ap.add_argument("--item_vecs", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--ann_method", required=True)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--metric_topk", type=int, default=10)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--ef", type=int, default=128)
    args = ap.parse_args()

    item_vecs_path = Path(args.item_vecs)
    item_vecs = np.load(item_vecs_path).astype("float32")
    N, D = item_vecs.shape

    # ANN
    fpath = _resolve_index_path(args.index)
    method, ann_search = _prepare_ann(fpath, D, N, nprobe=args.nprobe, ef=args.ef)

    # Exact (for ANN agreement baselines)
    exact = faiss.IndexFlatL2(D)
    exact.add(item_vecs)

    # Interactions + id mapping
    id2idx = _load_id_map(item_vecs_path, N)
    inter = _load_interactions(args.interactions)

    # Build leave-one-out queries (anchor item + positives from the same user history)
    anchor_idx, positives_list = _make_loo_queries(inter, id2idx, max_q=args.queries, seed=42)

    # Popularity from training part (for long-tail baseline)
    inter = inter.sort_values(["user_id", "timestamp"], kind="mergesort")
    train = inter.groupby("user_id", sort=False).apply(lambda g: g.iloc[:-1]).reset_index(drop=True)
    pop_counts = np.zeros(N, dtype=np.int64)
    for item in train["item_id"].tolist():
        j = id2idx.get(str(item))
        if j is not None:
            pop_counts[j] += 1

    # Metrics accumulators
    e2e_rec, e2e_prec, e2e_ndcg, e2e_hr, e2e_map, e2e_mrr = [], [], [], [], [], []
    ann_agree_recall = []
    exposure_counts = np.zeros(N, dtype=np.int64)

    k = int(args.metric_topk)
    topk = int(args.topk)

    for qi, pos in zip(anchor_idx, positives_list):
        q = item_vecs[qi : qi + 1]

        # exact neighbors (agreement baseline)
        _, gtI = exact.search(q, topk + 1)
        gt_ids = [int(i) for i in gtI[0] if int(i) != int(qi)][:topk]

        # ann neighbors
        annI = ann_search(q, topk + 1)[0]
        ann_ids = [int(i) for i in annI if int(i) != int(qi)][:topk]

        # agreement recall@k (ANN vs exact)
        ann_agree_recall.append(
            _recall_at_k(ann_ids, gt_ids, k)
        )

        # relevance metrics vs interactions (positives from same user)
        e2e_rec.append(_recall_at_k(ann_ids, pos, k))
        e2e_prec.append(_precision_at_k(ann_ids, pos, k))
        e2e_ndcg.append(_ndcg_at_k(ann_ids, pos, k))
        e2e_hr.append(_hit_rate_at_k(ann_ids, pos, k))
        e2e_map.append(_average_precision_at_k(ann_ids, pos, k))
        e2e_mrr.append(_mrr_at_k(ann_ids, pos, k))

        # exposure for fairness/coverage
        for iid in ann_ids[:k]:
            exposure_counts[iid] += 1

    # Coverage & fairness
    coverage = float(np.count_nonzero(exposure_counts) / float(N))
    gini = _gini_exposure(exposure_counts)

    # Long-tail share & uplift (tail defined by training popularity)
    nz = pop_counts > 0
    if np.any(nz):
        thr = np.quantile(pop_counts[nz], 0.2)
        tail_mask = pop_counts <= thr
    else:
        tail_mask = pop_counts == 0

    total_exposures = int(np.sum(exposure_counts))
    tail_exposures = int(np.sum(exposure_counts[tail_mask]))
    tail_share_exposure = 0.0 if total_exposures == 0 else float(tail_exposures) / float(total_exposures)

    total_pop = int(np.sum(pop_counts))
    tail_pop = int(np.sum(pop_counts[tail_mask]))
    tail_share_pop = 0.0 if total_pop == 0 else float(tail_pop) / float(total_pop)

    long_tail_uplift = float(tail_share_exposure - tail_share_pop)

    out = {
        "method": method,
        "N": int(N),
        "D": int(D),
        "queries": int(len(anchor_idx)),
        "topk": int(topk),
        "metric_topk": int(k),

        # Relevance vs interactions (what you want for the paper)
        "recall_at_topk_mean": float(np.mean(e2e_rec)),
        "ndcg_at_k_mean": float(np.mean(e2e_ndcg)),
        "hr_at_k_mean": float(np.mean(e2e_hr)),
        "precision_at_k_mean": float(np.mean(e2e_prec)),
        "map_at_k_mean": float(np.mean(e2e_map)),
        "mrr_at_k_mean": float(np.mean(e2e_mrr)),

        # Exposure & fairness
        "coverage_at_k": coverage,
        "gini_exposure": gini,
        "long_tail_uplift": long_tail_uplift,

        # ANN agreement (keep, but do not interpret as user relevance)
        "ann_recall_vs_exact_at_k_mean": float(np.mean(ann_agree_recall)),
    }

    out_dir = Path(args.index)
    if out_dir.is_file():
        out_dir = out_dir.parent
    out_path = out_dir / f"end2end_{args.ann_method}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
