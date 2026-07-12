"""Modality-separated evaluation: user-to-item (U2I) and item-to-item (I2I).

Protocol (deterministic, temporal leave-one-out; see utils/splits.py):
  - For each user with >= 2 interactions, the chronologically last interaction
    is the held-out positive; the rest is the training history.
  - U2I: the query vector is the mean of the user's training item vectors;
    training items are excluded from the ranked list.
  - I2I: the query vector is the user's chronologically last *training* item;
    the anchor item itself is excluded from the ranked list.

Outputs (dim is read from the item-vector matrix, so filenames are truthful):
  - Aggregate JSON:      {out_dir}/aggregates/{dataset}__{weighting}__d{dim}__{modality}__{method}.json
  - Per-query metrics:   {out_dir}/perquery/{dataset}__{weighting}__d{dim}__{modality}__{method}.npz
    (consumed by bootstrap_significance.py and effect_size_tables.py; per-query
    arrays align across methods because query construction is method-independent.)

Long-tail terminology: long_tail_exposure, long_tail_uplift, exposure_proxy
(see utils/metrics.py).

Metric semantics — two distinct notions, never to be conflated:
  - agreement recall (`ann_recall_vs_exact*`): overlap between the ANN top-k
    and the exact Flat top-k. Measures index fidelity; used for calibration.
  - recommendation relevance (`recall/precision/hr/ndcg/map/mrr`): whether
    the held-out interaction appears in the top-k. Measures usefulness to
    the user. An index can have perfect agreement recall and poor relevance
    (that is an embedding problem, not an index problem).

Modality labels are normalized to the canonical 'u2i' / 'i2i'
(utils.common.normalize_modality_label) before anything is written.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import metrics as M
from utils.ann_io import load_ann_index, build_exact_index
from utils.paths import RESULTS
from utils.modality_queries import (build_query_population, build_query_vectors,
                                    item_id_map, load_interactions,
                                    load_query_cache)
from utils.common import set_global_seed, normalize_modality_label
from utils.config import ConfigError, cfg_get, load_config
from utils.result_io import (ResultExistsError, preflight_output,
                             write_json_atomic, write_npz_atomic)

SCRIPT = "eval_modalities"
DEFAULT_CONFIG = "configs/main_cpu.yml"


def _load_id_map(item_vecs_path: Path, N: int):
    return item_id_map(item_vecs_path, N)


def _load_interactions(csv_path: str) -> pd.DataFrame:
    return load_interactions(csv_path)


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
    return build_query_vectors(modality, item_vecs, train_idx_list, test_idx)


def _ranked_lists(index, Q, exclusions, topk, N, batch=1024):
    """Search in batches with extra depth, then strip excluded items.

    Retrieval depth is computed per batch from that batch's longest exclusion
    set, not from a single global maximum: searching topk + |excluded| is
    sufficient to return topk unseen items whenever the catalog contains that
    many, and using the per-batch maximum avoids one long-history user forcing
    unnecessarily deep retrieval for every query. This yields byte-for-byte
    identical ranked lists (each row still requests >= topk + its own
    exclusions, capped at N) while retrieving less on average.
    """
    ranked = []
    for s in range(0, Q.shape[0], batch):
        batch_exclusions = exclusions[s:s + batch]
        batch_max_excl = max((len(e) for e in batch_exclusions), default=0)
        depth = min(N, topk + batch_max_excl)
        I = index.search(Q[s:s + batch], depth)
        for r, row in enumerate(I):
            ex = exclusions[s + r]
            ranked.append([int(x) for x in row if int(x) >= 0 and int(x) not in ex][:topk])
    return ranked


def evaluate_modality(modality, item_vecs, ann, exact, train_idx_list, test_idx,
                      pop_counts, topk, metric_topk, tail_frac, query_vectors=None):
    N = item_vecs.shape[0]
    built_q, exclusions, positives = _build_queries(
        modality, item_vecs, train_idx_list, test_idx)
    Q = built_q if query_vectors is None else np.ascontiguousarray(
        query_vectors, dtype=np.float32)
    if Q.shape != built_q.shape:
        raise ValueError(f"query vector shape mismatch: {Q.shape} != {built_q.shape}")

    ann_ranked = _ranked_lists(ann, Q, exclusions, topk, N)
    exact_ranked = _ranked_lists(exact, Q, exclusions, topk, N)

    k = int(metric_topk)
    k100 = min(100, int(topk))
    n_q = len(ann_ranked)
    per_query = {name: np.zeros(n_q, dtype=np.float64)
                 for name in M.PER_QUERY_METRICS}
    ann_agreement = np.zeros(n_q, dtype=np.float64)
    recall_100 = np.zeros(n_q, dtype=np.float64)
    ann_agreement_100 = np.zeros(n_q, dtype=np.float64)
    exposure_counts = np.zeros(N, dtype=np.int64)
    exposure_counts_at_100 = np.zeros(N, dtype=np.int64)
    recs_at_k = np.full((n_q, k), -1, dtype=np.int32)
    hist_pop_mean = np.zeros(n_q, dtype=np.float64)

    for i, (ranked, gt_ranked, pos) in enumerate(zip(ann_ranked, exact_ranked, positives)):
        for name, fn in M.PER_QUERY_METRICS.items():
            per_query[name][i] = fn(ranked, pos, k)
        ann_agreement[i] = M.recall_at_k(ranked, set(gt_ranked[:k]), k)
        recall_100[i] = M.recall_at_k(ranked, pos, k100)
        ann_agreement_100[i] = M.recall_at_k(ranked, set(gt_ranked[:k100]), k100)
        top_k = ranked[:k]
        recs_at_k[i, :len(top_k)] = top_k
        hist_pop_mean[i] = float(np.mean(pop_counts[train_idx_list[i]]))
        for iid in top_k:
            exposure_counts[iid] += 1
        for iid in ranked[:k100]:
            exposure_counts_at_100[iid] += 1

    tail_mask = M.tail_mask_from_popularity(pop_counts, tail_frac)
    aggregate = {
        "modality": modality,
        "queries": int(len(ann_ranked)),
        "topk": int(topk),
        "metric_topk": k,
        **{f"{name}_at_k_mean": float(v.mean()) for name, v in per_query.items()},
        "ann_recall_vs_exact_at_k_mean": float(ann_agreement.mean()),
        "recall_at_100_mean": float(recall_100.mean()),
        "ann_recall_vs_exact_at_100_mean": float(ann_agreement_100.mean()),
        "coverage_at_k": M.coverage_at_k(exposure_counts),
        "gini_exposure": M.gini_exposure(exposure_counts),
        "long_tail_exposure": M.long_tail_exposure(exposure_counts, tail_mask),
        "long_tail_uplift": M.long_tail_uplift(exposure_counts, pop_counts, tail_frac),
        "tail_frac": float(tail_frac),
    }
    per_query["ann_recall_vs_exact"] = ann_agreement
    per_query["recall_at_100"] = recall_100
    per_query["ann_recall_vs_exact_at_100"] = ann_agreement_100
    # extras consumed by exposure_analysis.py (raw counts, not just proxies)
    extras = {
        "exposure_counts_at_k": exposure_counts,
        "exposure_counts_at_100": exposure_counts_at_100,
        "pop_counts": pop_counts.astype(np.int64),
        "recs_at_k": recs_at_k,
        "hist_pop_mean": hist_pop_mean,
    }
    return aggregate, per_query, exposure_counts, extras


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="resolved scientific defaults for standalone use")
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--item_vecs", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--ann_method", required=True,
                    help="label recorded in outputs (flat/hnsw/ivfflat/ivfpq/flatpq)")
    ap.add_argument("--modality", choices=["u2i", "i2i", "both"], default="both")
    ap.add_argument("--query_cache", default=None,
                    help="validated shared modality-query NPZ; avoids rebuilding the split")
    ap.add_argument("--queries", default=None,
                    help="int, or 'full' to evaluate every eligible user")
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--metric_topk", type=int, default=None)
    ap.add_argument("--ef", type=int, default=None)
    ap.add_argument("--nprobe", type=int, default=None)
    ap.add_argument("--tail_frac", type=float, default=None)
    ap.add_argument("--min_user_interactions", type=int, default=None,
                    help="k-core filter override; defaults to "
                         "data.min_user_interactions from the resolved config")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dataset", default=None, help="dataset tag; defaults to interactions stem")
    ap.add_argument("--weighting", default=None,
                    help="embedding weighting tag (metadata only)")
    ap.add_argument("--aggregate_dir", default=RESULTS["aggregates"])
    ap.add_argument("--perquery_dir", default=RESULTS["perquery"])
    ap.add_argument("--write_mode", default="fail_if_exists",
                    choices=["fail_if_exists", "replace", "merge"])
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    try:
        cfg = load_config(args.config, cli_overrides={
            "retrieval.topk": args.topk,
            "retrieval.metric_topk": args.metric_topk,
            "retrieval.runtime_defaults.ef": args.ef,
            "retrieval.runtime_defaults.nprobe": args.nprobe,
            "evaluation.tail_fraction": args.tail_frac,
            "embedding.weighting": args.weighting,
            "data.min_user_interactions": args.min_user_interactions,
            "reproducibility.seed": args.seed,
        })
    except ConfigError as exc:
        print(f"[{SCRIPT}] ERROR: {exc}")
        return 1

    dataset = args.dataset or Path(args.interactions).stem
    agg_dir = Path(args.aggregate_dir)
    perquery_dir = Path(args.perquery_dir)
    weighting = cfg_get(cfg, "embedding.weighting", required=True)
    topk = cfg_get(cfg, "retrieval.topk", type=int, required=True)
    metric_topk = cfg_get(cfg, "retrieval.metric_topk", type=int,
                          required=True)
    ef = cfg_get(cfg, "retrieval.runtime_defaults.ef", type=int,
                 required=True)
    nprobe = cfg_get(cfg, "retrieval.runtime_defaults.nprobe", type=int,
                     required=True)
    tail_frac = cfg_get(cfg, "evaluation.tail_fraction", type=float,
                        required=True)
    min_user_interactions = cfg_get(cfg, "data.min_user_interactions",
                                    type=int, required=True)
    seed = cfg_get(cfg, "reproducibility.seed", type=int, required=True)
    queries = args.queries
    if queries is None:
        query_key = ("evaluation.queries_ml1m" if dataset == "ml-1m"
                     else "evaluation.queries_large")
        queries = cfg_get(cfg, query_key, required=True)

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.interactions}")
    print(f"[{SCRIPT}] input path: {args.item_vecs}")
    print(f"[{SCRIPT}] input path: {args.index}")
    print(f"[{SCRIPT}] output path: {agg_dir}")
    print(f"[{SCRIPT}] output path: {perquery_dir}")

    set_global_seed(seed)

    item_vecs_path = Path(args.item_vecs)
    item_vecs = np.load(item_vecs_path).astype("float32")
    N, D = item_vecs.shape
    id2idx = _load_id_map(item_vecs_path, N)

    modalities = (["u2i", "i2i"] if args.modality == "both"
                  else [normalize_modality_label(args.modality)])
    stems = {
        mod: f"{dataset}__{weighting}__d{D}__{mod}__{args.ann_method}"
        for mod in modalities
    }
    try:
        for stem in stems.values():
            preflight_output(agg_dir / f"{stem}.json", args.write_mode)
            preflight_output(perquery_dir / f"{stem}.npz", args.write_mode)
    except (ResultExistsError, ValueError) as exc:
        print(f"[{SCRIPT}] ERROR: {exc}")
        return 1

    max_q = "full" if str(queries).lower() == "full" else int(queries)

    ann = load_ann_index(args.index, D, N, ef=ef, nprobe=nprobe)
    if ann.method != args.ann_method:
        print(f"[{SCRIPT}] ERROR: requested method label {args.ann_method!r} "
              f"does not match detected index method {ann.method!r}.")
        return 1
    exact = build_exact_index(item_vecs)

    agg_dir.mkdir(parents=True, exist_ok=True)
    perquery_dir.mkdir(parents=True, exist_ok=True)

    for mod in modalities:
        if args.query_cache:
            population = load_query_cache(args.query_cache, {
                "modality": mod, "seed": int(seed),
                "min_user_interactions": int(min_user_interactions),
                "max_queries": str(max_q), "dim": int(D), "n_items": int(N),
            })
        else:
            population = build_query_population(
                interactions_path=args.interactions, item_vecs=item_vecs,
                id2idx=id2idx, modality=mod, max_queries=max_q, seed=seed,
                min_user_interactions=min_user_interactions,
                metadata={"query_fingerprint": "standalone-unfingerprinted"})
        users = population.query_ids
        train_idx_list = population.train_indices
        test_idx = population.test_indices
        pop_counts = population.pop_counts
        print(f"[{SCRIPT}] query population: queries={len(users)} "
              f"modality={mod} method={args.ann_method}")
        aggregate, per_query, exposure_counts, extras = evaluate_modality(
            mod, item_vecs, ann, exact, train_idx_list, test_idx,
            pop_counts, topk, metric_topk, tail_frac,
            query_vectors=population.query_vectors)
        aggregate.update({
            "dataset": dataset,
            "weighting": weighting,
            "method": args.ann_method,
            "detected_method": ann.method,
            "N": int(N), "D": int(D), "dim": int(D),
            "ef": int(ef), "nprobe": int(nprobe),
            "min_user_interactions": int(min_user_interactions),
            "seed": int(seed),
            "query_fingerprint": population.metadata.get("query_fingerprint"),
        })

        stem = stems[mod]
        agg_path = agg_dir / f"{stem}.json"
        write_json_atomic(aggregate, agg_path, mode=args.write_mode)

        npz_path = perquery_dir / f"{stem}.npz"
        write_npz_atomic(
            npz_path, mode=args.write_mode,
            meta=json.dumps({"dataset": dataset, "weighting": weighting,
                             "dim": int(D),
                             "modality": mod, "method": args.ann_method,
                             "metric_topk": int(metric_topk),
                             "seed": int(seed),
                             "query_fingerprint": population.metadata.get(
                                 "query_fingerprint")}),
            query_ids=np.asarray(users, dtype=str),
            exposure_proxy=M.exposure_proxy(exposure_counts),
            **extras,
            **per_query,
        )
        print(f"[{SCRIPT}] output path: {agg_path}")
        print(f"[{SCRIPT}] output path: {npz_path}")
        print(json.dumps(aggregate, indent=2))

    print(f"[{SCRIPT}] completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
