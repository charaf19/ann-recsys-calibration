"""Stage-specific compatibility fingerprints for reusable experiment artifacts."""
import hashlib
import json


FINGERPRINT_VERSION = 1


def fingerprint(stage, fields):
    """Hash only fields that can change the named stage's scientific output."""
    payload = {"fingerprint_version": FINGERPRINT_VERSION,
               "stage": str(stage), "fields": fields}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def embedding_fingerprint(*, dataset, min_user_interactions, weighting, dim,
                          normalize, bm25_k1, bm25_b, seed):
    return fingerprint("embedding", {
        "dataset": dataset, "min_user_interactions": int(min_user_interactions),
        "weighting": weighting, "dim": int(dim), "normalize": normalize,
        "bm25_k1": float(bm25_k1), "bm25_b": float(bm25_b), "seed": int(seed),
    })


def index_fingerprint(*, embedding_fp, method, budget_mb, omp_threads,
                      index_config, seed):
    relevant = {}
    if method == "hnsw":
        relevant = {"hnsw": index_config.get("hnsw", {})}
    elif method == "ivfflat":
        relevant = {"ivf": index_config.get("ivf", {})}
    elif method == "ivfpq":
        relevant = {key: index_config.get(key, {})
                    for key in ("ivf", "pq", "ivfpq")}
    elif method == "flatpq":
        relevant = {"pq": index_config.get("pq", {})}
    return fingerprint("index", {
        "embedding_fingerprint": embedding_fp, "method": method,
        "budget_mb": int(budget_mb), "omp_threads": int(omp_threads),
        "index_config": relevant, "seed": int(seed), "faiss_cpu": True,
    })


def query_fingerprint(*, embedding_fp, dataset, modality, weighting, dim,
                      min_user_interactions, max_queries, topk, metric_topk,
                      seed):
    return fingerprint("query_evaluation", {
        "embedding_fingerprint": embedding_fp, "dataset": dataset,
        "population_filter": {"min_user_interactions": int(min_user_interactions)},
        "split": "temporal_leave_one_out_v1", "modality": modality,
        "query_construction": ("mean_training_history_v1" if modality == "u2i"
                               else "last_training_item_v1"),
        "weighting": weighting, "dim": int(dim),
        "evaluation_queries": str(max_queries), "topk": int(topk),
        "metric_topk": int(metric_topk), "seed": int(seed),
    })


def calibration_fingerprint(*, query_fp, targets, n_queries, param_grid,
                            topk, seed):
    return fingerprint("calibration", {
        "query_fingerprint": query_fp,
        "targets": [float(v) for v in targets], "queries": int(n_queries),
        "param_grid": param_grid, "topk": int(topk), "seed": int(seed),
    })


def scale_stress_fingerprint(*, catalog_sizes, dimensions, methods,
                             calibration_target, calibration_queries,
                             timed_queries, seed, index_config,
                             n_clusters=None, cluster_std=None, topk=None,
                             budget_mb=None, omp_threads=None):
    return fingerprint("scale_stress", {
        "catalog_sizes": [int(v) for v in catalog_sizes],
        "dimensions": [int(v) for v in dimensions], "methods": list(methods),
        "calibration_target": float(calibration_target),
        "calibration_queries": int(calibration_queries),
        "timed_queries": int(timed_queries), "seed": int(seed),
        "index_config": index_config, "quality_measured": False,
        "n_clusters": n_clusters, "cluster_std": cluster_std,
        "topk": topk, "budget_mb": budget_mb, "omp_threads": omp_threads,
    })
