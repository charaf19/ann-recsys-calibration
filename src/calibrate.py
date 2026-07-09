"""Calibrate an ANN index against the exact Flat baseline.

Sweeps the method's runtime parameter (ef for HNSW, nprobe for IVF-*) over an
ascending grid and reports the smallest value whose recall@k against exact
search meets the requested target. Latency (mean/p50/p95) is measured at each
grid point on the same query sample. Methods without a tunable parameter
(flat, flatpq) are measured as-is.

Recall here is *ANN agreement recall* (overlap with exact top-k), the standard
calibration criterion — not end-to-end user relevance.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from utils.ann_io import (load_ann_index, build_exact_index,
                          CALIBRATION_PARAM, DEFAULT_PARAM_GRIDS)
from utils.common import set_global_seed

SCRIPT = "calibrate"


def sample_queries(item_vecs, n_queries, seed):
    rng = np.random.default_rng(seed)
    N = item_vecs.shape[0]
    idx = rng.choice(N, size=min(n_queries, N), replace=False)
    return item_vecs[idx].astype("float32")


def agreement_recall(ann, exact_I, Q, topk):
    I = ann.search(Q, topk)
    hits = 0
    for r in range(Q.shape[0]):
        hits += len(set(int(x) for x in I[r]) & set(int(x) for x in exact_I[r]))
    return hits / float(Q.shape[0] * topk)


def measure_latency(ann, Q, topk, timed_queries=500):
    n = min(timed_queries, Q.shape[0])
    # warmup
    ann.search(Q[: min(50, n)], topk)
    lat = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t0 = time.perf_counter()
        ann.search(Q[i:i + 1], topk)
        lat[i] = (time.perf_counter() - t0) * 1000.0
    return {"mean": float(lat.mean()),
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95))}


def calibrate_index(ann, item_vecs, target, topk, n_queries, seed,
                    param_grid=None, timed_queries=500):
    """Core calibration routine (also used by run_calibration_sensitivity).

    Returns a result dict; 'calibrated_param_value' is None for untunable
    methods and for grids that never reach the target (then the best point is
    reported with target_reached=False).
    """
    Q = sample_queries(item_vecs, n_queries, seed)
    exact = build_exact_index(item_vecs)
    exact_I = exact.search(Q, topk)

    param_name = CALIBRATION_PARAM.get(ann.method)
    if param_name is not None and getattr(ann, "gpu_used", False):
        # GPU-cloned indexes have their runtime parameter frozen (applied on
        # CPU before cloning); sweeping is unsupported, so measure as-is.
        print(f"[{SCRIPT}] WARN: GPU index parameters are frozen; measuring "
              f"at the pre-set {param_name} instead of sweeping.")
        param_name = None
    sweep = []

    if param_name is None:
        rec = agreement_recall(ann, exact_I, Q, topk)
        lat = measure_latency(ann, Q, topk, timed_queries)
        sweep.append({"param_value": None, "recall_vs_exact": rec, "latency_ms": lat})
        chosen = sweep[0]
        reached = rec >= target
    else:
        grid = param_grid or DEFAULT_PARAM_GRIDS[ann.method]
        chosen, reached = None, False
        for v in grid:
            ann.set_calibration_param(v)
            rec = agreement_recall(ann, exact_I, Q, topk)
            lat = measure_latency(ann, Q, topk, timed_queries)
            point = {"param_value": int(v), "recall_vs_exact": rec, "latency_ms": lat}
            sweep.append(point)
            print(f"[{SCRIPT}]   {param_name}={v}: recall@{topk} vs exact = {rec:.4f} "
                  f"(p95 {lat['p95']:.3f} ms)")
            if rec >= target:
                chosen, reached = point, True
                break
        if chosen is None:  # target never reached; report best point
            chosen = max(sweep, key=lambda s: s["recall_vs_exact"])

    return {
        "method": ann.method,
        "param_name": param_name,
        "target_recall": float(target),
        "target_reached": bool(reached),
        "calibrated_param_value": chosen["param_value"],
        "achieved_recall_vs_exact": float(chosen["recall_vs_exact"]),
        "latency_ms_at_calibrated": chosen["latency_ms"],
        "topk": int(topk),
        "n_calibration_queries": int(Q.shape[0]),
        "seed": int(seed),
        "sweep": sweep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item_vecs", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--target", type=float, default=0.95,
                    help="target recall@topk vs exact Flat search")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--queries", type=int, default=1000,
                    help="number of calibration query vectors (sampled items)")
    ap.add_argument("--timed_queries", type=int, default=500)
    ap.add_argument("--param_grid", type=int, nargs="*", default=None,
                    help="override the ascending parameter grid")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default=None, help="dataset tag for the output record")
    ap.add_argument("--weighting", default="none", help="weighting tag for the output record")
    ap.add_argument("--use_gpu", default="false",
                    help="OPTIONAL exploratory GPU search (true/false; default "
                         "false; outputs go to results/gpu_experiments/)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: results/main/calibration/"
                         "{dataset}__{weighting}__{method}__t{target}.json; "
                         "results/gpu_experiments/calibration/ when --use_gpu)")
    args = ap.parse_args()
    use_gpu = str(args.use_gpu).strip().lower() in ("1", "true", "yes", "y")

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.item_vecs}")
    print(f"[{SCRIPT}] input path: {args.index}")
    if use_gpu:
        print(f"[{SCRIPT}] WARN: --use_gpu is an exploratory extension; the "
              f"canonical reproducible benchmark is CPU-only.")

    set_global_seed(args.seed)

    item_vecs = np.load(args.item_vecs).astype("float32")
    N, D = item_vecs.shape
    ann = load_ann_index(args.index, D, N, use_gpu=use_gpu)

    result = calibrate_index(ann, item_vecs, args.target, args.topk,
                             args.queries, args.seed,
                             param_grid=args.param_grid,
                             timed_queries=args.timed_queries)
    dataset = args.dataset or Path(args.index).name
    result.update({"dataset": dataset, "weighting": args.weighting,
                   "index_path": str(args.index), "N": int(N), "D": int(D),
                   "gpu_used": bool(getattr(ann, "gpu_used", False))})

    if args.out:
        out_path = Path(args.out)
    elif getattr(ann, "gpu_used", False):
        out_path = Path("results/gpu_experiments/calibration") / (
            f"{dataset}__{args.weighting}__{ann.method}__t{args.target:.2f}.json")
    else:
        out_path = Path("results/main/calibration") / (
            f"{dataset}__{args.weighting}__{ann.method}__t{args.target:.2f}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[{SCRIPT}] output path: {out_path}")
    print(json.dumps({k: v for k, v in result.items() if k != "sweep"}, indent=2))
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
