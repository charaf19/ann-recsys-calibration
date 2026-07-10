"""Optional GPU experiment runner (exploratory; NOT the canonical benchmark).

Reuses CPU-trained embeddings and CPU-built FAISS indexes, clones supported
index types (flat, ivfflat, ivfpq) to a selected GPU, and measures:

  - GPU single-query latency (p50/p95/mean),
  - agreement recall @100 of GPU retrieval vs exact CPU Flat search
    (`agreement_recall_vs_cpu_flat_at_100`),
  - agreement recall @100 of GPU retrieval vs the CPU version of the same
    index method (`agreement_recall_vs_cpu_method_at_100`),
  - speedup vs the CPU version of the same method at identical runtime
    parameters (`cpu_reference_method`).

These are index-fidelity and cost measurements, NOT recommendation quality.
GPU results may differ slightly from CPU due to hardware, the FAISS GPU
implementation, and floating-point reduction order.

Safety contract:
  - never touches results/main/ or any other CPU result directory; all
    outputs go under results/gpu_experiments/;
  - when FAISS GPU is unavailable: with --allow_cpu_fallback true (default)
    every planned combination is recorded as a skipped row
    (status=skipped_gpu_unavailable, NA measurements — never fake zeros);
    with false, the script exits with a clear error;
  - HNSW / Flat-PQ are not supported by FAISS GPU and are recorded as
    skipped_gpu_unsupported_method.

Outputs:
    results/gpu_experiments/gpu_experiment_summary.csv
    results/gpu_experiments/latency/{dataset}_{modality}_{method}_gpu.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from calibrate import measure_latency
from utils.ann_io import (load_ann_index, build_exact_index,
                          GPU_UNSUPPORTED_METHODS)
from utils.common import set_global_seed, normalize_modality_label, rss_mb
from utils.paths import emb_dir, index_dir, RESULTS

SCRIPT = "run_gpu_experiments"
NA = "NA"
OUT_ROOT = Path(RESULTS["gpu_experiments"])

SUMMARY_COLUMNS = [
    "dataset", "modality", "method", "weighting", "dim", "queries", "topk",
    "metric_topk", "gpu_available", "gpu_used", "gpu_device",
    "faiss_gpu_available", "cpu_reference_method",
    "agreement_recall_vs_cpu_flat_at_100",
    "agreement_recall_vs_cpu_method_at_100",
    "latency_p50_ms_gpu", "latency_p95_ms_gpu", "latency_mean_ms_gpu",
    "latency_p50_ms_cpu_reference", "latency_p95_ms_cpu_reference",
    "speedup_vs_cpu_reference_p50", "speedup_vs_cpu_reference_p95",
    "rss_mb_after", "gpu_memory_allocated_mb", "gpu_memory_reserved_mb",
    "status", "error_message", "seed",
]


def _str2bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def faiss_gpu_count():
    try:
        import faiss
        return int(getattr(faiss, "get_num_gpus", lambda: 0)())
    except Exception:
        return 0


def gpu_memory_mb(device):
    """torch.cuda allocated/reserved MB for the device, or (NA, NA)."""
    try:
        import torch
        if torch.cuda.is_available():
            d = int(device)
            return (round(torch.cuda.memory_allocated(d) / (1024 ** 2), 2),
                    round(torch.cuda.memory_reserved(d) / (1024 ** 2), 2))
    except Exception:
        pass
    return NA, NA


def load_config(path):
    p = Path(path)
    if not p.is_file():
        print(f"[{SCRIPT}] WARN: config {path} not found; using built-in defaults.")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    return doc.get("gpu_experiments", doc)


def build_queries(item_vecs, modality, n_queries, seed):
    """Deterministic query streams. i2i: sampled item vectors. u2i: seeded
    mean-of-5-items history proxies (same construction as the energy runner;
    documented — these are agreement/latency streams, not user evaluations)."""
    rng = np.random.default_rng(seed)
    N = item_vecs.shape[0]
    if modality == "i2i":
        idx = rng.choice(N, size=min(n_queries, N), replace=True)
        return item_vecs[idx].copy()
    Q = np.zeros((n_queries, item_vecs.shape[1]), dtype=np.float32)
    for r in range(n_queries):
        idx = rng.choice(N, size=min(5, N), replace=False)
        Q[r] = item_vecs[idx].mean(axis=0)
    return Q


def agreement_at(I_a, I_b, k):
    """Mean per-query overlap fraction between two id matrices at depth k."""
    ov = np.zeros(I_a.shape[0], dtype=np.float64)
    for r in range(I_a.shape[0]):
        a = set(int(x) for x in I_a[r][:k] if int(x) >= 0)
        b = set(int(x) for x in I_b[r][:k] if int(x) >= 0)
        ov[r] = len(a & b) / float(k)
    return float(ov.mean())


def skipped_row(base, status, error):
    row = {c: NA for c in SUMMARY_COLUMNS}
    row.update(base)
    row.update({"gpu_used": False, "status": status, "error_message": error})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu_experiments.yml")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--modalities", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--weighting", default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--queries", type=int, default=None)
    ap.add_argument("--timed_queries", type=int, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--metric_topk", type=int, default=None)
    ap.add_argument("--nprobe", type=int, default=None)
    ap.add_argument("--ef", type=int, default=None)
    ap.add_argument("--gpu_device", type=int, default=None)
    ap.add_argument("--allow_cpu_fallback", default=None,
                    help="true/false (default true): when FAISS GPU is "
                         "unavailable, record skipped rows instead of failing")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true",
                    help="print the experiment plan without loading anything")
    args = ap.parse_args()

    cfg = load_config(args.config)

    def opt(name, default):
        v = getattr(args, name, None)
        if v is not None:
            return v
        return cfg.get(name, default)

    datasets = opt("datasets", ["ml-1m"])
    modalities = [normalize_modality_label(m) for m in opt("modalities", ["u2i", "i2i"])]
    methods = opt("methods", ["flat", "ivfflat", "ivfpq"])
    weighting = opt("weighting", "bm25")
    dim = int(opt("dim", 128))
    n_queries = int(opt("queries", 5000))
    timed_queries = int(opt("timed_queries", 1000))
    topk = int(opt("topk", 100))
    metric_topk = int(opt("metric_topk", 10))
    nprobe = int(opt("nprobe", 16))
    ef = int(opt("ef", 128))
    seed = int(opt("seed", 42))
    gpu_device = args.gpu_device
    if gpu_device is None:
        ids = cfg.get("device_ids", [0])
        gpu_device = int(ids[0]) if ids else 0
    fallback_raw = args.allow_cpu_fallback
    if fallback_raw is None:
        fallback_raw = cfg.get("allow_cpu_fallback", True)
    allow_cpu_fallback = _str2bool(fallback_raw)

    summary_path = OUT_ROOT / "gpu_experiment_summary.csv"
    latency_dir = OUT_ROOT / "latency"

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.config}")
    print(f"[{SCRIPT}] output path: {summary_path}")
    print(f"[{SCRIPT}] NOTE: exploratory GPU layer; the canonical reproducible "
          f"benchmark is CPU-only. No CPU result file is read-modified or "
          f"overwritten.")
    print(f"[{SCRIPT}] plan: datasets={datasets} modalities={modalities} "
          f"methods={methods} weighting={weighting} dim={dim} "
          f"queries={n_queries} gpu_device={gpu_device} "
          f"allow_cpu_fallback={allow_cpu_fallback} seed={seed}")

    if args.dry_run:
        for dataset in datasets:
            for modality in modalities:
                for method in methods:
                    note = (" [unsupported by FAISS GPU -> skipped row]"
                            if method in GPU_UNSUPPORTED_METHODS else "")
                    print(f"[{SCRIPT}] (dry run) {dataset}/{modality}/{method}"
                          f" on GPU {gpu_device}{note}")
        print(f"[{SCRIPT}] completed.")
        return

    set_global_seed(seed)
    n_gpus = faiss_gpu_count()
    gpu_available = n_gpus > 0
    if not gpu_available and not allow_cpu_fallback:
        print(f"[{SCRIPT}] ERROR: FAISS GPU unavailable (faiss.get_num_gpus()=0 "
              f"or faiss-cpu build) and --allow_cpu_fallback false. Install a "
              f"faiss-gpu build in a separate environment "
              f"(docs/gpu_experiment_protocol.md) or re-run with fallback.")
        sys.exit(1)
    if not gpu_available:
        print(f"[{SCRIPT}] WARN: FAISS GPU unavailable; recording skipped rows "
              f"for every planned combination (no fake measurements).")

    rows = []
    for dataset in datasets:
        base_common = {"dataset": dataset, "weighting": weighting, "dim": dim,
                       "queries": n_queries, "topk": topk,
                       "metric_topk": metric_topk,
                       "gpu_available": gpu_available,
                       "gpu_device": gpu_device,
                       "faiss_gpu_available": gpu_available,
                       "seed": seed}
        vec_path = Path(emb_dir(dataset, weighting, dim)) / "item_vecs.npy"
        if not vec_path.is_file():
            print(f"[{SCRIPT}] WARN: missing CPU embeddings {vec_path}; run the "
                  f"CPU pipeline first. Skipping {dataset}.")
            for modality in modalities:
                for method in methods:
                    rows.append(skipped_row(
                        {**base_common, "modality": modality, "method": method,
                         "cpu_reference_method": method},
                        "skipped_missing_cpu_artifacts",
                        f"missing_embeddings:{vec_path}"))
            continue

        item_vecs = None
        exact = None
        if gpu_available:
            item_vecs = np.load(vec_path).astype("float32")
            exact = build_exact_index(item_vecs)

        for modality in modalities:
            Q = (build_queries(item_vecs, modality, n_queries, seed)
                 if gpu_available else None)
            for method in methods:
                base = {**base_common, "modality": modality, "method": method,
                        "cpu_reference_method": method}
                idx_dir = index_dir(dataset, weighting, dim, method)

                if not gpu_available:
                    rows.append(skipped_row(base, "skipped_gpu_unavailable",
                                            "faiss_gpu_unavailable"))
                    continue
                if method in GPU_UNSUPPORTED_METHODS:
                    print(f"[{SCRIPT}] {dataset}/{modality}/{method}: FAISS GPU "
                          f"does not support this index type; skipped row.")
                    rows.append(skipped_row(base, "skipped_gpu_unsupported_method",
                                            "faiss_gpu_does_not_support_method"))
                    continue
                if not Path(idx_dir).exists():
                    print(f"[{SCRIPT}] WARN: missing CPU index {idx_dir}; skipped.")
                    rows.append(skipped_row(base, "skipped_missing_cpu_artifacts",
                                            f"missing_index:{idx_dir}"))
                    continue

                N, D = item_vecs.shape
                print(f"[{SCRIPT}] measuring {dataset}/{modality}/{method} "
                      f"on GPU {gpu_device}")
                try:
                    cpu_ann = load_ann_index(idx_dir, D, N, ef=ef, nprobe=nprobe)
                    gpu_ann = load_ann_index(idx_dir, D, N, ef=ef, nprobe=nprobe,
                                             use_gpu=True, gpu_device=gpu_device)
                    if not getattr(gpu_ann, "gpu_used", False):
                        rows.append(skipped_row(base, "skipped_gpu_unavailable",
                                                "gpu_clone_failed"))
                        continue

                    k100 = min(100, topk, N)
                    depth = min(max(topk, 100), N)
                    I_gpu = gpu_ann.search(Q, depth)
                    I_cpu = cpu_ann.search(Q, depth)
                    I_flat = exact.search(Q, depth)
                    agree_flat = agreement_at(I_gpu, I_flat, k100)
                    agree_method = agreement_at(I_gpu, I_cpu, k100)

                    lat_gpu = measure_latency(gpu_ann, Q, topk,
                                              timed_queries=timed_queries)
                    lat_cpu = measure_latency(cpu_ann, Q, topk,
                                              timed_queries=timed_queries)
                    alloc_mb, reserved_mb = gpu_memory_mb(gpu_device)

                    row = {
                        **base,
                        "gpu_used": True,
                        "agreement_recall_vs_cpu_flat_at_100": agree_flat,
                        "agreement_recall_vs_cpu_method_at_100": agree_method,
                        "latency_p50_ms_gpu": lat_gpu["p50"],
                        "latency_p95_ms_gpu": lat_gpu["p95"],
                        "latency_mean_ms_gpu": lat_gpu["mean"],
                        "latency_p50_ms_cpu_reference": lat_cpu["p50"],
                        "latency_p95_ms_cpu_reference": lat_cpu["p95"],
                        "speedup_vs_cpu_reference_p50":
                            (lat_cpu["p50"] / lat_gpu["p50"]
                             if lat_gpu["p50"] > 0 else NA),
                        "speedup_vs_cpu_reference_p95":
                            (lat_cpu["p95"] / lat_gpu["p95"]
                             if lat_gpu["p95"] > 0 else NA),
                        "rss_mb_after": round(rss_mb(), 1),
                        "gpu_memory_allocated_mb": alloc_mb,
                        "gpu_memory_reserved_mb": reserved_mb,
                        "status": "ok",
                        "error_message": "",
                    }
                    rows.append(row)

                    latency_dir.mkdir(parents=True, exist_ok=True)
                    json_path = latency_dir / f"{dataset}_{modality}_{method}_gpu.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(row, f, indent=2)
                    print(f"[{SCRIPT}] output path: {json_path}")
                except Exception as e:
                    print(f"[{SCRIPT}] ERROR: {dataset}/{modality}/{method} "
                          f"failed on GPU path: {e}")
                    rows.append(skipped_row(base, "failed", f"runtime_error: {e}"))

    df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)
    print(f"[{SCRIPT}] output path: {summary_path} ({len(df)} rows)")
    print(f"[{SCRIPT}] NOTE: agreement columns are index-fidelity measures vs "
          f"CPU search, not recommendation quality.")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
