"""Measure single-query latency and memory for a built index.

Deterministic: the warmup and timed query streams are drawn with
numpy.random.default_rng(--seed), so every method sees the same queries.
Latency numbers remain machine-dependent (see docs/hardware_protocol.md).

IndexWise-Recsys is evaluated as a CPU-only framework. GPU-specific
acceleration is outside the present scope.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil  # kept for env parity; rss_mb comes from utils.common

from utils.ann_io import load_ann_index
from utils.common import percentiles, rss_mb, set_global_seed

SCRIPT = "run_device"


def main():
    print(f"[{SCRIPT}] starting...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--item_vecs", required=True)
    ap.add_argument("--query_vecs", default=None,
                    help="optional .npy/.npz modality query pool")
    ap.add_argument("--modality", choices=["u2i", "i2i"], default=None)
    ap.add_argument("--query_fingerprint", default=None)
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--queries", type=int, default=10000)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--ef", type=int, default=64)       # for HNSW
    ap.add_argument("--nprobe", type=int, default=16)   # for IVF
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    index_arg = Path(args.index)
    print(f"[{SCRIPT}] input path: {index_arg}")
    print(f"[{SCRIPT}] input path: {args.item_vecs}")

    rng = set_global_seed(args.seed)

    item_vecs = np.load(args.item_vecs).astype("float32")
    N, D = item_vecs.shape
    query_pool = item_vecs
    query_source = "sampled_item_vectors"
    if args.query_vecs:
        loaded = np.load(args.query_vecs)
        query_pool = (loaded["query_vectors"]
                      if isinstance(loaded, np.lib.npyio.NpzFile) else loaded)
        query_pool = np.ascontiguousarray(query_pool, dtype=np.float32)
        if query_pool.ndim != 2 or query_pool.shape[1] != D:
            raise ValueError(f"query vector shape {query_pool.shape} incompatible with d{D}")
        query_source = "modality_query_cache"

    ann = load_ann_index(str(index_arg), D, N, ef=args.ef, nprobe=args.nprobe)
    method = ann.method

    # deterministic warmup + timed query streams
    n_pool = len(query_pool)
    if n_pool == 0:
        raise ValueError("query vector pool is empty")
    warmup_idx = rng.integers(0, n_pool, size=200)
    timed_idx = rng.integers(0, n_pool, size=args.queries)

    rss_before = rss_mb()
    for qi in warmup_idx:
        ann.search(query_pool[int(qi)].reshape(1, -1), args.topk)

    lat_ms = []
    for qi in timed_idx:
        q = query_pool[int(qi)].reshape(1, -1)
        t0 = time.perf_counter()
        ann.search(q, args.topk)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    rss_after = rss_mb()

    stats = {
        "method": method,
        "N": int(N),
        "D": int(D),
        "queries": int(args.queries),
        "topk": int(args.topk),
        "ef": int(args.ef),
        "nprobe": int(args.nprobe),
        "seed": int(args.seed),
        "modality": args.modality,
        "query_source": query_source,
        "query_fingerprint": args.query_fingerprint,
        "latency_ms": {**percentiles(lat_ms), "mean": float(np.mean(lat_ms))},
        "rss_mb_delta": float(rss_after - rss_before),
        "rss_mb_after": float(rss_after),
    }

    out_dir = index_arg if index_arg.is_dir() else index_arg.parent
    out = (Path(args.out) if args.out else out_dir /
           f"latency_{args.modality + '_' if args.modality else ''}{method}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[{SCRIPT}] output path: {out}")

    print(json.dumps(stats, indent=2))
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
