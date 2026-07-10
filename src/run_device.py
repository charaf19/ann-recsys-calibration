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

    ann = load_ann_index(str(index_arg), D, N, ef=args.ef, nprobe=args.nprobe)
    method = ann.method

    # deterministic warmup + timed query streams
    warmup_idx = rng.integers(0, N, size=200)
    timed_idx = rng.integers(0, N, size=args.queries)

    rss_before = rss_mb()
    for qi in warmup_idx:
        ann.search(item_vecs[int(qi)].reshape(1, -1), args.topk)

    lat_ms = []
    for qi in timed_idx:
        q = item_vecs[int(qi)].reshape(1, -1)
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
        "latency_ms": {**percentiles(lat_ms), "mean": float(np.mean(lat_ms))},
        "rss_mb_delta": float(rss_after - rss_before),
        "rss_mb_after": float(rss_after),
    }

    out_dir = index_arg if index_arg.is_dir() else index_arg.parent
    out = out_dir / f"latency_{method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[{SCRIPT}] output path: {out}")

    print(json.dumps(stats, indent=2))
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
