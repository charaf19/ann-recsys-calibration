"""Optional ANN backend comparison (FAISS-HNSW vs hnswlib vs ScaNN vs NGT).

Reviewer concern addressed: "no ScaNN/NGT comparison". ScaNN and NGT are NOT
required dependencies (Linux-centric wheels); when a package is missing the
comparison degrades gracefully and records a row with

    backend_available = false
    backend_error_message = package_not_installed

so the output CSV always documents exactly which backends were compared.
FAISS-only results remain the paper's primary scope.

Query/vector source: real embeddings via --item_vecs when available,
otherwise seeded synthetic vectors (recorded in the vectors_source column).

Outputs:
    results/optional_backends/optional_ann_backend_comparison.csv
    results/paper_tables/optional_ann_backend_comparison.tex (+ .csv/.md)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backends import ALL_BACKENDS
from utils.ann_io import build_exact_index
from utils.common import set_global_seed
from utils.paths import RESULTS
from utils.reporting import write_table

SCRIPT = "run_optional_ann_backend_comparison"


def measure_backend(backend, vectors, Q, exact_I10, topk, timed_queries, seed):
    t0 = time.perf_counter()
    handle = backend["build"](vectors, seed=seed)
    build_secs = time.perf_counter() - t0

    I = handle.search(Q, 10)
    overlap = np.zeros(Q.shape[0], dtype=np.float64)
    for r in range(Q.shape[0]):
        overlap[r] = len(set(int(x) for x in I[r] if int(x) >= 0)
                         & set(int(x) for x in exact_I10[r])) / 10.0

    n = min(timed_queries, Q.shape[0])
    handle.search(Q[: min(50, n)], topk)  # warmup
    lat = np.zeros(n, dtype=np.float64)
    for r in range(n):
        t0 = time.perf_counter()
        handle.search(Q[r:r + 1], topk)
        lat[r] = (time.perf_counter() - t0) * 1000.0

    return {
        "build_time_sec": round(build_secs, 3),
        "recall_vs_exact_at_10": float(overlap.mean()),
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item_vecs", default=None,
                    help="npy of real item vectors; synthetic fallback if omitted")
    ap.add_argument("--dataset_label", default=None)
    ap.add_argument("--synthetic_items", type=int, default=100000)
    ap.add_argument("--synthetic_dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--timed_queries", type=int, default=500)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=RESULTS["optional_backends"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.item_vecs or '(synthetic vectors)'}")
    print(f"[{SCRIPT}] output path: {out_dir}")

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.item_vecs and Path(args.item_vecs).is_file():
        vectors = np.load(args.item_vecs).astype("float32")
        source = args.dataset_label or Path(args.item_vecs).parent.name
    else:
        if args.item_vecs:
            print(f"[{SCRIPT}] WARN: {args.item_vecs} not found; using synthetic vectors.")
        vectors = rng.standard_normal(
            (args.synthetic_items, args.synthetic_dim)).astype("float32")
        source = f"synthetic_n{args.synthetic_items}_d{args.synthetic_dim}"
    N, D = vectors.shape
    q_idx = rng.choice(N, size=min(args.queries, N), replace=False)
    Q = vectors[q_idx]

    exact = build_exact_index(vectors)
    exact_I10 = exact.search(Q, 10)

    rows = []
    for module in ALL_BACKENDS:
        backend = module.get_backend()
        base = {"backend": backend["name"],
                "backend_available": backend["available"],
                "backend_error_message": backend["error_message"],
                "vectors_source": source, "n_items": N, "dim": D,
                "topk": args.topk, "seed": args.seed}
        if not backend["available"]:
            print(f"[{SCRIPT}] backend {backend['name']}: NOT available "
                  f"({backend['error_message']}); recording placeholder row.")
            rows.append(base)
            continue
        print(f"[{SCRIPT}] measuring backend {backend['name']}...")
        try:
            rows.append({**base, **measure_backend(backend, vectors, Q, exact_I10,
                                                   args.topk, args.timed_queries,
                                                   args.seed)})
        except Exception as e:  # never let one backend kill the comparison
            print(f"[{SCRIPT}] WARN: backend {backend['name']} failed at "
                  f"runtime: {e}")
            base["backend_available"] = False
            base["backend_error_message"] = f"runtime_error: {e}"
            rows.append(base)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "optional_ann_backend_comparison.csv"
    if csv_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {csv_path}")
    df.to_csv(csv_path, index=False)
    print(f"[{SCRIPT}] output path: {csv_path}")

    written = write_table(df, Path(RESULTS["paper_tables"])
                          / "optional_ann_backend_comparison")
    for p in written:
        print(f"[{SCRIPT}] output path: {p}")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
