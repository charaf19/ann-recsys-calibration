"""Run energy measurement over the built indexes.

For each dataset x method, executes a fixed seeded query workload against the
index and records energy (Intel RAPL when available), wall time, CPU
utilization, and RSS. On platforms without a direct energy counter (e.g.
Windows), rows carry direct_energy_available=false and NA energy fields —
no energy values are ever estimated or fabricated.

Query streams per modality:
    i2i  sampled catalog item vectors
    u2i  means of small seeded item-vector groups (history proxy; noted)

Outputs:
    results/energy/energy_measurement_all.csv
    results/paper_tables/energy_measurement_summary.csv/.tex (+ .md)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import energy_measurement as EM
from utils.ann_io import load_ann_index
from utils.common import set_global_seed, normalize_modality_label
from utils.paths import emb_dir, index_dir, RESULTS
from utils.reporting import write_table

SCRIPT = "run_energy_measurement"


def build_queries(item_vecs, modality, n_queries, seed):
    rng = np.random.default_rng(seed)
    N = item_vecs.shape[0]
    if modality == "i2i":
        idx = rng.choice(N, size=min(n_queries, N), replace=True)
        return item_vecs[idx].copy()
    # u2i proxy: mean of 5 seeded items per query (documented in notes)
    Q = np.zeros((n_queries, item_vecs.shape[1]), dtype=np.float32)
    for r in range(n_queries):
        idx = rng.choice(N, size=min(5, N), replace=False)
        Q[r] = item_vecs[idx].mean(axis=0)
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["ml-1m", "ml-20m", "goodbooks"])
    ap.add_argument("--methods", nargs="*",
                    default=["flat", "hnsw", "ivfflat", "ivfpq", "flatpq"])
    ap.add_argument("--modalities", nargs="*", default=["u2i", "i2i"])
    ap.add_argument("--weighting", default="bm25")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=5000)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--ef", type=int, default=128)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=RESULTS["energy"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: data/index_* (built indexes)")
    print(f"[{SCRIPT}] output path: {out_dir}")

    args.modalities = [normalize_modality_label(m) for m in args.modalities]
    set_global_seed(args.seed)
    if EM.rapl_available():
        print(f"[{SCRIPT}] Intel RAPL detected: direct CPU energy will be recorded.")
    else:
        print(f"[{SCRIPT}] No direct energy counter on this platform: rows will "
              f"carry direct_energy_available=false and NA energy fields.")

    rows = []
    for dataset in args.datasets:
        vec_path = Path(emb_dir(dataset, args.weighting, args.dim)) / "item_vecs.npy"
        if not vec_path.is_file():
            print(f"[{SCRIPT}] WARN: missing embeddings {vec_path}; skipping {dataset}.")
            continue
        item_vecs = np.load(vec_path).astype("float32")
        N, D = item_vecs.shape

        for method in args.methods:
            idx_dir = index_dir(dataset, args.weighting, args.dim, method)
            if not Path(idx_dir).exists():
                print(f"[{SCRIPT}] WARN: missing index {idx_dir}; skipping.")
                continue
            ann = load_ann_index(idx_dir, D, N, ef=args.ef, nprobe=args.nprobe)

            for modality in args.modalities:
                Q = build_queries(item_vecs, modality, args.queries, args.seed)
                ann.search(Q[:100], args.topk)  # warmup outside measurement

                def workload(_ann=ann, _Q=Q, _k=args.topk):
                    for r in range(_Q.shape[0]):
                        _ann.search(_Q[r:r + 1], _k)
                    return _Q.shape[0]

                print(f"[{SCRIPT}] measuring {dataset}/{modality}/{method} "
                      f"({args.queries} queries)")
                m = EM.measure(workload)
                n_q = m.pop("workload_result")
                rows.append({
                    "dataset": dataset,
                    "modality": modality,
                    "method": method,
                    "measurement_backend": m["measurement_backend"],
                    "direct_energy_available": m["direct_energy_available"],
                    "cpu_energy_joules": m["cpu_energy_joules"],
                    "wall_time_sec": m["wall_time_sec"],
                    "queries": int(n_q),
                    "energy_per_query_joules": EM.per_query_energy(m, n_q),
                    "cpu_utilization_mean": m["cpu_utilization_mean"],
                    "rss_mb": m["rss_mb"],
                    "notes": m["notes"] + (
                        " | u2i queries are seeded mean-of-5-items history proxies."
                        if modality == "u2i" else ""),
                })

    if not rows:
        print(f"[{SCRIPT}] WARN: nothing measured (are indexes built?).")
        print(f"[{SCRIPT}] completed.")
        return

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "energy_measurement_all.csv"
    if all_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {all_path}")
    df.to_csv(all_path, index=False)
    print(f"[{SCRIPT}] output path: {all_path}")

    written = write_table(df, Path(RESULTS["paper_tables"]) / "energy_measurement_summary")
    for p in written:
        print(f"[{SCRIPT}] output path: {p}")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
