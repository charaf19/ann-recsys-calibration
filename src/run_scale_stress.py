"""Production-scale synthetic stress test (cost only, no quality claims).

Reviewer concern addressed: "MovieLens-20M catalog is below production/web
scale". This script builds indexes over seeded synthetic vectors at catalog
sizes up to 1M items and dims up to 256, and measures build time, on-disk
index size, process RSS, and calibrated latency. It measures NO
recommendation quality — every output row has quality_measured=false so the
results cannot be misread as end-to-end quality at scale.

Complements src/synthetic_scaling.py (which runs the full interaction ->
embedding pipeline at moderate scale); here vectors are generated directly
so 1M-item catalogs are feasible without SVD training.

Outputs:
    results/scale_stress/scale_stress_all.csv
    results/paper_tables/scale_stress_summary.csv/.tex (+ .md)
    results/figures_paper/scale_stress_latency.pdf
    results/figures_paper/scale_stress_memory.pdf
    results/figures_paper/scale_stress_index_size.pdf
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from calibrate import calibrate_index
from utils.ann_io import load_ann_index
from utils.common import set_global_seed, rss_mb
from utils.paths import RESULTS
from utils.reporting import write_table
from utils.figures_ext import (fig_scale_stress_latency, fig_scale_stress_memory,
                               fig_scale_stress_index_size)

SCRIPT = "run_scale_stress"


def run(cmd, dry_run=False):
    full = [sys.executable] + cmd[1:] if cmd[0] == "python" else cmd
    print(">>", " ".join(str(c) for c in full))
    if not dry_run:
        subprocess.run(full, check=True)


def load_config(path):
    p = Path(path)
    if not p.is_file():
        print(f"[{SCRIPT}] WARN: config {path} not found; using defaults.")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def synth_vectors(n_items, dim, n_clusters, cluster_std, seed):
    """Seeded Gaussian-mixture vectors (clustered like real item embeddings)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    assign = rng.integers(0, n_clusters, size=n_items)
    X = centers[assign] + cluster_std * rng.standard_normal((n_items, dim)).astype(np.float32)
    return X.astype(np.float32)


def dir_size_mb(path):
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)


def build_index_cmd(method, emb_dir, idx_dir, budget_mb):
    cmd = ["python", "src/build_index.py", "--method", method,
           "--item_vecs", f"{emb_dir}/item_vecs.npy",
           "--item_ids", f"{emb_dir}/item_ids.npy",
           "--out_dir", idx_dir, "--budget_mb", str(budget_mb)]
    if method == "hnsw":
        cmd += ["--M", "24", "--efc", "200"]
    elif method == "ivfflat":
        cmd += ["--nlist", "auto"]
    elif method == "ivfpq":
        cmd += ["--nlist", "auto", "--m", "32", "--bits", "8"]
    elif method == "flatpq":
        cmd += ["--m", "32", "--bits", "8"]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scale_stress.yml")
    ap.add_argument("--catalog_sizes", type=int, nargs="*", default=None)
    ap.add_argument("--dims", type=int, nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_dir", default=RESULTS["scale_stress"])
    ap.add_argument("--dry_run", action="store_true",
                    help="print the plan without generating/building anything")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sizes = args.catalog_sizes or cfg.get("catalog_sizes",
                                          [10000, 50000, 100000, 500000, 1000000])
    dims = args.dims or cfg.get("dims", [64, 128, 256])
    methods = args.methods or cfg.get("methods",
                                      ["flat", "hnsw", "ivfflat", "ivfpq", "flatpq"])
    n_clusters = int(cfg.get("n_clusters", 64))
    cluster_std = float(cfg.get("cluster_std", 0.3))
    target = float(cfg.get("calibration_target", 0.95))
    topk = int(cfg.get("topk", 100))
    cal_queries = int(cfg.get("calibration_queries", 500))
    timed_queries = int(cfg.get("timed_queries", 300))
    budget_mb = int(cfg.get("budget_mb", 100))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))

    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.config}")
    print(f"[{SCRIPT}] output path: {out_dir}")
    print(f"[{SCRIPT}] sizes={sizes} dims={dims} methods={methods} seed={seed}")
    print(f"[{SCRIPT}] NOTE: cost-only stress test; quality_measured=false on "
          f"every row by design.")

    set_global_seed(seed)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n_items in sizes:
        for dim in dims:
            tag = f"stress_n{n_items}_d{dim}"
            emb = Path(f"data/{tag}")
            if args.dry_run:
                print(f"[{SCRIPT}] (dry run) would generate {n_items}x{dim} "
                      f"vectors -> {emb}, then build {methods}")
                continue
            emb.mkdir(parents=True, exist_ok=True)
            vec_path = emb / "item_vecs.npy"
            if not vec_path.is_file():
                print(f"[{SCRIPT}] generating {n_items}x{dim} synthetic vectors")
                X = synth_vectors(n_items, dim, n_clusters, cluster_std, seed)
                np.save(vec_path, X)
                np.save(emb / "item_ids.npy", np.arange(n_items))
            item_vecs = np.load(vec_path).astype("float32")

            for method in methods:
                idx_dir = f"data/index_{tag}_{method}"
                t0 = time.perf_counter()
                run(build_index_cmd(method, str(emb), idx_dir, budget_mb))
                build_secs = time.perf_counter() - t0

                rss_before = rss_mb()
                ann = load_ann_index(idx_dir, dim, n_items)
                cal = calibrate_index(ann, item_vecs, target, topk,
                                      n_queries=cal_queries, seed=seed,
                                      timed_queries=timed_queries)
                rss_after = rss_mb()

                rows.append({
                    "n_items": n_items, "dim": dim, "method": method,
                    "build_wall_time_sec": round(build_secs, 3),
                    "index_size_mb": round(dir_size_mb(idx_dir), 2),
                    "rss_mb_after": round(rss_after, 1),
                    "rss_mb_delta": round(rss_after - rss_before, 1),
                    "calibration_target": target,
                    "target_reached": cal["target_reached"],
                    "param_name": cal["param_name"],
                    "calibrated_param_value": cal["calibrated_param_value"],
                    "achieved_recall_vs_exact": cal["achieved_recall_vs_exact"],
                    "latency_p50_ms": cal["latency_ms_at_calibrated"]["p50"],
                    "latency_p95_ms": cal["latency_ms_at_calibrated"]["p95"],
                    "quality_measured": False,
                    "seed": seed,
                })
                # checkpoint after every cell (long-running grid)
                pd.DataFrame(rows).to_csv(out_dir / "scale_stress_all.csv", index=False)

    if args.dry_run or not rows:
        print(f"[{SCRIPT}] completed.")
        return

    df = pd.DataFrame(rows)
    all_path = out_dir / "scale_stress_all.csv"
    df.to_csv(all_path, index=False)
    print(f"[{SCRIPT}] output path: {all_path}")

    summary = df[["n_items", "dim", "method", "build_wall_time_sec",
                  "index_size_mb", "rss_mb_after", "latency_p50_ms",
                  "latency_p95_ms", "achieved_recall_vs_exact",
                  "quality_measured"]]
    written = write_table(summary, Path(RESULTS["paper_tables"]) / "scale_stress_summary")
    for p in written:
        print(f"[{SCRIPT}] output path: {p}")

    fig_dir = Path(RESULTS["figures_paper"])
    for p in (fig_scale_stress_latency(df, fig_dir)
              + fig_scale_stress_memory(df, fig_dir)
              + fig_scale_stress_index_size(df, fig_dir)):
        print(f"[{SCRIPT}] output path: {p}")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
