"""Production-scale synthetic stress test (cost only, no quality claims).

Reviewer concern addressed: "MovieLens-20M catalog is below production/web
scale". This script builds indexes over seeded synthetic vectors at catalog
sizes up to 1M items and dims up to 256, and measures build time, on-disk
index size, process RSS, and calibrated latency. It measures NO
recommendation quality — every output row has quality_measured=false so the
results cannot be misread as end-to-end quality at scale.

Synthetic vectors are generated internally (seeded Gaussian mixture), so
1M-item catalogs are feasible without SVD training.

Configuration: configs/analyses.yml, section scale_stress.

Output: results/analyses/scale_stress/scale_stress_all.csv
(presentation tables/figures come from tables_paper.py / figures_paper.py)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate import calibrate_index
from utils.ann_io import load_ann_index
from utils.common import set_global_seed, rss_mb
from utils.config import load_config, cfg_get, ConfigError
from utils.paths import RESULTS
from utils.result_io import (preflight_output, write_dataframe_atomic,
                             ResultExistsError)

SCRIPT = "run_scale_stress"
DEFAULT_CONFIG = "configs/analyses.yml"
KEY = ["n_items", "dim", "method", "seed"]


def run(cmd):
    full = [sys.executable] + cmd[1:] if cmd[0] == "python" else cmd
    print(">>", " ".join(str(c) for c in full))
    subprocess.run(full, check=True)


def synth_vectors(n_items, dim, n_clusters, cluster_std, seed):
    """Seeded Gaussian-mixture vectors (clustered like real item embeddings)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    assign = rng.integers(0, n_clusters, size=n_items)
    X = centers[assign] + cluster_std * rng.standard_normal(
        (n_items, dim)).astype(np.float32)
    return X.astype(np.float32)


def dir_size_mb(path):
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)


def build_index_cmd(method, emb_dir, idx_dir, budget_mb, seed, omp_threads,
                    index_cfg):
    """Index build invocation from the RESOLVED index configuration."""
    cmd = ["python", "src/build_index.py", "--method", method,
           "--item_vecs", f"{emb_dir}/item_vecs.npy",
           "--item_ids", f"{emb_dir}/item_ids.npy",
           "--out_dir", idx_dir, "--budget_mb", str(budget_mb),
           "--seed", str(seed), "--omp_threads", str(omp_threads)]
    hnsw = index_cfg.get("hnsw", {})
    ivf = index_cfg.get("ivf", {})
    pq = index_cfg.get("pq", {})
    ivfpq = index_cfg.get("ivfpq", {})
    if method == "hnsw":
        cmd += ["--M", str(hnsw.get("M", 24)),
                "--efc", str(hnsw.get("ef_construction", 200))]
    elif method == "ivfflat":
        cmd += ["--nlist", str(ivf.get("nlist", "auto"))]
    elif method == "ivfpq":
        cmd += ["--nlist", str(ivf.get("nlist", "auto")),
                "--m", str(pq.get("m", 32)), "--bits", str(pq.get("bits", 8))]
        if ivfpq.get("use_opq", True):
            cmd += ["--opq"]
    elif method == "flatpq":
        cmd += ["--m", str(pq.get("m", 32)), "--bits", str(pq.get("bits", 8))]
    return cmd


def main():
    ap = argparse.ArgumentParser(
        description="Cost-only synthetic stress test: build time, index "
                    "size, RSS, and calibrated latency across catalog sizes "
                    "and dims. Measures NO recommendation quality.")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="analyses YAML (section scale_stress)")
    ap.add_argument("--catalog_sizes", type=int, nargs="*", default=None)
    ap.add_argument("--dimensions", type=int, nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_dir", default=RESULTS["scale_stress"])
    ap.add_argument("--write_mode", default="fail_if_exists",
                    choices=["fail_if_exists", "replace", "merge"])
    ap.add_argument("--measure_quality", default="false",
                    help="end-to-end recommendation quality at synthetic "
                         "scale (true/false; deliberately NOT implemented)")
    ap.add_argument("--dry_run", action="store_true",
                    help="print the plan without generating/building anything")
    args = ap.parse_args()

    if str(args.measure_quality).strip().lower() in ("1", "true", "yes", "y"):
        raise NotImplementedError(
            "--measure_quality true: end-to-end recommendation quality at "
            "synthetic scale is deliberately not implemented (synthetic "
            "vectors carry no user relevance signal). The stress test "
            "reports index fidelity via achieved_recall_vs_exact; for "
            "recommendation quality use the real-dataset pipeline "
            "(run_revision_experiments.py).")

    try:
        cfg = load_config(args.config, cli_overrides={
            "scale_stress.catalog_sizes": args.catalog_sizes,
            "scale_stress.dimensions": args.dimensions,
            "scale_stress.methods": args.methods,
            "reproducibility.seed": args.seed,
        })
    except ConfigError as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    ss = cfg_get(cfg, "scale_stress", default={})
    sizes = list(ss.get("catalog_sizes",
                        [10000, 50000, 100000, 500000, 1000000]))
    dims = list(ss.get("dimensions", [64, 128, 256]))
    methods = list(ss.get("methods",
                          ["flat", "hnsw", "ivfflat", "ivfpq", "flatpq"]))
    n_clusters = int(ss.get("n_clusters", 64))
    cluster_std = float(ss.get("cluster_std", 0.3))
    target = float(ss.get("calibration_target", 0.95))
    cal_queries = int(ss.get("calibration_queries", 500))
    timed_queries = int(ss.get("timed_queries", 300))
    topk = cfg_get(cfg, "retrieval.topk", type=int, default=100)
    budget_mb = cfg_get(cfg, "retrieval.budget_mb", type=int, default=100)
    omp_threads = cfg_get(cfg, "reproducibility.omp_threads", type=int,
                          default=1)
    seed = cfg_get(cfg, "reproducibility.seed", type=int, default=42)
    index_cfg = cfg_get(cfg, "index", default={})

    out_dir = Path(args.out_dir)
    all_path = out_dir / "scale_stress_all.csv"
    if not args.dry_run:
        try:
            preflight_output(all_path, args.write_mode)
        except (ResultExistsError, ValueError) as e:
            print(f"[{SCRIPT}] ERROR: {e}")
            sys.exit(1)

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.config}")
    print(f"[{SCRIPT}] output path: {out_dir}")
    print(f"[{SCRIPT}] sizes={sizes} dims={dims} methods={methods} seed={seed}")
    print(f"[{SCRIPT}] NOTE: cost-only stress test; quality_measured=false on "
          f"every row by design.")

    set_global_seed(seed)
    base_rows = None
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.write_mode == "merge" and all_path.exists():
            base_rows = pd.read_csv(all_path)

    def checkpoint(rows):
        df = pd.DataFrame(rows)
        if base_rows is not None:
            from utils.result_io import merge_dataframe
            df = merge_dataframe(base_rows, df, KEY)
        write_dataframe_atomic(df, all_path, mode="replace",
                               key=KEY, sort_by=KEY)

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
                run(build_index_cmd(method, str(emb), idx_dir, budget_mb,
                                    seed, omp_threads, index_cfg))
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
                    "quality_metric": "none",
                    "quality_notes": ("synthetic vectors carry no relevance "
                                      "signal; only index fidelity "
                                      "(achieved_recall_vs_exact) is reported"),
                    "seed": seed,
                })
                # checkpoint after every cell (long-running grid)
                checkpoint(rows)

    if args.dry_run:
        print(f"[{SCRIPT}] completed.")
        return
    if not rows:
        print(f"[{SCRIPT}] ERROR: no cells produced results.")
        sys.exit(1)

    checkpoint(rows)
    print(f"[{SCRIPT}] output path: {all_path} ({len(rows)} new rows)")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
