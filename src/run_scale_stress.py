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

Outputs:
    results/analyses/scale_stress/scale_stress_checkpoint.csv
        interruption-safe working state; never final paper evidence
    results/analyses/scale_stress/scale_stress_all.csv
        published only after the configured grid is complete
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
from utils.config import load_config, cfg_get, config_hash, ConfigError
from utils.index_config import build_index_command
from utils.paths import RESULTS
from utils.result_io import (preflight_output, write_dataframe_atomic,
                             validate_unique_keys, ResultExistsError)

SCRIPT = "run_scale_stress"
DEFAULT_CONFIG = "configs/analyses.yml"
KEY = ["n_items", "dim", "method", "seed"]
FINAL_FILENAME = "scale_stress_all.csv"
CHECKPOINT_FILENAME = "scale_stress_checkpoint.csv"


def _cell_key(n_items, dim, method, seed):
    """Canonical, type-stable natural key for one scale-stress cell."""
    return int(n_items), int(dim), str(method), int(seed)


def expected_grid_keys(sizes, dims, methods, seed):
    """Return the exact natural-key set required by the configured grid."""
    return {
        _cell_key(n_items, dim, method, seed)
        for n_items in sizes
        for dim in dims
        for method in methods
    }


def completed_grid_keys(df):
    """Validate checkpoint key uniqueness and return its completed cells."""
    validate_unique_keys(df, KEY, context="in scale-stress checkpoint")
    return {
        _cell_key(row.n_items, row.dim, row.method, row.seed)
        for row in df[KEY].itertuples(index=False)
    }


def pending_grid_cells(sizes, dims, methods, seed, completed=None):
    """Configured cells not already present in a compatible checkpoint."""
    completed = set(completed or ())
    return [
        (int(n_items), int(dim), str(method))
        for n_items in sizes
        for dim in dims
        for method in methods
        if _cell_key(n_items, dim, method, seed) not in completed
    ]


def _bool_value(value, label):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{label} must be true or false, got {value!r}")


def require_cost_only(quality_measured):
    """Reject any configuration that would imply synthetic quality evidence."""
    if _bool_value(quality_measured, "scale_stress.quality_measured"):
        raise ValueError(
            "scale_stress.quality_measured must be false: synthetic vectors "
            "support cost and index-fidelity measurements only")


def _require_checkpoint_columns(df, path):
    required = KEY + ["config_hash", "quality_measured"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"checkpoint {path} is missing required columns {missing}")


def load_resume_checkpoint(path, expected_config_hash):
    """Load and validate resumable state without mutating either result file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"--resume requested but checkpoint does not exist: {path}")
    # Preserve a rare all-numeric hash (including leading zeroes) as text.
    df = pd.read_csv(path, dtype={"config_hash": str})
    _require_checkpoint_columns(df, path)
    validate_unique_keys(df, KEY, context=f"in checkpoint {path}")

    hashes = set(df["config_hash"].dropna().astype(str))
    if hashes != {str(expected_config_hash)} or df["config_hash"].isna().any():
        raise ValueError(
            "incompatible checkpoint configuration hash: "
            f"expected {expected_config_hash}, found {sorted(hashes) or ['<missing>']}")

    try:
        quality_flags = [
            _bool_value(value, "checkpoint quality_measured")
            for value in df["quality_measured"]
        ]
    except ValueError as exc:
        raise ValueError(f"invalid checkpoint {path}: {exc}") from exc
    if any(quality_flags):
        raise ValueError(
            f"checkpoint {path} contains quality_measured=true; refusing to "
            "resume non-cost evidence")
    return df


def write_checkpoint(rows, path):
    """Atomically replace the non-final checkpoint with all completed rows."""
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        raise ValueError("cannot write an empty scale-stress checkpoint")
    write_dataframe_atomic(df, path, mode="replace", key=KEY, sort_by=KEY)
    return df


def validate_complete_grid(df, sizes, dims, methods, seed,
                           expected_config_hash):
    """Require exactly one cost-only row for every configured natural key."""
    _require_checkpoint_columns(df, "final scale-stress frame")
    actual = completed_grid_keys(df)
    expected = expected_grid_keys(sizes, dims, methods, seed)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        sample_missing = sorted(missing, key=str)[:5]
        sample_extra = sorted(extra, key=str)[:5]
        raise ValueError(
            "scale-stress grid is incomplete or incompatible: "
            f"expected {len(expected)} keys, found {len(actual)}; "
            f"missing={sample_missing}, extra={sample_extra}")

    hashes = set(df["config_hash"].dropna().astype(str))
    if hashes != {str(expected_config_hash)} or df["config_hash"].isna().any():
        raise ValueError(
            "scale-stress rows do not all match the resolved configuration "
            f"hash {expected_config_hash}: found {sorted(hashes)}")
    if any(_bool_value(value, "quality_measured")
           for value in df["quality_measured"]):
        raise ValueError("scale-stress final rows must all set quality_measured=false")
    return df


def finalize_results(rows, final_path, checkpoint_path, write_mode, sizes,
                     dims, methods, seed, expected_config_hash):
    """Validate and atomically publish final evidence, then remove checkpoint."""
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    validate_complete_grid(df, sizes, dims, methods, seed,
                           expected_config_hash)
    write_dataframe_atomic(df, final_path, mode=write_mode,
                           key=KEY, sort_by=KEY)
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return df


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
    ap.add_argument("--resume", action="store_true",
                    help="resume from scale_stress_checkpoint.csv; completed "
                         "natural keys are skipped after config-hash validation")
    ap.add_argument("--measure_quality", default="false",
                    help="end-to-end recommendation quality at synthetic "
                         "scale (true/false; deliberately NOT implemented)")
    ap.add_argument("--dry_run", action="store_true",
                    help="print the plan without generating/building anything")
    args = ap.parse_args()

    if _bool_value(args.measure_quality, "--measure_quality"):
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
    resolved_hash = config_hash(cfg)

    try:
        require_cost_only(cfg_get(
            cfg, "scale_stress.quality_measured", type=bool, required=True))
        if not sizes or not dims or not methods:
            raise ValueError(
                "scale-stress catalog_sizes, dimensions, and methods must "
                "all be non-empty")
        if any(int(value) <= 0 for value in sizes):
            raise ValueError("scale-stress catalog sizes must all be positive")
        if any(int(value) <= 0 for value in dims):
            raise ValueError("scale-stress dimensions must all be positive")
        if len(expected_grid_keys(sizes, dims, methods, seed)) != (
                len(sizes) * len(dims) * len(methods)):
            raise ValueError(
                "scale-stress grid lists contain duplicate sizes, dimensions, "
                "or methods")
    except (ConfigError, ValueError) as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    all_path = out_dir / FINAL_FILENAME
    checkpoint_path = out_dir / CHECKPOINT_FILENAME

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.config}")
    print(f"[{SCRIPT}] output path: {out_dir}")
    print(f"[{SCRIPT}] sizes={sizes} dims={dims} methods={methods} seed={seed}")
    print(f"[{SCRIPT}] config_hash={resolved_hash}")
    print(f"[{SCRIPT}] NOTE: cost-only stress test; quality_measured=false on "
          f"every row by design.")

    if args.dry_run:
        for n_items in sizes:
            for dim in dims:
                print(f"[{SCRIPT}] (dry run) would generate {n_items}x{dim} "
                      f"vectors, then build {methods}")
        if args.resume:
            print(f"[{SCRIPT}] (dry run) would validate and resume from "
                  f"{checkpoint_path}")
        print(f"[{SCRIPT}] completed.")
        return

    try:
        write_mode = preflight_output(all_path, args.write_mode)
        if args.resume:
            checkpoint_df = load_resume_checkpoint(checkpoint_path,
                                                   resolved_hash)
        else:
            if checkpoint_path.exists():
                raise ResultExistsError(
                    f"unfinished checkpoint exists: {checkpoint_path}\n"
                    "  use --resume to continue it; it will not be overwritten")
            checkpoint_df = pd.DataFrame()
        completed = (completed_grid_keys(checkpoint_df)
                     if not checkpoint_df.empty else set())
        unexpected = completed - expected_grid_keys(sizes, dims, methods, seed)
        if unexpected:
            raise ValueError(
                "checkpoint contains cells outside the configured grid: "
                f"{sorted(unexpected, key=str)[:5]}")
    except (FileNotFoundError, ResultExistsError, ValueError) as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(seed)
    rows = checkpoint_df.to_dict("records") if not checkpoint_df.empty else []
    pending = pending_grid_cells(sizes, dims, methods, seed, completed)
    if args.resume:
        print(f"[{SCRIPT}] resume: {len(completed)} completed cells loaded; "
              f"{len(pending)} cells remain")

    for n_items in sizes:
        for dim in dims:
            pending_methods = [
                method for method in methods
                if _cell_key(n_items, dim, method, seed) not in completed
            ]
            if not pending_methods:
                print(f"[{SCRIPT}] resume: skipping completed n_items={n_items} "
                      f"dim={dim}")
                continue
            tag = f"stress_n{n_items}_d{dim}"
            emb = Path(f"data/{tag}")
            emb.mkdir(parents=True, exist_ok=True)
            vec_path = emb / "item_vecs.npy"
            if not vec_path.is_file():
                print(f"[{SCRIPT}] generating {n_items}x{dim} synthetic vectors")
                X = synth_vectors(n_items, dim, n_clusters, cluster_std, seed)
                np.save(vec_path, X)
                np.save(emb / "item_ids.npy", np.arange(n_items))
            item_vecs = np.load(vec_path).astype("float32")

            for method in pending_methods:
                idx_dir = f"data/index_{tag}_{method}"
                t0 = time.perf_counter()
                run(build_index_command(
                    method, emb / "item_vecs.npy", emb / "item_ids.npy",
                    idx_dir, budget_mb, seed, omp_threads, index_cfg))
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
                    "config_hash": resolved_hash,
                })
                # checkpoint after every cell (long-running grid)
                write_checkpoint(rows, checkpoint_path)
                completed.add(_cell_key(n_items, dim, method, seed))

    if not rows:
        print(f"[{SCRIPT}] ERROR: no cells produced results.")
        sys.exit(1)

    try:
        final_df = finalize_results(
            rows, all_path, checkpoint_path, write_mode, sizes, dims, methods,
            seed, resolved_hash)
    except (ResultExistsError, ValueError, OSError) as e:
        print(f"[{SCRIPT}] ERROR: finalization failed; checkpoint retained at "
              f"{checkpoint_path}: {e}")
        sys.exit(1)
    print(f"[{SCRIPT}] output path: {all_path} ({len(final_df)} grid rows)")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
