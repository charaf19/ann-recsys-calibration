"""Calibration sensitivity study: recalibrate every tunable ANN method at
multiple recall targets (paper: 0.90 / 0.95 / 0.98) and record how the
calibrated parameter, achieved recall, and latency move with the target.

Reads the canonical experiment configuration (configs/main_cpu.yml): the
dataset list, the tunable subset of retrieval.methods, calibration.targets,
and calibration.queries. Requires embeddings and indexes to exist already
(built by run_revision_experiments.py).

Outputs:
    results/analyses/calibration_sensitivity/calibration_sensitivity.csv
    plus one JSON per (dataset, method, target) with the full sweep.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate import calibrate_index
from utils.ann_io import load_ann_index, CALIBRATION_PARAM
from utils.common import set_global_seed
from utils.config import load_config, cfg_get, config_hash, ConfigError
from utils.paths import emb_dir, index_dir, RESULTS
from utils.provenance import make_run_id, provenance_columns
from utils.result_io import (preflight_output, write_dataframe_atomic,
                             write_json_atomic, ResultExistsError)

SCRIPT = "run_calibration_sensitivity"
DEFAULT_CONFIG = "configs/main_cpu.yml"
KEY = ["dataset", "weighting", "dim", "method", "target_recall", "seed"]


def main():
    ap = argparse.ArgumentParser(
        description="Recalibrate every tunable ANN method (hnsw/ivfflat/"
                    "ivfpq) at each recall target and record parameter, "
                    "achieved agreement recall, and latency.")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="experiment YAML (inherits configs/defaults.yml)")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="tunable methods only (default: tunable subset of "
                         "the configured retrieval methods)")
    ap.add_argument("--targets", type=float, nargs="*", default=None)
    ap.add_argument("--weighting", default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--queries", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_dir", default=RESULTS["calibration_sensitivity"])
    ap.add_argument("--write_mode", default="fail_if_exists",
                    choices=["fail_if_exists", "replace", "merge"])
    ap.add_argument("--allow_missing_inputs", action="store_true",
                    help="skip combinations whose embeddings/indexes are missing "
                         "(default: fail before any calibration starts)")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config, cli_overrides={
            "datasets": args.datasets,
            "embedding.weighting": args.weighting,
            "embedding.dim": args.dim,
            "retrieval.topk": args.topk,
            "calibration.targets": args.targets,
            "calibration.queries": args.queries,
            "reproducibility.seed": args.seed,
        })
    except ConfigError as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    datasets = list(cfg_get(cfg, "datasets", required=True))
    all_methods = list(cfg_get(cfg, "retrieval.methods", required=True))
    methods = args.methods or [m for m in all_methods
                               if CALIBRATION_PARAM.get(m) is not None]
    untunable = [m for m in methods if CALIBRATION_PARAM.get(m) is None]
    if untunable:
        print(f"[{SCRIPT}] ERROR: methods {untunable} have no tunable "
              f"runtime parameter; calibration sensitivity applies to "
              f"hnsw/ivfflat/ivfpq only.")
        sys.exit(1)
    targets = [float(t) for t in cfg_get(cfg, "calibration.targets",
                                         required=True)]
    weighting = cfg_get(cfg, "embedding.weighting", required=True)
    dim = cfg_get(cfg, "embedding.dim", type=int, required=True)
    topk = cfg_get(cfg, "retrieval.topk", type=int, required=True)
    queries = cfg_get(cfg, "calibration.queries", type=int, required=True)
    seed = cfg_get(cfg, "reproducibility.seed", type=int, required=True)
    cfg_hash = config_hash(cfg)
    prov = provenance_columns(make_run_id(cfg_hash), cfg_hash)

    out_dir = Path(args.out_dir)
    csv_path = out_dir / "calibration_sensitivity.csv"
    try:
        preflight_output(csv_path, args.write_mode)
        for dataset in datasets:
            for method in methods:
                for target in targets:
                    preflight_output(
                        out_dir / (f"{dataset}__{weighting}__d{dim}__{method}"
                                   f"__target_{target:.2f}.json"),
                        args.write_mode)
    except (ResultExistsError, ValueError) as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    # fail fast on missing inputs before any calibration work
    missing = []
    for dataset in datasets:
        if not (Path(emb_dir(dataset, weighting, dim)) / "item_vecs.npy").is_file():
            missing.append(f"{dataset}: embeddings "
                           f"{emb_dir(dataset, weighting, dim)}")
            continue
        for method in methods:
            if not Path(index_dir(dataset, weighting, dim, method)).exists():
                missing.append(f"{dataset}/{method}: index "
                               f"{index_dir(dataset, weighting, dim, method)}")
    if missing and not args.allow_missing_inputs:
        print(f"[{SCRIPT}] ERROR: {len(missing)} required input(s) missing; "
              f"run run_revision_experiments.py first:")
        for m in missing:
            print(f"[{SCRIPT}]   {m}")
        sys.exit(1)

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.config}")
    print(f"[{SCRIPT}] output path: {out_dir}")
    print(f"[{SCRIPT}] datasets={datasets} methods={methods} targets={targets} "
          f"weighting={weighting} dim={dim} queries={queries} seed={seed}")

    set_global_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in datasets:
        vec_path = Path(emb_dir(dataset, weighting, dim)) / "item_vecs.npy"
        if not vec_path.is_file():
            print(f"[{SCRIPT}] WARN: missing embeddings {vec_path}; "
                  f"skipping {dataset} (--allow_missing_inputs).")
            continue
        item_vecs = np.load(vec_path).astype("float32")
        N, D = item_vecs.shape

        for method in methods:
            idx_dir = index_dir(dataset, weighting, dim, method)
            if not Path(idx_dir).exists():
                print(f"[{SCRIPT}] WARN: missing index {idx_dir}; skipping "
                      f"(--allow_missing_inputs).")
                continue
            for target in targets:
                print(f"[{SCRIPT}] calibrating {dataset}/{method} "
                      f"at target={target}")
                ann = load_ann_index(idx_dir, D, N)  # fresh load per target
                res = calibrate_index(ann, item_vecs, float(target), int(topk),
                                      int(queries), int(seed))
                res.update({"dataset": dataset, "weighting": weighting,
                            "dim": int(dim), **prov})

                detail_path = out_dir / (
                    f"{dataset}__{weighting}__d{dim}__{method}"
                    f"__target_{float(target):.2f}.json")
                write_json_atomic(res, detail_path, mode=args.write_mode)

                rows.append({
                    "dataset": dataset,
                    "weighting": weighting,
                    "dim": int(dim),
                    "method": method,
                    "target_recall": float(target),
                    "target_reached": res["target_reached"],
                    "param_name": res["param_name"],
                    "calibrated_param_value": res["calibrated_param_value"],
                    "achieved_recall_vs_exact": res["achieved_recall_vs_exact"],
                    "latency_mean_ms": res["latency_ms_at_calibrated"]["mean"],
                    "latency_p50_ms": res["latency_ms_at_calibrated"]["p50"],
                    "latency_p95_ms": res["latency_ms_at_calibrated"]["p95"],
                    "topk": int(topk),
                    "n_calibration_queries": res["n_calibration_queries"],
                    "seed": int(seed),
                    **prov,
                })

    if not rows:
        print(f"[{SCRIPT}] ERROR: no combinations produced results "
              f"(are embeddings/indexes built?).")
        sys.exit(1)

    df = pd.DataFrame(rows)
    expected_rows = len(datasets) * len(methods) * len(targets)
    if len(df) != expected_rows:
        message = (f"produced {len(df)}/{expected_rows} expected rows; "
                   f"evidence is partial")
        if not args.allow_missing_inputs:
            print(f"[{SCRIPT}] ERROR: {message}")
            sys.exit(1)
        print(f"[{SCRIPT}] WARN: {message} (--allow_missing_inputs).")
    write_dataframe_atomic(df, csv_path, mode=args.write_mode,
                           key=KEY, sort_by=KEY)
    print(f"[{SCRIPT}] output path: {csv_path} ({len(df)} rows)")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
