"""Calibration sensitivity study: recalibrate every ANN method at multiple
recall targets (default 0.90 / 0.95 / 0.98) and record how the calibrated
parameter, achieved recall, and latency move with the target.

Reads configs/calibration_thresholds.yml (CLI flags override). Requires
embeddings and indexes to exist already (built by run_revision_experiments.py
or manually via train_embeddings.py + build_index.py); combinations with
missing artifacts are skipped with a warning, so this script can be run on a
partial grid.

Output: results/calibration_sensitivity/calibration_sensitivity.csv
        plus one JSON per (dataset, method, target) with the full sweep.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from calibrate import calibrate_index
from utils.ann_io import load_ann_index
from utils.common import set_global_seed
from utils.paths import emb_dir, index_dir, RESULTS

SCRIPT = "run_calibration_sensitivity"
DEFAULT_CONFIG = "configs/calibration_thresholds.yml"


def load_config(path):
    p = Path(path)
    if not p.is_file():
        print(f"[{SCRIPT}] WARN: config {path} not found; using built-in defaults.")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--targets", type=float, nargs="*", default=None)
    ap.add_argument("--weighting", default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--queries", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_dir", default=RESULTS["calibration_sensitivity"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    datasets = args.datasets or cfg.get("datasets", ["ml-1m", "ml-20m", "goodbooks"])
    methods = args.methods or cfg.get("methods", ["hnsw", "ivfflat", "ivfpq"])
    targets = args.targets or cfg.get("targets", [0.90, 0.95, 0.98])
    weighting = args.weighting or cfg.get("weighting", "bm25")
    dim = args.dim or cfg.get("dim", 128)
    topk = args.topk or cfg.get("topk", 100)
    queries = args.queries or cfg.get("calibration_queries", 1000)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    param_grids = cfg.get("param_grids", {})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.config}")
    print(f"[{SCRIPT}] output path: {out_dir}")
    print(f"[{SCRIPT}] datasets={datasets} methods={methods} targets={targets} "
          f"weighting={weighting} dim={dim}")

    set_global_seed(seed)

    rows = []
    for dataset in datasets:
        vec_path = Path(emb_dir(dataset, weighting, dim)) / "item_vecs.npy"
        if not vec_path.is_file():
            print(f"[{SCRIPT}] WARN: missing embeddings {vec_path}; skipping {dataset}. "
                  f"Build them via run_revision_experiments.py or train_embeddings.py.")
            continue
        item_vecs = np.load(vec_path).astype("float32")
        N, D = item_vecs.shape

        for method in methods:
            idx_dir = index_dir(dataset, weighting, dim, method)
            if not Path(idx_dir).exists():
                print(f"[{SCRIPT}] WARN: missing index {idx_dir}; skipping.")
                continue
            for target in targets:
                print(f"[{SCRIPT}] calibrating {dataset}/{method} at target={target}")
                ann = load_ann_index(idx_dir, D, N)  # fresh load per target
                res = calibrate_index(ann, item_vecs, float(target), int(topk),
                                      int(queries), int(seed),
                                      param_grid=param_grids.get(method))
                res.update({"dataset": dataset, "weighting": weighting, "dim": int(dim)})

                detail_path = out_dir / f"{dataset}__{weighting}__{method}__t{float(target):.2f}.json"
                with open(detail_path, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2)

                rows.append({
                    "dataset": dataset,
                    "weighting": weighting,
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
                })

    if rows:
        df = pd.DataFrame(rows)
        csv_path = out_dir / "calibration_sensitivity.csv"
        df.to_csv(csv_path, index=False)
        print(f"[{SCRIPT}] output path: {csv_path}")
    else:
        print(f"[{SCRIPT}] WARN: no combinations produced results "
              f"(are embeddings/indexes built?).")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
