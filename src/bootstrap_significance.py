"""Bootstrap confidence intervals and paired significance tests.

Consumes the per-query metric arrays written by eval_modalities.py
(results/main/perquery/*.npz). Because query construction is deterministic
and method-independent (same split, same seed), per-query arrays are aligned
across methods within a (dataset, weighting, dim, modality) group, which
enables *paired* bootstrap comparisons of each ANN method against the exact
Flat baseline.

The bootstrap iteration count comes from the resolved configuration
(statistics.bootstrap_iterations, paper value 2000); --n_boot is an explicit
override. Non-positive counts are rejected.

Outputs (results/analyses/bootstrap/):
  bootstrap_cis.csv    percentile-bootstrap CI of the mean for every metric
  paired_tests.csv     mean difference vs baseline, 95% CI, bootstrap p-value
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import load_config, cfg_get, ConfigError
from utils.metrics import bootstrap_ci, paired_bootstrap_test
from utils.paths import RESULTS
from utils.result_io import (preflight_output, write_dataframe_atomic,
                             ResultExistsError)

SCRIPT = "bootstrap_significance"
DEFAULT_CONFIG = "configs/main_cpu.yml"
FALLBACK_N_BOOT = 2000  # emergency code fallback = the paper value
METRICS = ["recall", "precision", "hr", "ndcg", "map", "mrr",
           "ann_recall_vs_exact"]
CI_KEY = ["dataset", "weighting", "dim", "modality", "method", "metric",
          "seed", "n_boot"]
TEST_KEY = ["dataset", "weighting", "dim", "modality", "method", "baseline",
            "metric", "seed", "n_boot"]


def load_perquery(perquery_dir):
    """Return {(dataset, weighting, dim, modality, method):
               {"arrays": {metric: array}, "seed": int}}."""
    runs = {}
    for f in sorted(Path(perquery_dir).glob("*.npz")):
        with np.load(f, allow_pickle=True) as z:
            if "meta" not in z.files or "query_ids" not in z.files:
                raise ValueError(
                    f"per-query archive {f} lacks meta/query_ids alignment "
                    f"evidence; rerun eval_modalities.py")
            meta = json.loads(str(z["meta"]))
            required_meta = {"dataset", "weighting", "dim", "modality",
                             "method", "metric_topk", "seed"}
            missing_meta = sorted(required_meta - set(meta))
            if missing_meta:
                raise ValueError(f"per-query archive {f} lacks metadata "
                                 f"{missing_meta}")
            key = (meta["dataset"], meta["weighting"], meta.get("dim"),
                   meta["modality"], meta["method"])
            if key in runs:
                raise ValueError(
                    f"duplicate per-query run metadata key {key}: {f}")
            runs[key] = {
                "arrays": {m: np.asarray(z[m]) for m in METRICS if m in z.files},
                "seed": meta.get("seed"),
                "query_ids": np.asarray(z["query_ids"]).astype(str),
            }
    return runs


def validate_pairing_contract(runs, cfg, baseline="flat"):
    """Require the complete configured grid and identity-aligned queries."""
    weighting = cfg_get(cfg, "embedding.weighting", required=True)
    dim = cfg_get(cfg, "embedding.dim", type=int, required=True)
    datasets = list(cfg_get(cfg, "datasets", required=True))
    modalities = list(cfg_get(cfg, "retrieval.modalities", required=True))
    methods = list(cfg_get(cfg, "retrieval.methods", required=True))
    if baseline != "flat":
        raise ValueError("the scientific baseline is fixed to exact Flat")
    missing = []
    for dataset in datasets:
        for modality in modalities:
            for method in methods:
                key = (dataset, weighting, dim, modality, method)
                if key not in runs:
                    missing.append(key)
    if missing:
        raise ValueError(
            f"missing {len(missing)} required per-query runs, including "
            f"{missing[:3]}; run the complete main experiment first")

    for dataset in datasets:
        for modality in modalities:
            base_key = (dataset, weighting, dim, modality, baseline)
            base = runs.get(base_key)
            if base is None:
                raise ValueError(f"Flat baseline absent for required group "
                                 f"{base_key[:-1]}")
            if len(np.unique(base["query_ids"])) != len(base["query_ids"]):
                raise ValueError(f"duplicate query_ids in Flat baseline "
                                 f"{base_key[:-1]}")
            for method in methods:
                rec = runs[(dataset, weighting, dim, modality, method)]
                if not np.array_equal(rec["query_ids"], base["query_ids"]):
                    raise ValueError(
                        f"unaligned query identities for {dataset}/{weighting}/"
                        f"d{dim}/{modality}/{method}")
                missing_metrics = [m for m in METRICS
                                   if m not in rec["arrays"]]
                if missing_metrics:
                    raise ValueError(
                        f"per-query run {dataset}/{modality}/{method} lacks "
                        f"metrics {missing_metrics}")
                for metric, values in rec["arrays"].items():
                    if values.shape != rec["query_ids"].shape:
                        raise ValueError(
                            f"unaligned {metric} array for {dataset}/{modality}/"
                            f"{method}: {values.shape} vs "
                            f"{rec['query_ids'].shape}")


def main():
    ap = argparse.ArgumentParser(
        description="Percentile-bootstrap CIs and paired significance tests "
                    "of every ANN method vs the exact Flat baseline, over "
                    "the aligned per-query arrays.")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="experiment YAML providing "
                         "statistics.bootstrap_iterations")
    ap.add_argument("--perquery_dir", default=RESULTS["perquery"])
    ap.add_argument("--out_dir", default=RESULTS["bootstrap"])
    ap.add_argument("--baseline_method", default="flat", choices=["flat"],
                    help="fixed exact retrieval baseline")
    ap.add_argument("--n_boot", type=int, default=None,
                    help="explicit override of the configured bootstrap "
                         "iteration count (paper value: 2000)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--write_mode", default="fail_if_exists",
                    choices=["fail_if_exists", "replace", "merge"])
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)
    n_boot = (args.n_boot if args.n_boot is not None
              else cfg_get(cfg, "statistics.bootstrap_iterations", type=int,
                           default=FALLBACK_N_BOOT))
    seed = (args.seed if args.seed is not None
            else cfg_get(cfg, "reproducibility.seed", type=int, default=42))
    if n_boot <= 0:
        print(f"[{SCRIPT}] ERROR: bootstrap iteration count must be positive, "
              f"got {n_boot}.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    ci_path = out_dir / "bootstrap_cis.csv"
    test_path = out_dir / "paired_tests.csv"
    try:
        preflight_output(ci_path, args.write_mode)
        preflight_output(test_path, args.write_mode)
    except (ResultExistsError, ValueError) as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.perquery_dir}")
    print(f"[{SCRIPT}] output path: {out_dir}")
    print(f"[{SCRIPT}] n_boot={n_boot} seed={seed} "
          f"baseline={args.baseline_method}")

    try:
        runs = load_perquery(args.perquery_dir)
        validate_pairing_contract(runs, cfg, args.baseline_method)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[{SCRIPT}] ERROR: {exc}")
        sys.exit(1)
    if not runs:
        print(f"[{SCRIPT}] ERROR: no per-query files found in "
              f"{args.perquery_dir}. Run run_revision_experiments.py first.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) per-run bootstrap CIs
    ci_rows = []
    for (dataset, weighting, dim, modality, method), rec in sorted(
            runs.items(), key=lambda kv: tuple(map(str, kv[0]))):
        for metric, values in rec["arrays"].items():
            ci = bootstrap_ci(values, n_boot=n_boot, seed=seed)
            ci_rows.append({
                "dataset": dataset, "weighting": weighting, "dim": dim,
                "modality": modality, "method": method, "metric": metric,
                **ci, "seed": seed, "evaluation_seed": rec["seed"],
            })
    ci_df = pd.DataFrame(ci_rows)
    write_dataframe_atomic(ci_df, ci_path, mode=args.write_mode,
                           key=CI_KEY, sort_by=CI_KEY)
    print(f"[{SCRIPT}] output path: {ci_path} ({len(ci_df)} rows)")

    # 2) paired tests vs baseline within each (dataset, weighting, dim,
    #    modality)
    test_rows = []
    for (dataset, weighting, dim, modality, method), rec in sorted(
            runs.items(), key=lambda kv: tuple(map(str, kv[0]))):
        if method == args.baseline_method:
            continue
        base = runs.get((dataset, weighting, dim, modality,
                         args.baseline_method))
        if base is None:  # guarded by validate_pairing_contract
            raise RuntimeError("validated Flat baseline unexpectedly absent")
        for metric in METRICS:
            if metric == "ann_recall_vs_exact":
                continue  # trivially 1.0 for the flat baseline; not paired
            arrays, base_arrays = rec["arrays"], base["arrays"]
            if metric not in arrays or metric not in base_arrays:
                raise RuntimeError("validated metric unexpectedly absent")
            a, b = arrays[metric], base_arrays[metric]
            if a.shape != b.shape:
                raise RuntimeError("validated paired arrays unexpectedly differ")
            t = paired_bootstrap_test(a, b, n_boot=n_boot, seed=seed)
            test_rows.append({
                "dataset": dataset, "weighting": weighting, "dim": dim,
                "modality": modality, "method": method,
                "baseline": args.baseline_method, "metric": metric, **t,
                "significant_at_0.05": bool(t["p_value"] < 0.05),
                "seed": seed, "evaluation_seed": rec["seed"],
            })
    test_df = pd.DataFrame(test_rows)
    write_dataframe_atomic(test_df, test_path, mode=args.write_mode,
                           key=TEST_KEY, sort_by=TEST_KEY)
    print(f"[{SCRIPT}] output path: {test_path} ({len(test_df)} rows)")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
