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


def grouped_bootstrap_means(arrays, n_boot, seed, batch_size=128):
    """Resample aligned arrays with one deterministic index stream in batches."""
    arrays = {key: np.asarray(value, dtype=np.float64)
              for key, value in arrays.items()}
    lengths = {value.size for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"bootstrap arrays are not aligned: lengths={lengths}")
    n = lengths.pop() if lengths else 0
    out = {key: np.empty(n_boot, dtype=np.float64) for key in arrays}
    if n == 0:
        for value in out.values():
            value.fill(0.0)
        return out
    rng = np.random.default_rng(seed)
    for start in range(0, n_boot, batch_size):
        stop = min(start + batch_size, n_boot)
        idx = rng.integers(0, n, size=(stop - start, n))
        for key, value in arrays.items():
            out[key][start:stop] = value[idx].mean(axis=1)
    return out


def _ci(values, boot):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(np.mean(values)), "ci_low": float(lo),
            "ci_high": float(hi), "n": int(len(values)),
            "n_boot": int(len(boot))}


def _paired(values, baseline, boot):
    diff = np.asarray(values) - np.asarray(baseline)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "p_value": float(min(1.0, p)),
            "n": int(len(diff)), "n_boot": int(len(boot))}


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

    # Generate one deterministic, bounded bootstrap stream per aligned group
    # and reuse it for every method, metric, and paired difference.
    ci_rows = []
    test_rows = []
    groups = sorted({key[:4] for key in runs}, key=lambda key: tuple(map(str, key)))
    for dataset, weighting, dim, modality in groups:
        base = runs[(dataset, weighting, dim, modality, args.baseline_method)]
        samples = {}
        for method in cfg_get(cfg, "retrieval.methods", required=True):
            rec = runs[(dataset, weighting, dim, modality, method)]
            for metric, values in rec["arrays"].items():
                samples[("ci", method, metric)] = values
                if method != args.baseline_method and metric != "ann_recall_vs_exact":
                    samples[("diff", method, metric)] = (
                        values - base["arrays"][metric])
        boot = grouped_bootstrap_means(samples, n_boot, seed)
        for method in cfg_get(cfg, "retrieval.methods", required=True):
            rec = runs[(dataset, weighting, dim, modality, method)]
            for metric, values in rec["arrays"].items():
                ci_rows.append({
                    "dataset": dataset, "weighting": weighting, "dim": dim,
                    "modality": modality, "method": method, "metric": metric,
                    **_ci(values, boot[("ci", method, metric)]), "seed": seed,
                    "evaluation_seed": rec["seed"],
                })
                if method == args.baseline_method or metric == "ann_recall_vs_exact":
                    continue
                t = _paired(values, base["arrays"][metric],
                            boot[("diff", method, metric)])
                test_rows.append({
                    "dataset": dataset, "weighting": weighting, "dim": dim,
                    "modality": modality, "method": method,
                    "baseline": args.baseline_method, "metric": metric, **t,
                    "significant_at_0.05": bool(t["p_value"] < 0.05),
                    "seed": seed, "evaluation_seed": rec["seed"],
                })
    ci_df = pd.DataFrame(ci_rows)
    write_dataframe_atomic(ci_df, ci_path, mode=args.write_mode,
                           key=CI_KEY, sort_by=CI_KEY)
    print(f"[{SCRIPT}] output path: {ci_path} ({len(ci_df)} rows)")

    test_df = pd.DataFrame(test_rows)
    write_dataframe_atomic(test_df, test_path, mode=args.write_mode,
                           key=TEST_KEY, sort_by=TEST_KEY)
    print(f"[{SCRIPT}] output path: {test_path} ({len(test_df)} rows)")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
