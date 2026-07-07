"""Bootstrap confidence intervals and paired significance tests.

Consumes the per-query metric arrays written by eval_modalities.py
(results/main/perquery/*.npz). Because query construction is deterministic
and method-independent (same split, same seed), per-query arrays are aligned
across methods within a (dataset, weighting, modality) group, which enables
*paired* bootstrap comparisons of each ANN method against the exact Flat
baseline.

Outputs (results/bootstrap/):
  bootstrap_cis.csv    percentile-bootstrap CI of the mean for every metric
  paired_tests.csv     mean difference vs baseline, 95% CI, bootstrap p-value
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.metrics import bootstrap_ci, paired_bootstrap_test
from utils.paths import RESULTS

SCRIPT = "bootstrap_significance"
METRICS = ["recall", "precision", "hr", "ndcg", "map", "mrr", "ann_recall_vs_exact"]


def load_perquery(perquery_dir):
    """Return {(dataset, weighting, modality, method): {metric: array}}."""
    runs = {}
    for f in sorted(Path(perquery_dir).glob("*.npz")):
        with np.load(f, allow_pickle=True) as z:
            meta = json.loads(str(z["meta"]))
            key = (meta["dataset"], meta["weighting"], meta["modality"], meta["method"])
            runs[key] = {m: np.asarray(z[m]) for m in METRICS if m in z.files}
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perquery_dir", default=RESULTS["perquery"])
    ap.add_argument("--out_dir", default=RESULTS["bootstrap"])
    ap.add_argument("--baseline_method", default="flat")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.perquery_dir}")
    print(f"[{SCRIPT}] output path: {args.out_dir}")

    runs = load_perquery(args.perquery_dir)
    if not runs:
        print(f"[{SCRIPT}] WARN: no per-query files found in {args.perquery_dir}. "
              f"Run eval_modalities.py (or run_revision_experiments.py) first.")
        print(f"[{SCRIPT}] completed.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) per-run bootstrap CIs
    ci_rows = []
    for (dataset, weighting, modality, method), arrays in sorted(runs.items()):
        for metric, values in arrays.items():
            ci = bootstrap_ci(values, n_boot=args.n_boot, seed=args.seed)
            ci_rows.append({
                "dataset": dataset, "weighting": weighting, "modality": modality,
                "method": method, "metric": metric, **ci,
            })
    ci_df = pd.DataFrame(ci_rows)
    ci_path = out_dir / "bootstrap_cis.csv"
    ci_df.to_csv(ci_path, index=False)
    print(f"[{SCRIPT}] output path: {ci_path} ({len(ci_df)} rows)")

    # 2) paired tests vs baseline within each (dataset, weighting, modality)
    test_rows = []
    for (dataset, weighting, modality, method), arrays in sorted(runs.items()):
        if method == args.baseline_method:
            continue
        base_key = (dataset, weighting, modality, args.baseline_method)
        base = runs.get(base_key)
        if base is None:
            print(f"[{SCRIPT}] WARN: no {args.baseline_method} baseline for "
                  f"{dataset}/{weighting}/{modality}; skipping {method}.")
            continue
        for metric in METRICS:
            if metric == "ann_recall_vs_exact":
                continue  # trivially 1.0 for the flat baseline; not a paired quantity
            if metric not in arrays or metric not in base:
                continue
            a, b = arrays[metric], base[metric]
            if a.shape != b.shape:
                print(f"[{SCRIPT}] WARN: unaligned queries for "
                      f"{dataset}/{weighting}/{modality}/{method} metric={metric}; skipping.")
                continue
            t = paired_bootstrap_test(a, b, n_boot=args.n_boot, seed=args.seed)
            test_rows.append({
                "dataset": dataset, "weighting": weighting, "modality": modality,
                "method": method, "baseline": args.baseline_method,
                "metric": metric, **t,
                "significant_at_0.05": bool(t["p_value"] < 0.05),
            })
    test_df = pd.DataFrame(test_rows)
    test_path = out_dir / "paired_tests.csv"
    test_df.to_csv(test_path, index=False)
    print(f"[{SCRIPT}] output path: {test_path} ({len(test_df)} rows)")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
