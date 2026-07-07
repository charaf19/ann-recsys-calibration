"""Effect-size reporting alongside statistical significance.

For every ANN method vs the exact Flat baseline (paired per-query arrays from
eval_modalities.py), computes:
  - Cohen's d for paired samples: mean(diff) / sd(diff)
  - Cliff's delta (non-parametric, robust to the heavy zero-inflation of
    per-query ranking metrics)
with conventional magnitude interpretations (negligible/small/medium/large).

Outputs (results/effect_sizes/): effect_sizes.csv / .md / .tex
"""
import argparse
from pathlib import Path

import pandas as pd

from bootstrap_significance import load_perquery, METRICS
from utils.metrics import cohens_d_paired, cliffs_delta, effect_size_interpretation
from utils.paths import RESULTS
from utils.reporting import write_table

SCRIPT = "effect_size_tables"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perquery_dir", default=RESULTS["perquery"])
    ap.add_argument("--out_dir", default=RESULTS["effect_sizes"])
    ap.add_argument("--baseline_method", default="flat")
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

    rows = []
    for (dataset, weighting, modality, method), arrays in sorted(runs.items()):
        if method == args.baseline_method:
            continue
        base = runs.get((dataset, weighting, modality, args.baseline_method))
        if base is None:
            print(f"[{SCRIPT}] WARN: no {args.baseline_method} baseline for "
                  f"{dataset}/{weighting}/{modality}; skipping {method}.")
            continue
        for metric in METRICS:
            if metric == "ann_recall_vs_exact":
                continue
            if metric not in arrays or metric not in base:
                continue
            a, b = arrays[metric], base[metric]
            if a.shape != b.shape:
                continue
            d = cohens_d_paired(a, b)
            delta = cliffs_delta(a, b, seed=args.seed)
            rows.append({
                "dataset": dataset, "weighting": weighting, "modality": modality,
                "method": method, "baseline": args.baseline_method, "metric": metric,
                "mean_diff": float(a.mean() - b.mean()),
                "cohens_d": d,
                "cohens_d_magnitude": effect_size_interpretation(d, "cohens_d"),
                "cliffs_delta": delta,
                "cliffs_delta_magnitude": effect_size_interpretation(delta, "cliffs_delta"),
                "n": int(a.size),
            })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    written = write_table(df, out_dir / "effect_sizes")
    for p in written:
        print(f"[{SCRIPT}] output path: {p}")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
