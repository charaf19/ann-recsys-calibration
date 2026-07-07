"""Deployment guidance generator.

Turns the measured results (main summary + calibration sensitivity) into an
operational recommendation table: for each (dataset, modality) and a given
operating constraint set (p95 latency budget, ANN agreement-recall floor),
which index method to deploy and with which calibrated parameter.

Purely a post-processing step over existing CSVs — it runs in milliseconds
and performs no searches itself.

Outputs (results/deployment_guidance/):
  deployment_guidance.csv / .md / .tex   per-scenario recommendations
  deployment_notes.md                    rule-of-thumb narrative
"""
import argparse
from pathlib import Path

import pandas as pd

from utils.paths import RESULTS
from utils.reporting import write_table
from utils.common import set_global_seed

SCRIPT = "deployment_guidance"

NOTES = """# Deployment guidance notes (generated)

These rules are derived from the measured benchmark CSVs referenced above.
They apply to CPU-only serving of item-embedding indexes.

1. **Exact Flat is the default below ~100k items.** If measured Flat p95
   latency already fits the budget, ANN adds parameter-tuning and recall risk
   for no benefit.
2. **HNSW is the general-purpose ANN choice** when latency dominates and the
   full float32 vectors fit in memory (index RAM ~= vectors + graph
   overhead). Calibrate `ef` per the calibration-sensitivity table.
3. **IVF-PQ is the memory-constrained choice**: compressed codes shrink RAM
   at the cost of recall ceiling; verify the 0.98 target is reachable before
   choosing it for quality-sensitive surfaces.
4. **Calibration target selection**: 0.95 agreement recall is the balanced
   default; use 0.98 for U2I home-feed style surfaces where end-to-end NDCG
   effect sizes vs Flat should stay negligible, and 0.90 for I2I
   related-items widgets where latency budgets are tighter.
5. **Recalibrate after re-embedding.** Calibrated ef/nprobe values are only
   valid for the embedding geometry they were tuned on (weighting scheme,
   dim, dataset snapshot).
6. **Watch long_tail_exposure, not just accuracy.** PQ-compressed indexes can
   shift exposure toward head items; compare `long_tail_exposure` /
   `long_tail_uplift` columns against Flat before shipping.
"""


def recommend(group: pd.DataFrame, latency_budget_ms: float, recall_floor: float):
    """Pick the cheapest method meeting the constraints; fall back gracefully."""
    ok = group[(group["latency_p95_ms"] <= latency_budget_ms)
               & (group["achieved_recall_vs_exact"] >= recall_floor)]
    if len(ok):
        best = ok.sort_values(["latency_p95_ms"]).iloc[0]
        return best, "meets latency budget and recall floor"
    # constraint relaxation: best recall among those within latency budget
    in_budget = group[group["latency_p95_ms"] <= latency_budget_ms]
    if len(in_budget):
        best = in_budget.sort_values("achieved_recall_vs_exact", ascending=False).iloc[0]
        return best, "recall floor not reachable within latency budget (best-recall fallback)"
    best = group.sort_values("latency_p95_ms").iloc[0]
    return best, "latency budget not reachable (lowest-latency fallback)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration_csv",
                    default=f"{RESULTS['calibration_sensitivity']}/calibration_sensitivity.csv")
    ap.add_argument("--summary_csv", default=f"{RESULTS['main']}/summary_main.csv",
                    help="optional; adds end-to-end quality columns when present")
    ap.add_argument("--latency_budgets_ms", type=float, nargs="*", default=[1.0, 5.0, 20.0])
    ap.add_argument("--recall_floors", type=float, nargs="*", default=[0.90, 0.95, 0.98])
    ap.add_argument("--out_dir", default=RESULTS["deployment_guidance"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.calibration_csv}")
    print(f"[{SCRIPT}] input path: {args.summary_csv}")
    print(f"[{SCRIPT}] output path: {args.out_dir}")

    set_global_seed(args.seed)

    cal_path = Path(args.calibration_csv)
    if not cal_path.is_file():
        print(f"[{SCRIPT}] WARN: {cal_path} not found. "
              f"Run run_calibration_sensitivity.py first.")
        print(f"[{SCRIPT}] completed.")
        return
    cal = pd.read_csv(cal_path)

    summary = None
    sum_path = Path(args.summary_csv)
    if sum_path.is_file():
        summary = pd.read_csv(sum_path)
    else:
        print(f"[{SCRIPT}] WARN: {sum_path} not found; guidance will omit "
              f"end-to-end quality columns.")

    rows = []
    for (dataset, weighting), group in cal.groupby(["dataset", "weighting"]):
        # use the tightest calibration row per method (highest achieved recall
        # per target); each target is its own candidate operating point
        for budget in args.latency_budgets_ms:
            for floor in args.recall_floors:
                best, rationale = recommend(group, budget, floor)
                row = {
                    "dataset": dataset,
                    "weighting": weighting,
                    "latency_budget_p95_ms": budget,
                    "recall_floor": floor,
                    "recommended_method": best["method"],
                    "param_name": best.get("param_name"),
                    "param_value": best.get("calibrated_param_value"),
                    "calibration_target": best.get("target_recall"),
                    "achieved_recall_vs_exact": best["achieved_recall_vs_exact"],
                    "latency_p95_ms": best["latency_p95_ms"],
                    "rationale": rationale,
                }
                if summary is not None:
                    q = summary[(summary["dataset"] == dataset)
                                & (summary["weighting"] == weighting)
                                & (summary["method"] == best["method"])]
                    for mod in ("u2i", "i2i"):
                        qm = q[q["modality"] == mod] if "modality" in q.columns else q.iloc[0:0]
                        if len(qm):
                            row[f"{mod}_ndcg_at_k_mean"] = float(qm.iloc[0].get("ndcg_at_k_mean"))
                            row[f"{mod}_long_tail_exposure"] = float(
                                qm.iloc[0].get("long_tail_exposure"))
                rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    written = write_table(df, out_dir / "deployment_guidance")
    for p in written:
        print(f"[{SCRIPT}] output path: {p}")

    notes_path = out_dir / "deployment_notes.md"
    header = (f"Inputs: `{args.calibration_csv}`"
              + (f", `{args.summary_csv}`" if summary is not None else "") + "\n\n")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(NOTES.replace("(generated)\n", "(generated)\n\n" + header, 1))
    print(f"[{SCRIPT}] output path: {notes_path}")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
