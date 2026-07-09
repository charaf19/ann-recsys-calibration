"""Validate existence, schema, and completeness of pipeline outputs.

For every expected artifact this checks: (a) the file exists, (b) required
columns / JSON keys are present, (c) CSVs are non-empty. It never fabricates
or repairs anything — it only reports. Intended for reviewers ("did I run
everything?") and as a pre-submission gate.

Exit code: 0 when all required artifacts validate (optional ones may be
missing with --allow_missing_optional), 1 otherwise.

Outputs:
    results/validation/validation_report.csv
    results/validation/validation_report.md
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from utils.paths import RESULTS
from utils.reporting import df_to_markdown

SCRIPT = "validate_results"

# (artifact path, stage that produces it, required?, kind, required columns/keys)
ARTIFACTS = [
    (f"{RESULTS['hardware']}/hardware.json", "capture_hardware.py", True, "json",
     ["main_experiments_gpu_used", "cuda_available", "faiss_gpu_available",
      "platform", "cpu", "packages"]),
    (f"{RESULTS['main']}/summary_main.csv", "run_revision_experiments.py", True, "csv",
     ["dataset", "weighting", "modality", "method", "ndcg_at_k_mean",
      "recall_at_k_mean", "ann_recall_vs_exact_at_k_mean", "long_tail_exposure",
      "long_tail_uplift", "latency_p95_ms", "seed"]),
    (f"{RESULTS['main']}/run_config.json", "run_revision_experiments.py", True, "json",
     ["datasets", "methods", "modalities", "seed"]),
    (f"{RESULTS['calibration_sensitivity']}/calibration_sensitivity.csv",
     "run_calibration_sensitivity.py", True, "csv",
     ["dataset", "method", "target_recall", "target_reached",
      "calibrated_param_value", "achieved_recall_vs_exact", "latency_p95_ms"]),
    (f"{RESULTS['bootstrap']}/bootstrap_cis.csv", "bootstrap_significance.py", True, "csv",
     ["dataset", "modality", "method", "metric", "mean", "ci_low", "ci_high", "n_boot"]),
    (f"{RESULTS['bootstrap']}/paired_tests.csv", "bootstrap_significance.py", True, "csv",
     ["dataset", "modality", "method", "baseline", "metric", "mean_diff", "p_value"]),
    (f"{RESULTS['effect_sizes']}/effect_sizes.csv", "effect_size_tables.py", True, "csv",
     ["dataset", "modality", "method", "metric", "cohens_d", "cliffs_delta"]),
    (f"{RESULTS['deployment_guidance']}/ann_decision_framework_scores.csv",
     "ann_decision_framework.py", True, "csv",
     ["dataset", "modality", "method", "effect_size_label", "quality_retention",
      "latency_score", "memory_score", "exposure_score", "deployment_score",
      "deployment_rank", "recommended_use_case", "recommendation_reason"]),
    (f"{RESULTS['paper_tables']}/claim_support_audit.csv", "claim_support_audit.py",
     True, "csv",
     ["claim_area", "claim_strength_allowed", "required_evidence_file",
      "evidence_available", "safe_interpretation",
      "unsafe_interpretation_to_avoid"]),
    # ---- optional modules ----
    (f"{RESULTS['pq_diagnostics']}/pq_diagnostics_all.csv", "run_pq_diagnostics.py",
     False, "csv", ["dataset", "method", "metric", "value"]),
    (f"{RESULTS['exposure_analysis']}/exposure_analysis_all.csv",
     "run_exposure_analysis.py", False, "csv",
     ["dataset", "modality", "method", "metric", "value", "fairness_scope"]),
    (f"{RESULTS['embedding_sensitivity']}/embedding_backbone_sensitivity_all.csv",
     "run_embedding_backbone_sensitivity.py", False, "csv",
     ["dataset", "backbone", "backend_available", "status", "error_message",
      "ann_ranking_stability"]),
    (f"{RESULTS['scale_stress']}/scale_stress_all.csv", "run_scale_stress.py",
     False, "csv",
     ["n_items", "dim", "method", "latency_p95_ms", "quality_measured",
      "quality_metric", "quality_notes"]),
    (f"{RESULTS['optional_backends']}/optional_ann_backend_comparison.csv",
     "run_optional_ann_backend_comparison.py", False, "csv",
     ["backend", "backend_available", "backend_error_message"]),
    (f"{RESULTS['energy']}/energy_measurement_all.csv", "run_energy_measurement.py",
     False, "csv",
     ["dataset", "modality", "method", "measurement_backend",
      "direct_energy_available", "cpu_energy_joules", "energy_per_query_joules",
      "wall_time_sec", "queries", "cpu_utilization_mean", "rss_mb", "notes"]),
    (f"{RESULTS['scaling']}/scaling.csv", "synthetic_scaling.py", False, "csv",
     ["n_items", "method", "latency_p95_ms"]),
]

MODALITY_VALUES = {"u2i", "i2i"}


def check_artifact(path, stage, required, kind, needed):
    p = Path(path)
    row = {"artifact": path, "stage": stage,
           "required": bool(required), "status": "ok",
           "rows": None, "missing_fields": "", "detail": ""}
    if not p.is_file():
        row["status"] = "missing_required" if required else "missing_optional"
        row["detail"] = f"run: python src/{stage.split()[0]}" if stage else ""
        return row
    try:
        if kind == "json":
            with open(p, "r", encoding="utf-8") as f:
                doc = json.load(f)
            missing = [k for k in needed if k not in doc]
            if missing:
                row["status"] = "schema_mismatch"
                row["missing_fields"] = ";".join(missing)
        else:
            df = pd.read_csv(p)
            row["rows"] = int(len(df))
            missing = [c for c in needed if c not in df.columns]
            if missing:
                row["status"] = "schema_mismatch"
                row["missing_fields"] = ";".join(missing)
            elif len(df) == 0:
                row["status"] = "empty"
            elif "modality" in df.columns:
                bad = set(df["modality"].dropna().astype(str)) - MODALITY_VALUES
                if bad:
                    row["status"] = "schema_mismatch"
                    row["detail"] = f"non-canonical modality labels: {sorted(bad)}"
    except Exception as e:
        row["status"] = "unreadable"
        row["detail"] = str(e)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow_missing_optional", action="store_true",
                    help="missing optional-module outputs are warnings, not failures")
    ap.add_argument("--out_dir", default=RESULTS["validation"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: results/ (expected artifacts)")
    print(f"[{SCRIPT}] output path: {out_dir}")

    rows = [check_artifact(*spec) for spec in ARTIFACTS]
    df = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "validation_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"[{SCRIPT}] output path: {csv_path}")

    failures = df[(df["status"] == "missing_required")
                  | (df["status"].isin(["schema_mismatch", "empty", "unreadable"])
                     & df["required"])]
    opt_missing = df[df["status"] == "missing_optional"]
    if not args.allow_missing_optional:
        failures = pd.concat([failures, opt_missing])

    md_path = out_dir / "validation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Validation report\n\n")
        f.write(f"- Artifacts checked: {len(df)}\n")
        f.write(f"- OK: {int((df['status'] == 'ok').sum())}\n")
        f.write(f"- Missing (required): "
                f"{int((df['status'] == 'missing_required').sum())}\n")
        f.write(f"- Missing (optional): {len(opt_missing)}"
                f"{' (allowed)' if args.allow_missing_optional else ''}\n")
        f.write(f"- Schema mismatches / empty / unreadable: "
                f"{int(df['status'].isin(['schema_mismatch', 'empty', 'unreadable']).sum())}\n\n")
        f.write(df_to_markdown(df))
    print(f"[{SCRIPT}] output path: {md_path}")

    if len(failures):
        print(f"[{SCRIPT}] FAIL: {len(failures)} artifact(s) missing or invalid:")
        for _, r in failures.iterrows():
            print(f"[{SCRIPT}]   {r['status']}: {r['artifact']} "
                  f"{('(' + r['missing_fields'] + ')') if r['missing_fields'] else ''}")
        print(f"[{SCRIPT}] completed.")
        sys.exit(1)
    print(f"[{SCRIPT}] PASS: all required artifacts present and schema-consistent.")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
