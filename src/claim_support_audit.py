"""Claim-support audit: link every claim area to the evidence that bounds it.

Prevents overclaiming: for each claim area the audit states the *maximum*
claim strength the artifact can support, which result file is required as
evidence, whether that evidence currently exists on disk, and the safe vs
unsafe phrasings. `evidence_available` reflects the actual filesystem at
audit time — running this before results exist yields honest `False` rows.

Schema: docs/claim_support_schema.md
Outputs: results/paper_tables/claim_support_audit.csv/.tex (+ .md)
"""
import argparse
from pathlib import Path

import pandas as pd

from utils.paths import RESULTS, first_existing
from utils.reporting import write_table

SCRIPT = "claim_support_audit"

# claim_area, claim_strength_allowed, required_evidence_file(s),
# safe_interpretation, unsafe_interpretation_to_avoid
CLAIMS = [
    ("ANN method selection",
     "strong (measured, calibrated, effect-size-tested)",
     [f"{RESULTS['main']}/summary_main.csv",
      f"{RESULTS['deployment_guidance']}/ann_decision_framework_scores.csv"],
     "At calibrated agreement-recall targets, the measured latency/quality/"
     "memory trade-offs support the per-scenario method recommendations.",
     "Claiming one index method is universally best regardless of catalog "
     "size, embedding, or serving constraints."),

    ("U2I vs I2I modality effect",
     "strong (both modalities measured under one protocol)",
     [f"{RESULTS['main']}/summary_main.csv"],
     "ANN effects are reported separately for U2I and I2I; differences "
     "between modalities are measured on the same split and seed.",
     "Generalizing a modality-specific finding to 'recommendation quality' "
     "in general."),

    ("SVD-based embedding scope",
     "strong within scope (primary results are SVD-embedding-specific)",
     [f"{RESULTS['main']}/summary_main.csv"],
     "Conclusions describe index behavior over TruncatedSVD embeddings "
     "(optionally BM25/TF-IDF weighted).",
     "Presenting the results as embedding-agnostic or state-of-the-art "
     "recommendation quality."),

    ("neural embedding generalization",
     "weak (sensitivity check only)",
     [f"{RESULTS['embedding_sensitivity']}/embedding_backbone_sensitivity_all.csv"],
     "The ANN method *ranking* stability across backbones (ann_ranking_"
     "stability) indicates whether conclusions survive an embedding swap.",
     "Claiming the benchmark evaluates neural recommenders or that quality "
     "numbers transfer to production two-tower models."),

    ("PQ compression behavior",
     "moderate (diagnostics, not mechanism)",
     [f"{RESULTS['pq_diagnostics']}/pq_diagnostics_all.csv"],
     "PQ reconstruction error, neighbor overlap, and popularity-decile "
     "effects are measured; quality deltas carry diagnostic labels.",
     "Asserting that PQ acts as implicit regularization; only "
     "'compression_may_smooth_noise' evidence labels are supported."),

    ("long-tail exposure",
     "strong for the proxy (top-k slot exposure)",
     [f"{RESULTS['exposure_analysis']}/exposure_analysis_all.csv"],
     "long_tail_exposure / long_tail_uplift quantify how index choice "
     "shifts top-k slot exposure toward or away from tail items.",
     "Equating slot-exposure proxies with realized user impressions or "
     "position-weighted attention."),

    ("fairness",
     "weak (exposure proxies only; see fairness_scope column)",
     [f"{RESULTS['exposure_analysis']}/exposure_analysis_all.csv"],
     "Item-side exposure and popularity-calibration proxies are reported "
     "with explicit fairness_scope values.",
     "Any claim of a full fairness evaluation (no user-group, provider-"
     "group, or outcome-level fairness analysis is performed)."),

    ("production-scale catalogs",
     "moderate for cost, none for quality",
     [f"{RESULTS['scale_stress']}/scale_stress_all.csv"],
     "Build time, index size, RSS, and calibrated latency are measured on "
     "synthetic catalogs up to 1M items (quality_measured=false).",
     "Claiming recommendation quality at production scale; real datasets "
     "top out at ML-20M."),

    ("energy consumption",
     "conditional on platform (direct_energy_available)",
     [f"{RESULTS['energy']}/energy_measurement_all.csv"],
     "Where Intel RAPL is readable, CPU package energy per query is a "
     "direct measurement; elsewhere only timing/utilization are reported.",
     "Reporting energy numbers from rows with direct_energy_available="
     "false, or extrapolating package energy to wall power."),

    ("FAISS-specific scope",
     "strong for FAISS; weak beyond it",
     [f"{RESULTS['optional_backends']}/optional_ann_backend_comparison.csv"],
     "Primary conclusions are FAISS(+hnswlib-adapter) specific; the "
     "optional backend comparison documents which other libraries were "
     "actually measured (backend_available column).",
     "Extending conclusions to ScaNN/NGT when their rows show "
     "backend_available=false."),

    ("Flat-PQ deployment role",
     "strong (policy + measurements)",
     [f"{RESULTS['deployment_guidance']}/ann_decision_framework_scores.csv"],
     "Flat-PQ is labeled offline_batch_only by policy (full compressed scan; "
     "no sublinear structure) unless explicitly overridden in "
     "configs/decision_framework.yml.",
     "Recommending Flat-PQ for latency-sensitive online serving."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base",
                    default=f"{RESULTS['paper_tables']}/claim_support_audit")
    args = ap.parse_args()

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: results/ (evidence existence check)")
    print(f"[{SCRIPT}] output path: {args.out_base}.csv")

    rows = []
    for (area, strength, evidence_files, safe, unsafe) in CLAIMS:
        primary = first_existing(*evidence_files)
        available = all(Path(f).is_file() for f in evidence_files) or \
            Path(primary).is_file()
        rows.append({
            "claim_area": area,
            "claim_strength_allowed": strength,
            "required_evidence_file": "; ".join(evidence_files),
            "evidence_available": bool(available),
            "safe_interpretation": safe,
            "unsafe_interpretation_to_avoid": unsafe,
        })

    df = pd.DataFrame(rows)
    written = write_table(df, args.out_base)
    for p in written:
        print(f"[{SCRIPT}] output path: {p}")
    n_missing = int((~df["evidence_available"]).sum())
    if n_missing:
        print(f"[{SCRIPT}] NOTE: {n_missing}/{len(df)} claim areas currently lack "
              f"evidence files — run the corresponding pipeline stages first.")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
