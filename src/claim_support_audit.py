"""Claim-support audit: link every paper claim area to validated evidence.

Prevents overclaiming. A claim is marked supported ONLY when the strict
validator (validate_paper_evidence.py) reports its required sections as
passing — never merely because a file exists on disk. Run the validator
first; before results exist every claim is honestly unsupported.

For each claim area the audit states the *maximum* claim strength the
artifact can support, the validation sections required as evidence, whether
those sections currently pass, and the safe vs unsafe phrasings.

Outputs:
    results/paper/tables/claim_support_audit.csv
    results/paper/tables/claim_support_audit.md
    results/paper/tables/claim_support_audit.tex
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from utils.paths import RESULTS
from utils.reporting import write_table
from utils.result_io import ResultExistsError

SCRIPT = "claim_support_audit"

# claim_area, claim_strength_allowed, required validation sections,
# safe_interpretation, unsafe_interpretation_to_avoid
CLAIMS = [
    ("ANN method selection",
     "strong (measured, calibrated, effect-size-tested)",
     ["main", "calibration_sensitivity", "effect_sizes"],
     "At calibrated agreement-recall targets, the measured latency/quality/"
     "memory trade-offs support the per-scenario method recommendations.",
     "Claiming one index method is universally best regardless of catalog "
     "size, embedding, or serving constraints."),

    ("larger-catalog generalization (Amazon Books)",
     "strong within the four evaluated catalogs",
     ["main"],
     "Findings hold across ML-1M, ML-20M, GoodBooks and Amazon Books under "
     "one protocol; Amazon Books rows are complete in the main evidence.",
     "Citing Amazon Books scale without complete amazon-books main rows, or "
     "extrapolating beyond the evaluated catalog sizes."),

    ("U2I vs I2I modality effect",
     "strong (both modalities measured under one protocol)",
     ["main"],
     "ANN effects are reported separately for U2I and I2I; differences "
     "between modalities are measured on the same split and seed.",
     "Generalizing a modality-specific finding to 'recommendation quality' "
     "in general."),

    ("statistical significance",
     "strong (paired bootstrap, n_boot=2000, seeded)",
     ["bootstrap", "effect_sizes"],
     "Paired bootstrap CIs/p-values (2000 iterations) plus paired Cohen's d "
     "and Cliff's delta against the exact Flat baseline.",
     "Reporting significance from fewer than the contracted 2000 bootstrap "
     "iterations, or unpaired comparisons."),

    ("embedding-sensitivity generalization",
     "weak (sensitivity check only)",
     ["embedding_sensitivity"],
     "The ANN method *ranking* stability across backbones (including BPR "
     "and all five methods) indicates whether conclusions survive an "
     "embedding swap; stability values are reported honestly.",
     "Claiming the benchmark evaluates neural recommenders, or citing the "
     "check without BPR rows and all five methods."),

    ("PQ compression behavior",
     "moderate (diagnostics, not mechanism)",
     ["pq_diagnostics"],
     "PQ reconstruction error, neighbor overlap, and popularity-decile "
     "effects are measured; quality deltas carry diagnostic labels.",
     "Asserting that PQ acts as implicit regularization; only "
     "'compression_may_smooth_noise' evidence labels are supported."),

    ("long-tail exposure",
     "strong for the proxy (top-k slot exposure)",
     ["exposure", "main"],
     "long_tail_exposure / long_tail_uplift quantify how index choice "
     "shifts top-k slot exposure toward or away from tail items, on the "
     "complete exposure analysis with an explicit fairness_scope field.",
     "Equating slot-exposure proxies with realized user impressions, "
     "position-weighted attention, or a full fairness evaluation."),

    ("production-scale catalogs",
     "moderate for cost, none for quality",
     ["scale_stress"],
     "Build time, index size, RSS, and calibrated latency are measured on "
     "the complete 75-cell synthetic grid up to 1M items "
     "(quality_measured=false on every row).",
     "Claiming recommendation quality at production scale; real datasets "
     "top out at ML-20M/Amazon Books."),

    ("CPU-only scope",
     "strong (hardware/provenance declaration)",
     ["cpu_scope"],
     "All canonical experiments ran on CPU; accelerator presence is "
     "disclosed passively and main_experiments_accelerator_used=false.",
     "Citing any GPU behavior, or omitting the CPU-only boundary when "
     "discussing latency."),
]


def load_validation_report(path):
    p = Path(path)
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="Audit which paper claims the VALIDATED evidence "
                    "supports (consumes the strict validation report; run "
                    "validate_paper_evidence.py first).")
    ap.add_argument("--validation_report",
                    default=str(Path(RESULTS["meta"])
                                / "validation_report.json"))
    ap.add_argument("--out_base",
                    default=str(Path(RESULTS["paper_tables"])
                                / "claim_support_audit"))
    ap.add_argument("--write_mode", default="replace",
                    choices=["fail_if_exists", "replace"],
                    help="regenerated paper table; default replace")
    args = ap.parse_args()

    out_base = Path(args.out_base)
    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.validation_report}")
    print(f"[{SCRIPT}] output path: {out_base}.csv")

    report = load_validation_report(args.validation_report)
    if report is None:
        print(f"[{SCRIPT}] ERROR: critical input missing: "
              f"{args.validation_report}; run validate_paper_evidence.py "
              f"first.")
        sys.exit(1)
    sections = report.get("sections", {})

    def section_status(name):
        return sections.get(name, {}).get("status", "missing")

    def failing_checks(name):
        checks = sections.get(name, {}).get("checks", [])
        return [c["check"] for c in checks if c.get("status") == "fail"]

    rows = []
    for (area, strength, required_sections, safe, unsafe) in CLAIMS:
        statuses = {s: section_status(s) for s in required_sections}
        supported = all(v == "pass" for v in statuses.values())
        fails = []
        for s in required_sections:
            fails.extend(f"{s}:{c}" for c in failing_checks(s))
        rows.append({
            "claim_area": area,
            "claim_strength_allowed": strength,
            "required_validation_sections": "; ".join(required_sections),
            "evidence_supported": supported,
            "failing_checks": "; ".join(fails),
            "safe_interpretation": safe,
            "unsafe_interpretation_to_avoid": unsafe,
        })

    df = pd.DataFrame(rows)
    try:
        written = write_table(
            df, out_base, mode=args.write_mode,
            source_files=[args.validation_report], script=SCRIPT)
    except ResultExistsError as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)
    for p in written:
        print(f"[{SCRIPT}] output path: {p}")
    n_unsupported = int((~df["evidence_supported"]).sum())
    if n_unsupported:
        print(f"[{SCRIPT}] NOTE: {n_unsupported}/{len(df)} claim areas are "
              f"currently UNSUPPORTED — run the pipeline and the validator "
              f"before citing them.")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
