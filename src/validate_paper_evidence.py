"""Validate completeness and contract invariants of a fresh paper evidence set."""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

from utils.paths import RESULTS

DATASETS = {"ml-1m", "ml-20m", "goodbooks", "amazon-books"}
METHODS = {"flat", "hnsw", "ivfflat", "ivfpq", "flatpq"}
MODALITIES = {"u2i", "i2i"}
TARGETS = {0.90, 0.95, 0.98}
REQUIRED_ANALYSES = {
    "calibration sensitivity": Path(RESULTS["calibration_sensitivity"]) / "calibration_sensitivity.csv",
    "bootstrap confidence intervals": Path(RESULTS["bootstrap"]) / "bootstrap_cis.csv",
    "paired significance": Path(RESULTS["bootstrap"]) / "paired_tests.csv",
    "effect sizes": Path(RESULTS["effect_sizes"]) / "effect_sizes.csv",
    "embedding sensitivity": Path(RESULTS["embedding_sensitivity"]) / "embedding_backbone_sensitivity_all.csv",
    "exposure analysis": Path(RESULTS["exposure_analysis"]) / "exposure_analysis_all.csv",
    "PQ diagnostics": Path(RESULTS["pq_diagnostics"]) / "pq_diagnostics_all.csv",
    "scale stress": Path(RESULTS["scale_stress"]) / "scale_stress_all.csv",
    "decision framework": Path(RESULTS["deployment_guidance"]) / "ann_decision_framework_scores.csv",
    "claim-support audit": Path(RESULTS["paper_tables"]) / "claim_support_audit.csv",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main_cpu.yml")
    ap.add_argument("--out", default=f"{RESULTS['validation']}/paper_evidence.json")
    args = ap.parse_args()
    errors = []

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    configured = set(cfg["datasets"]["main"]) | set(cfg["datasets"].get("optional", []))
    checks = {
        "datasets": configured == DATASETS,
        "methods": set(cfg["methods"]) == METHODS,
        "modalities": set(cfg["modalities"]) == MODALITIES,
        "embedding": (cfg["weighting"] == "bm25" and cfg["dim"] == 128
                      and cfg["normalize"] == "l2" and cfg["bm25_k1"] == 1.2
                      and cfg["bm25_b"] == 0.75 and cfg["seed"] == 42),
        "calibration": (set(map(float, cfg["calibration_targets"])) == TARGETS
                        and float(cfg["primary_target"]) == 0.95),
    }
    errors.extend(f"configuration contract failed: {name}"
                  for name, passed in checks.items() if not passed)

    summary_path = Path(RESULTS["main"]) / "summary_main.csv"
    if not summary_path.is_file():
        errors.append(f"missing main evidence: {summary_path}")
    else:
        summary = pd.read_csv(summary_path)
        needed = {"dataset", "method", "modality", "ndcg_at_k_mean",
                  "recall_at_k_mean", "ann_recall_vs_exact_at_k_mean"}
        missing_columns = needed - set(summary.columns)
        if missing_columns:
            errors.append(f"main summary missing columns: {sorted(missing_columns)}")
        else:
            actual = set(map(tuple, summary[["dataset", "method", "modality"]].astype(str).values))
            expected = set(product(sorted(DATASETS), sorted(METHODS), sorted(MODALITIES)))
            missing = expected - actual
            duplicates = summary.duplicated(["dataset", "method", "modality"]).sum()
            if missing:
                errors.append(f"main summary missing {len(missing)} dataset/method/modality rows")
            if duplicates:
                errors.append(f"main summary has {int(duplicates)} duplicate keys")

    for name, path in REQUIRED_ANALYSES.items():
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty {name}: {path}")

    report = {"valid": not errors, "configuration_checks": checks, "errors": errors}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
