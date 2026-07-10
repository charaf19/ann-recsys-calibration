"""Generate the paper tables (CSV + Markdown + LaTeX) from measured results.

Pure post-processing: reads ONLY the canonical evidence under results/main/
and results/analyses/ and writes ONLY under results/paper/tables/. Never
fabricates values. Every generated format gets a
<full-filename>.sources.json sidecar recording source files, hashes, the
canonical run's config hash, git commit, and timestamp.

The main summary is a critical input (clear error when absent); analysis
inputs are optional (visible warning, only the dependent table is skipped).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from utils.paths import RESULTS
from utils.reporting import write_table
from utils.result_io import resolve_write_mode, ResultExistsError

SCRIPT = "tables_paper"

# canonical inputs (all produced by the pipeline)
IN = {
    "summary": Path(RESULTS["main"]) / "summary_main.csv",
    "calibration": Path(RESULTS["calibration_sensitivity"]) / "calibration_sensitivity.csv",
    "bootstrap_cis": Path(RESULTS["bootstrap"]) / "bootstrap_cis.csv",
    "paired_tests": Path(RESULTS["bootstrap"]) / "paired_tests.csv",
    "effect_sizes": Path(RESULTS["effect_sizes"]) / "effect_sizes.csv",
    "embedding": Path(RESULTS["embedding_sensitivity"]) / "embedding_backbone_sensitivity_all.csv",
    "pq_summary": Path(RESULTS["pq_diagnostics"]) / "pq_diagnostics_summary.csv",
    "exposure": Path(RESULTS["exposure_analysis"]) / "exposure_analysis_all.csv",
    "scale": Path(RESULTS["scale_stress"]) / "scale_stress_all.csv",
    "optional_backends": Path(RESULTS["optional_backends"]) / "optional_ann_backend_comparison.csv",
    "energy": Path(RESULTS["energy"]) / "energy_measurement_all.csv",
    "decision": Path(RESULTS["decision_framework"]) / "ann_decision_framework_scores.csv",
}


def _read_optional(path):
    p = Path(path)
    if not p.is_file():
        print(f"[{SCRIPT}] WARN: {p} not found; skipping the dependent table.")
        return None
    return pd.read_csv(p)


class TableEmitter:
    def __init__(self, out_dir, write_mode):
        self.out_dir = Path(out_dir)
        self.mode = write_mode
        self.written = []

    def emit(self, df, name, sources):
        base = self.out_dir / name
        self.written += write_table(
            df, base, mode=self.mode, source_files=sources, script=SCRIPT)


def table_main_quality(em, summary):
    cols = ["dataset", "weighting", "dim", "modality", "method",
            "ndcg_at_k_mean", "recall_at_k_mean", "hr_at_k_mean",
            "mrr_at_k_mean", "ann_recall_vs_exact_at_k_mean",
            "latency_p50_ms", "latency_p95_ms"]
    df = summary[[c for c in cols if c in summary.columns]].copy()
    df = df.sort_values(["dataset", "modality", "method"])
    em.emit(df, "table_main_quality", [IN["summary"]])


def table_long_tail(em, summary):
    cols = ["dataset", "weighting", "dim", "modality", "method",
            "coverage_at_k", "gini_exposure", "long_tail_exposure",
            "long_tail_uplift"]
    df = summary[[c for c in cols if c in summary.columns]].copy()
    df = df.sort_values(["dataset", "modality", "method"])
    em.emit(df, "table_long_tail_exposure", [IN["summary"]])


def table_calibration(em, cal):
    df = cal.copy().sort_values(["dataset", "method", "target_recall"])
    cols = ["dataset", "weighting", "dim", "method", "target_recall",
            "target_reached", "param_name", "calibrated_param_value",
            "achieved_recall_vs_exact", "latency_p50_ms", "latency_p95_ms"]
    df = df[[c for c in cols if c in df.columns]]
    em.emit(df, "table_calibration_sensitivity", [IN["calibration"]])


def table_significance(em, tests, effects):
    """Merged significance + effect-size table (NDCG focus)."""
    keys = ["dataset", "weighting", "modality", "method", "metric"]
    t = tests.copy()
    sources = [IN["paired_tests"]]
    if effects is not None:
        e = effects[keys + ["cohens_d", "cohens_d_magnitude",
                            "cliffs_delta", "cliffs_delta_magnitude"]]
        t = t.merge(e, on=keys, how="left")
        sources.append(IN["effect_sizes"])
    t = t[t["metric"].isin(["ndcg", "recall", "hr"])]
    cols = keys + ["mean_diff", "ci_low", "ci_high", "p_value", "n_boot",
                   "significant_at_0.05", "cohens_d", "cohens_d_magnitude",
                   "cliffs_delta", "cliffs_delta_magnitude"]
    t = t[[c for c in cols if c in t.columns]].sort_values(keys)
    em.emit(t, "table_significance_effect_sizes", sources)


def table_ci(em, cis):
    df = cis[cis["metric"].isin(["ndcg", "recall", "hr"])].copy()
    df = df.sort_values(["dataset", "modality", "metric", "method"])
    em.emit(df, "table_bootstrap_cis", [IN["bootstrap_cis"]])


def table_embedding_sensitivity(em, df):
    summary = (df.groupby(["dataset", "backbone"])
               .agg(ndcg_at_10_best=("ndcg_at_10", "max"),
                    ann_ranking_stability=("ann_ranking_stability", "first"),
                    backend_available=("backend_available", "all"),
                    n_methods=("method", "nunique")).reset_index())
    em.emit(summary, "embedding_backbone_sensitivity_summary",
            [IN["embedding"]])


def table_exposure_summary(em, df):
    headline = df[df["metric"].isin(
        ["long_tail_exposure", "long_tail_uplift", "gini_exposure",
         "coverage", "user_popularity_calibration_error"])].copy()
    headline["metric_k"] = headline.apply(
        lambda r: f"{r['metric']}_at_{int(r['k'])}" if pd.notna(r["k"])
        else r["metric"], axis=1)
    summary = headline.pivot_table(
        index=["dataset", "weighting", "modality", "method"],
        columns="metric_k", values="value").reset_index()
    summary["fairness_scope"] = "not_full_fairness_evaluation"
    em.emit(summary, "exposure_analysis_summary", [IN["exposure"]])


def table_scale_stress(em, df):
    cols = ["n_items", "dim", "method", "build_wall_time_sec",
            "index_size_mb", "rss_mb_after", "latency_p50_ms",
            "latency_p95_ms", "achieved_recall_vs_exact", "quality_measured"]
    summary = df[[c for c in cols if c in df.columns]]
    em.emit(summary, "scale_stress_summary", [IN["scale"]])


def main():
    ap = argparse.ArgumentParser(
        description="Generate every paper table (CSV+MD+LaTeX with source "
                    "sidecars) from the canonical results directories.")
    ap.add_argument("--out_dir", default=RESULTS["paper_tables"])
    ap.add_argument("--write_mode", default="replace",
                    choices=["fail_if_exists", "replace"],
                    help="regenerated presentation tables; default replace")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    write_mode = resolve_write_mode(args.write_mode)
    print(f"[{SCRIPT}] starting...")
    for p in IN.values():
        print(f"[{SCRIPT}] input path: {p}")
    print(f"[{SCRIPT}] output path: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    em = TableEmitter(out_dir, write_mode)

    # main summary is CRITICAL: without it there is no paper
    if not IN["summary"].is_file():
        print(f"[{SCRIPT}] ERROR: canonical input missing: {IN['summary']}\n"
              f"  produce it first with: python src/run_revision_experiments.py"
              f" --config configs/main_cpu.yml")
        sys.exit(1)

    try:
        summary = pd.read_csv(IN["summary"])
        table_main_quality(em, summary)
        table_long_tail(em, summary)

        cal = _read_optional(IN["calibration"])
        if cal is not None:
            table_calibration(em, cal)

        cis = _read_optional(IN["bootstrap_cis"])
        if cis is not None:
            table_ci(em, cis)

        tests = _read_optional(IN["paired_tests"])
        effects = _read_optional(IN["effect_sizes"])
        if tests is not None:
            table_significance(em, tests, effects)

        embedding = _read_optional(IN["embedding"])
        if embedding is not None:
            table_embedding_sensitivity(em, embedding)

        pq = _read_optional(IN["pq_summary"])
        if pq is not None:
            em.emit(pq, "pq_diagnostics_summary", [IN["pq_summary"]])

        exposure = _read_optional(IN["exposure"])
        if exposure is not None:
            table_exposure_summary(em, exposure)

        scale = _read_optional(IN["scale"])
        if scale is not None:
            table_scale_stress(em, scale)

        backends = _read_optional(IN["optional_backends"])
        if backends is not None:
            em.emit(backends, "optional_ann_backend_comparison",
                    [IN["optional_backends"]])

        energy = _read_optional(IN["energy"])
        if energy is not None:
            em.emit(energy, "energy_measurement_summary", [IN["energy"]])

        decision = _read_optional(IN["decision"])
        if decision is not None:
            em.emit(decision, "ann_decision_framework_scores", [IN["decision"]])
    except ResultExistsError as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    for p in em.written:
        print(f"[{SCRIPT}] output path: {p}")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
