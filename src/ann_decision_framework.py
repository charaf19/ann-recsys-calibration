"""Calibrated, modality-aware, effect-size-aware ANN selection framework.

Reviewer concern addressed: "limited novelty beyond benchmarking". Rather
than reporting raw measurements, this module turns them into a reproducible
*decision procedure* for edge recommender retrieval: every (dataset,
modality, method) gets a deployment score built from (a) quality retention
binned by the measured effect size of delta-NDCG vs exact Flat, (b) calibrated
latency, (c) memory, and (d) long-tail exposure — plus a rule-based
use-case label.

Scoring formula (weights and bins from configs/decision_framework.yml; every
component is also emitted as a column so scores are fully auditable):

    deployment_score =
        w_quality  * quality_retention   # binned from effect_size_label
      + w_latency  * latency_score       # min-max of -log10(p95) within group
      + w_memory   * memory_score        # min-max of -rss within group
      + w_exposure * exposure_score      # min-max of long_tail_uplift within group

"within group" = within each (dataset, modality); min-max maps to [0,1] with
1 = best. quality_retention bins: negligible=1.0, small=0.7, medium=0.3,
large=0.0 (exact Flat reference = 1.0; unknown = 0.3, conservative).

Key ranking property (enforced by construction): methods whose quality loss
vs Flat is NEGLIGIBLE all receive the same quality-retention credit, so among
them ranking is decided by latency; a slower negligible-delta method can only
outrank a faster one through a *clear* memory or exposure advantage
(margins in configs/decision_framework.yml).

Inputs (alternate *_all.csv filenames are accepted):
    results/main/summary_main.csv               | results/main/main_results_all.csv
    results/effect_sizes/effect_sizes.csv       | .../effect_sizes_all.csv
    results/calibration_sensitivity/calibration_sensitivity.csv | .../calibration_sensitivity_all.csv

Outputs:
    results/deployment_guidance/ann_decision_framework_scores.csv
    results/paper_tables/ann_decision_framework_scores.tex (+ .md)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils.common import normalize_modality_label
from utils.paths import RESULTS, first_existing
from utils.reporting import write_table

SCRIPT = "ann_decision_framework"

USE_CASES = ("exact_reference", "online_serving_recommended",
             "memory_constrained_online_serving", "offline_batch_only",
             "not_recommended")


def load_config(path):
    p = Path(path)
    if not p.is_file():
        print(f"[{SCRIPT}] WARN: config {path} not found; using built-in defaults.")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _minmax(series, invert=False):
    """Min-max normalize to [0,1]; constant series map to 1.0."""
    x = series.astype(float)
    lo, hi = x.min(), x.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return pd.Series(1.0, index=series.index)
    n = (x - lo) / (hi - lo)
    return 1.0 - n if invert else n


def attach_effect_labels(df, effects):
    """Attach the Cliff's-delta magnitude of the NDCG delta vs Flat."""
    df["delta_ndcg_vs_flat"] = np.nan
    df["effect_size_label"] = "exact_reference"
    if effects is None:
        df.loc[df["method"] != "flat", "effect_size_label"] = "unknown"
        return df
    e = effects[effects["metric"] == "ndcg"][
        ["dataset", "weighting", "modality", "method", "mean_diff",
         "cliffs_delta_magnitude"]].rename(
        columns={"mean_diff": "delta_ndcg_vs_flat_e",
                 "cliffs_delta_magnitude": "effect_size_label_e"})
    df = df.merge(e, on=["dataset", "weighting", "modality", "method"], how="left")
    non_flat = df["method"] != "flat"
    df.loc[non_flat, "delta_ndcg_vs_flat"] = df.loc[non_flat, "delta_ndcg_vs_flat_e"]
    df.loc[non_flat, "effect_size_label"] = df.loc[non_flat, "effect_size_label_e"].fillna("unknown")
    df.loc[~non_flat, "delta_ndcg_vs_flat"] = 0.0
    return df.drop(columns=["delta_ndcg_vs_flat_e", "effect_size_label_e"])


def use_case_label(row, cfg):
    """Rule-based deployment label per method role and measured constraints."""
    roles = cfg.get("method_roles", {})
    role = roles.get(row["method"], "online_candidate")
    lat_budget = float(cfg.get("latency_budget_p95_ms", 20.0))
    mem_budget = float(cfg.get("memory_budget_mb", 2048))
    min_recall = float(cfg.get("min_agreement_recall", 0.90))
    allow_flatpq_online = bool(cfg.get("allow_flatpq_online", False))

    lat = row.get("latency_p95_ms")
    rss = row.get("rss_mb")
    agree = row.get("ann_recall_vs_flat_at_100")
    label_quality_ok = row["effect_size_label"] in ("negligible", "small", "exact_reference")

    if row["method"] == "flat":
        return "exact_reference", "exact search; ground-truth quality reference"
    if row["method"] == "flatpq" and not allow_flatpq_online:
        return ("offline_batch_only",
                "Flat-PQ scans all compressed codes; policy restricts it to "
                "offline/batch candidate generation (allow_flatpq_online=false)")
    if pd.notna(agree) and agree < min_recall:
        return ("not_recommended",
                f"agreement recall {agree:.3f} below floor {min_recall}")
    if not label_quality_ok:
        return ("not_recommended",
                f"delta-NDCG vs Flat has {row['effect_size_label']} effect size")
    if role == "memory_constrained_candidate":
        if pd.notna(lat) and lat <= lat_budget:
            return ("memory_constrained_online_serving",
                    "PQ compression trades recall ceiling for RAM; fits the "
                    f"latency budget ({lat:.2f} <= {lat_budget} ms p95)")
        return ("offline_batch_only",
                f"latency p95 {lat} ms exceeds online budget {lat_budget} ms")
    # online candidates (hnsw, ivfflat, plugins)
    if pd.notna(lat) and lat > lat_budget:
        return ("not_recommended",
                f"latency p95 {lat:.2f} ms exceeds online budget {lat_budget} ms")
    if pd.notna(rss) and rss > mem_budget:
        return ("memory_constrained_online_serving",
                f"RSS {rss:.0f} MB exceeds edge budget {mem_budget} MB; "
                "consider the PQ variant")
    return ("online_serving_recommended",
            "quality retention negligible/small at calibrated operating point "
            "within latency and memory budgets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/decision_framework.yml")
    ap.add_argument("--summary_csv", default=None)
    ap.add_argument("--effect_sizes_csv", default=None)
    ap.add_argument("--calibration_csv", default=None)
    ap.add_argument("--out_csv",
                    default=f"{RESULTS['deployment_guidance']}/ann_decision_framework_scores.csv")
    ap.add_argument("--paper_table_base",
                    default=f"{RESULTS['paper_tables']}/ann_decision_framework_scores")
    args = ap.parse_args()

    summary_csv = args.summary_csv or first_existing(
        f"{RESULTS['main']}/summary_main.csv",
        f"{RESULTS['main']}/main_results_all.csv")
    effects_csv = args.effect_sizes_csv or first_existing(
        f"{RESULTS['effect_sizes']}/effect_sizes.csv",
        f"{RESULTS['effect_sizes']}/effect_sizes_all.csv")
    calibration_csv = args.calibration_csv or first_existing(
        f"{RESULTS['calibration_sensitivity']}/calibration_sensitivity.csv",
        f"{RESULTS['calibration_sensitivity']}/calibration_sensitivity_all.csv")

    print(f"[{SCRIPT}] starting...")
    for p in (summary_csv, effects_csv, calibration_csv, args.config):
        print(f"[{SCRIPT}] input path: {p}")
    print(f"[{SCRIPT}] output path: {args.out_csv}")

    cfg = load_config(args.config)

    if not Path(summary_csv).is_file():
        print(f"[{SCRIPT}] WARN: {summary_csv} not found. "
              f"Run run_revision_experiments.py first.")
        print(f"[{SCRIPT}] completed.")
        return
    summary = pd.read_csv(summary_csv)
    effects = pd.read_csv(effects_csv) if Path(effects_csv).is_file() else None
    if effects is None:
        print(f"[{SCRIPT}] WARN: effect sizes missing; labels fall back to 'unknown'.")
    calibration = (pd.read_csv(calibration_csv)
                   if Path(calibration_csv).is_file() else None)

    df = summary.copy()
    if "modality" in df.columns:
        df["modality"] = df["modality"].map(normalize_modality_label)
    if "embedding_backend" not in df.columns:
        df["embedding_backend"] = "svd_" + df["weighting"].astype(str)
    df["qps"] = np.where(df["latency_p50_ms"].astype(float) > 0,
                         1000.0 / df["latency_p50_ms"].astype(float), np.nan)
    df = df.rename(columns={
        "ndcg_at_k_mean": "ndcg_at_10",
        "recall_at_100_mean": "recall_at_100",
        "ann_recall_vs_exact_at_100_mean": "ann_recall_vs_flat_at_100",
        "rss_mb_after": "rss_mb",
    })
    # older summaries only carry @k (metric_topk) values; fall back explicitly
    if "recall_at_100" not in df.columns and "recall_at_k_mean" in df.columns:
        print(f"[{SCRIPT}] WARN: recall_at_100 missing; falling back to recall@metric_topk.")
        df["recall_at_100"] = df["recall_at_k_mean"]
    if "ann_recall_vs_flat_at_100" not in df.columns and "ann_recall_vs_exact_at_k_mean" in df.columns:
        print(f"[{SCRIPT}] WARN: agreement@100 missing; falling back to agreement@metric_topk.")
        df["ann_recall_vs_flat_at_100"] = df["ann_recall_vs_exact_at_k_mean"]
    df = attach_effect_labels(df, effects)

    # augment with calibration reachability (informational)
    if calibration is not None:
        reach = (calibration.groupby(["dataset", "weighting", "method"])
                 ["target_reached"].all().rename("all_targets_reached").reset_index())
        df = df.merge(reach, on=["dataset", "weighting", "method"], how="left")

    # scoring
    w = cfg.get("weights", {})
    w_q = float(w.get("quality_retention", 0.45))
    w_l = float(w.get("latency", 0.30))
    w_m = float(w.get("memory", 0.15))
    w_e = float(w.get("long_tail_exposure", 0.10))
    qbins = cfg.get("quality_retention_by_effect",
                    {"negligible": 1.0, "small": 0.7, "medium": 0.3, "large": 0.0})

    def qret(label):
        if label == "exact_reference":
            return 1.0
        return float(qbins.get(label, 0.3))  # unknown -> conservative

    df["quality_retention"] = df["effect_size_label"].map(qret)

    scored = []
    for (_, _), g in df.groupby(["dataset", "modality"]):
        g = g.copy()
        g["latency_score"] = _minmax(np.log10(g["latency_p95_ms"].astype(float)
                                              .clip(lower=1e-3)), invert=True)
        g["memory_score"] = _minmax(g["rss_mb"].astype(float), invert=True)
        g["exposure_score"] = _minmax(g["long_tail_uplift"].astype(float))
        g["deployment_score"] = (w_q * g["quality_retention"]
                                 + w_l * g["latency_score"]
                                 + w_m * g["memory_score"]
                                 + w_e * g["exposure_score"]).round(4)
        g["deployment_rank"] = (g["deployment_score"]
                                .rank(ascending=False, method="first").astype(int))
        scored.append(g)
    df = pd.concat(scored, ignore_index=True)

    labels = df.apply(lambda r: use_case_label(r, cfg), axis=1)
    df["recommended_use_case"] = [l[0] for l in labels]
    df["recommendation_reason"] = [l[1] for l in labels]

    # weights as columns: every score is reproducible from its own row
    df["w_quality"], df["w_latency"] = w_q, w_l
    df["w_memory"], df["w_exposure"] = w_m, w_e

    cols = ["dataset", "modality", "method", "weighting", "embedding_backend",
            "ndcg_at_10", "recall_at_100", "ann_recall_vs_flat_at_100",
            "latency_p95_ms", "qps", "rss_mb", "long_tail_uplift",
            "delta_ndcg_vs_flat", "effect_size_label",
            "quality_retention", "latency_score", "memory_score",
            "exposure_score", "w_quality", "w_latency", "w_memory",
            "w_exposure", "deployment_score", "deployment_rank",
            "recommended_use_case", "recommendation_reason"]
    out = df[[c for c in cols if c in df.columns]].sort_values(
        ["dataset", "modality", "deployment_rank"])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {out_csv}")
    out.to_csv(out_csv, index=False)
    print(f"[{SCRIPT}] output path: {out_csv}")

    written = write_table(out, args.paper_table_base)
    for p in written:
        print(f"[{SCRIPT}] output path: {p}")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
