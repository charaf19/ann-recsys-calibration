"""Generate the paper figures (PNG + PDF, 300 dpi) from measured results.

Pure post-processing: reads ONLY the canonical evidence under results/main/
and results/analyses/ and writes ONLY under results/paper/figures/. Uses
the Agg backend (headless, CPU-only). Every figure gets a
distinct <full-filename>.sources.json sidecar recording source files, hashes,
the canonical run's config hash, git commit, and timestamp.

The main summary is a critical input (clear error when absent); analysis
inputs are optional (visible warning, only the dependent figures skipped).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.paths import RESULTS
from utils.result_io import resolve_write_mode, ResultExistsError
from utils import figures_ext as FX

SCRIPT = "figures_paper"

IN = {
    "summary": Path(RESULTS["main"]) / "summary_main.csv",
    "calibration": Path(RESULTS["calibration_sensitivity"]) / "calibration_sensitivity.csv",
    "effect_sizes": Path(RESULTS["effect_sizes"]) / "effect_sizes.csv",
    "embedding": Path(RESULTS["embedding_sensitivity"]) / "embedding_backbone_sensitivity_all.csv",
    "pq_summary": Path(RESULTS["pq_diagnostics"]) / "pq_diagnostics_summary.csv",
    "pq_all": Path(RESULTS["pq_diagnostics"]) / "pq_diagnostics_all.csv",
    "exposure": Path(RESULTS["exposure_analysis"]) / "exposure_analysis_all.csv",
    "scale": Path(RESULTS["scale_stress"]) / "scale_stress_all.csv",
}


def _read_optional(path):
    p = Path(path)
    if not p.is_file():
        print(f"[{SCRIPT}] WARN: {p} not found; skipping dependent figures.")
        return None
    return pd.read_csv(p)


def _save(fig, out_dir, name, sources, write_mode="replace"):
    paths = FX.save_figure_artifacts(
        fig, out_dir, name, write_mode=write_mode, source_files=sources,
        script=SCRIPT)
    for p in paths:
        print(f"[{SCRIPT}] output path: {p}")
    return paths


def fig_latency_vs_ndcg(summary, out_dir, write_mode="replace"):
    modalities = sorted(summary["modality"].dropna().unique())
    fig, axes = plt.subplots(1, max(1, len(modalities)),
                             figsize=(5.5 * max(1, len(modalities)), 4.2),
                             squeeze=False)
    for ax, mod in zip(axes[0], modalities):
        sub = summary[summary["modality"] == mod]
        for method in sorted(sub["method"].unique()):
            s = sub[sub["method"] == method]
            ax.scatter(s["latency_p95_ms"], s["ndcg_at_k_mean"],
                       label=method, s=40)
        ax.set_xscale("log")
        ax.set_xlabel("Latency p95 (ms, log)")
        ax.set_ylabel("NDCG@k (mean)")
        ax.set_title(f"Modality: {mod.upper()}")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Quality vs latency at the calibrated operating point")
    return _save(fig, out_dir, "fig_latency_vs_ndcg", [IN["summary"]],
                 write_mode=write_mode)


def fig_calibration_sensitivity(cal, out_dir, write_mode="replace"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (dataset, method), g in cal.groupby(["dataset", "method"]):
        g = g.sort_values("target_recall")
        axes[0].plot(g["target_recall"], g["calibrated_param_value"],
                     marker="o", label=f"{dataset}/{method}")
        axes[1].plot(g["target_recall"], g["latency_p95_ms"],
                     marker="o", label=f"{dataset}/{method}")
    axes[0].set_xlabel("Calibration target (recall vs exact)")
    axes[0].set_ylabel("Calibrated parameter (ef / nprobe)")
    axes[0].set_yscale("log")
    axes[1].set_xlabel("Calibration target (recall vs exact)")
    axes[1].set_ylabel("Latency p95 (ms)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.suptitle("Calibration sensitivity across recall targets")
    return _save(fig, out_dir, "fig_calibration_sensitivity",
                 [IN["calibration"]], write_mode=write_mode)


def fig_long_tail_exposure(summary, out_dir, write_mode="replace"):
    sub = summary.dropna(subset=["long_tail_exposure"])
    datasets = sorted(sub["dataset"].unique())
    fig, axes = plt.subplots(1, max(1, len(datasets)),
                             figsize=(4.2 * max(1, len(datasets)), 4.0),
                             squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        d = sub[(sub["dataset"] == dataset)]
        piv = d.pivot_table(index="method", columns="modality",
                            values="long_tail_exposure", aggfunc="mean")
        piv.plot(kind="bar", ax=ax, legend=False)
        ax.set_title(dataset)
        ax.set_ylabel("long_tail_exposure")
        ax.grid(alpha=0.3, axis="y")
    axes[0][-1].legend(title="modality", fontsize=8)
    fig.suptitle("Long-tail exposure by index method")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_long_tail_exposure", [IN["summary"]],
                 write_mode=write_mode)


def fig_effect_sizes(effects, out_dir, write_mode="replace"):
    sub = effects[effects["metric"] == "ndcg"].copy()
    if sub.empty:
        print(f"[{SCRIPT}] WARN: no NDCG effect sizes; skipping "
              f"fig_effect_sizes.")
        return
    sub["label"] = (sub["dataset"] + "/" + sub["modality"] + "/"
                    + sub["method"])
    sub = sub.sort_values("cliffs_delta")
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(sub) + 1.5))
    ax.barh(sub["label"], sub["cliffs_delta"])
    for thr in (-0.147, 0.147):
        ax.axvline(thr, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Cliff's delta vs Flat (NDCG@k); dashed = negligible band")
    ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Effect sizes of ANN vs exact search")
    return _save(fig, out_dir, "fig_effect_sizes", [IN["effect_sizes"]],
                 write_mode=write_mode)


def generate_figures(out_dir, write_mode):
    """Generate every figure whose canonical inputs are available."""
    summary = pd.read_csv(IN["summary"])
    fig_latency_vs_ndcg(summary, out_dir, write_mode)
    fig_long_tail_exposure(summary, out_dir, write_mode)

    cal = _read_optional(IN["calibration"])
    if cal is not None:
        fig_calibration_sensitivity(cal, out_dir, write_mode)

    effects = _read_optional(IN["effect_sizes"])
    if effects is not None:
        fig_effect_sizes(effects, out_dir, write_mode)

    # Analysis-module figures share the same atomic renderer and sidecar
    # contract as the headline figures.
    module_figures = [
        (IN["pq_summary"], [FX.fig_pq_reconstruction_error_by_dataset,
                            FX.fig_pq_neighbor_overlap_vs_quality_delta]),
        (IN["pq_all"], [FX.fig_pq_popularity_decile_effect]),
        (IN["exposure"], [FX.fig_exposure_by_popularity_decile,
                          FX.fig_user_popularity_calibration_error,
                          FX.fig_exposure_gini_by_method]),
        (IN["embedding"], [FX.fig_embedding_backbone_sensitivity]),
        (IN["scale"], [FX.fig_scale_stress_latency,
                       FX.fig_scale_stress_memory,
                       FX.fig_scale_stress_index_size]),
    ]
    for csv_path, fig_fns in module_figures:
        df = _read_optional(csv_path)
        if df is None:
            continue
        for fn in fig_fns:
            written = fn(
                df, out_dir, write_mode=write_mode,
                source_files=[csv_path], script=SCRIPT)
            for path in written:
                print(f"[{SCRIPT}] output path: {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate every paper figure (PNG+PDF, 300 dpi, source "
                    "sidecars) from the canonical results directories.")
    ap.add_argument("--out_dir", default=RESULTS["paper_figures"])
    ap.add_argument("--write_mode", default="replace",
                    choices=["fail_if_exists", "replace"],
                    help="regenerated presentation figures; default replace")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] starting...")
    for p in IN.values():
        print(f"[{SCRIPT}] input path: {p}")
    print(f"[{SCRIPT}] output path: {out_dir}")

    # main summary is CRITICAL: without it there are no headline figures
    if not IN["summary"].is_file():
        print(f"[{SCRIPT}] ERROR: canonical input missing: {IN['summary']}\n"
              f"  produce it first with: python src/run_revision_experiments.py"
              f" --config configs/main_cpu.yml")
        sys.exit(1)
    write_mode = resolve_write_mode(args.write_mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        generate_figures(out_dir, write_mode)
    except ResultExistsError as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
