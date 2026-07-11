"""Shared plotting functions for the reviewer-limitation modules.

Used by run_pq_diagnostics.py, run_exposure_analysis.py,
run_embedding_backbone_sensitivity.py, run_scale_stress.py, and
figures_paper.py so figures are identical no matter which entry point
generates them. Every function takes a DataFrame + output directory,
writes <name>.pdf and <name>.png, and returns the written paths.
Empty/incompatible inputs return [] with a warning instead of raising.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.provenance import sources_sidecar_path, write_sources_sidecar
from utils.result_io import (atomic_output_path, preflight_output,
                             resolve_write_mode)

SCRIPT = "figures_ext"


def save_figure_artifacts(fig, fig_dir, name, *, write_mode="replace",
                          source_files=None, script=SCRIPT, cfg_hash=None):
    """Atomically publish a figure as 300-DPI PNG and vector PDF.

    Binary figure formats do not have meaningful merge semantics.  If source
    files are supplied, each format receives an independent provenance
    sidecar retaining the full artifact filename.
    """
    write_mode = resolve_write_mode(write_mode)
    if write_mode == "merge":
        raise ValueError("merge mode is not supported for rendered figures")
    fig_dir = Path(fig_dir)
    paths = [fig_dir / f"{name}.{ext}" for ext in ("png", "pdf")]
    sources = None if source_files is None else list(source_files)
    planned = list(paths)
    if sources is not None:
        planned.extend(sources_sidecar_path(path) for path in paths)
    for path in planned:
        preflight_output(path, write_mode)

    try:
        for path in paths:
            with atomic_output_path(path, mode=write_mode) as temp_path:
                fig.savefig(temp_path, format=path.suffix.lstrip("."),
                            dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)

    if sources is not None:
        for path in paths:
            write_sources_sidecar(path, sources, script, cfg_hash=cfg_hash,
                                  mode=write_mode)
    return [str(path) for path in paths]


def _save(fig, fig_dir, name, **artifact_options):
    return save_figure_artifacts(fig, fig_dir, name, **artifact_options)


def _guard(df, cols, name):
    if df is None or df.empty or any(c not in df.columns for c in cols):
        print(f"[{SCRIPT}] WARN: input for {name} missing/empty; skipping.")
        return False
    return True


# ----------------------------
# PQ diagnostics
# ----------------------------

def fig_pq_reconstruction_error_by_dataset(summary_df, fig_dir,
                                            **artifact_options):
    if not _guard(summary_df, ["dataset", "method", "reconstruction_error_rel_mean"],
                  "pq_reconstruction_error_by_dataset"):
        return []
    piv = summary_df.pivot_table(index="dataset", columns="method",
                                 values="reconstruction_error_rel_mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("Relative reconstruction error (mean)")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("PQ reconstruction error by dataset")
    return _save(fig, fig_dir, "pq_reconstruction_error_by_dataset",
                 **artifact_options)


def fig_pq_neighbor_overlap_vs_quality_delta(summary_df, fig_dir,
                                              **artifact_options):
    if not _guard(summary_df, ["neighbor_overlap_at_10", "delta_ndcg_vs_flat"],
                  "pq_neighbor_overlap_vs_quality_delta"):
        return []
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for method, g in summary_df.groupby("method"):
        ax.scatter(g["neighbor_overlap_at_10"], g["delta_ndcg_vs_flat"],
                   label=method, s=60)
        for _, r in g.iterrows():
            ax.annotate(r["dataset"], (r["neighbor_overlap_at_10"],
                                       r["delta_ndcg_vs_flat"]), fontsize=7)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Top-10 neighbor overlap with Flat")
    ax.set_ylabel("Delta NDCG@k vs Flat")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Neighbor overlap vs quality delta (diagnostic, not causal)")
    return _save(fig, fig_dir, "pq_neighbor_overlap_vs_quality_delta",
                 **artifact_options)


def fig_pq_popularity_decile_effect(long_df, fig_dir, **artifact_options):
    if not _guard(long_df, ["metric", "decile", "value"],
                  "pq_popularity_decile_effect"):
        return []
    sub = long_df[long_df["metric"].isin(
        ["reconstruction_error_rel_decile", "neighbor_overlap_at_10_decile"])]
    sub = sub.dropna(subset=["decile"])
    if sub.empty:
        print(f"[{SCRIPT}] WARN: no per-decile PQ rows; skipping decile figure.")
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(
            axes,
            ["reconstruction_error_rel_decile", "neighbor_overlap_at_10_decile"],
            ["Reconstruction error", "Top-10 overlap with Flat"]):
        m = sub[sub["metric"] == metric]
        for (dataset, method), g in m.groupby(["dataset", "method"]):
            g = g.sort_values("decile")
            ax.plot(g["decile"], g["value"], marker="o",
                    label=f"{dataset}/{method}")
        ax.set_xlabel("Popularity decile (0 = least popular)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.suptitle("PQ effects across item-popularity deciles")
    return _save(fig, fig_dir, "pq_popularity_decile_effect",
                 **artifact_options)


# ----------------------------
# Exposure analysis
# ----------------------------

def fig_exposure_by_popularity_decile(long_df, fig_dir, **artifact_options):
    if not _guard(long_df, ["metric", "decile", "value"],
                  "exposure_by_popularity_decile"):
        return []
    sub = long_df[long_df["metric"] == "exposure_share_decile"].dropna(subset=["decile"])
    if sub.empty:
        print(f"[{SCRIPT}] WARN: no decile exposure rows; skipping.")
        return []
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (dataset, modality, method), g in sub.groupby(["dataset", "modality", "method"]):
        g = g.sort_values("decile")
        ax.plot(g["decile"], g["value"], marker="o", alpha=0.8,
                label=f"{dataset}/{modality}/{method}")
    ax.set_xlabel("Item popularity decile (0 = least popular)")
    ax.set_ylabel("Share of top-k exposure")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.suptitle("Exposure by popularity decile (exposure proxy)")
    return _save(fig, fig_dir, "exposure_by_popularity_decile",
                 **artifact_options)


def fig_user_popularity_calibration_error(df, fig_dir, **artifact_options):
    if not _guard(df, ["method", "value"], "user_popularity_calibration_error"):
        return []
    sub = df[df["metric"] == "user_popularity_calibration_error"] if "metric" in df.columns else df
    if sub.empty:
        print(f"[{SCRIPT}] WARN: no calibration-error rows; skipping.")
        return []
    piv = sub.pivot_table(index="method", columns="dataset", values="value")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("User popularity calibration error\n(mean |log1p rec pop − log1p hist pop|)")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("User-level popularity calibration error (proxy)")
    return _save(fig, fig_dir, "user_popularity_calibration_error",
                 **artifact_options)


def fig_exposure_gini_by_method(df, fig_dir, **artifact_options):
    if not _guard(df, ["method", "value"], "exposure_gini_by_method"):
        return []
    sub = df[df["metric"] == "gini_exposure"] if "metric" in df.columns else df
    if sub.empty:
        print(f"[{SCRIPT}] WARN: no exposure-gini rows; skipping.")
        return []
    piv = sub.pivot_table(index="method", columns=["dataset", "modality"], values="value")
    fig, ax = plt.subplots(figsize=(7, 4))
    piv.plot(kind="bar", ax=ax)
    ax.set_ylabel("Exposure Gini (0 = uniform)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=6)
    fig.suptitle("Exposure concentration by index method")
    return _save(fig, fig_dir, "exposure_gini_by_method",
                 **artifact_options)


# ----------------------------
# Embedding backbone sensitivity
# ----------------------------

def fig_embedding_backbone_sensitivity(df, fig_dir, **artifact_options):
    if not _guard(df, ["backbone", "method", "ndcg_at_10"],
                  "embedding_backbone_sensitivity"):
        return []
    datasets = sorted(df["dataset"].unique()) if "dataset" in df.columns else ["all"]
    fig, axes = plt.subplots(1, max(1, len(datasets)),
                             figsize=(5 * max(1, len(datasets)), 4.2), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = df[df["dataset"] == dataset] if "dataset" in df.columns else df
        piv = sub.pivot_table(index="backbone", columns="method", values="ndcg_at_10")
        piv.plot(kind="bar", ax=ax)
        ax.set_ylabel("NDCG@10")
        ax.set_title(dataset)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=7)
    fig.suptitle("ANN method quality across embedding backbones")
    fig.tight_layout()
    return _save(fig, fig_dir, "embedding_backbone_sensitivity",
                 **artifact_options)


# ----------------------------
# Scale stress
# ----------------------------

def _scale_stress_lines(df, ycol, ylabel, name, fig_dir, logy=True,
                        **artifact_options):
    if not _guard(df, ["n_items", "dim", "method", ycol], name):
        return []
    dims = sorted(df["dim"].unique())
    fig, axes = plt.subplots(1, max(1, len(dims)),
                             figsize=(4.5 * max(1, len(dims)), 4), squeeze=False)
    for ax, dim in zip(axes[0], dims):
        sub = df[df["dim"] == dim]
        for method, g in sub.groupby("method"):
            g = g.sort_values("n_items")
            ax.plot(g["n_items"], g[ycol], marker="o", label=method)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Catalog size (items)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"dim={dim}")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=7)
    fig.suptitle(f"{ylabel} vs synthetic catalog size (quality_measured=false)")
    fig.tight_layout()
    return _save(fig, fig_dir, name, **artifact_options)


def fig_scale_stress_latency(df, fig_dir, **artifact_options):
    return _scale_stress_lines(df, "latency_p95_ms", "Latency p95 (ms)",
                               "scale_stress_latency", fig_dir,
                               **artifact_options)


def fig_scale_stress_memory(df, fig_dir, **artifact_options):
    return _scale_stress_lines(df, "rss_mb_after", "Process RSS (MB)",
                               "scale_stress_memory", fig_dir, logy=False,
                               **artifact_options)


def fig_scale_stress_index_size(df, fig_dir, **artifact_options):
    return _scale_stress_lines(df, "index_size_mb", "Index size on disk (MB)",
                               "scale_stress_index_size", fig_dir,
                               **artifact_options)
