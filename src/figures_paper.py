"""Generate the paper figures (PNG + PDF) from measured result CSVs.

Pure post-processing over existing CSVs; missing inputs are skipped with a
warning. Uses the Agg backend (headless, CPU-only).

Figures (results/figures_paper/):
    fig_latency_vs_ndcg          quality/latency trade-off per modality
    fig_calibration_sensitivity  calibrated param + latency vs recall target
    fig_long_tail_exposure       long_tail_exposure by method and dataset
    fig_effect_sizes             Cliff's delta vs Flat (NDCG)
    fig_scaling                  latency vs catalog size (if scaling ran)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.paths import RESULTS

SCRIPT = "figures_paper"


def _read(path):
    p = Path(path)
    if not p.is_file():
        print(f"[{SCRIPT}] WARN: {p} not found; skipping dependent figures.")
        return None
    return pd.read_csv(p)


def _save(fig, out_dir, name):
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    for p in paths:
        print(f"[{SCRIPT}] output path: {p}")


def fig_latency_vs_ndcg(summary, out_dir):
    modalities = sorted(summary["modality"].dropna().unique())
    fig, axes = plt.subplots(1, max(1, len(modalities)),
                             figsize=(5.5 * max(1, len(modalities)), 4.2), squeeze=False)
    for ax, mod in zip(axes[0], modalities):
        sub = summary[summary["modality"] == mod]
        for method in sorted(sub["method"].unique()):
            s = sub[sub["method"] == method]
            ax.scatter(s["latency_p95_ms"], s["ndcg_at_k_mean"], label=method, s=40)
        ax.set_xscale("log")
        ax.set_xlabel("Latency p95 (ms, log)")
        ax.set_ylabel("NDCG@k (mean)")
        ax.set_title(f"Modality: {mod.upper()}")
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=8)
    fig.suptitle("Quality vs latency at the calibrated operating point")
    _save(fig, out_dir, "fig_latency_vs_ndcg")


def fig_calibration_sensitivity(cal, out_dir):
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
    _save(fig, out_dir, "fig_calibration_sensitivity")


def fig_long_tail_exposure(summary, out_dir):
    sub = summary.dropna(subset=["long_tail_exposure"])
    datasets = sorted(sub["dataset"].unique())
    fig, axes = plt.subplots(1, max(1, len(datasets)),
                             figsize=(4.2 * max(1, len(datasets)), 4.0), squeeze=False)
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
    _save(fig, out_dir, "fig_long_tail_exposure")


def fig_effect_sizes(effects, out_dir):
    sub = effects[effects["metric"] == "ndcg"].copy()
    if sub.empty:
        print(f"[{SCRIPT}] WARN: no NDCG effect sizes; skipping fig_effect_sizes.")
        return
    sub["label"] = (sub["dataset"] + "/" + sub["modality"] + "/" + sub["method"])
    sub = sub.sort_values("cliffs_delta")
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(sub) + 1.5))
    ax.barh(sub["label"], sub["cliffs_delta"])
    for thr in (-0.147, 0.147):
        ax.axvline(thr, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Cliff's delta vs Flat (NDCG@k); dashed = negligible band")
    ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Effect sizes of ANN vs exact search")
    _save(fig, out_dir, "fig_effect_sizes")


def fig_scaling(scaling, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for method, g in scaling.groupby("method"):
        g = g.sort_values("n_items")
        ax.plot(g["n_items"], g["latency_p95_ms"], marker="o", label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Catalog size (items, log)")
    ax.set_ylabel("Latency p95 (ms, log)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.suptitle("Synthetic scaling at fixed calibration target")
    _save(fig, out_dir, "fig_scaling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", default=f"{RESULTS['main']}/summary_main.csv")
    ap.add_argument("--calibration_csv",
                    default=f"{RESULTS['calibration_sensitivity']}/calibration_sensitivity.csv")
    ap.add_argument("--effect_sizes_csv", default=f"{RESULTS['effect_sizes']}/effect_sizes.csv")
    ap.add_argument("--scaling_csv", default=f"{RESULTS['scaling']}/scaling.csv")
    ap.add_argument("--out_dir", default=RESULTS["figures_paper"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] starting...")
    for p in (args.summary_csv, args.calibration_csv, args.effect_sizes_csv, args.scaling_csv):
        print(f"[{SCRIPT}] input path: {p}")
    print(f"[{SCRIPT}] output path: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _read(args.summary_csv)
    if summary is not None:
        fig_latency_vs_ndcg(summary, out_dir)
        fig_long_tail_exposure(summary, out_dir)

    cal = _read(args.calibration_csv)
    if cal is not None:
        fig_calibration_sensitivity(cal, out_dir)

    effects = _read(args.effect_sizes_csv)
    if effects is not None:
        fig_effect_sizes(effects, out_dir)

    scaling = Path(args.scaling_csv)
    if scaling.is_file():
        fig_scaling(pd.read_csv(scaling), out_dir)
    else:
        print(f"[{SCRIPT}] INFO: {scaling} not found (scaling study optional).")

    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
