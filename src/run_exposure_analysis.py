"""Run the exposure-proxy analysis over saved per-query evaluation artifacts.

Pure post-processing over results/main/perquery/*.npz (written by
eval_modalities.py) — no searches are re-run. Provider-side exposure is
computed only when item metadata exists at data/{stem}_item_meta.csv with
columns item_id,provider; otherwise a clear warning is emitted and the
provider rows are omitted.

Outputs:
    results/exposure_analysis/exposure_analysis_all.csv       (long format)
    results/paper_tables/exposure_analysis_summary.csv/.tex   (+ .md)
    results/figures_paper/exposure_by_popularity_decile.pdf
    results/figures_paper/user_popularity_calibration_error.pdf
    results/figures_paper/exposure_gini_by_method.pdf
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import exposure_analysis as EA
from utils.paths import RESULTS, dataset_stem, emb_dir
from utils.reporting import write_table
from utils.figures_ext import (fig_exposure_by_popularity_decile,
                               fig_user_popularity_calibration_error,
                               fig_exposure_gini_by_method)

SCRIPT = "run_exposure_analysis"
REQUIRED_KEYS = ["exposure_counts_at_k", "exposure_counts_at_100",
                 "pop_counts", "recs_at_k", "hist_pop_mean"]


def _load_provider_map(dataset, weighting, dim):
    """Optional provider metadata: data/{stem}_item_meta.csv (item_id,provider),
    aligned to the embedding's item index via item_ids.npy."""
    meta_path = Path(f"data/{dataset_stem(dataset)}_item_meta.csv")
    ids_path = Path(emb_dir(dataset, weighting, dim)) / "item_ids.npy"
    if not meta_path.is_file():
        print(f"[{SCRIPT}] WARN: provider metadata {meta_path} not available; "
              f"provider-side exposure proxy skipped for {dataset}. "
              f"(fairness_scope=provider_proxy_if_metadata_available rows omitted)")
        return None
    if not ids_path.is_file():
        print(f"[{SCRIPT}] WARN: {ids_path} missing; cannot align provider metadata.")
        return None
    meta = pd.read_csv(meta_path)
    if "item_id" not in meta.columns or "provider" not in meta.columns:
        print(f"[{SCRIPT}] WARN: {meta_path} lacks item_id/provider columns; skipped.")
        return None
    prov_by_id = dict(zip(meta["item_id"].astype(str), meta["provider"]))
    ids = np.load(ids_path, allow_pickle=True)
    return [prov_by_id.get(str(x)) for x in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perquery_dir", default=RESULTS["perquery"])
    ap.add_argument("--tail_frac", type=float, default=0.2)
    ap.add_argument("--head_frac", type=float, default=0.1)
    ap.add_argument("--dim", type=int, default=128,
                    help="embedding dim (to locate item_ids for provider metadata)")
    ap.add_argument("--out_dir", default=RESULTS["exposure_analysis"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.perquery_dir}")
    print(f"[{SCRIPT}] output path: {out_dir}")

    files = sorted(Path(args.perquery_dir).glob("*.npz"))
    if not files:
        print(f"[{SCRIPT}] WARN: no per-query files in {args.perquery_dir}. "
              f"Run eval_modalities.py / run_revision_experiments.py first.")
        print(f"[{SCRIPT}] completed.")
        return

    provider_cache = {}
    rows = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            meta = json.loads(str(z["meta"]))
            missing = [k for k in REQUIRED_KEYS if k not in z.files]
            if missing:
                print(f"[{SCRIPT}] WARN: {f.name} lacks {missing} "
                      f"(produced by an older eval run); skipping.")
                continue
            arrays = {k: z[k] for k in REQUIRED_KEYS}

        key = (meta["dataset"], meta["weighting"])
        if key not in provider_cache:
            provider_cache[key] = _load_provider_map(meta["dataset"],
                                                     meta["weighting"], args.dim)
        run_rows = EA.analyze_run(arrays, tail_frac=args.tail_frac,
                                  head_frac=args.head_frac,
                                  item_provider=provider_cache[key])
        for r in run_rows:
            rows.append({"dataset": meta["dataset"], "weighting": meta["weighting"],
                         "modality": meta["modality"], "method": meta["method"], **r})

    if not rows:
        print(f"[{SCRIPT}] WARN: no analyzable runs found.")
        print(f"[{SCRIPT}] completed.")
        return

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "exposure_analysis_all.csv"
    if all_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {all_path}")
    df.to_csv(all_path, index=False)
    print(f"[{SCRIPT}] output path: {all_path}")

    # summary: headline metrics wide, one row per run; overall scope disclaimer
    headline = df[df["metric"].isin(["long_tail_exposure", "long_tail_uplift",
                                     "gini_exposure", "coverage",
                                     "user_popularity_calibration_error"])].copy()
    headline["metric_k"] = headline.apply(
        lambda r: f"{r['metric']}_at_{int(r['k'])}" if pd.notna(r["k"]) else r["metric"],
        axis=1)
    summary = headline.pivot_table(index=["dataset", "weighting", "modality", "method"],
                                   columns="metric_k", values="value").reset_index()
    summary["fairness_scope"] = "not_full_fairness_evaluation"
    written = write_table(summary,
                          Path(RESULTS["paper_tables"]) / "exposure_analysis_summary")
    for p in written:
        print(f"[{SCRIPT}] output path: {p}")

    fig_dir = Path(RESULTS["figures_paper"])
    for p in (fig_exposure_by_popularity_decile(df, fig_dir)
              + fig_user_popularity_calibration_error(df, fig_dir)
              + fig_exposure_gini_by_method(df, fig_dir)):
        print(f"[{SCRIPT}] output path: {p}")

    print(f"[{SCRIPT}] NOTE: all quantities are exposure *proxies*; this is "
          f"not a full fairness evaluation (see fairness_scope column).")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
