"""Run the exposure-proxy analysis over saved per-query evaluation artifacts.

Pure post-processing over results/main/perquery/*.npz (written by
eval_modalities.py) — no searches are re-run. Every row carries a
fairness_scope value restricting interpretation to exposure/popularity
proxies; this is NOT a full fairness evaluation. Provider-side exposure is
computed only when item metadata exists at data/{stem}_item_meta.csv with
columns item_id,provider.

Output: results/analyses/exposure/exposure_analysis_all.csv (long format)
(presentation tables/figures come from tables_paper.py / figures_paper.py)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import exposure_analysis as EA
from utils.paths import RESULTS, dataset_stem, emb_dir
from utils.result_io import (preflight_output, write_dataframe_atomic,
                             ResultExistsError)

SCRIPT = "run_exposure_analysis"
REQUIRED_KEYS = ["exposure_counts_at_k", "exposure_counts_at_100",
                 "pop_counts", "recs_at_k", "hist_pop_mean"]
KEY = ["dataset", "weighting", "dim", "modality", "method", "metric", "k",
       "decile", "group", "seed"]


def _load_provider_map(dataset, weighting, dim):
    """Optional provider metadata: data/{stem}_item_meta.csv (item_id,provider),
    aligned to the embedding's item index via item_ids.npy."""
    meta_path = Path(f"data/{dataset_stem(dataset)}_item_meta.csv")
    ids_path = Path(emb_dir(dataset, weighting, dim)) / "item_ids.npy"
    if not meta_path.is_file():
        print(f"[{SCRIPT}] WARN: provider metadata {meta_path} not available; "
              f"provider-side exposure proxy skipped for {dataset}.")
        return None
    if not ids_path.is_file():
        print(f"[{SCRIPT}] WARN: {ids_path} missing; cannot align provider "
              f"metadata.")
        return None
    meta = pd.read_csv(meta_path)
    if "item_id" not in meta.columns or "provider" not in meta.columns:
        print(f"[{SCRIPT}] WARN: {meta_path} lacks item_id/provider columns; "
              f"skipped.")
        return None
    prov_by_id = dict(zip(meta["item_id"].astype(str), meta["provider"]))
    ids = np.load(ids_path, allow_pickle=True)
    return [prov_by_id.get(str(x)) for x in ids]


def main():
    ap = argparse.ArgumentParser(
        description="Exposure-proxy analysis over the per-query evaluation "
                    "artifacts (long-tail exposure, popularity calibration, "
                    "decile shares). Proxies only; not a fairness audit.")
    ap.add_argument("--perquery_dir", default=RESULTS["perquery"])
    ap.add_argument("--tail_frac", type=float, default=0.2)
    ap.add_argument("--head_frac", type=float, default=0.1)
    ap.add_argument("--out_dir", default=RESULTS["exposure_analysis"])
    ap.add_argument("--write_mode", default="fail_if_exists",
                    choices=["fail_if_exists", "replace", "merge"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    all_path = out_dir / "exposure_analysis_all.csv"
    try:
        preflight_output(all_path, args.write_mode)
    except (ResultExistsError, ValueError) as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.perquery_dir}")
    print(f"[{SCRIPT}] output path: {out_dir}")

    files = sorted(Path(args.perquery_dir).glob("*.npz"))
    if not files:
        print(f"[{SCRIPT}] ERROR: no per-query files in {args.perquery_dir}. "
              f"Run run_revision_experiments.py first.")
        sys.exit(1)

    provider_cache = {}
    rows = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            meta = json.loads(str(z["meta"]))
            missing = [k for k in REQUIRED_KEYS if k not in z.files]
            if missing:
                print(f"[{SCRIPT}] WARN: {f.name} lacks {missing}; skipping.")
                continue
            arrays = {k: z[k] for k in REQUIRED_KEYS}

        dim = meta.get("dim", 128)
        key = (meta["dataset"], meta["weighting"], dim)
        if key not in provider_cache:
            provider_cache[key] = _load_provider_map(meta["dataset"],
                                                     meta["weighting"], dim)
        run_rows = EA.analyze_run(arrays, tail_frac=args.tail_frac,
                                  head_frac=args.head_frac,
                                  item_provider=provider_cache[key])
        for r in run_rows:
            rows.append({"dataset": meta["dataset"],
                         "weighting": meta["weighting"],
                         "dim": dim,
                         "modality": meta["modality"],
                         "method": meta["method"],
                         "seed": meta.get("seed"),
                         **r})

    if not rows:
        print(f"[{SCRIPT}] ERROR: no analyzable runs found.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Natural-key columns cannot contain null values. These dimensions are
    # intentionally absent for metrics where they do not apply, so use explicit
    # sentinel values while preserving metric-level uniqueness.
    df["k"] = pd.to_numeric(df["k"], errors="coerce").fillna(-1).astype(int)
    df["decile"] = (
        pd.to_numeric(df["decile"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    df["group"] = df["group"].fillna("__all__").astype(str)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").fillna(-1).astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_dataframe_atomic(df, all_path, mode=args.write_mode,
                           key=KEY, sort_by=KEY)
    print(f"[{SCRIPT}] output path: {all_path} ({len(df)} rows)")
    print(f"[{SCRIPT}] NOTE: all quantities are exposure *proxies*; this is "
          f"not a full fairness evaluation (see fairness_scope column).")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
