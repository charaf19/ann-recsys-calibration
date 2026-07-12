"""Run PQ diagnostics over built indexes (Flat vs Flat-PQ vs IVF-PQ).

Requires embeddings + indexes from run_revision_experiments.py and,
optionally, the interactions CSVs for popularity deciles and
results/main/summary_main.csv for delta-NDCG correlations. Interpretation
labels are diagnostic evidence only — no claim of implicit regularization
is made or implied.

Outputs:
    results/analyses/pq_diagnostics/pq_diagnostics_all.csv       (long format)
    results/analyses/pq_diagnostics/pq_diagnostics_summary.csv   (wide format)
(presentation tables/figures come from tables_paper.py / figures_paper.py)
"""
import argparse
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

import pq_diagnostics as PQ
from utils.ann_io import resolve_index_path, load_ann_index, build_exact_index
from utils.common import set_global_seed
from utils.config import load_config, cfg_get, ConfigError
from utils.paths import dataset_csv, emb_dir, index_dir, RESULTS
from utils.result_io import (preflight_output, write_dataframe_atomic,
                             ResultExistsError)

SCRIPT = "run_pq_diagnostics"
DEFAULT_CONFIG = "configs/main_cpu.yml"
PQ_METHODS = ["flatpq", "ivfpq"]
KEY = ["dataset", "weighting", "dim", "method", "metric", "decile", "seed"]


def _popularity(csv_path, item_ids_path):
    if not Path(csv_path).is_file() or not Path(item_ids_path).is_file():
        return None
    ids = np.load(item_ids_path, allow_pickle=True)
    id2idx = {str(ids[i]): i for i in range(len(ids))}
    df = pd.read_csv(csv_path, usecols=["item_id"])
    pop = np.zeros(len(ids), dtype=np.int64)
    for item, cnt in df["item_id"].astype(str).value_counts().items():
        j = id2idx.get(item)
        if j is not None:
            pop[j] += int(cnt)
    return pop


def _delta_ndcg(summary, dataset, weighting, method):
    if summary is None:
        return float("nan")
    flat = summary[(summary["dataset"] == dataset)
                   & (summary["weighting"] == weighting)
                   & (summary["method"] == "flat")]
    pq = summary[(summary["dataset"] == dataset)
                 & (summary["weighting"] == weighting)
                 & (summary["method"] == method)]
    if flat.empty or pq.empty:
        return float("nan")
    # average over modalities present
    return float(pq["ndcg_at_k_mean"].mean() - flat["ndcg_at_k_mean"].mean())


def _diagnostic_nprobe(summary, dataset, weighting, method):
    """Return the calibrated I2I nprobe for item-vector diagnostics."""
    if summary is None or "nprobe" not in summary.columns:
        return 1

    rows = summary[
        (summary["dataset"] == dataset)
        & (summary["weighting"] == weighting)
        & (summary["method"] == method)
    ]

    if "modality" in rows.columns:
        i2i_rows = rows[rows["modality"] == "i2i"]
        if not i2i_rows.empty:
            rows = i2i_rows

    values = pd.to_numeric(rows["nprobe"], errors="coerce").dropna()
    if values.empty:
        return 1

    return max(1, int(values.max()))


def main():
    ap = argparse.ArgumentParser(
        description="PQ codec diagnostics: reconstruction error, neighbor "
                    "overlap, score variance, popularity-decile effects, "
                    "and long-tail exposure shift for flatpq/ivfpq.")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="experiment YAML (datasets, weighting, dim, seed)")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None, choices=PQ_METHODS)
    ap.add_argument("--weighting", default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--sample_vectors", type=int, default=5000)
    ap.add_argument("--sample_queries", type=int, default=1000)
    ap.add_argument("--tail_frac", type=float, default=0.2)
    ap.add_argument("--summary_csv", default=None,
                    help="main summary for delta-NDCG correlations "
                         "(default: results/main/summary_main.csv)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_dir", default=RESULTS["pq_diagnostics"])
    ap.add_argument("--write_mode", default="fail_if_exists",
                    choices=["fail_if_exists", "replace", "merge"])
    args = ap.parse_args()

    try:
        cfg = load_config(args.config, cli_overrides={
            "datasets": args.datasets,
            "embedding.weighting": args.weighting,
            "embedding.dim": args.dim,
            "reproducibility.seed": args.seed,
        })
    except ConfigError as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    datasets = list(cfg_get(cfg, "datasets", required=True))
    methods = args.methods or PQ_METHODS
    weighting = cfg_get(cfg, "embedding.weighting", required=True)
    dim = cfg_get(cfg, "embedding.dim", type=int, required=True)
    seed = cfg_get(cfg, "reproducibility.seed", type=int, default=42)

    summary_csv = args.summary_csv or str(Path(RESULTS["main"])
                                          / "summary_main.csv")
    out_dir = Path(args.out_dir)
    all_path = out_dir / "pq_diagnostics_all.csv"
    summary_path = out_dir / "pq_diagnostics_summary.csv"
    try:
        preflight_output(all_path, args.write_mode)
        preflight_output(summary_path, args.write_mode)
    except (ResultExistsError, ValueError) as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {summary_csv}")
    print(f"[{SCRIPT}] output path: {out_dir}")

    set_global_seed(seed)
    summary = pd.read_csv(summary_csv) if Path(summary_csv).is_file() else None
    if summary is None:
        print(f"[{SCRIPT}] WARN: {summary_csv} missing; delta-NDCG "
              f"correlations will be NaN (insufficient_evidence labels).")

    long_rows, summary_rows = [], []
    for dataset in datasets:
        emb = emb_dir(dataset, weighting, dim)
        vec_path = Path(emb) / "item_vecs.npy"
        if not vec_path.is_file():
            print(f"[{SCRIPT}] WARN: missing embeddings {vec_path}; "
                  f"skipping {dataset}.")
            continue
        item_vecs = np.load(vec_path).astype("float32")
        N, D = item_vecs.shape
        rng = np.random.default_rng(seed)
        v_idx = rng.choice(N, size=min(args.sample_vectors, N), replace=False)
        q_idx = rng.choice(N, size=min(args.sample_queries, N), replace=False)
        X = item_vecs[v_idx]
        Q = item_vecs[q_idx]

        pop = _popularity(dataset_csv(dataset), Path(emb) / "item_ids.npy")
        deciles = PQ.popularity_deciles(pop) if pop is not None else None

        exact = build_exact_index(item_vecs)
        flat_faiss = faiss.IndexFlatL2(D)
        flat_faiss.add(item_vecs)
        flat_score_var = PQ.score_variance(flat_faiss, Q, k=min(100, N))

        for method in methods:
            idx_dir = index_dir(dataset, weighting, dim, method)
            if not Path(idx_dir).exists():
                print(f"[{SCRIPT}] WARN: missing index {idx_dir}; skipping.")
                continue
            print(f"[{SCRIPT}] diagnosing {dataset}/{method}")
            fpath = resolve_index_path(idx_dir)
            raw_index = faiss.read_index(str(fpath))

            nprobe_used = None
            if method == "ivfpq":
                nprobe_used = _diagnostic_nprobe(
                    summary, dataset, weighting, method
                )
                ivf = faiss.extract_index_ivf(raw_index)
                ivf.nprobe = int(nprobe_used)
                print(
                    f"[{SCRIPT}] using calibrated I2I "
                    f"nprobe={nprobe_used} for {dataset}/{method}"
                )

            ann = load_ann_index(
                idx_dir, D, N, nprobe=nprobe_used
            )

            # 1-3: codec geometry
            X_hat = PQ.reconstruct(raw_index, X)
            rerr = PQ.reconstruction_error(X, X_hat)
            ndist = PQ.norm_distortion(X, X_hat)
            pdist = PQ.pairwise_distance_distortion(X, X_hat, seed=seed)

            # 4: neighbor overlap
            ov10 = PQ.neighbor_overlap(exact, ann, Q, k=min(10, N))
            ov100 = PQ.neighbor_overlap(exact, ann, Q, k=min(100, N))

            # 5: score variance
            pq_score_var = PQ.score_variance(raw_index, Q, k=min(100, N))

            # 6: popularity-decile effects
            decile_rerr, decile_ov10 = {}, {}
            if deciles is not None:
                decile_rerr = PQ.per_decile(rerr, deciles[v_idx])
                decile_ov10 = PQ.per_decile(ov10, deciles[q_idx])

            # 7: long-tail exposure shift
            shift = {}
            if pop is not None:
                shift = PQ.exposure_shift(exact, ann, Q, pop, k=10,
                                          tail_frac=args.tail_frac)

            delta = _delta_ndcg(summary, dataset, weighting, method)
            label = PQ.interpretation_label(delta)

            base = {"dataset": dataset, "weighting": weighting,
                    "dim": dim, "method": method, "seed": seed}
            headline = {
                "reconstruction_error_rel_mean": float(rerr.mean()),
                "norm_distortion_mean": float(np.mean(np.abs(ndist))),
                "pairwise_distance_distortion": pdist,
                "neighbor_overlap_at_10": float(ov10.mean()),
                "neighbor_overlap_at_100": float(ov100.mean()),
                "score_variance_flat": flat_score_var,
                "score_variance_pq": pq_score_var,
                "score_variance_ratio": (pq_score_var / flat_score_var
                                         if flat_score_var > 0 else float("nan")),
                **shift,
                "delta_ndcg_vs_flat": delta,
                "interpretation_label": label,
            }
            summary_rows.append({
                **base,
                **headline,
                "nprobe_used": (
                    int(nprobe_used)
                    if nprobe_used is not None
                    else -1
                ),
                "n_sample_vectors": len(v_idx),
                "n_sample_queries": len(q_idx),
            })
            for metric, value in headline.items():
                if isinstance(value, (int, float)):
                    long_rows.append({**base, "metric": metric, "decile": None,
                                      "value": float(value)})
            for d, v in decile_rerr.items():
                long_rows.append({**base,
                                  "metric": "reconstruction_error_rel_decile",
                                  "decile": d, "value": v})
            for d, v in decile_ov10.items():
                long_rows.append({**base,
                                  "metric": "neighbor_overlap_at_10_decile",
                                  "decile": d, "value": v})

    if not summary_rows:
        print(f"[{SCRIPT}] ERROR: no PQ indexes diagnosed (build them first "
              f"with run_revision_experiments.py).")
        sys.exit(1)

    # 8-9: cross-dataset correlations with delta-NDCG
    sdf = pd.DataFrame(summary_rows)
    rho_rerr, n1 = PQ.safe_corr(sdf["reconstruction_error_rel_mean"],
                                sdf["delta_ndcg_vs_flat"])
    rho_ov, n2 = PQ.safe_corr(sdf["neighbor_overlap_at_10"],
                              sdf["delta_ndcg_vs_flat"])
    sdf["corr_recon_error_vs_delta_ndcg"] = rho_rerr
    sdf["corr_neighbor_overlap_vs_delta_ndcg"] = rho_ov
    sdf["corr_n_points"] = min(n1, n2)
    if min(n1, n2) < 3:
        print(f"[{SCRIPT}] WARN: <3 (dataset,method) points with delta-NDCG; "
              f"correlations reported as NaN.")

    out_dir.mkdir(parents=True, exist_ok=True)
    long_df = pd.DataFrame(long_rows)

    # Global metrics do not belong to a popularity decile. Represent that
    # dimension explicitly because natural-key columns cannot contain nulls.
    long_df["decile"] = (
        pd.to_numeric(long_df["decile"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    long_df["dim"] = (
        pd.to_numeric(long_df["dim"], errors="raise").astype(int)
    )
    long_df["seed"] = (
        pd.to_numeric(long_df["seed"], errors="raise").astype(int)
    )

    write_dataframe_atomic(long_df, all_path, mode=args.write_mode,
                           key=KEY, sort_by=KEY)
    print(f"[{SCRIPT}] output path: {all_path} ({len(long_df)} rows)")

    write_dataframe_atomic(sdf, summary_path, mode=args.write_mode,
                           key=["dataset", "weighting", "dim", "method", "seed"],
                           sort_by=["dataset", "method"])
    print(f"[{SCRIPT}] output path: {summary_path} ({len(sdf)} rows)")

    print(f"[{SCRIPT}] NOTE: labels are diagnostic evidence only; no claim of "
          f"implicit regularization is made or implied.")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
