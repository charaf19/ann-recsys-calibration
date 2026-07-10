"""Run PQ diagnostics over built indexes (Flat vs Flat-PQ vs IVF-PQ).

Requires embeddings + indexes from run_revision_experiments.py (or manual
train_embeddings.py + build_index.py) and, optionally, the interactions CSVs
for popularity deciles and results/main/summary_main.csv for delta-NDCG
correlations. Missing artifacts are skipped with warnings.

Outputs:
    results/pq_diagnostics/pq_diagnostics_all.csv        (long format)
    results/paper_tables/pq_diagnostics_summary.csv/.tex (+ .md)
    results/figures_paper/pq_reconstruction_error_by_dataset.pdf
    results/figures_paper/pq_neighbor_overlap_vs_quality_delta.pdf
    results/figures_paper/pq_popularity_decile_effect.pdf
"""
import argparse
import faiss
import numpy as np
import pandas as pd
from pathlib import Path

import pq_diagnostics as PQ
from utils.ann_io import resolve_index_path, load_ann_index, build_exact_index
from utils.common import set_global_seed
from utils.paths import dataset_csv, emb_dir, index_dir, RESULTS, first_existing
from utils.reporting import write_table
from utils.figures_ext import (fig_pq_reconstruction_error_by_dataset,
                               fig_pq_neighbor_overlap_vs_quality_delta,
                               fig_pq_popularity_decile_effect)

SCRIPT = "run_pq_diagnostics"
PQ_METHODS = ["flatpq", "ivfpq"]


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
    flat = summary[(summary["dataset"] == dataset) & (summary["weighting"] == weighting)
                   & (summary["method"] == "flat")]
    pq = summary[(summary["dataset"] == dataset) & (summary["weighting"] == weighting)
                 & (summary["method"] == method)]
    if flat.empty or pq.empty:
        return float("nan")
    # average over modalities present
    return float(pq["ndcg_at_k_mean"].mean() - flat["ndcg_at_k_mean"].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["ml-1m", "ml-20m", "goodbooks"])
    ap.add_argument("--methods", nargs="*", default=PQ_METHODS,
                    choices=PQ_METHODS)
    ap.add_argument("--weighting", default="bm25")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--sample_vectors", type=int, default=5000)
    ap.add_argument("--sample_queries", type=int, default=1000)
    ap.add_argument("--tail_frac", type=float, default=0.2)
    ap.add_argument("--summary_csv", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=RESULTS["pq_diagnostics"])
    args = ap.parse_args()

    summary_csv = args.summary_csv or first_existing(
        f"{RESULTS['main']}/summary_main.csv", f"{RESULTS['main']}/main_results_all.csv")
    out_dir = Path(args.out_dir)

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {summary_csv}")
    print(f"[{SCRIPT}] output path: {out_dir}")

    set_global_seed(args.seed)
    summary = pd.read_csv(summary_csv) if Path(summary_csv).is_file() else None
    if summary is None:
        print(f"[{SCRIPT}] WARN: {summary_csv} missing; delta-NDCG correlations "
              f"will be NaN (insufficient_evidence labels).")

    long_rows, summary_rows = [], []
    for dataset in args.datasets:
        emb = emb_dir(dataset, args.weighting, args.dim)
        vec_path = Path(emb) / "item_vecs.npy"
        if not vec_path.is_file():
            print(f"[{SCRIPT}] WARN: missing embeddings {vec_path}; skipping {dataset}.")
            continue
        item_vecs = np.load(vec_path).astype("float32")
        N, D = item_vecs.shape
        rng = np.random.default_rng(args.seed)
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

        for method in args.methods:
            idx_dir = index_dir(dataset, args.weighting, args.dim, method)
            if not Path(idx_dir).exists():
                print(f"[{SCRIPT}] WARN: missing index {idx_dir}; skipping.")
                continue
            print(f"[{SCRIPT}] diagnosing {dataset}/{method}")
            fpath = resolve_index_path(idx_dir)
            raw_index = faiss.read_index(str(fpath))
            ann = load_ann_index(idx_dir, D, N)

            # 1-3: codec geometry
            X_hat = PQ.reconstruct(raw_index, X)
            rerr = PQ.reconstruction_error(X, X_hat)
            ndist = PQ.norm_distortion(X, X_hat)
            pdist = PQ.pairwise_distance_distortion(X, X_hat, seed=args.seed)

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

            delta = _delta_ndcg(summary, dataset, args.weighting, method)
            label = PQ.interpretation_label(delta)

            base = {"dataset": dataset, "weighting": args.weighting,
                    "dim": args.dim, "method": method, "seed": args.seed}
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
            summary_rows.append({**base, **headline,
                                 "n_sample_vectors": len(v_idx),
                                 "n_sample_queries": len(q_idx)})
            for metric, value in headline.items():
                if isinstance(value, (int, float)):
                    long_rows.append({**base, "metric": metric, "decile": None,
                                      "value": float(value)})
            for d, v in decile_rerr.items():
                long_rows.append({**base, "metric": "reconstruction_error_rel_decile",
                                  "decile": d, "value": v})
            for d, v in decile_ov10.items():
                long_rows.append({**base, "metric": "neighbor_overlap_at_10_decile",
                                  "decile": d, "value": v})

    if not summary_rows:
        print(f"[{SCRIPT}] WARN: no PQ indexes diagnosed (build them first).")
        print(f"[{SCRIPT}] completed.")
        return

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
    all_path = out_dir / "pq_diagnostics_all.csv"
    if all_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {all_path}")
    pd.DataFrame(long_rows).to_csv(all_path, index=False)
    print(f"[{SCRIPT}] output path: {all_path}")

    written = write_table(sdf, Path(RESULTS["paper_tables"]) / "pq_diagnostics_summary")
    for p in written:
        print(f"[{SCRIPT}] output path: {p}")

    fig_dir = Path(RESULTS["figures_paper"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    long_df = pd.DataFrame(long_rows)
    for p in (fig_pq_reconstruction_error_by_dataset(sdf, fig_dir)
              + fig_pq_neighbor_overlap_vs_quality_delta(sdf, fig_dir)
              + fig_pq_popularity_decile_effect(long_df, fig_dir)):
        print(f"[{SCRIPT}] output path: {p}")

    print(f"[{SCRIPT}] NOTE: labels are diagnostic evidence only; no claim of "
          f"implicit regularization is made or implied.")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
