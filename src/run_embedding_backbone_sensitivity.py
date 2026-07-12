"""Embedding backbone sensitivity study.

Reviewer concern addressed: "SVD-only embeddings". For each backbone
(svd_bm25 / svd_tfidf / svd_none / bpr_matrix_factorization, plus the
optional two_tower_mlp) this trains embeddings, builds indexes, calibrates,
and runs the U2I evaluation — then reports `ann_ranking_stability`: the
Spearman correlation of the ANN-method ranking (by NDCG@10) under each
backbone against the ranking under the reference backbone `svd_bm25`.
Stability is REPORTED, never required: unstable rankings are honest results.

This is a sensitivity check, not the main pipeline. `two_tower_mlp` needs
PyTorch (optional dependency); if torch is missing that backbone is recorded
with backend_available=false and the required backbones still run.

Configuration: configs/analyses.yml, section embedding_sensitivity
(inherits scientific defaults from configs/defaults.yml).

Output: results/analyses/embedding_sensitivity/embedding_backbone_sensitivity_all.csv
(presentation tables/figures come from tables_paper.py / figures_paper.py)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from calibrate import calibrate_index
from utils.ann_io import load_ann_index, CALIBRATION_PARAM
from utils.common import set_global_seed
from utils.config import load_config, cfg_get, config_hash, ConfigError
from utils.index_config import build_index_command
from utils.fingerprints import fingerprint
from utils.modality_queries import (build_query_population, item_id_map,
                                    write_query_cache)
from utils.paths import dataset_csv, dataset_stem, RESULTS
from utils.provenance import make_run_id, provenance_columns
from utils.result_io import (preflight_output, write_dataframe_atomic,
                             ResultExistsError)

SCRIPT = "run_embedding_backbone_sensitivity"
DEFAULT_CONFIG = "configs/analyses.yml"
REFERENCE_BACKBONE = "svd_bm25"
SVD_BACKBONES = {"svd_bm25": "bm25", "svd_tfidf": "tfidf", "svd_none": "none"}
KEY = ["dataset", "backbone", "modality", "method", "dim", "seed"]


def reuse_main_bm25_rows(dataset, modality, methods, dim, seed,
                         sensitivity_queries, main_dir=None, perquery_dir=None):
    """Transform main rows only after proving query identity and fingerprints."""
    main_dir = Path(main_dir or RESULTS["main"])
    perquery_dir = Path(perquery_dir or RESULTS["perquery"])
    summary = main_dir / "summary_main.csv"
    run_config = main_dir / "run_config.json"
    if not summary.is_file() or not run_config.is_file():
        return None, "main summary/run configuration is missing"
    frame = pd.read_csv(summary)
    required = {"dataset", "weighting", "dim", "modality", "method", "seed",
                "ndcg_at_k_mean", "recall_at_k_mean",
                "ann_recall_vs_exact_at_k_mean", "long_tail_uplift",
                "query_fingerprint"}
    if not required <= set(frame.columns):
        return None, f"main rows lack reuse fields {sorted(required - set(frame.columns))}"
    selected = frame[(frame["dataset"] == dataset)
                     & (frame["weighting"] == "bm25")
                     & (frame["dim"] == dim) & (frame["modality"] == modality)
                     & (frame["seed"] == seed) & (frame["method"].isin(methods))]
    if len(selected) != len(methods) or selected["method"].nunique() != len(methods):
        return None, "main BM25 method grid is incomplete or duplicated"
    cfg = json.loads(run_config.read_text(encoding="utf-8"))
    if cfg.get("evaluation_protocol_version") != "modality_queries_v1":
        return None, "main evaluation protocol predates shared modality queries"
    expected_ids = None
    expected_fp = None
    for method in methods:
        path = perquery_dir / (
            f"{dataset}__bm25__d{dim}__{modality}__{method}.npz")
        if not path.is_file():
            return None, f"missing main per-query evidence {path}"
        with np.load(path, allow_pickle=True) as z:
            if "meta" not in z or "query_ids" not in z:
                return None, f"unalignable main per-query evidence {path}"
            meta = json.loads(str(z["meta"]))
            ids = np.asarray(z["query_ids"]).astype(str)
        fp = meta.get("query_fingerprint")
        row_fp = selected.loc[selected["method"] == method,
                              "query_fingerprint"].iloc[0]
        if not fp or fp != row_fp:
            return None, f"query fingerprint mismatch for {method}"
        if expected_ids is None:
            expected_ids, expected_fp = ids, fp
        elif not np.array_equal(ids, expected_ids) or fp != expected_fp:
            return None, f"query identities/fingerprint differ for {method}"
    if str(sensitivity_queries).lower() != "full" and int(sensitivity_queries) < len(expected_ids):
        return None, "sensitivity query cap would select a different population"
    rows = []
    for row in selected.to_dict("records"):
        rows.append({
            "dataset": dataset, "backbone": REFERENCE_BACKBONE,
            "embedding_backend": REFERENCE_BACKBONE, "modality": modality,
            "method": row["method"], "dim": dim,
            "ndcg_at_10": row["ndcg_at_k_mean"],
            "recall_at_10": row["recall_at_k_mean"],
            "ann_recall_vs_exact_at_k_mean": row["ann_recall_vs_exact_at_k_mean"],
            "long_tail_uplift": row["long_tail_uplift"],
            "backend_available": True, "status": "ok", "error_message": "",
            "query_fingerprint": expected_fp, "reused_from_main": True,
            "seed": seed,
        })
    return rows, None


def run(cmd, check=True):
    full = [sys.executable] + cmd[1:] if cmd[0] == "python" else cmd
    print(">>", " ".join(str(c) for c in full))
    return subprocess.run(full, check=check).returncode


def _require_embedding_metadata(emb, expected, backbone):
    path = Path(emb) / "embedding_meta.json"
    if not path.is_file():
        raise RuntimeError(f"metadata missing for reusable {backbone}: {path}")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid embedding metadata {path}: {exc}") from exc
    mismatch = {k: {"expected": v, "found": meta.get(k)}
                for k, v in expected.items() if meta.get(k) != v}
    if mismatch:
        raise RuntimeError(
            f"refusing stale {backbone} embeddings in {emb}: {mismatch}")


def ensure_embeddings(backbone, dataset, dim, normalize, seed, es_cfg,
                      embedding_cfg, cfg_hash):
    """Train (or reuse) embeddings for a backbone; returns emb dir or None
    when an optional dependency is missing."""
    csv = dataset_csv(dataset)
    stem = dataset_stem(dataset)
    if backbone in SVD_BACKBONES:
        weighting = SVD_BACKBONES[backbone]
        emb = f"data/emb_{stem}_{weighting}_d{dim}"
        expected = {"dim_requested": dim, "weighting": weighting,
                    "normalize": normalize, "seed": seed}
        if weighting == "bm25":
            expected.update({
                "bm25_k1": cfg_get(embedding_cfg, "bm25_k1", type=float,
                                   required=True),
                "bm25_b": cfg_get(embedding_cfg, "bm25_b", type=float,
                                  required=True),
            })
        if Path(f"{emb}/item_vecs.npy").is_file():
            _require_embedding_metadata(emb, expected, backbone)
        else:
            cmd = ["python", "src/train_embeddings.py", "--interactions", csv,
                   "--dim", str(dim), "--weighting", weighting,
                   "--normalize", normalize, "--seed", str(seed),
                   "--config_hash", cfg_hash,
                   "--out_dir", emb]
            if weighting == "bm25":
                cmd += [
                    "--bm25_k1", str(cfg_get(
                        embedding_cfg, "bm25_k1", type=float, required=True)),
                    "--bm25_b", str(cfg_get(
                        embedding_cfg, "bm25_b", type=float, required=True)),
                ]
            run(cmd)
        return emb

    emb = f"data/emb_{stem}_{backbone}_d{dim}"
    if Path(f"{emb}/item_vecs.npy").is_file():
        hp_name = "bpr" if backbone == "bpr_matrix_factorization" else "two_tower"
        hp = cfg_get(es_cfg, hp_name, required=True)
        expected = {"embedding_backend": backbone, "dim": dim,
                    "normalize": normalize, "seed": seed,
                    "epochs": int(cfg_get(hp, "epochs", required=True)),
                    "lr": float(cfg_get(hp, "lr", required=True))}
        if backbone == "bpr_matrix_factorization":
            expected.update({
                "reg": float(cfg_get(hp, "reg", required=True)),
                "n_negatives": int(cfg_get(hp, "n_negatives", required=True)),
            })
        else:
            expected.update({
                "hidden_dim": int(cfg_get(hp, "hidden_dim", required=True)),
                "batch_size": int(cfg_get(hp, "batch_size", required=True)),
            })
        _require_embedding_metadata(emb, expected, backbone)
        return emb
    if backbone == "two_tower_mlp":
        try:
            import torch  # noqa: F401
        except ImportError:
            print(f"[{SCRIPT}] WARN: PyTorch not installed; skipping optional "
                  f"backbone two_tower_mlp (pip install torch to enable).")
            return None
    hp = es_cfg.get("bpr" if backbone == "bpr_matrix_factorization"
                    else "two_tower", {})
    cmd = ["python", "src/train_neural_embeddings.py", "--interactions", csv,
           "--backbone", backbone, "--dim", str(dim), "--normalize", normalize,
           "--seed", str(seed), "--config_hash", cfg_hash,
           "--out_dir", emb,
           "--epochs", str(cfg_get(hp, "epochs", required=True)),
           "--lr", str(cfg_get(hp, "lr", required=True))]
    if backbone == "bpr_matrix_factorization":
        cmd += ["--reg", str(cfg_get(hp, "reg", required=True)),
                "--n_negatives", str(cfg_get(hp, "n_negatives",
                                             required=True))]
    else:
        cmd += ["--hidden_dim", str(cfg_get(hp, "hidden_dim", required=True)),
                "--batch_size", str(cfg_get(hp, "batch_size", required=True))]
    run(cmd)
    return emb


def main():
    ap = argparse.ArgumentParser(
        description="ANN-method-ranking stability across embedding backbones "
                    "(U2I; reference backbone svd_bm25).")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="analyses YAML (section embedding_sensitivity)")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--backbones", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_dir", default=RESULTS["embedding_sensitivity"])
    ap.add_argument("--write_mode", default="fail_if_exists",
                    choices=["fail_if_exists", "replace", "merge"])
    ap.add_argument("--include_optional_backbones", action="store_true",
                    help="also run configured optional backbones such as "
                         "two_tower_mlp (off by default)")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config, cli_overrides={
            "embedding_sensitivity.datasets": args.datasets,
            "embedding_sensitivity.backbones": args.backbones,
            "embedding_sensitivity.methods": args.methods,
            "embedding.dim": args.dim,
            "embedding_sensitivity.queries": args.queries,
            "reproducibility.seed": args.seed,
        })
    except ConfigError as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    es = cfg_get(cfg, "embedding_sensitivity", default={})
    datasets = list(es.get("datasets", ["ml-1m"]))
    backbones = list(es.get("backbones", list(SVD_BACKBONES)
                            + ["bpr_matrix_factorization"]))
    optional_backbones = list(es.get("optional_backbones", ["two_tower_mlp"]))
    if args.backbones is None and args.include_optional_backbones:
        backbones = backbones + [b for b in optional_backbones
                                 if b not in backbones]
    methods = list(es.get("methods",
                          ["flat", "hnsw", "ivfflat", "ivfpq", "flatpq"]))
    modalities = list(es.get("modalities", ["u2i"]))
    queries = str(es.get("queries", 10000))
    dim = cfg_get(cfg, "embedding.dim", type=int, default=128)
    embedding_cfg = cfg_get(cfg, "embedding", required=True)
    normalize = cfg_get(cfg, "embedding.normalize", required=True)
    topk = cfg_get(cfg, "retrieval.topk", type=int, required=True)
    metric_topk = cfg_get(cfg, "retrieval.metric_topk", type=int, required=True)
    min_user_interactions = cfg_get(cfg, "data.min_user_interactions",
                                    type=int, required=True)
    budget_mb = cfg_get(cfg, "retrieval.budget_mb", type=int, required=True)
    cal_queries = cfg_get(cfg, "calibration.queries", type=int, required=True)
    primary_target = cfg_get(cfg, "calibration.primary_target", type=float,
                             required=True)
    omp_threads = cfg_get(cfg, "reproducibility.omp_threads", type=int,
                          required=True)
    seed = cfg_get(cfg, "reproducibility.seed", type=int, required=True)
    default_ef = cfg_get(cfg, "retrieval.runtime_defaults.ef", type=int,
                         required=True)
    default_nprobe = cfg_get(cfg, "retrieval.runtime_defaults.nprobe", type=int,
                             required=True)
    index_cfg = cfg_get(cfg, "index", required=True)
    cfg_hash = config_hash(cfg)
    prov = provenance_columns(make_run_id(cfg_hash), cfg_hash)

    required_backbones = list(es.get("backbones", []))
    expected_datasets = ["ml-1m"]
    expected_modalities = ["u2i"]
    canonical_methods = list(cfg_get(cfg, "retrieval.methods", required=True))
    if datasets != expected_datasets or modalities != expected_modalities:
        print(f"[{SCRIPT}] ERROR: canonical sensitivity scope is "
              f"datasets={expected_datasets}, modalities={expected_modalities}; "
              f"resolved {datasets}/{modalities}.")
        sys.exit(1)
    if methods != canonical_methods:
        print(f"[{SCRIPT}] ERROR: embedding sensitivity must use the canonical "
              f"five methods {canonical_methods}; resolved {methods}.")
        sys.exit(1)
    if any(b not in backbones for b in required_backbones):
        print(f"[{SCRIPT}] ERROR: required backbones cannot be omitted: "
              f"{required_backbones}.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    all_path = out_dir / "embedding_backbone_sensitivity_all.csv"
    eval_dir = out_dir / "eval"
    eval_aggregate_dir = eval_dir / "aggregates"
    eval_perquery_dir = eval_dir / "perquery"
    try:
        preflight_output(all_path, args.write_mode)
    except (ResultExistsError, ValueError) as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    missing = [d for d in datasets if not Path(dataset_csv(d)).is_file()]
    if missing:
        print(f"[{SCRIPT}] ERROR: dataset CSV(s) missing: "
              f"{[dataset_csv(d) for d in missing]}; prepare them first.")
        sys.exit(1)

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.config}")
    print(f"[{SCRIPT}] output path: {out_dir}")
    print(f"[{SCRIPT}] datasets={datasets} backbones={backbones} "
          f"methods={methods} modalities={modalities} queries={queries}")

    set_global_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in datasets:
        csv = dataset_csv(dataset)
        stem = dataset_stem(dataset)
        for backbone in backbones:
            if backbone == REFERENCE_BACKBONE:
                reused, reason = reuse_main_bm25_rows(
                    dataset, modalities[0], methods, dim, seed, queries)
                if reused is not None:
                    rows.extend([{**row, **prov} for row in reused])
                    print(f"[{SCRIPT}] reused {len(reused)} validated main BM25 rows")
                    continue
                print(f"[{SCRIPT}] main BM25 rows not reusable: {reason}; recomputing")
            emb = ensure_embeddings(backbone, dataset, dim, normalize, seed, es,
                                    embedding_cfg, cfg_hash)
            if emb is None:
                # optional backbone with missing dependency: keep it visible
                # in the schema instead of silently dropping it
                for modality in modalities:
                    for method in methods:
                        rows.append({
                            "dataset": dataset, "backbone": backbone,
                            "embedding_backend": backbone,
                            "modality": modality, "method": method,
                            "dim": dim, "ndcg_at_10": None,
                            "recall_at_10": None,
                            "ann_recall_vs_exact_at_k_mean": None,
                            "long_tail_uplift": None,
                            "backend_available": False,
                            "status": "skipped_missing_dependency",
                            "error_message": "torch_not_installed",
                            "seed": seed, **prov,
                        })
                continue
            item_vecs = np.load(f"{emb}/item_vecs.npy").astype("float32")
            N, D = item_vecs.shape
            if D != dim:
                raise RuntimeError(
                    f"embedding dimension mismatch for {dataset}/{backbone}: "
                    f"configured d{dim}, loaded d{D}")

            modality = modalities[0]
            qfp = fingerprint("embedding_sensitivity_query", {
                "dataset": dataset, "backbone": backbone, "modality": modality,
                "dim": dim, "queries": queries,
                "min_user_interactions": min_user_interactions,
                "split": "temporal_leave_one_out_v1", "seed": seed,
            })
            population = build_query_population(
                interactions_path=csv, item_vecs=item_vecs,
                id2idx=item_id_map(Path(emb) / "item_vecs.npy", N),
                modality=modality,
                max_queries="full" if queries.lower() == "full" else int(queries),
                seed=seed, min_user_interactions=min_user_interactions,
                metadata={"query_fingerprint": qfp, "dataset": dataset,
                          "weighting": backbone, "dim": dim})
            query_cache = out_dir / "query_cache" / (
                f"{dataset}__{backbone}__d{dim}__{modality}.npz")
            write_query_cache(population, query_cache, mode="replace")

            for method in methods:
                idx_dir = f"data/index_{stem}_{backbone}_d{dim}_{method}"
                if not (Path(idx_dir).exists() and any(Path(idx_dir).iterdir())):
                    run(build_index_command(
                        method, Path(emb) / "item_vecs.npy",
                        Path(emb) / "item_ids.npy", idx_dir, budget_mb,
                        seed, omp_threads, index_cfg,
                        configuration_hash=cfg_hash))

                ef, nprobe = default_ef, default_nprobe
                if CALIBRATION_PARAM.get(method) is not None:
                    ann = load_ann_index(idx_dir, D, N)
                    cal = calibrate_index(ann, item_vecs, primary_target, topk,
                                          n_queries=cal_queries, seed=seed,
                                          query_vectors=population.query_vectors,
                                          modality=modality,
                                          query_source="embedding_sensitivity_query_cache",
                                          query_fingerprint=qfp)
                    if cal["calibrated_param_value"] is not None:
                        if CALIBRATION_PARAM[method] == "ef":
                            ef = int(cal["calibrated_param_value"])
                        else:
                            nprobe = int(cal["calibrated_param_value"])

                for modality in modalities:
                    run(["python", "src/eval_modalities.py",
                         "--interactions", csv, "--item_vecs",
                         f"{emb}/item_vecs.npy",
                         "--index", idx_dir, "--ann_method", method,
                         "--modality", modality, "--queries", queries,
                         "--query_cache", str(query_cache),
                         "--topk", str(topk), "--metric_topk", str(metric_topk),
                         "--ef", str(ef), "--nprobe", str(nprobe),
                         "--seed", str(seed), "--dataset", dataset,
                         "--weighting", backbone, "--config", args.config,
                         "--aggregate_dir", str(eval_aggregate_dir),
                         "--perquery_dir", str(eval_perquery_dir),
                         "--write_mode", args.write_mode])

                    agg = (eval_aggregate_dir
                           / f"{dataset}__{backbone}__d{D}__{modality}__{method}.json")
                    if not agg.is_file():
                        raise RuntimeError(
                            f"evaluation completed without aggregate {agg}")
                    with open(agg, encoding="utf-8") as handle:
                        e = json.load(handle)
                    rows.append({
                        "dataset": dataset, "backbone": backbone,
                        "embedding_backend": backbone, "modality": modality,
                        "method": method, "dim": dim,
                        "ndcg_at_10": e.get("ndcg_at_k_mean"),
                        "recall_at_10": e.get("recall_at_k_mean"),
                        "ann_recall_vs_exact_at_k_mean": e.get("ann_recall_vs_exact_at_k_mean"),
                        "long_tail_uplift": e.get("long_tail_uplift"),
                        "backend_available": True,
                        "status": "ok",
                        "error_message": "",
                        "seed": seed, **prov,
                    })

    if not rows:
        print(f"[{SCRIPT}] ERROR: nothing evaluated.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    required_df = df[df["backbone"].isin(required_backbones)]
    expected_required_rows = (len(datasets) * len(required_backbones)
                              * len(modalities) * len(methods))
    successful = ((required_df["backend_available"] == True)  # noqa: E712
                  & (required_df["status"] == "ok"))
    if len(required_df) != expected_required_rows or not successful.all():
        print(f"[{SCRIPT}] ERROR: required sensitivity grid incomplete: "
              f"expected {expected_required_rows} successful rows, found "
              f"{int(successful.sum())}/{len(required_df)}.")
        sys.exit(1)

    # ann_ranking_stability: Spearman of the method ranking vs the reference
    # backbone. Reported honestly — never required to be stable.
    df["ann_ranking_stability"] = np.nan
    for dataset, g in df.groupby("dataset"):
        ref = g[g["backbone"] == REFERENCE_BACKBONE].set_index("method")["ndcg_at_10"]
        if ref.empty:
            print(f"[{SCRIPT}] WARN: reference backbone {REFERENCE_BACKBONE} "
                  f"missing for {dataset}; stability is NaN.")
            continue
        for backbone, gb in g.groupby("backbone"):
            cur = gb.set_index("method")["ndcg_at_10"]
            common = [m for m in ref.index if m in cur.index]
            if len(common) < 3:
                continue
            rho, _ = spearmanr(ref.loc[common].rank(), cur.loc[common].rank())
            df.loc[(df["dataset"] == dataset) & (df["backbone"] == backbone),
                   "ann_ranking_stability"] = float(rho)

    write_dataframe_atomic(df, all_path, mode=args.write_mode,
                           key=KEY, sort_by=KEY)
    print(f"[{SCRIPT}] output path: {all_path} ({len(df)} rows)")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
