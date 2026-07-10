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
from utils.config import load_config, cfg_get, ConfigError
from utils.paths import dataset_csv, dataset_stem, RESULTS
from utils.result_io import (preflight_output, write_dataframe_atomic,
                             ResultExistsError)

SCRIPT = "run_embedding_backbone_sensitivity"
DEFAULT_CONFIG = "configs/analyses.yml"
REFERENCE_BACKBONE = "svd_bm25"
SVD_BACKBONES = {"svd_bm25": "bm25", "svd_tfidf": "tfidf", "svd_none": "none"}
KEY = ["dataset", "backbone", "modality", "method", "dim", "seed"]


def run(cmd, check=True):
    full = [sys.executable] + cmd[1:] if cmd[0] == "python" else cmd
    print(">>", " ".join(str(c) for c in full))
    return subprocess.run(full, check=check).returncode


def build_index_cmd(method, emb, idx_dir, budget_mb, seed, omp_threads,
                    index_cfg):
    """Index build invocation from the RESOLVED index configuration."""
    cmd = ["python", "src/build_index.py", "--method", method,
           "--item_vecs", f"{emb}/item_vecs.npy",
           "--item_ids", f"{emb}/item_ids.npy",
           "--out_dir", idx_dir, "--budget_mb", str(budget_mb),
           "--seed", str(seed), "--omp_threads", str(omp_threads)]
    hnsw = index_cfg.get("hnsw", {})
    ivf = index_cfg.get("ivf", {})
    pq = index_cfg.get("pq", {})
    ivfpq = index_cfg.get("ivfpq", {})
    if method == "hnsw":
        cmd += ["--M", str(hnsw.get("M", 24)),
                "--efc", str(hnsw.get("ef_construction", 200))]
    elif method == "ivfflat":
        cmd += ["--nlist", str(ivf.get("nlist", "auto"))]
    elif method == "ivfpq":
        cmd += ["--nlist", str(ivf.get("nlist", "auto")),
                "--m", str(pq.get("m", 32)), "--bits", str(pq.get("bits", 8))]
        if ivfpq.get("use_opq", True):
            cmd += ["--opq"]
    elif method == "flatpq":
        cmd += ["--m", str(pq.get("m", 32)), "--bits", str(pq.get("bits", 8))]
    return cmd


def ensure_embeddings(backbone, dataset, dim, normalize, seed, es_cfg):
    """Train (or reuse) embeddings for a backbone; returns emb dir or None
    when an optional dependency is missing."""
    csv = dataset_csv(dataset)
    stem = dataset_stem(dataset)
    if backbone in SVD_BACKBONES:
        weighting = SVD_BACKBONES[backbone]
        emb = f"data/emb_{stem}_{weighting}_d{dim}"
        if not Path(f"{emb}/item_vecs.npy").is_file():
            run(["python", "src/train_embeddings.py", "--interactions", csv,
                 "--dim", str(dim), "--weighting", weighting,
                 "--normalize", normalize, "--seed", str(seed),
                 "--out_dir", emb])
        return emb

    emb = f"data/emb_{stem}_{backbone}_d{dim}"
    if Path(f"{emb}/item_vecs.npy").is_file():
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
           "--seed", str(seed), "--out_dir", emb]
    if "epochs" in hp:
        cmd += ["--epochs", str(hp["epochs"])]
    if "lr" in hp:
        cmd += ["--lr", str(hp["lr"])]
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
    if args.backbones is None:
        backbones = backbones + [b for b in optional_backbones
                                 if b not in backbones]
    methods = list(es.get("methods",
                          ["flat", "hnsw", "ivfflat", "ivfpq", "flatpq"]))
    modalities = list(es.get("modalities", ["u2i"]))
    queries = str(es.get("queries", 10000))
    dim = cfg_get(cfg, "embedding.dim", type=int, default=128)
    normalize = cfg_get(cfg, "embedding.normalize", default="l2")
    topk = cfg_get(cfg, "retrieval.topk", type=int, default=100)
    metric_topk = cfg_get(cfg, "retrieval.metric_topk", type=int, default=10)
    budget_mb = cfg_get(cfg, "retrieval.budget_mb", type=int, default=100)
    cal_queries = cfg_get(cfg, "calibration.queries", type=int, default=1000)
    primary_target = cfg_get(cfg, "calibration.primary_target", type=float,
                             default=0.95)
    omp_threads = cfg_get(cfg, "reproducibility.omp_threads", type=int,
                          default=1)
    seed = cfg_get(cfg, "reproducibility.seed", type=int, default=42)
    index_cfg = cfg_get(cfg, "index", default={})

    out_dir = Path(args.out_dir)
    all_path = out_dir / "embedding_backbone_sensitivity_all.csv"
    eval_dir = out_dir / "eval"
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
            emb = ensure_embeddings(backbone, dataset, dim, normalize, seed, es)
            if emb is None:
                # optional backbone with missing dependency: keep it visible
                # in the schema instead of silently dropping it
                rows.append({
                    "dataset": dataset, "backbone": backbone,
                    "embedding_backend": backbone, "modality": "u2i",
                    "method": None, "dim": dim,
                    "ndcg_at_10": None, "recall_at_10": None,
                    "ann_recall_vs_exact_at_k_mean": None,
                    "long_tail_uplift": None,
                    "backend_available": False,
                    "status": "skipped_missing_dependency",
                    "error_message": "torch_not_installed",
                    "seed": seed,
                })
                continue
            item_vecs = np.load(f"{emb}/item_vecs.npy").astype("float32")
            N, D = item_vecs.shape

            for method in methods:
                idx_dir = f"data/index_{stem}_{backbone}_d{dim}_{method}"
                if not (Path(idx_dir).exists() and any(Path(idx_dir).iterdir())):
                    run(build_index_cmd(method, emb, idx_dir, budget_mb,
                                        seed, omp_threads, index_cfg))

                ef, nprobe = 128, 16
                if CALIBRATION_PARAM.get(method) is not None:
                    ann = load_ann_index(idx_dir, D, N)
                    cal = calibrate_index(ann, item_vecs, primary_target, topk,
                                          n_queries=cal_queries, seed=seed)
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
                         "--topk", str(topk), "--metric_topk", str(metric_topk),
                         "--ef", str(ef), "--nprobe", str(nprobe),
                         "--seed", str(seed), "--dataset", dataset,
                         "--weighting", backbone, "--out_dir", str(eval_dir)])

                    agg = (eval_dir / "aggregates"
                           / f"{dataset}__{backbone}__d{D}__{modality}__{method}.json")
                    e = json.load(open(agg)) if agg.is_file() else {}
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
                        "seed": seed,
                    })

    if not rows:
        print(f"[{SCRIPT}] ERROR: nothing evaluated.")
        sys.exit(1)

    df = pd.DataFrame(rows)

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
