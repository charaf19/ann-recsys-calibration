"""Embedding backbone sensitivity study.

Reviewer concern addressed: "SVD-only embeddings". For each backbone
(svd_bm25 / svd_tfidf / svd_none / bpr_matrix_factorization /
two_tower_mlp) this trains embeddings, builds indexes, calibrates, and runs
the U2I evaluation — then reports `ann_ranking_stability`: the Spearman
correlation of the ANN-method ranking (by NDCG@10) under each backbone
against the ranking under the reference backbone `svd_bm25`.

This is a sensitivity check, not the main pipeline. `two_tower_mlp` needs
PyTorch (optional dependency); if torch is missing, that backbone is skipped
with a warning and the others still run.

Outputs:
    results/embedding_sensitivity/embedding_backbone_sensitivity_all.csv
    results/paper_tables/embedding_backbone_sensitivity_summary.csv/.tex (+ .md)
    results/figures_paper/embedding_backbone_sensitivity.pdf
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from calibrate import calibrate_index
from utils.ann_io import load_ann_index, CALIBRATION_PARAM
from utils.common import set_global_seed
from utils.paths import dataset_csv, dataset_stem, RESULTS
from utils.reporting import write_table
from utils.figures_ext import fig_embedding_backbone_sensitivity

SCRIPT = "run_embedding_backbone_sensitivity"
REFERENCE_BACKBONE = "svd_bm25"
SVD_BACKBONES = {"svd_bm25": "bm25", "svd_tfidf": "tfidf", "svd_none": "none"}


def run(cmd, check=True):
    full = [sys.executable] + cmd[1:] if cmd[0] == "python" else cmd
    print(">>", " ".join(str(c) for c in full))
    return subprocess.run(full, check=check).returncode


def load_config(path):
    p = Path(path)
    if not p.is_file():
        print(f"[{SCRIPT}] WARN: config {path} not found; using defaults.")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_embeddings(backbone, dataset, dim, seed, cfg):
    """Train (or reuse) embeddings for a backbone; returns emb dir or None."""
    csv = dataset_csv(dataset)
    stem = dataset_stem(dataset)
    if backbone in SVD_BACKBONES:
        weighting = SVD_BACKBONES[backbone]
        emb = f"data/emb_{stem}_{weighting}_d{dim}"
        if not Path(f"{emb}/item_vecs.npy").is_file():
            run(["python", "src/train_embeddings.py", "--interactions", csv,
                 "--dim", str(dim), "--weighting", weighting,
                 "--normalize", "l2", "--seed", str(seed), "--out_dir", emb])
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
    hp = cfg.get("bpr" if backbone == "bpr_matrix_factorization" else "two_tower", {})
    cmd = ["python", "src/train_neural_embeddings.py", "--interactions", csv,
           "--backbone", backbone, "--dim", str(dim), "--normalize", "l2",
           "--seed", str(seed), "--out_dir", emb]
    if "epochs" in hp:
        cmd += ["--epochs", str(hp["epochs"])]
    if "lr" in hp:
        cmd += ["--lr", str(hp["lr"])]
    run(cmd)
    return emb


def build_index_cmd(method, emb, idx_dir, budget_mb):
    cmd = ["python", "src/build_index.py", "--method", method,
           "--item_vecs", f"{emb}/item_vecs.npy",
           "--item_ids", f"{emb}/item_ids.npy",
           "--out_dir", idx_dir, "--budget_mb", str(budget_mb)]
    if method == "hnsw":
        cmd += ["--M", "24", "--efc", "200"]
    elif method == "ivfflat":
        cmd += ["--nlist", "auto"]
    elif method == "ivfpq":
        cmd += ["--nlist", "auto", "--m", "32", "--bits", "8"]
    elif method == "flatpq":
        cmd += ["--m", "32", "--bits", "8"]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/embedding_backbones.yml")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--backbones", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_dir", default=RESULTS["embedding_sensitivity"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    datasets = args.datasets or cfg.get("datasets", ["ml-1m"])
    backbones = args.backbones or cfg.get("backbones", list(SVD_BACKBONES)
                                          + ["bpr_matrix_factorization", "two_tower_mlp"])
    methods = args.methods or cfg.get("methods",
                                      ["flat", "hnsw", "ivfflat", "ivfpq", "flatpq"])
    dim = args.dim or int(cfg.get("dim", 128))
    queries = str(args.queries or cfg.get("queries", 2000))
    topk = int(cfg.get("topk", 100))
    metric_topk = int(cfg.get("metric_topk", 10))
    budget_mb = int(cfg.get("budget_mb", 100))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))

    out_dir = Path(args.out_dir)
    eval_dir = out_dir / "eval"

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.config}")
    print(f"[{SCRIPT}] output path: {out_dir}")
    print(f"[{SCRIPT}] datasets={datasets} backbones={backbones} methods={methods}")

    set_global_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in datasets:
        csv = dataset_csv(dataset)
        if not Path(csv).is_file():
            print(f"[{SCRIPT}] WARN: {csv} missing; prepare it first. Skipping.")
            continue
        stem = dataset_stem(dataset)
        for backbone in backbones:
            emb = ensure_embeddings(backbone, dataset, dim, seed, cfg)
            if emb is None:
                continue
            item_vecs = np.load(f"{emb}/item_vecs.npy").astype("float32")
            N, D = item_vecs.shape

            for method in methods:
                idx_dir = f"data/index_{stem}_{backbone}_d{dim}_{method}"
                if not (Path(idx_dir).exists() and any(Path(idx_dir).iterdir())):
                    run(build_index_cmd(method, emb, idx_dir, budget_mb))

                ef, nprobe = 128, 16
                if CALIBRATION_PARAM.get(method) is not None:
                    ann = load_ann_index(idx_dir, D, N)
                    cal = calibrate_index(ann, item_vecs, 0.95, topk,
                                          n_queries=1000, seed=seed)
                    if cal["calibrated_param_value"] is not None:
                        if CALIBRATION_PARAM[method] == "ef":
                            ef = int(cal["calibrated_param_value"])
                        else:
                            nprobe = int(cal["calibrated_param_value"])

                run(["python", "src/eval_modalities.py",
                     "--interactions", csv, "--item_vecs", f"{emb}/item_vecs.npy",
                     "--index", idx_dir, "--ann_method", method,
                     "--modality", "u2i", "--queries", queries,
                     "--topk", str(topk), "--metric_topk", str(metric_topk),
                     "--ef", str(ef), "--nprobe", str(nprobe),
                     "--seed", str(seed), "--dataset", dataset,
                     "--weighting", backbone, "--out_dir", str(eval_dir)])

                agg = eval_dir / f"{dataset}__{backbone}__u2i__{method}.json"
                e = json.load(open(agg)) if agg.is_file() else {}
                rows.append({
                    "dataset": dataset, "backbone": backbone,
                    "embedding_backend": backbone, "modality": "u2i",
                    "method": method, "dim": dim,
                    "ndcg_at_10": e.get("ndcg_at_k_mean"),
                    "recall_at_10": e.get("recall_at_k_mean"),
                    "ann_recall_vs_exact_at_k_mean": e.get("ann_recall_vs_exact_at_k_mean"),
                    "long_tail_uplift": e.get("long_tail_uplift"),
                    "seed": seed,
                })

    if not rows:
        print(f"[{SCRIPT}] WARN: nothing evaluated.")
        print(f"[{SCRIPT}] completed.")
        return

    df = pd.DataFrame(rows)

    # ann_ranking_stability: Spearman of method ranking vs the reference backbone
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

    all_path = out_dir / "embedding_backbone_sensitivity_all.csv"
    df.to_csv(all_path, index=False)
    print(f"[{SCRIPT}] output path: {all_path}")

    summary = (df.groupby(["dataset", "backbone"])
               .agg(ndcg_at_10_best=("ndcg_at_10", "max"),
                    ann_ranking_stability=("ann_ranking_stability", "first"),
                    n_methods=("method", "nunique")).reset_index())
    written = write_table(summary, Path(RESULTS["paper_tables"])
                          / "embedding_backbone_sensitivity_summary")
    for p in written:
        print(f"[{SCRIPT}] output path: {p}")

    for p in fig_embedding_backbone_sensitivity(df, Path(RESULTS["figures_paper"])):
        print(f"[{SCRIPT}] output path: {p}")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
