"""Main revision experiment orchestrator (CPU-only canonical benchmark).

For every dataset x method it: trains weighted embeddings, builds the index,
calibrates the ANN runtime parameter against exact Flat at the primary
target, measures latency at the calibrated operating point, and runs the
modality-separated evaluation (U2I and I2I). Additional calibration targets
are recorded for the sensitivity study. Everything is seeded and all outputs
land under results/.

Terminology: "agreement recall" (ann_recall_vs_exact_*) is overlap with the
exact Flat top-k — an index-fidelity measure used for calibration. It is NOT
recommendation relevance; relevance metrics (recall/ndcg/... vs the held-out
interaction) are reported separately.

Status tracking: each (dataset, method) writes
results/status/status_{dataset}_{method}.json with per-step outcomes
(ok / failed / skipped / pending) and error messages. A failing step marks
downstream steps skipped and the orchestrator continues with the next
method instead of aborting the grid.

This script does NOT download datasets. Prepare them first, e.g.:
    python src/prepare_dataset.py --dataset ml-1m --out data/ml1m.csv

Defaults come from configs/main_cpu.yml; CLI flags override. The config's
`datasets:` may be a flat list or a {main: [...], optional: [...]} mapping —
only `main` runs by default; optional datasets (e.g. amazon-books) run when
named explicitly via --datasets.

Outputs:
    results/main/summary_main.csv            one row per dataset x modality x method
    results/main/*.json                      aggregate eval per combination
    results/main/perquery/*.npz              per-query metrics (bootstrap input)
    results/main/calibration/*.json          calibration records per target
    results/status/status_{dataset}_{method}.json
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from calibrate import calibrate_index
from utils.ann_io import load_ann_index, CALIBRATION_PARAM
from utils.common import set_global_seed, normalize_modality_label
from utils.paths import dataset_csv, emb_dir, index_dir, RESULTS

SCRIPT = "run_revision_experiments"
DEFAULT_CONFIG = "configs/main_cpu.yml"
ALL_METHODS = ["flat", "hnsw", "ivfflat", "ivfpq", "flatpq"]


class StepError(Exception):
    """A pipeline step failed; downstream steps for this method are skipped."""


def run(cmd, env=None):
    full = [sys.executable] + cmd[1:] if cmd[0] == "python" else cmd
    print(">>", " ".join(str(c) for c in full))
    try:
        subprocess.run(full, check=True, env=env)
    except subprocess.CalledProcessError as e:
        raise StepError(f"command failed (exit {e.returncode}): "
                        + " ".join(str(c) for c in full)) from e


def load_config(path):
    p = Path(path)
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_datasets(cfg_value, cli_value):
    """Support both flat-list and {main:[...], optional:[...]} config shapes.

    CLI always wins verbatim (so optional datasets can be requested by name).
    """
    if cli_value is not None:
        return list(cli_value)
    if isinstance(cfg_value, dict):
        main = list(cfg_value.get("main", []))
        optional = list(cfg_value.get("optional", []))
        if optional:
            print(f"[{SCRIPT}] optional datasets not run by default: {optional} "
                  f"(request explicitly via --datasets)")
        return main
    if cfg_value is None:
        return ["ml-1m", "ml-20m", "goodbooks"]
    return list(cfg_value)


def build_index_cmd(method, emb, idx_dir, budget_mb, seed, omp_threads):
    cmd = ["python", "src/build_index.py", "--method", method,
           "--item_vecs", f"{emb}/item_vecs.npy",
           "--item_ids", f"{emb}/item_ids.npy",
           "--out_dir", idx_dir, "--budget_mb", str(budget_mb),
           "--seed", str(seed), "--omp_threads", str(omp_threads)]
    if method == "hnsw":
        cmd += ["--M", "24", "--efc", "200"]
    elif method == "ivfflat":
        cmd += ["--nlist", "auto"]
    elif method == "ivfpq":
        cmd += ["--nlist", "auto", "--m", "32", "--bits", "8", "--opq"]
    elif method == "flatpq":
        cmd += ["--m", "32", "--bits", "8"]
    return cmd


class StatusTracker:
    """Writes results/status/status_{dataset}_{method}.json after every step."""

    def __init__(self, status_dir, dataset, method, steps):
        self.path = Path(status_dir) / f"status_{dataset}_{method}.json"
        self.doc = {
            "dataset": dataset,
            "method": method,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "overall": "running",
            "steps": {s: {"status": "pending", "error": None, "finished_at_utc": None}
                      for s in steps},
        }
        self._flush()

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.doc, f, indent=2)

    def mark(self, step, status, error=None):
        self.doc["steps"][step] = {
            "status": status,
            "error": str(error) if error else None,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._flush()

    def finish(self):
        statuses = {s["status"] for s in self.doc["steps"].values()}
        if "failed" in statuses:
            self.doc["overall"] = "failed"
        elif statuses <= {"ok", "skipped"}:
            self.doc["overall"] = "ok" if "ok" in statuses else "skipped"
        else:
            self.doc["overall"] = "partial"
        self.doc["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._flush()
        print(f"[{SCRIPT}] status written: {self.path} (overall={self.doc['overall']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--modalities", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--weighting", default=None, choices=["none", "tfidf", "bm25"])
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--budget_mb", type=int, default=None)
    ap.add_argument("--calibration_targets", type=float, nargs="*", default=None)
    ap.add_argument("--primary_target", type=float, default=None,
                    help="calibration target used for latency + quality runs")
    ap.add_argument("--queries_large", default=None,
                    help="eval queries for large datasets (int or 'full')")
    ap.add_argument("--queries_ml1m", default=None,
                    help="eval queries for ml-1m (int or 'full')")
    ap.add_argument("--latency_queries", type=int, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--metric_topk", type=int, default=None)
    ap.add_argument("--omp_threads", type=int, default=None,
                    help="FAISS OMP threads for index builds (default 1)")
    ap.add_argument("--cpu_only", action="store_true", default=None)
    ap.add_argument("--reuse_existing", action="store_true",
                    help="skip embedding/index building when outputs already exist")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)

    def opt(name, default):
        v = getattr(args, name, None)
        if v is not None:
            return v
        return cfg.get(name, default)

    datasets = resolve_datasets(cfg.get("datasets"), args.datasets)
    modalities = [normalize_modality_label(m) for m in opt("modalities", ["u2i", "i2i"])]
    methods = opt("methods", ALL_METHODS)
    weighting = opt("weighting", "bm25")
    dim = int(opt("dim", 128))
    budget_mb = int(opt("budget_mb", 100))
    targets = [float(t) for t in opt("calibration_targets", [0.90, 0.95, 0.98])]
    primary_target = float(opt("primary_target", 0.95))
    queries_large = str(opt("queries_large", "10000"))
    queries_ml1m = str(opt("queries_ml1m", "full"))
    latency_queries = int(opt("latency_queries", 2000))
    topk = int(opt("topk", 100))
    metric_topk = int(opt("metric_topk", 10))
    omp_threads = int(opt("omp_threads", 1))
    cpu_only = bool(opt("cpu_only", True))
    seed = int(opt("seed", 42))

    main_dir = Path(RESULTS["main"])
    cal_dir = Path(RESULTS["calibration"])
    status_dir = Path(RESULTS["status"])

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.config}")
    print(f"[{SCRIPT}] output path: {main_dir}")
    print(f"[{SCRIPT}] output path: {status_dir}")
    print(f"[{SCRIPT}] datasets={datasets} methods={methods} modalities={modalities} "
          f"weighting={weighting} dim={dim} targets={targets} primary={primary_target} "
          f"omp_threads={omp_threads} cpu_only={cpu_only} seed={seed}")

    set_global_seed(seed)
    env = os.environ.copy()
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""  # belt-and-braces; faiss-cpu has no GPU anyway

    main_dir.mkdir(parents=True, exist_ok=True)
    cal_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "datasets": datasets, "modalities": modalities, "methods": methods,
        "weighting": weighting, "dim": dim, "budget_mb": budget_mb,
        "calibration_targets": targets, "primary_target": primary_target,
        "queries_large": queries_large, "queries_ml1m": queries_ml1m,
        "latency_queries": latency_queries, "topk": topk,
        "metric_topk": metric_topk, "omp_threads": omp_threads,
        "cpu_only": cpu_only, "seed": seed,
    }
    with open(main_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    rows = []
    for dataset in datasets:
        csv = dataset_csv(dataset)
        if not Path(csv).is_file():
            print(f"[{SCRIPT}] ERROR: {csv} not found. Prepare it first with:\n"
                  f"    python src/prepare_dataset.py --dataset {dataset} --out {csv}\n"
                  f"[{SCRIPT}] skipping dataset {dataset}.")
            continue

        n_queries = queries_ml1m if dataset == "ml-1m" else queries_large

        # 1) embeddings (a failure here skips the whole dataset)
        emb = emb_dir(dataset, weighting, dim)
        try:
            if args.reuse_existing and Path(f"{emb}/item_vecs.npy").is_file():
                print(f"[{SCRIPT}] reusing embeddings {emb}")
            else:
                run(["python", "src/train_embeddings.py", "--interactions", csv,
                     "--dim", str(dim), "--weighting", weighting,
                     "--bm25_k1", "1.2", "--bm25_b", "0.75",
                     "--normalize", "l2", "--seed", str(seed),
                     "--out_dir", emb], env=env)
            item_vecs = np.load(f"{emb}/item_vecs.npy").astype("float32")
        except (StepError, OSError) as e:
            print(f"[{SCRIPT}] ERROR: embeddings failed for {dataset}: {e}; "
                  f"skipping dataset.")
            continue
        N, D = item_vecs.shape

        for method in methods:
            steps = (["build", "calibration", "latency"]
                     + [f"eval_{m}" for m in modalities])
            tracker = StatusTracker(status_dir, dataset, method, steps)
            idx_dir = index_dir(dataset, weighting, dim, method)
            failed = False

            # 2) build index
            try:
                if args.reuse_existing and Path(idx_dir).exists() \
                        and any(Path(idx_dir).iterdir()):
                    print(f"[{SCRIPT}] reusing index {idx_dir}")
                else:
                    run(build_index_cmd(method, emb, idx_dir, budget_mb,
                                        seed, omp_threads), env=env)
                tracker.mark("build", "ok")
            except StepError as e:
                print(f"[{SCRIPT}] ERROR: build failed for {dataset}/{method}: {e}")
                tracker.mark("build", "failed", e)
                failed = True

            # 3) calibration at every target (primary target drives eval params)
            ef, nprobe = 128, 16  # defaults for untunable/uncalibrated paths
            primary_cal = None
            if failed:
                tracker.mark("calibration", "skipped")
            elif CALIBRATION_PARAM.get(method) is None:
                tracker.mark("calibration", "skipped")  # flat/flatpq: untunable
            else:
                try:
                    for target in sorted(set(targets + [primary_target])):
                        ann = load_ann_index(idx_dir, D, N)
                        res = calibrate_index(ann, item_vecs, target, topk,
                                              n_queries=1000, seed=seed)
                        res.update({"dataset": dataset, "weighting": weighting,
                                    "dim": dim})
                        cal_path = cal_dir / (f"{dataset}__{weighting}__{method}"
                                              f"__t{target:.2f}.json")
                        with open(cal_path, "w", encoding="utf-8") as f:
                            json.dump(res, f, indent=2)
                        if abs(target - primary_target) < 1e-9:
                            primary_cal = res
                    if primary_cal and primary_cal["calibrated_param_value"] is not None:
                        if CALIBRATION_PARAM[method] == "ef":
                            ef = int(primary_cal["calibrated_param_value"])
                        else:
                            nprobe = int(primary_cal["calibrated_param_value"])
                    tracker.mark("calibration", "ok")
                except Exception as e:
                    print(f"[{SCRIPT}] ERROR: calibration failed for "
                          f"{dataset}/{method}: {e}")
                    tracker.mark("calibration", "failed", e)
                    failed = True

            # 4) latency at the calibrated operating point
            lstats = {}
            if failed:
                tracker.mark("latency", "skipped")
            else:
                try:
                    run(["python", "src/run_device.py", "--index", idx_dir,
                         "--item_vecs", f"{emb}/item_vecs.npy",
                         "--queries", str(latency_queries), "--topk", str(topk),
                         "--ef", str(ef), "--nprobe", str(nprobe),
                         "--seed", str(seed)], env=env)
                    lat_file = Path(idx_dir) / f"latency_{method}.json"
                    lstats = json.load(open(lat_file)) if lat_file.is_file() else {}
                    tracker.mark("latency", "ok")
                except StepError as e:
                    print(f"[{SCRIPT}] ERROR: latency failed for "
                          f"{dataset}/{method}: {e}")
                    tracker.mark("latency", "failed", e)
                    # latency failure does not block quality evaluation

            # 5) modality-separated evaluation
            for modality in modalities:
                step = f"eval_{modality}"
                if failed:
                    tracker.mark(step, "skipped")
                    continue
                try:
                    run(["python", "src/eval_modalities.py",
                         "--interactions", csv,
                         "--item_vecs", f"{emb}/item_vecs.npy",
                         "--index", idx_dir,
                         "--ann_method", method,
                         "--modality", modality,
                         "--queries", n_queries,
                         "--topk", str(topk),
                         "--metric_topk", str(metric_topk),
                         "--ef", str(ef), "--nprobe", str(nprobe),
                         "--seed", str(seed),
                         "--dataset", dataset,
                         "--weighting", weighting,
                         "--out_dir", str(main_dir)], env=env)
                    tracker.mark(step, "ok")
                except StepError as e:
                    print(f"[{SCRIPT}] ERROR: {step} failed for "
                          f"{dataset}/{method}: {e}")
                    tracker.mark(step, "failed", e)
                    continue

                agg_path = main_dir / f"{dataset}__{weighting}__{modality}__{method}.json"
                estats = json.load(open(agg_path)) if agg_path.is_file() else {}
                rows.append({
                    "dataset": dataset, "weighting": weighting,
                    "modality": modality, "method": method,
                    "dim": dim, "budget_mb": budget_mb,
                    "queries": estats.get("queries"),
                    "recall_at_k_mean": estats.get("recall_at_k_mean"),
                    "ndcg_at_k_mean": estats.get("ndcg_at_k_mean"),
                    "hr_at_k_mean": estats.get("hr_at_k_mean"),
                    "precision_at_k_mean": estats.get("precision_at_k_mean"),
                    "map_at_k_mean": estats.get("map_at_k_mean"),
                    "mrr_at_k_mean": estats.get("mrr_at_k_mean"),
                    "ann_recall_vs_exact_at_k_mean": estats.get("ann_recall_vs_exact_at_k_mean"),
                    "recall_at_100_mean": estats.get("recall_at_100_mean"),
                    "ann_recall_vs_exact_at_100_mean": estats.get("ann_recall_vs_exact_at_100_mean"),
                    "coverage_at_k": estats.get("coverage_at_k"),
                    "gini_exposure": estats.get("gini_exposure"),
                    "long_tail_exposure": estats.get("long_tail_exposure"),
                    "long_tail_uplift": estats.get("long_tail_uplift"),
                    "calibration_target": primary_target if primary_cal else None,
                    "calibrated_param_name": CALIBRATION_PARAM.get(method),
                    "calibrated_param_value": (primary_cal or {}).get("calibrated_param_value"),
                    "latency_p50_ms": (lstats.get("latency_ms") or {}).get("p50"),
                    "latency_p95_ms": (lstats.get("latency_ms") or {}).get("p95"),
                    "rss_mb_after": lstats.get("rss_mb_after"),
                    "omp_threads": omp_threads,
                    "seed": seed,
                })

            tracker.finish()

        # checkpoint the summary after each dataset
        if rows:
            pd.DataFrame(rows).to_csv(main_dir / "summary_main.csv", index=False)
            print(f"[{SCRIPT}] checkpoint: {main_dir / 'summary_main.csv'} "
                  f"({len(rows)} rows)")

    if rows:
        pd.DataFrame(rows).to_csv(main_dir / "summary_main.csv", index=False)
        print(f"[{SCRIPT}] output path: {main_dir / 'summary_main.csv'}")
    else:
        print(f"[{SCRIPT}] WARN: no rows produced (missing datasets or all "
              f"steps failed — see {status_dir}).")
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
