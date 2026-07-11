"""Main revision experiment orchestrator (CPU-only canonical benchmark).

For every dataset x method it: trains weighted embeddings, builds the index,
calibrates the ANN runtime parameter against exact Flat at every target,
measures latency at the primary calibrated operating point, and runs the
modality-separated evaluation (U2I and I2I). Everything is seeded; all
scientific values come from the resolved configuration (configs/main_cpu.yml
inheriting configs/defaults.yml), never from hardcoded constants.

Terminology: "agreement recall" (ann_recall_vs_exact_*) is overlap with the
exact Flat top-k — an index-fidelity measure used for calibration. It is NOT
recommendation relevance; relevance metrics (recall/ndcg/... vs the held-out
interaction) are reported separately and must never be conflated with it.

This script does NOT download datasets. Prepare them first, e.g.:
    python src/prepare_dataset.py --dataset ml-1m --out data/ml1m.csv

All four canonical datasets must exist before any experiment starts unless
--allow_missing_datasets is passed explicitly.

Outputs:
    results/main/summary_main.csv          consolidated summary (result_io)
    results/main/aggregates/{ds}__{w}__d{dim}__{mod}__{m}.json
    results/main/perquery/{ds}__{w}__d{dim}__{mod}__{m}.npz
    results/main/calibration/{ds}__{w}__d{dim}__{m}__target_{t:.2f}.json
    results/main/status/{ds}__{w}__d{dim}__{m}.status.json
    results/main/run_config.json           resolved run configuration
    results/_meta/run_manifest.json        full provenance manifest
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate import calibrate_index
from utils.ann_io import load_ann_index, CALIBRATION_PARAM
from utils.common import set_global_seed, normalize_modality_label
from utils.config import load_config, cfg_get, config_hash, ConfigError
from utils.index_config import build_index_command
from utils.paths import dataset_csv, emb_dir, index_dir, RESULTS
from utils.provenance import RunManifest, provenance_columns, utc_now
from utils.result_io import (preflight_output, write_dataframe_atomic,
                             write_json_atomic, ResultExistsError)

SCRIPT = "run_revision_experiments"
DEFAULT_CONFIG = "configs/main_cpu.yml"
MAIN_KEY = ["dataset", "weighting", "dim", "modality", "method", "seed"]


def run_config_path() -> Path:
    """Canonical location of the resolved run configuration.

    Must equal the manifest's main.run_config_file
    (results/main/run_config.json); the validator reads the same path.
    """
    return Path(RESULTS["main"]) / "run_config.json"


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


def _require_matching_metadata(path, expected, artifact):
    """Reject stale reusable artifacts instead of trusting path existence."""
    path = Path(path)
    if not path.is_file():
        raise StepError(f"cannot reuse {artifact}: metadata missing at {path}")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StepError(f"cannot reuse {artifact}: invalid metadata {path}: "
                        f"{exc}") from exc
    mismatches = {k: {"expected": v, "found": meta.get(k)}
                  for k, v in expected.items() if meta.get(k) != v}
    if mismatches:
        raise StepError(
            f"refusing to reuse incompatible {artifact} at {path.parent}: "
            f"{mismatches}")


class StatusTracker:
    """Writes results/main/status/{ds}__{w}__d{dim}__{m}.status.json after
    every step so partial runs are inspectable."""

    def __init__(self, status_dir, dataset, weighting, dim, method, steps):
        self.path = (Path(status_dir)
                     / f"{dataset}__{weighting}__d{dim}__{method}.status.json")
        self.doc = {
            "dataset": dataset, "weighting": weighting, "dim": dim,
            "method": method,
            "started_at_utc": utc_now(),
            "overall": "running",
            "steps": {s: {"status": "pending", "error": None,
                          "finished_at_utc": None} for s in steps},
        }
        self._flush()

    def _flush(self):
        write_json_atomic(self.doc, self.path, mode="replace")

    def mark(self, step, status, error=None):
        self.doc["steps"][step] = {
            "status": status,
            "error": str(error) if error else None,
            "finished_at_utc": utc_now(),
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
        self.doc["finished_at_utc"] = utc_now()
        self._flush()
        print(f"[{SCRIPT}] status written: {self.path} "
              f"(overall={self.doc['overall']})")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run the canonical CPU-only revision experiment grid "
                    "(embeddings, indexes, calibration, latency, U2I/I2I "
                    "evaluation) for every dataset x method.")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="experiment YAML (inherits configs/defaults.yml)")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="override the configured dataset list")
    ap.add_argument("--modalities", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--weighting", default=None,
                    choices=["none", "tfidf", "bm25"])
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
                    help="FAISS OMP threads for index builds (1 = reproducible)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--write_mode", default="fail_if_exists",
                    choices=["fail_if_exists", "replace", "merge"],
                    help="behavior when results/main/summary_main.csv exists")
    ap.add_argument("--allow_missing_datasets", action="store_true",
                    help="continue when a configured dataset CSV is missing "
                         "(default: fail before any experiment starts)")
    ap.add_argument("--reuse_existing", action="store_true",
                    help="skip embedding/index building when outputs exist")
    return ap.parse_args()


def main():
    args = parse_args()
    try:
        cfg = load_config(args.config, cli_overrides={
            "datasets": args.datasets,
            "retrieval.modalities": args.modalities,
            "retrieval.methods": args.methods,
            "embedding.weighting": args.weighting,
            "embedding.dim": args.dim,
            "retrieval.budget_mb": args.budget_mb,
            "calibration.targets": args.calibration_targets,
            "calibration.primary_target": args.primary_target,
            "evaluation.queries_large": args.queries_large,
            "evaluation.queries_ml1m": args.queries_ml1m,
            "evaluation.latency_queries": args.latency_queries,
            "retrieval.topk": args.topk,
            "retrieval.metric_topk": args.metric_topk,
            "reproducibility.omp_threads": args.omp_threads,
            "reproducibility.seed": args.seed,
        })
    except ConfigError as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    datasets = list(cfg_get(cfg, "datasets", required=True))
    modalities = [normalize_modality_label(m)
                  for m in cfg_get(cfg, "retrieval.modalities", required=True)]
    methods = list(cfg_get(cfg, "retrieval.methods", required=True))
    weighting = cfg_get(cfg, "embedding.weighting", required=True)
    dim = cfg_get(cfg, "embedding.dim", type=int, required=True)
    normalize = cfg_get(cfg, "embedding.normalize", required=True)
    bm25_k1 = cfg_get(cfg, "embedding.bm25_k1", type=float, required=True)
    bm25_b = cfg_get(cfg, "embedding.bm25_b", type=float, required=True)
    budget_mb = cfg_get(cfg, "retrieval.budget_mb", type=int, required=True)
    targets = [float(t) for t in cfg_get(cfg, "calibration.targets",
                                         required=True)]
    primary_target = cfg_get(cfg, "calibration.primary_target", type=float,
                             required=True)
    cal_queries = cfg_get(cfg, "calibration.queries", type=int, required=True)
    queries_large = str(cfg_get(cfg, "evaluation.queries_large",
                                required=True))
    queries_ml1m = str(cfg_get(cfg, "evaluation.queries_ml1m", required=True))
    latency_queries = cfg_get(cfg, "evaluation.latency_queries", type=int,
                              required=True)
    topk = cfg_get(cfg, "retrieval.topk", type=int, required=True)
    metric_topk = cfg_get(cfg, "retrieval.metric_topk", type=int, required=True)
    omp_threads = cfg_get(cfg, "reproducibility.omp_threads", type=int,
                          required=True)
    seed = cfg_get(cfg, "reproducibility.seed", type=int, required=True)
    default_ef = cfg_get(cfg, "retrieval.runtime_defaults.ef", type=int,
                         required=True)
    default_nprobe = cfg_get(cfg, "retrieval.runtime_defaults.nprobe", type=int,
                             required=True)
    cpu_only = cfg_get(cfg, "hardware.cpu_only", type=bool, required=True)
    if not cpu_only:
        print(f"[{SCRIPT}] ERROR: canonical main configuration must set "
              f"hardware.cpu_only=true")
        sys.exit(1)
    index_cfg = cfg_get(cfg, "index", required=True)
    cfg_hash = config_hash(cfg)

    main_dir = Path(RESULTS["main"])
    meta_dir = Path(RESULTS["meta"])
    agg_dir = Path(RESULTS["aggregates"])
    cal_dir = Path(RESULTS["calibration"])
    status_dir = Path(RESULTS["status"])
    summary_path = main_dir / "summary_main.csv"
    checkpoint_path = status_dir / "summary_main_checkpoint.csv"
    run_config_file = run_config_path()  # canonical: results/main/run_config.json

    # ---- fail-fast argument/input validation (before any expensive work) --
    try:
        write_mode = preflight_output(summary_path, args.write_mode)
        preflight_output(run_config_file, args.write_mode)
        for dataset in datasets:
            for method in methods:
                if write_mode == "fail_if_exists":
                    preflight_output(
                        status_dir / (f"{dataset}__{weighting}__d{dim}__"
                                      f"{method}.status.json"), write_mode)
                if CALIBRATION_PARAM.get(method) is not None:
                    for target in sorted(set(targets + [primary_target])):
                        preflight_output(
                            cal_dir / (f"{dataset}__{weighting}__d{dim}__"
                                       f"{method}__target_{target:.2f}.json"),
                            write_mode)
                for modality in modalities:
                    stem = (f"{dataset}__{weighting}__d{dim}__{modality}__"
                            f"{method}")
                    preflight_output(agg_dir / f"{stem}.json", write_mode)
                    preflight_output(Path(RESULTS["perquery"]) / f"{stem}.npz",
                                     write_mode)
    except (ResultExistsError, ValueError) as e:
        print(f"[{SCRIPT}] ERROR: {e}")
        sys.exit(1)

    missing = [d for d in datasets if not Path(dataset_csv(d)).is_file()]
    if missing and not args.allow_missing_datasets:
        print(f"[{SCRIPT}] ERROR: {len(missing)} configured dataset(s) not "
              f"prepared; refusing to start a supposedly complete run.")
        for d in missing:
            print(f"[{SCRIPT}]   missing {dataset_csv(d)} — prepare with: "
                  f"python src/prepare_dataset.py --dataset {d} "
                  f"--out {dataset_csv(d)}")
        print(f"[{SCRIPT}] pass --allow_missing_datasets to run the others "
              f"anyway (the evidence validator will then fail on the gap).")
        sys.exit(1)
    if missing:
        print(f"[{SCRIPT}] WARN: proceeding WITHOUT {missing} "
              f"(--allow_missing_datasets).")
        datasets = [d for d in datasets if d not in missing]

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.config}")
    print(f"[{SCRIPT}] output path: {main_dir}")
    print(f"[{SCRIPT}] datasets={datasets} methods={methods} "
          f"modalities={modalities} weighting={weighting} dim={dim} "
          f"targets={targets} primary={primary_target} "
          f"omp_threads={omp_threads} seed={seed} write_mode={write_mode}")

    set_global_seed(seed)
    env = None
    for d in (main_dir, meta_dir, agg_dir, cal_dir, status_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest.start(SCRIPT, cfg, cfg_hash, datasets=datasets,
                                 methods=methods, modalities=modalities)
    prov = provenance_columns(manifest.run_id, cfg_hash)

    run_meta = {
        "run_id": manifest.run_id, "config_hash": cfg_hash,
        "datasets": datasets, "modalities": modalities, "methods": methods,
        "weighting": weighting, "dim": dim, "normalize": normalize,
        "bm25_k1": bm25_k1, "bm25_b": bm25_b, "budget_mb": budget_mb,
        "calibration_targets": targets, "primary_target": primary_target,
        "calibration_queries": cal_queries,
        "queries_large": queries_large, "queries_ml1m": queries_ml1m,
        "latency_queries": latency_queries, "topk": topk,
        "metric_topk": metric_topk, "omp_threads": omp_threads,
        "cpu_only": cpu_only, "seed": seed,
    }
    write_json_atomic(run_meta, run_config_file, mode=write_mode)
    manifest.add_output(str(run_config_file))

    def checkpoint(rows):
        df = pd.DataFrame(rows)
        for col, val in prov.items():
            df[col] = val
        write_dataframe_atomic(df, checkpoint_path, mode="replace",
                               key=MAIN_KEY, sort_by=MAIN_KEY)
        print(f"[{SCRIPT}] non-final checkpoint: {checkpoint_path} "
              f"({len(df)} rows)")

    rows = []
    any_failure = False
    for dataset in datasets:
        csv = dataset_csv(dataset)
        n_queries = queries_ml1m if dataset == "ml-1m" else queries_large

        # 1) embeddings (a failure here skips the whole dataset)
        emb = emb_dir(dataset, weighting, dim)
        try:
            if args.reuse_existing and Path(f"{emb}/item_vecs.npy").is_file():
                _require_matching_metadata(
                    Path(emb) / "embedding_meta.json",
                    {"dim_requested": dim, "weighting": weighting,
                     "bm25_k1": bm25_k1, "bm25_b": bm25_b,
                     "normalize": normalize, "seed": seed,
                     "config_hash": cfg_hash},
                    f"{dataset} embeddings")
                print(f"[{SCRIPT}] reusing embeddings {emb}")
            else:
                run(["python", "src/train_embeddings.py", "--interactions", csv,
                     "--dim", str(dim), "--weighting", weighting,
                     "--bm25_k1", str(bm25_k1), "--bm25_b", str(bm25_b),
                     "--normalize", normalize, "--seed", str(seed),
                     "--config_hash", cfg_hash,
                     "--out_dir", emb], env=env)
            item_vecs = np.load(f"{emb}/item_vecs.npy").astype("float32")
            if item_vecs.ndim != 2 or item_vecs.shape[1] != dim:
                raise StepError(
                    f"embedding dimension mismatch for {dataset}: configured "
                    f"d{dim}, loaded shape {item_vecs.shape}")
        except (StepError, OSError) as e:
            print(f"[{SCRIPT}] ERROR: embeddings failed for {dataset}: {e}; "
                  f"skipping dataset.")
            manifest.record_failure({"dataset": dataset, "step": "embeddings"}, e)
            any_failure = True
            continue
        N, D = item_vecs.shape

        for method in methods:
            steps = (["build", "calibration", "latency"]
                     + [f"eval_{m}" for m in modalities])
            tracker = StatusTracker(status_dir, dataset, weighting, dim,
                                    method, steps)
            idx_dir = index_dir(dataset, weighting, dim, method)
            failed = False

            # 2) build index (parameters come from the resolved index config)
            try:
                if args.reuse_existing and Path(idx_dir).exists() \
                        and any(Path(idx_dir).iterdir()):
                    _require_matching_metadata(
                        Path(idx_dir) / "index_meta.json",
                        {"method": method, "budget_mb": budget_mb,
                         "omp_threads": omp_threads, "seed": seed,
                         "config_hash": cfg_hash},
                        f"{dataset}/{method} index")
                    print(f"[{SCRIPT}] reusing index {idx_dir}")
                else:
                    run(build_index_command(
                        method, Path(emb) / "item_vecs.npy",
                        Path(emb) / "item_ids.npy", idx_dir, budget_mb,
                        seed, omp_threads, index_cfg,
                        configuration_hash=cfg_hash), env=env)
                tracker.mark("build", "ok")
            except StepError as e:
                print(f"[{SCRIPT}] ERROR: build failed for {dataset}/{method}: {e}")
                tracker.mark("build", "failed", e)
                manifest.record_failure(
                    {"dataset": dataset, "method": method, "step": "build"}, e)
                failed = True

            # 3) calibration at every target (primary target drives eval)
            ef, nprobe = default_ef, default_nprobe
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
                                              n_queries=cal_queries, seed=seed)
                        res.update({"dataset": dataset, "weighting": weighting,
                                    "dim": dim})
                        cal_path = cal_dir / (
                            f"{dataset}__{weighting}__d{dim}__{method}"
                            f"__target_{target:.2f}.json")
                        write_json_atomic(res, cal_path, mode=write_mode)
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
                    manifest.record_failure(
                        {"dataset": dataset, "method": method,
                         "step": "calibration"}, e)
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
                    manifest.record_failure(
                        {"dataset": dataset, "method": method,
                         "step": "latency"}, e)
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
                         "--config", args.config,
                         "--aggregate_dir", str(agg_dir),
                         "--perquery_dir", RESULTS["perquery"],
                         "--write_mode", write_mode], env=env)
                    tracker.mark(step, "ok")
                except StepError as e:
                    print(f"[{SCRIPT}] ERROR: {step} failed for "
                          f"{dataset}/{method}: {e}")
                    tracker.mark(step, "failed", e)
                    manifest.record_failure(
                        {"dataset": dataset, "method": method, "step": step}, e)
                    any_failure = True
                    continue

                agg_path = agg_dir / (f"{dataset}__{weighting}__d{dim}"
                                      f"__{modality}__{method}.json")
                estats = json.load(open(agg_path)) if agg_path.is_file() else {}
                rows.append({
                    "dataset": dataset, "weighting": weighting, "dim": dim,
                    "modality": modality, "method": method,
                    "budget_mb": budget_mb,
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
            if any(s["status"] == "failed" for s in tracker.doc["steps"].values()):
                any_failure = True
            tracker.finish()

        # checkpoint the summary after each dataset
        if rows:
            checkpoint(rows)

    expected_rows = len(datasets) * len(methods) * len(modalities)
    if rows and not any_failure and len(rows) == expected_rows:
        final_df = pd.DataFrame(rows)
        for col, val in prov.items():
            final_df[col] = val
        write_dataframe_atomic(final_df, summary_path, mode=write_mode,
                               key=MAIN_KEY, sort_by=MAIN_KEY)
        if checkpoint_path.exists():
            os.unlink(checkpoint_path)
        manifest.add_output(str(summary_path))
        print(f"[{SCRIPT}] output path: {summary_path}")
    elif rows:
        checkpoint(rows)
        any_failure = True
        print(f"[{SCRIPT}] WARN: produced {len(rows)}/{expected_rows} rows; "
              f"partial rows remain only in {checkpoint_path}, not in final "
              f"paper evidence.")
    else:
        print(f"[{SCRIPT}] WARN: no rows produced (missing datasets or all "
              f"steps failed — see {status_dir}).")

    manifest.finish("completed_with_failures" if any_failure else "completed")
    print(f"[{SCRIPT}] completed.")
    if any_failure:
        print(f"[{SCRIPT}] NOTE: some combinations failed — see "
              f"{status_dir} and results/_meta/run_manifest.json.")
        sys.exit(1)


if __name__ == "__main__":
    main()
