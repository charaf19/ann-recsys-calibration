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
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate import calibrate_index
from utils.ann_io import load_ann_index, CALIBRATION_PARAM, DEFAULT_PARAM_GRIDS
from utils.common import set_global_seed, normalize_modality_label
from utils.config import load_config, cfg_get, config_hash, ConfigError
from utils.fingerprints import (fingerprint, embedding_fingerprint,
                                index_fingerprint, query_fingerprint,
                                calibration_fingerprint)
from utils.index_config import build_index_command
from utils.modality_queries import (build_query_population, item_id_map,
                                    load_query_cache, write_query_cache)
from utils.paths import dataset_csv, emb_dir, index_dir, RESULTS
from utils.provenance import RunManifest, provenance_columns, utc_now
from utils.result_io import (preflight_output, write_dataframe_atomic,
                             write_json_atomic, validate_unique_keys,
                             ResultExistsError)

SCRIPT = "run_revision_experiments"
DEFAULT_CONFIG = "configs/main_cpu.yml"
MAIN_KEY = ["dataset", "weighting", "dim", "modality", "method", "seed"]
EVALUATION_PROTOCOL_VERSION = "modality_queries_v1"
CHECKPOINT_META_VERSION = 1


def expected_main_grid(datasets, weighting, dim, modalities, methods, seed):
    return [{"dataset": d, "weighting": weighting, "dim": int(dim),
             "modality": mod, "method": method, "seed": int(seed)}
            for d in datasets for method in methods for mod in modalities]


def _row_key(row):
    return tuple(row[field] for field in MAIN_KEY)


def checkpoint_sidecar_path(checkpoint_path):
    path = Path(checkpoint_path)
    return path.with_name(path.name + ".meta.json")


def archive_legacy_checkpoint(checkpoint_path, archive_dir=None):
    """Preserve a sidecar-less checkpoint from the pre-cell-resume protocol."""
    path = Path(checkpoint_path)
    sidecar = checkpoint_sidecar_path(path)
    if not path.exists() or sidecar.exists():
        return None
    archive_dir = Path(archive_dir or path.parent / "archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().replace(":", "-").replace("+", "_")
    destination = archive_dir / f"summary_main_checkpoint_legacy_{stamp}.csv"
    shutil.move(str(path), str(destination))
    return destination


def write_main_checkpoint(rows, checkpoint_path, checkpoint_fingerprint,
                          expected_grid, provenance, created_at=None):
    """Atomically checkpoint after one modality cell and update its sidecar."""
    path = Path(checkpoint_path)
    frame = pd.DataFrame(rows)
    for column, value in provenance.items():
        if column not in frame.columns:
            frame[column] = value
    validate_unique_keys(frame, MAIN_KEY, context="in main checkpoint")
    write_dataframe_atomic(frame, path, mode="replace", key=MAIN_KEY,
                           sort_by=MAIN_KEY)
    now = utc_now()
    doc = {
        "checkpoint_meta_version": CHECKPOINT_META_VERSION,
        "configuration_fingerprint": checkpoint_fingerprint,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "expected_grid": expected_grid, "natural_key": MAIN_KEY,
        "created_at_utc": created_at or now, "updated_at_utc": now,
        "completed_rows": int(len(frame)),
    }
    write_json_atomic(doc, checkpoint_sidecar_path(path), mode="replace")
    return doc


def load_main_checkpoint(checkpoint_path, checkpoint_fingerprint, expected_grid,
                         verify_cell=None):
    """Validate resumable rows and every cell artifact before any skip."""
    path = Path(checkpoint_path)
    meta_path = checkpoint_sidecar_path(path)
    if not path.is_file() or not meta_path.is_file():
        raise ValueError(f"resume requires checkpoint and metadata sidecar: "
                         f"{path}, {meta_path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint metadata {meta_path}: {exc}") from exc
    required = {"checkpoint_meta_version", "configuration_fingerprint",
                "evaluation_protocol_version", "expected_grid", "natural_key",
                "created_at_utc", "updated_at_utc"}
    missing = required - set(meta)
    if missing:
        raise ValueError(f"checkpoint sidecar lacks {sorted(missing)}")
    if meta["checkpoint_meta_version"] != CHECKPOINT_META_VERSION:
        raise ValueError("incompatible main checkpoint metadata version")
    if meta["configuration_fingerprint"] != checkpoint_fingerprint:
        raise ValueError("incompatible main checkpoint configuration fingerprint: "
                         f"expected {checkpoint_fingerprint}, found "
                         f"{meta['configuration_fingerprint']}")
    if meta["evaluation_protocol_version"] != EVALUATION_PROTOCOL_VERSION:
        raise ValueError("incompatible main checkpoint evaluation protocol")
    if meta["natural_key"] != MAIN_KEY or meta["expected_grid"] != expected_grid:
        raise ValueError("incompatible main checkpoint grid or natural key")
    frame = pd.read_csv(path)
    validate_unique_keys(frame, MAIN_KEY, context=f"in checkpoint {path}")
    allowed = {_row_key(row) for row in expected_grid}
    valid_rows = []
    for row in frame.to_dict("records"):
        if _row_key(row) not in allowed:
            raise ValueError(f"checkpoint contains cell outside expected grid: {_row_key(row)}")
        if verify_cell is not None:
            try:
                verify_cell(row)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"[{SCRIPT}] resume: cell {_row_key(row)} is incomplete or "
                      f"invalid and will be rerun: {exc}")
                continue
        valid_rows.append(row)
    return pd.DataFrame(valid_rows, columns=frame.columns), meta


def verify_main_cell(row, agg_dir, perquery_dir, calibration_dir,
                     targets, query_fp, calibration_fp):
    """Validate aggregate, per-query, latency, and tunable calibration evidence."""
    dataset, weighting, dim, modality, method, seed = (
        row[field] for field in MAIN_KEY)
    stem = f"{dataset}__{weighting}__d{int(dim)}__{modality}__{method}"
    agg_path = Path(agg_dir) / f"{stem}.json"
    perquery_path = Path(perquery_dir) / f"{stem}.npz"
    latency_path = Path(index_dir(dataset, weighting, int(dim), method)) / (
        f"latency_{modality}_{method}.json")
    if not agg_path.is_file() or not perquery_path.is_file() or not latency_path.is_file():
        raise ValueError(f"checkpoint cell {stem} lacks required aggregate/per-query/latency artifact")
    agg = json.loads(agg_path.read_text(encoding="utf-8"))
    expected = {"dataset": dataset, "weighting": weighting, "dim": int(dim),
                "modality": modality, "method": method, "seed": int(seed),
                "query_fingerprint": query_fp}
    bad = {key: (value, agg.get(key)) for key, value in expected.items()
           if agg.get(key) != value}
    if bad:
        raise ValueError(f"incompatible aggregate {agg_path}: {bad}")
    latency = json.loads(latency_path.read_text(encoding="utf-8"))
    latency_expected = {"method": method, "N": agg.get("N"), "D": int(dim),
                        "modality": modality, "seed": int(seed),
                        "query_source": "modality_query_cache",
                        "query_fingerprint": query_fp}
    latency_bad = {key: (value, latency.get(key))
                   for key, value in latency_expected.items()
                   if latency.get(key) != value}
    if latency_bad:
        raise ValueError(f"incompatible latency artifact {latency_path}: {latency_bad}")
    with np.load(perquery_path, allow_pickle=True) as z:
        if "meta" not in z or "query_ids" not in z:
            raise ValueError(f"invalid per-query artifact {perquery_path}")
        meta = json.loads(str(z["meta"]))
        if any(meta.get(key) != value for key, value in expected.items()
               if key in {"dataset", "weighting", "dim", "modality", "method",
                          "seed", "query_fingerprint"}):
            raise ValueError(f"incompatible per-query metadata {perquery_path}")
    if CALIBRATION_PARAM.get(method) is not None:
        for target in targets:
            path = Path(calibration_dir) / (
                f"{dataset}__{weighting}__d{int(dim)}__{modality}__{method}"
                f"__target_{float(target):.2f}.json")
            if not path.is_file():
                raise ValueError(f"missing calibration artifact {path}")
            doc = json.loads(path.read_text(encoding="utf-8"))
            cal_expected = {
                "dataset": dataset, "weighting": weighting, "dim": int(dim),
                "modality": modality, "method": method,
                "target_recall": float(target), "seed": int(seed),
                "query_fingerprint": query_fp,
                "calibration_fingerprint": calibration_fp,
                "query_source": "shared_modality_query_cache",
            }
            if (any(doc.get(key) != value for key, value in cal_expected.items())
                    or "target_reached" not in doc
                    or "selected_param_value" not in doc):
                raise ValueError(f"incompatible calibration artifact {path}")
    return True


def run_config_path() -> Path:
    """Canonical location of the resolved run configuration.

    Must equal the manifest's main.run_config_file
    (results/main/run_config.json); the validator reads the same path.
    """
    return Path(RESULTS["main"]) / "run_config.json"


class StepError(Exception):
    """A pipeline step failed; downstream steps for this method are skipped."""


def _legacy_index_fields(method, index_cfg):
    """Construction fields available in pre-fingerprint index metadata."""
    if method == "hnsw":
        return {"M": int(index_cfg["hnsw"]["M"]),
                "efConstruction": int(index_cfg["hnsw"]["ef_construction"])}
    if method in {"ivfpq", "flatpq"}:
        fields = {"m": int(index_cfg["pq"]["m"]),
                  "bits": int(index_cfg["pq"]["bits"])}
        if method == "ivfpq":
            fields["opq"] = bool(index_cfg["ivfpq"]["use_opq"])
        return fields
    return {}


# How many trailing child-output lines to keep for the failure message. The
# deque is bounded so a multi-hour child log never accumulates in RAM.
_TAIL_LINES = 200


def run(cmd, env=None, tail_lines=_TAIL_LINES):
    """Run a child command, streaming its output live to this console.

    stdout and stderr are merged in execution order (stderr -> stdout) and
    forwarded line by line as they arrive, so long-running children still show
    progress. Only the last ``tail_lines`` lines are retained (bounded memory);
    on a nonzero exit that tail — the real child traceback — is embedded in the
    raised StepError so it lands in the run manifest, not just the command.
    """
    full = [sys.executable] + cmd[1:] if cmd[0] == "python" else cmd
    printable = " ".join(str(c) for c in full)
    print(">>", printable, flush=True)

    # Force unbuffered child output so we see progress immediately and the
    # traceback tail is complete even if the child dies mid-write.
    child_env = dict(os.environ if env is None else env)
    child_env["PYTHONUNBUFFERED"] = "1"

    tail = deque(maxlen=max(1, int(tail_lines)))
    proc = subprocess.Popen(
        full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=child_env)
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            tail.append(line.rstrip("\n"))
    finally:
        proc.stdout.close()
        returncode = proc.wait()

    if returncode != 0:
        tail_text = "\n".join(tail).strip()
        raise StepError(
            f"command failed (exit {returncode}): {printable}\n"
            f"--- last {len(tail)} line(s) of child output ---\n"
            f"{tail_text}")


def _require_matching_metadata(path, expected, artifact,
                               allow_legacy_fingerprint=False):
    """Reject stale reusable artifacts instead of trusting path existence."""
    path = Path(path)
    if not path.is_file():
        raise StepError(f"cannot reuse {artifact}: metadata missing at {path}")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StepError(f"cannot reuse {artifact}: invalid metadata {path}: "
                        f"{exc}") from exc
    mismatches = {}
    for key, value in expected.items():
        if (allow_legacy_fingerprint and key.endswith("_fingerprint")
                and meta.get(key) in (None, "")):
            continue
        if meta.get(key) != value:
            mismatches[key] = {"expected": value, "found": meta.get(key)}
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
    ap.add_argument("--resume", action="store_true",
                    help="resume verified modality cells from the main checkpoint")
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
    min_user_interactions = cfg_get(cfg, "data.min_user_interactions",
                                    type=int, required=True)
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
    tail_frac = cfg_get(cfg, "evaluation.tail_fraction", type=float, required=True)
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
    query_cache_dir = main_dir / "query_cache"
    summary_path = main_dir / "summary_main.csv"
    checkpoint_path = status_dir / "summary_main_checkpoint.csv"
    run_config_file = run_config_path()  # canonical: results/main/run_config.json

    expected_grid = expected_main_grid(
        datasets, weighting, dim, modalities, methods, seed)
    checkpoint_fp = fingerprint("main_checkpoint", {
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "grid": expected_grid, "min_user_interactions": min_user_interactions,
        "normalize": normalize, "bm25_k1": bm25_k1, "bm25_b": bm25_b,
        "budget_mb": budget_mb, "targets": targets,
        "primary_target": primary_target, "calibration_queries": cal_queries,
        "queries_large": queries_large, "queries_ml1m": queries_ml1m,
        "latency_queries": latency_queries, "topk": topk,
        "metric_topk": metric_topk, "tail_fraction": tail_frac,
        "omp_threads": omp_threads,
        "index": index_cfg,
    })

    # ---- fail-fast argument/input validation (before any expensive work) --
    try:
        write_mode = preflight_output(summary_path, args.write_mode)
        if not args.resume:
            preflight_output(run_config_file, args.write_mode)
        for dataset in datasets:
            for method in methods:
                if write_mode == "fail_if_exists" and not args.resume:
                    preflight_output(
                        status_dir / (f"{dataset}__{weighting}__d{dim}__"
                                      f"{method}.status.json"), write_mode)
                if CALIBRATION_PARAM.get(method) is not None:
                    for modality in modalities:
                        for target in sorted(set(targets + [primary_target])):
                            if not args.resume:
                                preflight_output(
                                    cal_dir / (f"{dataset}__{weighting}__d{dim}__"
                                               f"{modality}__{method}"
                                               f"__target_{target:.2f}.json"),
                                    write_mode)
                for modality in modalities:
                    stem = (f"{dataset}__{weighting}__d{dim}__{modality}__"
                            f"{method}")
                    if not args.resume:
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
    for d in (main_dir, meta_dir, agg_dir, cal_dir, status_dir, query_cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    checkpoint_meta_path = checkpoint_sidecar_path(checkpoint_path)
    if checkpoint_meta_path.exists() and not checkpoint_path.exists():
        print(f"[{SCRIPT}] ERROR: orphaned checkpoint sidecar exists without "
              f"its CSV: {checkpoint_meta_path}")
        sys.exit(1)
    if checkpoint_path.exists() and not checkpoint_meta_path.exists():
        archived = archive_legacy_checkpoint(checkpoint_path)
        print(f"[{SCRIPT}] archived legacy checkpoint (not protocol-safe): {archived}")
    elif checkpoint_path.exists() and not args.resume:
        print(f"[{SCRIPT}] ERROR: unfinished compatible-protocol checkpoint "
              f"exists: {checkpoint_path}; use --resume")
        sys.exit(1)

    manifest = RunManifest.start(SCRIPT, cfg, cfg_hash, datasets=datasets,
                                 methods=methods, modalities=modalities)
    prov = provenance_columns(manifest.run_id, cfg_hash)

    run_meta = {
        "run_id": manifest.run_id, "config_hash": cfg_hash,
        "datasets": datasets, "modalities": modalities, "methods": methods,
        "weighting": weighting, "min_user_interactions": min_user_interactions,
        "dim": dim, "normalize": normalize,
        "bm25_k1": bm25_k1, "bm25_b": bm25_b, "budget_mb": budget_mb,
        "calibration_targets": targets, "primary_target": primary_target,
        "calibration_queries": cal_queries,
        "queries_large": queries_large, "queries_ml1m": queries_ml1m,
        "latency_queries": latency_queries, "topk": topk,
        "metric_topk": metric_topk, "tail_fraction": tail_frac,
        "omp_threads": omp_threads,
        "cpu_only": cpu_only, "seed": seed,
    }
    run_meta["checkpoint_fingerprint"] = checkpoint_fp
    run_meta["evaluation_protocol_version"] = EVALUATION_PROTOCOL_VERSION
    write_json_atomic(run_meta, run_config_file,
                      mode="replace" if args.resume else write_mode)
    manifest.add_output(str(run_config_file))

    checkpoint_created = None

    def checkpoint(rows):
        nonlocal checkpoint_created
        meta = write_main_checkpoint(rows, checkpoint_path, checkpoint_fp,
                                     expected_grid, prov,
                                     created_at=checkpoint_created)
        checkpoint_created = meta["created_at_utc"]
        print(f"[{SCRIPT}] non-final checkpoint: {checkpoint_path} "
              f"({len(rows)} rows)")

    rows = []
    if args.resume and checkpoint_path.exists():
        # Per-cell artifact verification is deferred until dataset query
        # fingerprints are known below; schema/grid/fingerprint are strict here.
        resumed, checkpoint_meta = load_main_checkpoint(
            checkpoint_path, checkpoint_fp, expected_grid)
        rows = resumed.to_dict("records")
        checkpoint_created = checkpoint_meta["created_at_utc"]
        print(f"[{SCRIPT}] resume: loaded {len(rows)} checkpoint rows for verification")
    any_failure = False
    for dataset in datasets:
        csv = dataset_csv(dataset)
        n_queries = queries_ml1m if dataset == "ml-1m" else queries_large
        max_queries = "full" if str(n_queries).lower() == "full" else int(n_queries)
        emb_fp = embedding_fingerprint(
            dataset=dataset, min_user_interactions=min_user_interactions,
            weighting=weighting, dim=dim, normalize=normalize,
            bm25_k1=bm25_k1, bm25_b=bm25_b, seed=seed)

        # 1) embeddings (a failure here skips the whole dataset)
        emb = emb_dir(dataset, weighting, dim)
        try:
            if args.reuse_existing and Path(f"{emb}/item_vecs.npy").is_file():
                _require_matching_metadata(
                    Path(emb) / "embedding_meta.json",
                    {"dim_requested": dim, "weighting": weighting,
                     "bm25_k1": bm25_k1, "bm25_b": bm25_b,
                     "normalize": normalize, "seed": seed,
                     "min_user_interactions": min_user_interactions,
                     "embedding_fingerprint": emb_fp},
                    f"{dataset} embeddings", allow_legacy_fingerprint=True)
                print(f"[{SCRIPT}] reusing embeddings {emb}")
            else:
                run(["python", "src/train_embeddings.py", "--interactions", csv,
                     "--dim", str(dim), "--weighting", weighting,
                     "--bm25_k1", str(bm25_k1), "--bm25_b", str(bm25_b),
                     "--normalize", normalize, "--seed", str(seed),
                     "--min_user_interactions", str(min_user_interactions),
                     "--config_hash", cfg_hash,
                     "--embedding_fingerprint", emb_fp,
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

        # Construct each modality population once per dataset/embedding stage.
        id2idx = item_id_map(Path(emb) / "item_vecs.npy", N)
        populations = {}
        query_fps = {}
        for modality in modalities:
            qfp = query_fingerprint(
                embedding_fp=emb_fp, dataset=dataset, modality=modality,
                weighting=weighting, dim=dim,
                min_user_interactions=min_user_interactions,
                max_queries=max_queries, topk=topk, metric_topk=metric_topk,
                seed=seed)
            query_fps[modality] = qfp
            cache_path = query_cache_dir / (
                f"{dataset}__{weighting}__d{dim}__{modality}.npz")
            expected_cache = {
                "dataset": dataset, "weighting": weighting, "dim": int(dim),
                "modality": modality, "embedding_fingerprint": emb_fp,
                "query_fingerprint": qfp, "seed": int(seed),
                "min_user_interactions": int(min_user_interactions),
                "max_queries": str(max_queries), "n_items": int(N),
                "topk": int(topk), "metric_topk": int(metric_topk),
                "split_protocol": "temporal_leave_one_out_v1",
                "query_construction": ("mean_training_history_v1"
                                       if modality == "u2i"
                                       else "last_training_item_v1"),
            }
            try:
                populations[modality] = load_query_cache(cache_path, expected_cache)
                print(f"[{SCRIPT}] reusing validated query cache {cache_path}")
            except ValueError as exc:
                if cache_path.exists() or cache_path.with_suffix(".meta.json").exists():
                    print(f"[{SCRIPT}] rebuilding incompatible query cache: {exc}")
                population = build_query_population(
                    interactions_path=csv, item_vecs=item_vecs, id2idx=id2idx,
                    modality=modality, max_queries=max_queries, seed=seed,
                    min_user_interactions=min_user_interactions,
                    metadata=expected_cache)
                write_query_cache(population, cache_path, mode="replace")
                populations[modality] = population

        # A checkpoint row is skippable only after its scientific artifacts
        # validate against this dataset's current query/calibration fingerprints.
        verified_rows = []
        for row in rows:
            if row["dataset"] != dataset:
                verified_rows.append(row)
                continue
            method = row["method"]
            modality = row["modality"]
            cal_fp = calibration_fingerprint(
                query_fp=query_fps[modality], targets=targets,
                n_queries=cal_queries,
                param_grid=DEFAULT_PARAM_GRIDS.get(method), topk=topk, seed=seed)
            try:
                verify_main_cell(
                    row, agg_dir, RESULTS["perquery"], cal_dir, targets,
                    query_fps[modality], cal_fp)
                verified_rows.append(row)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"[{SCRIPT}] resume: rerunning invalid cell {_row_key(row)}: {exc}")
        rows = verified_rows
        completed_keys = {_row_key(row) for row in rows}

        for method in methods:
            method_keys = {
                (dataset, weighting, dim, modality, method, seed)
                for modality in modalities}
            if method_keys <= completed_keys:
                print(f"[{SCRIPT}] resume: skipping verified completed method "
                      f"{dataset}/{method} (both modalities)")
                continue
            steps = (["build", "calibration", "latency"]
                     + [f"eval_{m}" for m in modalities])
            tracker = StatusTracker(status_dir, dataset, weighting, dim,
                                    method, steps)
            idx_dir = index_dir(dataset, weighting, dim, method)
            idx_fp = index_fingerprint(
                embedding_fp=emb_fp, method=method, budget_mb=budget_mb,
                omp_threads=omp_threads, index_config=index_cfg, seed=seed)
            failed = False

            # 2) build index (parameters come from the resolved index config)
            try:
                if args.reuse_existing and Path(idx_dir).exists() \
                        and any(Path(idx_dir).iterdir()):
                    _require_matching_metadata(
                        Path(idx_dir) / "index_meta.json",
                        {"method": method, "budget_mb": budget_mb,
                         "omp_threads": omp_threads, "seed": seed,
                         "embedding_fingerprint": emb_fp,
                         "index_fingerprint": idx_fp,
                         **_legacy_index_fields(method, index_cfg)},
                        f"{dataset}/{method} index",
                        allow_legacy_fingerprint=True)
                    print(f"[{SCRIPT}] reusing index {idx_dir}")
                else:
                    run(build_index_command(
                        method, Path(emb) / "item_vecs.npy",
                        Path(emb) / "item_ids.npy", idx_dir, budget_mb,
                        seed, omp_threads, index_cfg,
                        configuration_hash=cfg_hash,
                        embedding_fingerprint=emb_fp,
                        index_fingerprint=idx_fp), env=env)
                tracker.mark("build", "ok")
            except StepError as e:
                print(f"[{SCRIPT}] ERROR: build failed for {dataset}/{method}: {e}")
                tracker.mark("build", "failed", e)
                manifest.record_failure(
                    {"dataset": dataset, "method": method, "step": "build"}, e)
                failed = True

            # 3-5) calibrate, time, and evaluate each modality independently.
            for modality in modalities:
                step = f"eval_{modality}"
                cell_key = (dataset, weighting, dim, modality, method, seed)
                if cell_key in completed_keys:
                    tracker.mark(step, "skipped")
                    continue
                if failed:
                    tracker.mark(step, "skipped")
                    continue
                query_cache = query_cache_dir / (
                    f"{dataset}__{weighting}__d{dim}__{modality}.npz")
                qfp = query_fps[modality]
                cal_fp = calibration_fingerprint(
                    query_fp=qfp, targets=targets, n_queries=cal_queries,
                    param_grid=DEFAULT_PARAM_GRIDS.get(method), topk=topk,
                    seed=seed)
                ef, nprobe = default_ef, default_nprobe
                primary_cal = None
                cell_write_mode = "replace" if args.resume else write_mode

                if CALIBRATION_PARAM.get(method) is None:
                    tracker.mark("calibration", "skipped")
                else:
                    try:
                        for target in sorted(set(targets + [primary_target])):
                            ann = load_ann_index(idx_dir, D, N)
                            res = calibrate_index(
                                ann, item_vecs, target, topk,
                                n_queries=cal_queries, seed=seed,
                                query_vectors=populations[modality].query_vectors,
                                modality=modality,
                                query_source="shared_modality_query_cache",
                                query_fingerprint=qfp)
                            res.update({
                                "dataset": dataset, "weighting": weighting,
                                "dim": dim, "modality": modality,
                                "calibration_fingerprint": cal_fp,
                            })
                            cal_path = cal_dir / (
                                f"{dataset}__{weighting}__d{dim}__{modality}"
                                f"__{method}__target_{target:.2f}.json")
                            write_json_atomic(res, cal_path, mode=cell_write_mode)
                            if abs(target - primary_target) < 1e-9:
                                primary_cal = res
                        selected = (primary_cal or {}).get("selected_param_value")
                        if selected is not None:
                            if CALIBRATION_PARAM[method] == "ef":
                                ef = int(selected)
                            else:
                                nprobe = int(selected)
                        tracker.mark("calibration", "ok")
                    except Exception as exc:
                        print(f"[{SCRIPT}] ERROR: calibration failed for "
                              f"{dataset}/{method}/{modality}: {exc}")
                        tracker.mark("calibration", "failed", exc)
                        tracker.mark(step, "skipped")
                        manifest.record_failure(
                            {"dataset": dataset, "method": method,
                             "modality": modality, "step": "calibration"}, exc)
                        any_failure = True
                        continue

                lstats = {}
                try:
                    lat_file = Path(idx_dir) / f"latency_{modality}_{method}.json"
                    run(["python", "src/run_device.py", "--index", idx_dir,
                         "--item_vecs", f"{emb}/item_vecs.npy",
                         "--query_vecs", str(query_cache),
                         "--modality", modality,
                         "--query_fingerprint", qfp,
                         "--out", str(lat_file),
                         "--queries", str(latency_queries), "--topk", str(topk),
                         "--ef", str(ef), "--nprobe", str(nprobe),
                         "--seed", str(seed)], env=env)
                    lstats = json.loads(lat_file.read_text(encoding="utf-8"))
                    tracker.mark("latency", "ok")
                except (StepError, OSError, json.JSONDecodeError) as exc:
                    print(f"[{SCRIPT}] ERROR: latency failed for "
                          f"{dataset}/{method}/{modality}: {exc}")
                    tracker.mark("latency", "failed", exc)
                    manifest.record_failure(
                        {"dataset": dataset, "method": method,
                         "modality": modality, "step": "latency"}, exc)
                    any_failure = True
                    continue

                try:
                    run(["python", "src/eval_modalities.py",
                         "--interactions", csv,
                         "--item_vecs", f"{emb}/item_vecs.npy",
                         "--index", idx_dir,
                         "--ann_method", method,
                         "--modality", modality,
                         "--query_cache", str(query_cache),
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
                         "--write_mode", cell_write_mode], env=env)
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
                    "calibration_target_reached": (primary_cal or {}).get("target_reached"),
                    "calibration_achieved_recall_vs_exact": (primary_cal or {}).get(
                        "achieved_recall_vs_exact"),
                    "calibrated_param_name": CALIBRATION_PARAM.get(method),
                    "calibrated_param_value": (primary_cal or {}).get("selected_param_value"),
                    "latency_p50_ms": (lstats.get("latency_ms") or {}).get("p50"),
                    "latency_p95_ms": (lstats.get("latency_ms") or {}).get("p95"),
                    "latency_p99_ms": (lstats.get("latency_ms") or {}).get("p99"),
                    "latency_mean_ms": (lstats.get("latency_ms") or {}).get("mean"),
                    "rss_mb_after": lstats.get("rss_mb_after"),
                    "query_fingerprint": qfp,
                    "calibration_fingerprint": cal_fp,
                    "embedding_fingerprint": emb_fp,
                    "index_fingerprint": idx_fp,
                    "omp_threads": omp_threads,
                    "seed": seed,
                })
                completed_keys.add(cell_key)
                checkpoint(rows)  # interruption-safe after every modality cell
            if any(s["status"] == "failed" for s in tracker.doc["steps"].values()):
                any_failure = True
            tracker.finish()

    expected_rows = len(datasets) * len(methods) * len(modalities)
    actual_keys = {_row_key(row) for row in rows}
    expected_keys = {_row_key(row) for row in expected_grid}
    if rows and not any_failure and len(rows) == expected_rows \
            and actual_keys == expected_keys:
        final_df = pd.DataFrame(rows)
        for col, val in prov.items():
            final_df[col] = val
        write_dataframe_atomic(final_df, summary_path, mode=write_mode,
                               key=MAIN_KEY, sort_by=MAIN_KEY)
        if checkpoint_path.exists():
            os.unlink(checkpoint_path)
        sidecar = checkpoint_sidecar_path(checkpoint_path)
        if sidecar.exists():
            os.unlink(sidecar)
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
