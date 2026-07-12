"""Consolidate validated modality-specific main calibration evidence."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from utils.ann_io import CALIBRATION_PARAM
from utils.config import load_config, cfg_get, config_hash, ConfigError
from utils.paths import RESULTS
from utils.provenance import make_run_id, provenance_columns
from utils.result_io import preflight_output, write_dataframe_atomic, ResultExistsError

SCRIPT = "run_calibration_sensitivity"
DEFAULT_CONFIG = "configs/main_cpu.yml"
KEY = ["dataset", "weighting", "dim", "modality", "method",
       "target_recall", "seed"]


def calibration_filename(dataset, weighting, dim, modality, method, target):
    return (f"{dataset}__{weighting}__d{dim}__{modality}__{method}"
            f"__target_{float(target):.2f}.json")


def consolidate_calibration_artifacts(source_dir, datasets, modalities, methods,
                                      targets, weighting, dim, seed):
    """Read exactly the configured grid; reject missing or incompatible JSON."""
    source_dir = Path(source_dir)
    expected = {(d, mod, method, float(target), int(seed))
                for d in datasets for mod in modalities for method in methods
                for target in targets}
    records = {}
    for path in sorted(source_dir.glob("*.json")):
        # Preserve and ignore pre-modality legacy artifacts. Only files whose
        # names claim a configured modality participate in the new contract.
        if not any(f"__{modality}__" in path.name for modality in modalities):
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid calibration artifact {path}: {exc}") from exc
        if doc.get("method") not in methods:
            continue
        required = {"dataset", "weighting", "dim", "modality", "method",
                    "target_recall", "target_reached", "selected_param_value",
                    "achieved_recall_vs_exact", "n_calibration_queries",
                    "query_source", "seed", "calibration_fingerprint"}
        missing = sorted(required - set(doc))
        if missing:
            raise ValueError(f"calibration artifact {path} lacks {missing}")
        key = (doc["dataset"], doc["modality"], doc["method"],
               float(doc["target_recall"]), int(doc["seed"]))
        if key not in expected:
            continue
        if key in records:
            raise ValueError(f"duplicate calibration artifacts for {key}: "
                             f"{records[key][0]} and {path}")
        expected_name = calibration_filename(
            doc["dataset"], weighting, dim, doc["modality"], doc["method"],
            doc["target_recall"])
        mismatches = {}
        for field, value in {"weighting": weighting, "dim": int(dim),
                             "seed": int(seed)}.items():
            if doc.get(field) != value:
                mismatches[field] = {"expected": value, "found": doc.get(field)}
        if path.name != expected_name:
            mismatches["filename"] = {"expected": expected_name,
                                      "found": path.name}
        if mismatches:
            raise ValueError(f"incompatible calibration artifact {path}: {mismatches}")
        records[key] = (path, doc)
    missing = expected - set(records)
    if missing:
        raise ValueError(f"missing {len(missing)} main calibration artifacts, "
                         f"including {sorted(missing, key=str)[:3]}")

    rows = []
    for key in sorted(records, key=str):
        _, doc = records[key]
        latency = doc.get("latency_ms_at_calibrated") or {}
        rows.append({
            "dataset": doc["dataset"], "weighting": doc["weighting"],
            "dim": int(doc["dim"]), "modality": doc["modality"],
            "method": doc["method"], "target_recall": float(doc["target_recall"]),
            "target_reached": bool(doc["target_reached"]),
            "param_name": doc.get("param_name"),
            "selected_param_value": doc["selected_param_value"],
            "calibrated_param_value": doc["selected_param_value"],
            "achieved_recall_vs_exact": doc["achieved_recall_vs_exact"],
            "latency_mean_ms": latency.get("mean"),
            "latency_p50_ms": latency.get("p50"),
            "latency_p95_ms": latency.get("p95"),
            "latency_p99_ms": latency.get("p99"),
            "topk": doc.get("topk"),
            "n_calibration_queries": doc["n_calibration_queries"],
            "query_source": doc["query_source"],
            "query_fingerprint": doc.get("query_fingerprint"),
            "calibration_fingerprint": doc["calibration_fingerprint"],
            "seed": int(doc["seed"]),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Consolidate modality-specific main calibration artifacts; "
                    "does not execute FAISS searches.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--source_dir", default=RESULTS["calibration"])
    ap.add_argument("--out_dir", default=RESULTS["calibration_sensitivity"])
    ap.add_argument("--write_mode", default="fail_if_exists",
                    choices=["fail_if_exists", "replace", "merge"])
    args = ap.parse_args()
    try:
        cfg = load_config(args.config)
        datasets = list(cfg_get(cfg, "datasets", required=True))
        modalities = list(cfg_get(cfg, "retrieval.modalities", required=True))
        methods = [m for m in cfg_get(cfg, "retrieval.methods", required=True)
                   if CALIBRATION_PARAM.get(m) is not None]
        targets = [float(v) for v in cfg_get(cfg, "calibration.targets", required=True)]
        weighting = cfg_get(cfg, "embedding.weighting", required=True)
        dim = cfg_get(cfg, "embedding.dim", type=int, required=True)
        seed = cfg_get(cfg, "reproducibility.seed", type=int, required=True)
    except ConfigError as exc:
        print(f"[{SCRIPT}] ERROR: {exc}")
        return 1
    out = Path(args.out_dir) / "calibration_sensitivity.csv"
    try:
        preflight_output(out, args.write_mode)
        rows = consolidate_calibration_artifacts(
            args.source_dir, datasets, modalities, methods, targets,
            weighting, dim, seed)
    except (ResultExistsError, ValueError) as exc:
        print(f"[{SCRIPT}] ERROR: {exc}")
        return 1
    prov = provenance_columns(make_run_id(config_hash(cfg)), config_hash(cfg))
    frame = pd.DataFrame([{**row, **prov} for row in rows])
    write_dataframe_atomic(frame, out, mode=args.write_mode, key=KEY, sort_by=KEY)
    print(f"[{SCRIPT}] output path: {out} ({len(frame)} rows; reused main evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
