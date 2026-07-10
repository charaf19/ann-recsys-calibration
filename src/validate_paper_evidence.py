"""Strict paper-evidence validator (the ONLY validator in this repository).

Validates COMPLETENESS of the evidence, not mere file existence: every
dataset/method/modality combination, expected row counts, uniform contract
values (weighting/dim/seed), bootstrap iteration counts, cost-only scale
rows, CPU-only scope, and schema invariants.

The contract itself lives in configs/paper_evidence_manifest.yml — this
script only interprets it. Because results/ starts empty, missing evidence
is EXPECTED to fail until the final experiments are rerun; the validator
never fabricates or repairs anything.

Outputs:
    results/_meta/validation_report.csv
    results/_meta/validation_report.md
    results/_meta/validation_report.json
Exit status: non-zero when any critical check fails.
"""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

from utils.paths import RESULTS
from utils.reporting import df_to_markdown
from utils.result_io import write_json_atomic, _atomic_write_text

SCRIPT = "validate_paper_evidence"
DEFAULT_MANIFEST = "configs/paper_evidence_manifest.yml"


class Checker:
    """Collects (section, check, status, detail) rows."""

    def __init__(self):
        self.rows = []

    def record(self, section, check, ok, detail=""):
        self.rows.append({"section": section, "check": check,
                          "status": "pass" if ok else "fail",
                          "detail": str(detail)})
        return ok

    def fail(self, section, check, detail=""):
        return self.record(section, check, False, detail)

    def ok(self, section, check, detail=""):
        return self.record(section, check, True, detail)


def _read_csv(ck, section, path):
    p = Path(path)
    if not p.is_file():
        ck.fail(section, "file_exists",
                f"missing evidence: {p} (expected before the final rerun)")
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:
        ck.fail(section, "file_readable", f"{p}: {e}")
        return None
    if len(df) == 0:
        ck.fail(section, "file_nonempty", f"{p} has 0 rows")
        return None
    ck.ok(section, "file_exists", str(p))
    return df


def _check_columns(ck, section, df, required):
    missing = [c for c in required if c not in df.columns]
    return ck.record(section, "required_columns", not missing,
                     f"missing columns: {missing}" if missing else "all present")


def _check_unique(ck, section, df, key):
    key = [k for k in key if k in df.columns]
    dups = int(df.duplicated(subset=key).sum()) if key else -1
    return ck.record(section, "no_duplicate_keys", dups == 0,
                     f"{dups} duplicate natural keys on {key}" if dups else "")


def _check_uniform(ck, section, df, column, expected):
    if column not in df.columns:
        return ck.fail(section, f"uniform_{column}", f"column {column} absent")
    vals = set(df[column].dropna().unique().tolist())
    norm = {str(v) for v in vals}
    ok = norm == {str(expected)}
    return ck.record(section, f"uniform_{column}", ok,
                     f"expected {expected!r} on every row, found {sorted(norm)}")


def validate_scope(ck, m):
    sec = "cpu_scope"
    spec = m.get("scope", {})
    hw_path = Path(spec.get("hardware_file", "results/_meta/hardware.json"))
    if not hw_path.is_file():
        ck.fail(sec, "hardware_declared",
                f"missing {hw_path}; run capture_hardware.py")
    else:
        try:
            hw = json.loads(hw_path.read_text(encoding="utf-8"))
        except Exception as e:
            hw = None
            ck.fail(sec, "hardware_readable", f"{hw_path}: {e}")
        if hw is not None:
            for field, expected in (spec.get("require_fields") or {}).items():
                ck.record(sec, f"hardware_{field}",
                          field in hw and hw[field] == expected,
                          f"expected {field}={expected}, "
                          f"found {hw.get(field, '<absent>')}")
    for p in spec.get("forbidden_result_paths", []):
        ck.record(sec, "no_gpu_result_paths", not Path(p).exists(),
                  f"forbidden path present: {p}" if Path(p).exists() else p)


def validate_main(ck, m):
    sec = "main"
    spec = m.get("main", {})
    df = _read_csv(ck, sec, spec.get("file", ""))
    # run-config cross-check happens even when the summary is missing
    rc_path = Path(spec.get("run_config_file", ""))
    if rc_path.name:
        if not rc_path.is_file():
            ck.fail(sec, "run_config_exists", f"missing {rc_path}")
        else:
            rc = json.loads(rc_path.read_text(encoding="utf-8"))
            for k, v in (spec.get("run_config_must_match") or {}).items():
                ck.record(sec, f"run_config_{k}", str(rc.get(k)) == str(v),
                          f"expected {k}={v}, found {rc.get(k, '<absent>')}")
    if df is None:
        return
    if not _check_columns(ck, sec, df, spec.get("required_columns", [])):
        return
    _check_unique(ck, sec, df, spec.get("key", []))
    _check_uniform(ck, sec, df, "weighting", spec.get("weighting"))
    _check_uniform(ck, sec, df, "dim", spec.get("dim"))
    _check_uniform(ck, sec, df, "seed", spec.get("seed"))

    datasets = set(spec.get("datasets", []))
    modalities = set(spec.get("modalities", []))
    methods = set(spec.get("methods", []))
    actual = set(map(tuple, df[["dataset", "modality", "method"]]
                     .astype(str).values))
    expected = set(product(sorted(datasets), sorted(modalities),
                           sorted(methods)))
    missing = expected - actual
    unexpected_ds = set(df["dataset"].astype(str)) - datasets
    ck.record(sec, "all_combinations_present", not missing,
              f"{len(missing)} of {len(expected)} combinations missing "
              f"(e.g. {sorted(missing)[:3]})" if missing else
              f"all {len(expected)} present")
    ck.record(sec, "no_unexpected_datasets", not unexpected_ds,
              f"unexpected datasets: {sorted(unexpected_ds)}")
    ck.record(sec, "expected_row_count",
              len(df) == int(spec.get("expected_rows", len(df))),
              f"expected {spec.get('expected_rows')}, found {len(df)}")

    forbidden = [c for c in spec.get("forbidden_columns", [])
                 if c in df.columns]
    ck.record(sec, "no_gpu_columns", not forbidden,
              f"forbidden GPU columns present: {forbidden}")
    for col in spec.get("metric_columns_not_all_null", []):
        if col in df.columns:
            ck.record(sec, f"metric_not_all_null_{col}",
                      df[col].notna().any(), f"{col} is entirely null")


def validate_calibration_sensitivity(ck, m):
    sec = "calibration_sensitivity"
    spec = m.get(sec, {})
    df = _read_csv(ck, sec, spec.get("file", ""))
    if df is None:
        return
    if not _check_columns(ck, sec, df, spec.get("required_columns", [])):
        return
    _check_unique(ck, sec, df, spec.get("key", []))
    datasets = set(spec.get("datasets", []))
    methods = set(spec.get("methods", []))
    targets = {f"{float(t):.2f}" for t in spec.get("targets", [])}
    actual = {(str(r.dataset), str(r.method), f"{float(r.target_recall):.2f}")
              for r in df.itertuples()}
    expected = set(product(sorted(datasets), sorted(methods), sorted(targets)))
    missing = expected - actual
    ck.record(sec, "all_combinations_present", not missing,
              f"{len(missing)} of {len(expected)} combinations missing"
              if missing else f"all {len(expected)} present")
    ck.record(sec, "expected_row_count",
              len(df) == int(spec.get("expected_rows", len(df))),
              f"expected {spec.get('expected_rows')}, found {len(df)}")
    found_targets = {f"{float(t):.2f}" for t in df["target_recall"].unique()}
    ck.record(sec, "targets_match", found_targets == targets,
              f"expected targets {sorted(targets)}, found {sorted(found_targets)}")


def validate_embedding_sensitivity(ck, m):
    sec = "embedding_sensitivity"
    spec = m.get(sec, {})
    df = _read_csv(ck, sec, spec.get("file", ""))
    if df is None:
        return
    if not _check_columns(ck, sec, df, spec.get("required_columns", [])):
        return
    datasets = set(spec.get("datasets", []))
    modalities = set(spec.get("modalities", []))
    backbones = set(spec.get("backbones", []))
    optional = set(spec.get("optional_backbones", []))
    methods = set(spec.get("methods", []))

    required_rows = df[df["backbone"].isin(backbones)]
    _check_unique(ck, sec, required_rows, spec.get("key", []))
    actual = set(map(tuple, required_rows[["dataset", "backbone", "modality",
                                           "method"]].astype(str).values))
    expected = set(product(sorted(datasets), sorted(backbones),
                           sorted(modalities), sorted(methods)))
    missing = expected - actual
    ck.record(sec, "required_backbones_complete", not missing,
              f"{len(missing)} of {len(expected)} required backbone rows "
              f"missing (BPR and all five methods are required)"
              if missing else f"all {len(expected)} present")
    ck.record(sec, "expected_row_count",
              len(required_rows) == int(spec.get("expected_rows",
                                                 len(required_rows))),
              f"expected {spec.get('expected_rows')} required-backbone rows, "
              f"found {len(required_rows)}")
    unexpected = (set(df["backbone"].astype(str)) - backbones - optional)
    ck.record(sec, "no_unexpected_backbones", not unexpected,
              f"unexpected backbones: {sorted(unexpected)}")
    # stability must be REPORTED (column exists); values are never judged
    ck.record(sec, "ranking_stability_reported",
              "ann_ranking_stability" in df.columns,
              "ann_ranking_stability column present"
              if "ann_ranking_stability" in df.columns else
              "ann_ranking_stability column missing")


def validate_scale_stress(ck, m):
    sec = "scale_stress"
    spec = m.get(sec, {})
    df = _read_csv(ck, sec, spec.get("file", ""))
    if df is None:
        return
    if not _check_columns(ck, sec, df, spec.get("required_columns", [])):
        return
    _check_unique(ck, sec, df, spec.get("key", []))
    sizes = {int(s) for s in spec.get("catalog_sizes", [])}
    dims = {int(d) for d in spec.get("dimensions", [])}
    methods = set(spec.get("methods", []))
    actual = {(int(r.n_items), int(r.dim), str(r.method))
              for r in df.itertuples()}
    expected = set(product(sorted(sizes), sorted(dims), sorted(methods)))
    missing = expected - actual
    ck.record(sec, "all_cells_present", not missing,
              f"{len(missing)} of {len(expected)} grid cells missing"
              if missing else f"all {len(expected)} present")
    ck.record(sec, "expected_row_count",
              len(df) == int(spec.get("expected_rows", len(df))),
              f"expected {spec.get('expected_rows')}, found {len(df)}")
    for col, val in (spec.get("require_values") or {}).items():
        if col not in df.columns:
            ck.fail(sec, f"require_{col}", f"column {col} absent")
            continue
        bad = df[df[col].astype(str).str.lower() != str(val).lower()]
        ck.record(sec, f"require_{col}_{val}", len(bad) == 0,
                  f"{len(bad)} rows violate {col}={val} "
                  f"(cost-only contract)" if len(bad) else "all rows comply")
    forbidden = [c for c in spec.get("forbidden_quality_columns", [])
                 if c in df.columns and df[c].notna().any()]
    ck.record(sec, "no_synthetic_quality_columns", not forbidden,
              f"recommendation-quality columns with values on synthetic "
              f"data: {forbidden}")


def validate_bootstrap(ck, m):
    sec = "bootstrap"
    spec = m.get(sec, {})
    n_boot = int(spec.get("n_boot", 2000))
    cis = _read_csv(ck, sec, spec.get("cis_file", ""))
    tests = _read_csv(ck, sec, spec.get("tests_file", ""))
    if cis is not None:
        _check_columns(ck, sec, cis, spec.get("required_columns_cis", []))
        if "n_boot" in cis.columns:
            bad = int((cis["n_boot"].astype(int) != n_boot).sum())
            ck.record(sec, "cis_n_boot", bad == 0,
                      f"{bad} rows with n_boot != {n_boot}" if bad
                      else f"n_boot={n_boot} on every row")
        datasets = set(spec.get("datasets", []))
        modalities = set(spec.get("modalities", []))
        methods = set(spec.get("methods", []))
        metrics = set(spec.get("metrics", []))
        actual = set(map(tuple, cis[["dataset", "modality", "method"]]
                         .astype(str).values))
        expected = set(product(sorted(datasets), sorted(modalities),
                               sorted(methods)))
        missing = expected - actual
        ck.record(sec, "cis_coverage", not missing,
                  f"{len(missing)} of {len(expected)} groups missing"
                  if missing else f"all {len(expected)} groups present")
        found_metrics = set(cis["metric"].astype(str))
        ck.record(sec, "cis_metrics_reported", metrics <= found_metrics,
                  f"expected at least {sorted(metrics)}, found "
                  f"{sorted(found_metrics)}")
    if tests is not None:
        _check_columns(ck, sec, tests, spec.get("required_columns_tests", []))
        if "n_boot" in tests.columns:
            bad = int((tests["n_boot"].astype(int) != n_boot).sum())
            ck.record(sec, "tests_n_boot", bad == 0,
                      f"{bad} rows with n_boot != {n_boot}" if bad
                      else f"n_boot={n_boot} on every row")
        comp = set(spec.get("comparison_methods", []))
        found = set(tests["method"].astype(str))
        ck.record(sec, "tests_comparisons_present", comp <= found,
                  f"expected comparisons {sorted(comp)}, found {sorted(found)}")
        if "baseline" in tests.columns:
            baselines = set(tests["baseline"].astype(str))
            ck.record(sec, "tests_baseline_flat", baselines == {"flat"},
                      f"baselines found: {sorted(baselines)}")


def validate_effect_sizes(ck, m):
    sec = "effect_sizes"
    spec = m.get(sec, {})
    df = _read_csv(ck, sec, spec.get("file", ""))
    if df is None:
        return
    if not _check_columns(ck, sec, df, spec.get("required_columns", [])):
        return
    baselines = set(df["baseline"].astype(str))
    ck.record(sec, "baseline_is_flat",
              baselines == {str(spec.get("baseline", "flat"))},
              f"baselines found: {sorted(baselines)}")
    for col in ("cohens_d", "cliffs_delta"):
        ck.record(sec, f"{col}_reported", df[col].notna().any(),
                  f"{col} entirely null" if df[col].isna().all() else "")


def validate_exposure(ck, m):
    sec = "exposure"
    spec = m.get(sec, {})
    df = _read_csv(ck, sec, spec.get("file", ""))
    if df is None:
        return
    if not _check_columns(ck, sec, df, spec.get("required_columns", [])):
        return
    datasets = set(spec.get("datasets", []))
    modalities = set(spec.get("modalities", []))
    methods = set(spec.get("methods", []))
    actual = set(map(tuple, df[["dataset", "modality", "method"]]
                     .astype(str).values))
    expected = set(product(sorted(datasets), sorted(modalities),
                           sorted(methods)))
    missing = expected - actual
    ck.record(sec, "all_runs_analyzed", not missing,
              f"{len(missing)} of {len(expected)} runs missing"
              if missing else f"all {len(expected)} runs present")
    scope_col = spec.get("scope_column", "fairness_scope")
    if scope_col in df.columns:
        empty = int(df[scope_col].isna().sum())
        ck.record(sec, "scope_field_populated", empty == 0,
                  f"{empty} rows lack {scope_col} (interpretation must be "
                  f"restricted to exposure/popularity proxies)")
    else:
        ck.fail(sec, "scope_field_populated", f"column {scope_col} absent")


def validate_pq_diagnostics(ck, m):
    sec = "pq_diagnostics"
    spec = m.get(sec, {})
    df = _read_csv(ck, sec, spec.get("file", ""))
    if df is None:
        return
    if not _check_columns(ck, sec, df, spec.get("required_columns", [])):
        return
    methods = set(spec.get("methods", []))
    found = set(df["method"].astype(str))
    ck.record(sec, "pq_methods_present", methods <= found,
              f"expected {sorted(methods)}, found {sorted(found)}")
    metrics = set(spec.get("required_metrics", []))
    found_metrics = set(df["metric"].astype(str))
    ck.record(sec, "diagnostic_metrics_present", metrics <= found_metrics,
              f"expected at least {sorted(metrics)}, found "
              f"{sorted(found_metrics)[:12]}")


SECTION_VALIDATORS = [
    validate_scope,
    validate_main,
    validate_calibration_sensitivity,
    validate_embedding_sensitivity,
    validate_scale_stress,
    validate_bootstrap,
    validate_effect_sizes,
    validate_exposure,
    validate_pq_diagnostics,
]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Strictly validate the paper evidence set against "
                    "configs/paper_evidence_manifest.yml (completeness, "
                    "contract values, CPU-only scope).")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out_dir", default=RESULTS["meta"])
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"[{SCRIPT}] ERROR: manifest not found: {manifest_path}")
        return 2
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {manifest_path}")
    print(f"[{SCRIPT}] output path: {out_dir}")

    ck = Checker()
    for validator in SECTION_VALIDATORS:
        validator(ck, manifest)

    df = pd.DataFrame(ck.rows)
    n_fail = int((df["status"] == "fail").sum())
    valid = n_fail == 0

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "validation_report.csv"
    _atomic_write_text(df.to_csv(index=False), csv_path)

    sections = {}
    for section, g in df.groupby("section"):
        sections[section] = {
            "status": "pass" if (g["status"] == "pass").all() else "fail",
            "checks": g.drop(columns=["section"]).to_dict("records"),
        }
    report = {
        "valid": valid,
        "manifest": str(manifest_path),
        "contract_version": manifest.get("contract_version"),
        "n_checks": int(len(df)),
        "n_failures": n_fail,
        "sections": sections,
    }
    json_path = out_dir / "validation_report.json"
    write_json_atomic(report, json_path, mode="replace")

    md = ["# Paper evidence validation report", "",
          f"- Manifest: `{manifest_path}`",
          f"- Checks: {len(df)}",
          f"- Failures: **{n_fail}**",
          f"- Verdict: {'PASS' if valid else 'FAIL'}", ""]
    if not valid:
        md.append("Missing evidence is expected while `results/` is empty; "
                  "rerun the canonical pipeline to produce it. The validator "
                  "never fabricates evidence.")
        md.append("")
    md.append(df_to_markdown(df))
    md_path = out_dir / "validation_report.md"
    _atomic_write_text("\n".join(md), md_path)

    for p in (csv_path, md_path, json_path):
        print(f"[{SCRIPT}] output path: {p}")
    if valid:
        print(f"[{SCRIPT}] PASS: evidence contract fully satisfied.")
    else:
        print(f"[{SCRIPT}] FAIL: {n_fail} check(s) failed "
              f"(see {md_path}).")
    print(f"[{SCRIPT}] completed.")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
