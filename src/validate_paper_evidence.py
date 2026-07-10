"""Strict, manifest-driven validator for revised-paper evidence.

The scientific contract lives in ``configs/paper_evidence_manifest.yml``.
This module supplies defensive interpreters for each artifact type: malformed
or incomplete evidence is recorded as a failed check rather than raising an
uncaught exception.  Only failed *critical* checks make the process exit
non-zero; optional artifacts remain visible in the report without creating a
validation cycle (notably the claim-support audit, which consumes this
validator's JSON report).

Outputs are always written atomically to ``results/_meta`` by default:

* ``validation_report.csv``
* ``validation_report.md``
* ``validation_report.json``
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from utils.config import config_hash
from utils.paths import RESULTS
from utils.reporting import df_to_markdown
from utils.result_io import _atomic_write_text, write_json_atomic

SCRIPT = "validate_paper_evidence"
DEFAULT_MANIFEST = "configs/paper_evidence_manifest.yml"

# These are artifact *types*, not scientific values.  Counts, grids, methods,
# metrics, paths, and seeds remain solely in the manifest.
VALIDATOR_SECTIONS = (
    "scope",
    "main",
    "calibration_sensitivity",
    "embedding_sensitivity",
    "scale_stress",
    "bootstrap",
    "effect_sizes",
    "exposure",
    "pq_diagnostics",
    "decision_framework",
    "dataset_stats",
    "hardware",
    "run_manifest",
    "claim_support_audit",
)


class Checker:
    """Collect severity-aware validation rows."""

    def __init__(self, manifest: dict[str, Any]):
        self.manifest = manifest
        self.rows: list[dict[str, str]] = []

    def _metadata(self, section: str) -> tuple[str, str]:
        spec = self.manifest.get(section)
        if not isinstance(spec, dict):
            return "critical", ""
        severity = str(spec.get("status", "critical")).strip().lower()
        if severity not in {"critical", "optional"}:
            severity = "critical"
        evidence = spec.get("associated_paper_evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        return severity, "; ".join(map(str, evidence or []))

    def record(self, section: str, check: str, ok: bool,
               detail: Any = "") -> bool:
        severity, evidence = self._metadata(section)
        self.rows.append({
            "section": section,
            "severity": severity,
            "associated_paper_evidence": evidence,
            "check": check,
            "status": "pass" if bool(ok) else "fail",
            "detail": str(detail),
        })
        return bool(ok)

    def fail(self, section: str, check: str, detail: Any = "") -> bool:
        return self.record(section, check, False, detail)

    def ok(self, section: str, check: str, detail: Any = "") -> bool:
        return self.record(section, check, True, detail)


def _spec(ck: Checker, section: str) -> dict[str, Any]:
    value = ck.manifest.get(section)
    if not isinstance(value, dict):
        ck.fail(section, "manifest_section", "section missing or not a mapping")
        return {}
    return value


def _path_from(spec: dict[str, Any], key: str) -> Path | None:
    value = spec.get(key)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    return Path(value)


def _read_csv(ck: Checker, section: str, spec: dict[str, Any],
              key: str = "file") -> pd.DataFrame | None:
    path = _path_from(spec, key)
    if path is None:
        ck.fail(section, f"{key}_declared", f"manifest key {key!r} is absent")
        return None
    if not path.is_file():
        ck.fail(section, f"{key}_exists", f"missing evidence: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # malformed CSV is evidence failure, not a crash
        ck.fail(section, f"{key}_readable", f"{path}: {type(exc).__name__}: {exc}")
        return None
    if df.empty:
        ck.fail(section, f"{key}_nonempty", f"{path} has 0 rows")
        return None
    ck.ok(section, f"{key}_exists", str(path))
    return df


def _read_json(ck: Checker, section: str, spec: dict[str, Any],
               key: str = "file") -> dict[str, Any] | None:
    path = _path_from(spec, key)
    if path is None:
        ck.fail(section, f"{key}_declared", f"manifest key {key!r} is absent")
        return None
    if not path.is_file():
        ck.fail(section, f"{key}_exists", f"missing evidence: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        ck.fail(section, f"{key}_readable", f"{path}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(value, dict):
        ck.fail(section, f"{key}_mapping", f"{path} must contain a JSON object")
        return None
    ck.ok(section, f"{key}_exists", str(path))
    return value


def _check_columns(ck: Checker, section: str, df: pd.DataFrame,
                   required: Iterable[str], label: str = "required_columns") -> bool:
    required = list(required or [])
    missing = [column for column in required if column not in df.columns]
    return ck.record(section, label, not missing,
                     f"missing columns: {missing}" if missing else "all present")


def _check_unique(ck: Checker, section: str, df: pd.DataFrame,
                  key: Iterable[str], label: str = "no_duplicate_keys") -> bool:
    key = list(key or [])
    missing = [column for column in key if column not in df.columns]
    if not key or missing:
        return ck.fail(section, label,
                       f"natural key invalid; key={key}, missing={missing}")
    duplicates = df.duplicated(subset=key, keep=False)
    count = int(duplicates.sum())
    sample = (df.loc[duplicates, key].head(5).to_dict("records")
              if count else [])
    return ck.record(section, label, count == 0,
                     f"{count} rows have duplicate key {key}; sample={sample}")


def _scalar_equal(actual: Any, expected: Any) -> bool:
    if pd.isna(actual):
        return False
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual is expected
        return str(actual).strip().lower() in ({"true", "1"} if expected
                                                else {"false", "0"})
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected),
                                rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def _check_uniform(ck: Checker, section: str, df: pd.DataFrame,
                   column: str, expected: Any, label: str | None = None) -> bool:
    label = label or f"uniform_{column}"
    if column not in df.columns:
        return ck.fail(section, label, f"column {column!r} absent")
    bad = [value for value in df[column].tolist()
           if not _scalar_equal(value, expected)]
    return ck.record(section, label, not bad,
                     (f"expected {column}={expected!r} on every row; "
                      f"{len(bad)} violation(s), sample={bad[:5]}") if bad
                     else f"{column}={expected!r} on every row")


def _check_no_nulls(ck: Checker, section: str, df: pd.DataFrame,
                    columns: Iterable[str], prefix: str = "no_nulls") -> bool:
    ok = True
    for column in columns or []:
        if column not in df.columns:
            ck.fail(section, f"{prefix}_{column}", f"column {column!r} absent")
            ok = False
            continue
        count = int(df[column].isna().sum())
        if not ck.record(section, f"{prefix}_{column}", count == 0,
                         f"{count} null value(s)"):
            ok = False
    return ok


def _check_row_count(ck: Checker, section: str, df: pd.DataFrame,
                     spec: dict[str, Any], key: str = "expected_rows",
                     label: str = "expected_row_count") -> bool:
    try:
        expected = int(spec[key])
    except (KeyError, TypeError, ValueError):
        return ck.fail(section, label, f"manifest {key!r} is absent/invalid")
    return ck.record(section, label, len(df) == expected,
                     f"expected {expected}, found {len(df)}")


def _check_min_rows(ck: Checker, section: str, df: pd.DataFrame,
                    spec: dict[str, Any]) -> bool:
    try:
        expected = int(spec["expected_rows_min"])
    except (KeyError, TypeError, ValueError):
        return ck.fail(section, "minimum_row_count",
                       "manifest 'expected_rows_min' is absent/invalid")
    return ck.record(section, "minimum_row_count", len(df) >= expected,
                     f"expected at least {expected}, found {len(df)}")


def _as_tuples(df: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    return {tuple(map(str, row)) for row in df[columns].itertuples(index=False,
                                                                   name=None)}


def _check_grid(ck: Checker, section: str, df: pd.DataFrame,
                columns: list[str], expected_axes: list[Iterable[Any]],
                label: str = "exact_grid") -> bool:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        return ck.fail(section, label, f"missing grid columns: {missing_columns}")
    actual = _as_tuples(df, columns)
    expected = {tuple(map(str, row)) for row in product(*expected_axes)}
    missing = expected - actual
    unexpected = actual - expected
    return ck.record(
        section, label, not missing and not unexpected,
        f"expected={len(expected)}, actual={len(actual)}, missing={len(missing)} "
        f"sample_missing={sorted(missing)[:3]}, unexpected={len(unexpected)} "
        f"sample_unexpected={sorted(unexpected)[:3]}")


def _check_required_values(ck: Checker, section: str, df: pd.DataFrame,
                           values: dict[str, Any]) -> bool:
    ok = True
    for column, expected in (values or {}).items():
        if not _check_uniform(ck, section, df, column, expected,
                              label=f"required_value_{column}"):
            ok = False
    return ok


def _check_boolean_column(ck: Checker, section: str, df: pd.DataFrame,
                          column: str) -> bool:
    if column not in df.columns:
        return ck.fail(section, f"boolean_{column}", f"column {column!r} absent")
    allowed = {"true", "false", "1", "0"}
    values = {str(value).strip().lower() for value in df[column].dropna()}
    nulls = int(df[column].isna().sum())
    return ck.record(section, f"boolean_{column}", not nulls and values <= allowed,
                     f"values={sorted(values)}, nulls={nulls}")


def _nested_get(value: dict[str, Any], dotted: str) -> Any:
    node: Any = value
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _all_manifest_paths(manifest: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for spec in manifest.values():
        if not isinstance(spec, dict):
            continue
        for key, value in spec.items():
            if (key == "file" or key.endswith("_file")) and isinstance(value, str):
                paths.append(Path(value))
    return paths


def _recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys |= _recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            keys |= _recursive_keys(child)
    return keys


def validate_manifest_contract(ck: Checker) -> None:
    section = "manifest_contract"
    # This report section is always critical because it governs interpretation
    # of every artifact.  Give it explicit metadata without polluting YAML.
    ck.manifest.setdefault(section, {
        "status": "critical",
        "associated_paper_evidence": ["Evidence-contract integrity"],
    })
    ck.record(section, "manifest_is_mapping", isinstance(ck.manifest, dict), "")
    for name in VALIDATOR_SECTIONS:
        spec = ck.manifest.get(name)
        if not isinstance(spec, dict):
            ck.fail(section, f"section_{name}", "missing or not a mapping")
            continue
        status = spec.get("status")
        ck.record(section, f"section_{name}_status",
                  status in {"critical", "optional"},
                  f"status={status!r}")
        evidence = spec.get("associated_paper_evidence")
        ck.record(section, f"section_{name}_paper_evidence",
                  isinstance(evidence, list) and bool(evidence),
                  f"associated_paper_evidence={evidence!r}")


def validate_scope(ck: Checker) -> None:
    section = "scope"
    spec = _spec(ck, section)
    ck.record(section, "cpu_only_contract", spec.get("cpu_only") is True,
              f"cpu_only={spec.get('cpu_only')!r}")
    for path_value in spec.get("forbidden_result_paths", []):
        path = Path(path_value)
        ck.record(section, f"forbidden_path_absent:{path.as_posix()}",
                  not path.exists(),
                  f"forbidden result path {'present' if path.exists() else 'absent'}: {path}")

    forbidden = set(map(str, spec.get("forbidden_columns", [])))
    offenders: list[str] = []
    for path in _all_manifest_paths(ck.manifest):
        if path.suffix.lower() != ".csv" or not path.is_file():
            continue
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            # The artifact-specific validator records readability in detail.
            offenders.append(f"{path}: unreadable ({type(exc).__name__})")
            continue
        found = sorted(columns & forbidden)
        if found:
            offenders.append(f"{path}: {found}")
    ck.record(section, "no_gpu_columns_in_canonical_csvs", not offenders,
              "; ".join(offenders) if offenders else "none found")


def validate_main(ck: Checker) -> None:
    section = "main"
    spec = _spec(ck, section)
    df = _read_csv(ck, section, spec)

    run_config = _read_json(ck, section, spec, "run_config_file")
    if run_config is not None:
        for key, expected in (spec.get("run_config_must_match") or {}).items():
            actual = run_config.get(key)
            ck.record(section, f"run_config_{key}", _scalar_equal(actual, expected),
                      f"expected {key}={expected!r}, found {actual!r}")
        for key in ("datasets", "modalities", "methods"):
            expected = spec.get(key, [])
            actual = run_config.get(key)
            ck.record(section, f"run_config_{key}_exact",
                      isinstance(actual, list) and set(map(str, actual)) == set(map(str, expected)),
                      f"expected={expected}, found={actual}")

    if df is None or not _check_columns(ck, section, df,
                                        spec.get("required_columns", [])):
        return
    _check_unique(ck, section, df, spec.get("key", []))
    _check_uniform(ck, section, df, "weighting", spec.get("weighting"))
    _check_uniform(ck, section, df, "dim", spec.get("dim"))
    _check_uniform(ck, section, df, "seed", spec.get("seed"))
    _check_grid(ck, section, df, ["dataset", "modality", "method"],
                [spec.get("datasets", []), spec.get("modalities", []),
                 spec.get("methods", [])])
    _check_row_count(ck, section, df, spec)
    _check_no_nulls(ck, section, df,
                    ["queries", *spec.get("metric_columns_no_nulls", [])])
    forbidden = set(ck.manifest.get("scope", {}).get("forbidden_columns", []))
    present = sorted(set(df.columns) & forbidden)
    ck.record(section, "no_gpu_columns", not present,
              f"forbidden columns present: {present}")


def validate_calibration_sensitivity(ck: Checker) -> None:
    section = "calibration_sensitivity"
    spec = _spec(ck, section)
    df = _read_csv(ck, section, spec)
    if df is None or not _check_columns(ck, section, df,
                                        spec.get("required_columns", [])):
        return
    _check_unique(ck, section, df, spec.get("key", []))
    for column in ("weighting", "dim", "seed"):
        _check_uniform(ck, section, df, column, spec.get(column))
    try:
        target_strings = [f"{float(value):.2f}" for value in df["target_recall"]]
        grid_df = df.assign(_target=target_strings)
        expected_targets = [f"{float(value):.2f}" for value in spec.get("targets", [])]
        _check_grid(ck, section, grid_df, ["dataset", "method", "_target"],
                    [spec.get("datasets", []), spec.get("methods", []),
                     expected_targets])
    except (TypeError, ValueError) as exc:
        ck.fail(section, "exact_grid", f"invalid target_recall value: {exc}")
    _check_row_count(ck, section, df, spec)
    _check_no_nulls(ck, section, df, spec.get("required_non_null", []))
    _check_boolean_column(ck, section, df, "target_reached")
    if "calibration_queries" in spec:
        _check_uniform(ck, section, df, "n_calibration_queries",
                       spec["calibration_queries"])


def validate_embedding_sensitivity(ck: Checker) -> None:
    section = "embedding_sensitivity"
    spec = _spec(ck, section)
    df = _read_csv(ck, section, spec)
    if df is None or not _check_columns(ck, section, df,
                                        spec.get("required_columns", [])):
        return

    required_backbones = set(map(str, spec.get("backbones", [])))
    optional_backbones = set(map(str, spec.get("optional_backbones", [])))
    known = required_backbones | optional_backbones
    unexpected = set(df["backbone"].astype(str)) - known
    ck.record(section, "no_unexpected_backbones", not unexpected,
              f"unexpected={sorted(unexpected)}")

    required = df[df["backbone"].astype(str).isin(required_backbones)].copy()
    _check_unique(ck, section, required, spec.get("key", []))
    _check_grid(ck, section, required,
                ["dataset", "backbone", "modality", "method"],
                [spec.get("datasets", []), spec.get("backbones", []),
                 spec.get("modalities", []), spec.get("methods", [])])
    _check_row_count(ck, section, required, spec)
    _check_uniform(ck, section, required, "dim", spec.get("dim"))
    _check_uniform(ck, section, required, "seed", spec.get("seed"))
    _check_uniform(ck, section, required, "backend_available", True)
    _check_uniform(ck, section, required, "status",
                   spec.get("required_success_status", "ok"))
    _check_no_nulls(ck, section, required,
                    ["ndcg_at_10", "recall_at_10",
                     "ann_recall_vs_exact_at_k_mean"])

    stability_ok = True
    stability_detail = []
    for backbone in sorted(required_backbones):
        subset = required[required["backbone"].astype(str) == backbone]
        count = int(subset["ann_ranking_stability"].notna().sum())
        stability_detail.append(f"{backbone}:{count}")
        stability_ok &= count > 0
    ck.record(section, "ranking_stability_reported", stability_ok,
              ", ".join(stability_detail) + " (values are not judged by sign)")

    optional = df[df["backbone"].astype(str).isin(optional_backbones)]
    if not optional.empty:
        _check_unique(ck, section, optional, spec.get("key", []),
                      label="optional_no_duplicate_keys")


def validate_scale_stress(ck: Checker) -> None:
    section = "scale_stress"
    spec = _spec(ck, section)
    df = _read_csv(ck, section, spec)
    if df is None or not _check_columns(ck, section, df,
                                        spec.get("required_columns", [])):
        return
    _check_unique(ck, section, df, spec.get("key", []))
    _check_uniform(ck, section, df, "seed", spec.get("seed"))
    _check_grid(ck, section, df, ["n_items", "dim", "method"],
                [spec.get("catalog_sizes", []), spec.get("dimensions", []),
                 spec.get("methods", [])])
    _check_row_count(ck, section, df, spec)
    _check_no_nulls(ck, section, df, spec.get("required_non_null", []))
    _check_boolean_column(ck, section, df, "target_reached")
    _check_required_values(ck, section, df, spec.get("require_values", {}))
    forbidden = [column for column in spec.get("forbidden_quality_columns", [])
                 if column in df.columns and df[column].notna().any()]
    ck.record(section, "no_synthetic_recommendation_quality", not forbidden,
              f"quality columns containing values: {forbidden}")


def _validate_bootstrap_frame(ck: Checker, spec: dict[str, Any],
                              df: pd.DataFrame | None, kind: str) -> None:
    section = "bootstrap"
    if df is None:
        return
    required_key = ("required_columns_cis" if kind == "cis"
                    else "required_columns_tests")
    if not _check_columns(ck, section, df, spec.get(required_key, []),
                          label=f"{kind}_required_columns"):
        return
    key = spec.get(f"{kind}_key", [])
    _check_unique(ck, section, df, key,
                  label=f"{kind}_no_duplicate_keys")
    for column in ("weighting", "dim", "seed", "n_boot"):
        expected = spec.get(column)
        _check_uniform(ck, section, df, column, expected,
                       label=f"{kind}_{column}")

    methods = (spec.get("methods", []) if kind == "cis"
               else spec.get("comparison_methods", []))
    metrics = (spec.get("metrics", []) if kind == "cis"
               else spec.get("comparison_metrics", []))
    _check_grid(ck, section, df,
                ["dataset", "modality", "method", "metric"],
                [spec.get("datasets", []), spec.get("modalities", []),
                 methods, metrics], label=f"{kind}_exact_grid")

    row_key = "expected_cis_rows" if kind == "cis" else "expected_tests_rows"
    _check_row_count(ck, section, df, spec, row_key,
                     label=f"{kind}_expected_row_count")
    numeric = (["mean", "ci_low", "ci_high", "n", "n_boot"] if kind == "cis"
               else ["mean_diff", "ci_low", "ci_high", "p_value", "n",
                     "n_boot"])
    _check_no_nulls(ck, section, df, numeric, prefix=f"{kind}_no_nulls")
    if kind == "tests":
        _check_uniform(ck, section, df, "baseline", "flat",
                       label="tests_flat_baseline")


def validate_bootstrap(ck: Checker) -> None:
    section = "bootstrap"
    spec = _spec(ck, section)
    cis = _read_csv(ck, section, spec, "cis_file")
    tests = _read_csv(ck, section, spec, "tests_file")
    _validate_bootstrap_frame(ck, spec, cis, "cis")
    _validate_bootstrap_frame(ck, spec, tests, "tests")


def validate_effect_sizes(ck: Checker) -> None:
    section = "effect_sizes"
    spec = _spec(ck, section)
    df = _read_csv(ck, section, spec)
    if df is None or not _check_columns(ck, section, df,
                                        spec.get("required_columns", [])):
        return
    _check_unique(ck, section, df, spec.get("key", []))
    for column in ("weighting", "dim", "seed"):
        _check_uniform(ck, section, df, column, spec.get(column))
    _check_uniform(ck, section, df, "baseline", spec.get("baseline", "flat"))
    _check_grid(ck, section, df,
                ["dataset", "modality", "method", "metric"],
                [spec.get("datasets", []), spec.get("modalities", []),
                 spec.get("methods", []), spec.get("metrics", [])])
    _check_row_count(ck, section, df, spec)
    _check_no_nulls(ck, section, df,
                    ["cohens_d", "cliffs_delta", "cohens_d_magnitude",
                     "cliffs_delta_magnitude", "n"])


def validate_exposure(ck: Checker) -> None:
    section = "exposure"
    spec = _spec(ck, section)
    df = _read_csv(ck, section, spec)
    if df is None or not _check_columns(ck, section, df,
                                        spec.get("required_columns", [])):
        return
    _check_unique(ck, section, df, spec.get("key", []))
    for column in ("weighting", "dim", "seed"):
        _check_uniform(ck, section, df, column, spec.get(column))
    _check_min_rows(ck, section, df, spec)
    _check_grid(ck, section, df, ["dataset", "modality", "method"],
                [spec.get("datasets", []), spec.get("modalities", []),
                 spec.get("methods", [])], label="all_runs_present")

    core = df[df["metric"].astype(str).isin(set(map(str,
                                    spec.get("required_metrics", []))))]
    _check_grid(ck, section, core,
                ["dataset", "modality", "method", "metric"],
                [spec.get("datasets", []), spec.get("modalities", []),
                 spec.get("methods", []), spec.get("required_metrics", [])],
                label="required_metrics_per_run")
    _check_no_nulls(ck, section, df, ["value", spec.get("scope_column",
                                                        "fairness_scope")])
    scope_column = spec.get("scope_column", "fairness_scope")
    if scope_column in df.columns:
        found = set(df[scope_column].dropna().astype(str))
        allowed = set(map(str, spec.get("allowed_scope_values", [])))
        invalid = found - allowed
        ck.record(section, "scope_restricted_to_proxies", not invalid,
                  f"allowed={sorted(allowed)}, invalid={sorted(invalid)}")


def validate_pq_diagnostics(ck: Checker) -> None:
    section = "pq_diagnostics"
    spec = _spec(ck, section)
    df = _read_csv(ck, section, spec)
    if df is None or not _check_columns(ck, section, df,
                                        spec.get("required_columns", [])):
        return
    _check_unique(ck, section, df, spec.get("key", []))
    for column in ("weighting", "dim", "seed"):
        _check_uniform(ck, section, df, column, spec.get(column))
    _check_min_rows(ck, section, df, spec)
    required = df[df["metric"].astype(str).isin(set(map(str,
                                        spec.get("required_metrics", []))))]
    _check_grid(ck, section, required, ["dataset", "method", "metric"],
                [spec.get("datasets", []), spec.get("methods", []),
                 spec.get("required_metrics", [])],
                label="required_diagnostics_per_dataset_method")
    _check_no_nulls(ck, section, required, ["value"])


def validate_decision_framework(ck: Checker) -> None:
    section = "decision_framework"
    spec = _spec(ck, section)
    df = _read_csv(ck, section, spec)
    if df is None or not _check_columns(ck, section, df,
                                        spec.get("required_columns", [])):
        return
    _check_unique(ck, section, df, spec.get("key", []))
    for column in ("weighting", "dim", "seed"):
        _check_uniform(ck, section, df, column, spec.get(column))
    _check_grid(ck, section, df, ["dataset", "modality", "method"],
                [spec.get("datasets", []), spec.get("modalities", []),
                 spec.get("methods", [])])
    _check_row_count(ck, section, df, spec)
    _check_no_nulls(ck, section, df,
                    ["deployment_score", "deployment_rank",
                     "recommended_use_case", "recommendation_reason"])
    _check_required_values(ck, section, df, spec.get("required_values", {}))


def validate_dataset_stats(ck: Checker) -> None:
    section = "dataset_stats"
    spec = _spec(ck, section)
    df = _read_csv(ck, section, spec)
    if df is None or not _check_columns(ck, section, df,
                                        spec.get("required_columns", [])):
        return
    _check_unique(ck, section, df, spec.get("key", []))
    actual = set(df["dataset"].astype(str))
    expected = set(map(str, spec.get("datasets", [])))
    ck.record(section, "exact_datasets", actual == expected,
              f"expected={sorted(expected)}, found={sorted(actual)}")
    _check_row_count(ck, section, df, spec)
    _check_required_values(ck, section, df, spec.get("required_values", {}))
    _check_no_nulls(ck, section, df,
                    [column for column in spec.get("required_columns", [])
                     if column != "dataset"])


def validate_hardware(ck: Checker) -> None:
    section = "hardware"
    spec = _spec(ck, section)
    doc = _read_json(ck, section, spec)
    if doc is None:
        return
    required = list(spec.get("required_fields", []))
    missing = [field for field in required if field not in doc]
    ck.record(section, "required_fields", not missing,
              f"missing fields: {missing}")
    for field, expected in (spec.get("required_values") or {}).items():
        actual = doc.get(field)
        ck.record(section, f"required_value_{field}",
                  _scalar_equal(actual, expected),
                  f"expected={expected!r}, found={actual!r}")
    ck.record(section, "accelerator_present_is_boolean",
              isinstance(doc.get("accelerator_present"), bool),
              f"found={doc.get('accelerator_present')!r}")
    forbidden = sorted(set(spec.get("forbidden_fields", [])) &
                       _recursive_keys(doc))
    ck.record(section, "no_gpu_execution_fields", not forbidden,
              f"forbidden fields present: {forbidden}")


def validate_run_manifest(ck: Checker) -> None:
    section = "run_manifest"
    spec = _spec(ck, section)
    doc = _read_json(ck, section, spec)
    if doc is None:
        return
    required = list(spec.get("required_fields", []))
    missing = [field for field in required if field not in doc]
    ck.record(section, "required_fields", not missing,
              f"missing fields: {missing}")
    for field, expected in (spec.get("required_values") or {}).items():
        actual = doc.get(field)
        ck.record(section, f"required_value_{field}",
                  _scalar_equal(actual, expected),
                  f"expected={expected!r}, found={actual!r}")
    for field in ("datasets", "methods", "modalities"):
        actual = doc.get(field)
        expected = spec.get(field, [])
        ck.record(section, f"exact_{field}",
                  isinstance(actual, list) and set(map(str, actual)) == set(map(str, expected)),
                  f"expected={expected}, found={actual}")
    ck.record(section, "no_failed_combinations",
              doc.get("failed_combinations") == [],
              f"failed_combinations={doc.get('failed_combinations')!r}")
    required_output = str(spec.get("required_output", "")).replace("\\", "/")
    outputs = [str(value).replace("\\", "/")
               for value in doc.get("outputs", [])] if isinstance(doc.get("outputs"), list) else []
    ck.record(section, "required_output_recorded", required_output in outputs,
              f"required={required_output!r}, outputs={outputs}")

    resolved = doc.get("resolved_configuration")
    ck.record(section, "resolved_configuration_mapping",
              isinstance(resolved, dict), f"type={type(resolved).__name__}")
    if isinstance(resolved, dict):
        expected_resolved = {
            "project.name": "IndexWise-Recsys",
            "reproducibility.seed": 42,
            "embedding.weighting": "bm25",
            "embedding.dim": 128,
            "hardware.cpu_only": True,
        }
        for dotted, expected in expected_resolved.items():
            actual = _nested_get(resolved, dotted)
            ck.record(section, f"resolved_{dotted}",
                      _scalar_equal(actual, expected),
                      f"expected={expected!r}, found={actual!r}")
        for dotted, expected in (("datasets", spec.get("datasets", [])),
                                 ("retrieval.methods", spec.get("methods", [])),
                                 ("retrieval.modalities", spec.get("modalities", []))):
            actual = _nested_get(resolved, dotted)
            ck.record(section, f"resolved_{dotted}_exact",
                      isinstance(actual, list) and set(map(str, actual)) == set(map(str, expected)),
                      f"expected={expected}, found={actual}")
        actual_hash = doc.get("configuration_hash")
        expected_hash = config_hash(resolved)
        ck.record(section, "configuration_hash_matches",
                  actual_hash == expected_hash,
                  f"expected={expected_hash}, found={actual_hash}")
        run_id = str(doc.get("run_id", ""))
        ck.record(section, "run_id_contains_config_hash",
                  run_id.endswith(f"__{expected_hash}"), f"run_id={run_id!r}")

    for field in ("started_at_utc", "finished_at_utc", "git_commit",
                  "python_version"):
        value = doc.get(field)
        ck.record(section, f"{field}_populated",
                  value is not None and str(value).strip() != "",
                  f"found={value!r}")
    hardware = doc.get("hardware")
    used = (hardware.get("main_experiments_accelerator_used")
            if isinstance(hardware, dict) else None)
    ck.record(section, "manifest_declares_cpu_only", used is False,
              f"main_experiments_accelerator_used={used!r}")
    forbidden = set(ck.manifest.get("scope", {}).get("forbidden_columns", []))
    present = sorted(forbidden & _recursive_keys(doc))
    ck.record(section, "no_gpu_execution_fields", not present,
              f"forbidden fields present: {present}")


def validate_claim_support_audit(ck: Checker) -> None:
    section = "claim_support_audit"
    spec = _spec(ck, section)
    df = _read_csv(ck, section, spec)
    if df is None:
        return
    if not _check_columns(ck, section, df, spec.get("required_columns", [])):
        return
    _check_unique(ck, section, df, spec.get("key", []))
    _check_row_count(ck, section, df, spec)
    _check_boolean_column(ck, section, df, "evidence_supported")


SECTION_VALIDATORS = (
    ("manifest_contract", validate_manifest_contract),
    ("scope", validate_scope),
    ("main", validate_main),
    ("calibration_sensitivity", validate_calibration_sensitivity),
    ("embedding_sensitivity", validate_embedding_sensitivity),
    ("scale_stress", validate_scale_stress),
    ("bootstrap", validate_bootstrap),
    ("effect_sizes", validate_effect_sizes),
    ("exposure", validate_exposure),
    ("pq_diagnostics", validate_pq_diagnostics),
    ("decision_framework", validate_decision_framework),
    ("dataset_stats", validate_dataset_stats),
    ("hardware", validate_hardware),
    ("run_manifest", validate_run_manifest),
    ("claim_support_audit", validate_claim_support_audit),
)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"manifest {path} must contain a YAML mapping")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Strictly validate the complete paper-evidence set from "
                    "configs/paper_evidence_manifest.yml.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", default=RESULTS["meta"])
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"[{SCRIPT}] ERROR: manifest not found: {manifest_path}")
        return 2
    try:
        manifest = _load_manifest(manifest_path)
    except ValueError as exc:
        print(f"[{SCRIPT}] ERROR: {exc}")
        return 2

    out_dir = Path(args.out_dir)
    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {manifest_path}")
    print(f"[{SCRIPT}] output path: {out_dir}")

    checker = Checker(manifest)
    for section, validator in SECTION_VALIDATORS:
        try:
            validator(checker)
        except Exception as exc:  # final containment for malformed evidence
            checker.fail(section, "validator_exception",
                         f"{type(exc).__name__}: {exc}")

    report_df = pd.DataFrame(checker.rows)
    failed = report_df[report_df["status"] == "fail"]
    critical_failures = failed[failed["severity"] == "critical"]
    optional_failures = failed[failed["severity"] == "optional"]
    valid = critical_failures.empty

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "validation_report.csv"
    _atomic_write_text(report_df.to_csv(index=False), csv_path)

    sections: dict[str, Any] = {}
    for section, group in report_df.groupby("section", sort=False):
        severity = str(group["severity"].iloc[0])
        sections[section] = {
            "status": "pass" if (group["status"] == "pass").all() else "fail",
            "severity": severity,
            "associated_paper_evidence": [value for value in
                str(group["associated_paper_evidence"].iloc[0]).split("; ") if value],
            "checks": group.drop(columns=["section", "severity",
                                            "associated_paper_evidence"])
                           .to_dict("records"),
        }
    report = {
        "valid": valid,
        "manifest": str(manifest_path),
        "contract_version": manifest.get("contract_version"),
        "n_checks": int(len(report_df)),
        "n_failures": int(len(failed)),
        "n_critical_failures": int(len(critical_failures)),
        "n_optional_failures": int(len(optional_failures)),
        "sections": sections,
    }
    json_path = out_dir / "validation_report.json"
    write_json_atomic(report, json_path, mode="replace")

    verdict = ("PASS" if valid and optional_failures.empty else
               "PASS WITH OPTIONAL GAPS" if valid else "FAIL")
    markdown = [
        "# Paper evidence validation report",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Checks: {len(report_df)}",
        f"- Critical failures: **{len(critical_failures)}**",
        f"- Optional failures: **{len(optional_failures)}**",
        f"- Verdict: **{verdict}**",
        "",
    ]
    if not valid:
        markdown.extend([
            "Missing evidence is expected while `results/` is empty. The "
            "validator reports the gap and never fabricates evidence.",
            "",
        ])
    markdown.append(df_to_markdown(report_df))
    md_path = out_dir / "validation_report.md"
    _atomic_write_text("\n".join(markdown), md_path)

    for path in (csv_path, md_path, json_path):
        print(f"[{SCRIPT}] output path: {path}")
    if valid:
        print(f"[{SCRIPT}] {verdict}: critical evidence contract satisfied; "
              f"optional failures={len(optional_failures)}.")
    else:
        print(f"[{SCRIPT}] FAIL: {len(critical_failures)} critical check(s) "
              f"failed (see {md_path}).")
    print(f"[{SCRIPT}] completed.")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
