"""Safe, atomic result writing shared by every canonical script.

Write modes (choose per script via --write_mode):

    fail_if_exists  refuse to overwrite existing evidence (default for
                    experiment-producing scripts)
    replace         atomically replace the file (default for regenerated
                    paper tables/figures)
    merge           merge new rows into the existing file by natural key;
                    unrelated rows are preserved, identical duplicates are
                    deduplicated, and conflicting rows (same key, different
                    values) raise MergeConflictError

All writes go through a temporary file in the target directory followed by
os.replace(), so a failed write never damages the previous file.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

WRITE_MODES = ("fail_if_exists", "replace", "merge")


class ResultExistsError(FileExistsError):
    """Target evidence file exists and write mode is fail_if_exists."""


class MergeConflictError(ValueError):
    """Two rows share a natural key but disagree on at least one value."""


def resolve_write_mode(mode: str) -> str:
    """Validate a --write_mode value (raises on anything unknown)."""
    m = str(mode).strip().lower()
    if m not in WRITE_MODES:
        raise ValueError(f"unknown write mode '{mode}' "
                         f"(expected one of {WRITE_MODES})")
    return m


def _atomic_write_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json_atomic(obj, path, mode: str = "fail_if_exists"):
    """Atomically write a JSON document under the given write mode."""
    mode = resolve_write_mode(mode)
    path = Path(path)
    if path.exists() and mode == "fail_if_exists":
        raise ResultExistsError(
            f"refusing to overwrite existing evidence: {path} "
            f"(use --write_mode replace or merge)")
    if path.exists() and mode == "merge":
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, dict) and isinstance(obj, dict):
            conflicts = {k for k in existing.keys() & obj.keys()
                         if existing[k] != obj[k]}
            if conflicts:
                raise MergeConflictError(
                    f"JSON merge conflict in {path} on keys {sorted(conflicts)}")
            obj = {**existing, **obj}
        elif existing != obj:
            raise MergeConflictError(
                f"cannot merge non-mapping JSON documents in {path}")
    _atomic_write_text(json.dumps(obj, indent=2, sort_keys=True, default=str),
                       path)
    return path


def validate_unique_keys(df: pd.DataFrame, key, context: str = ""):
    """Raise if the natural key does not uniquely identify rows."""
    key = [k for k in key if k in df.columns]
    if not key:
        raise ValueError(f"none of the key columns present in frame {context}")
    dup = df.duplicated(subset=key, keep=False)
    if dup.any():
        sample = df.loc[dup, key].drop_duplicates().head(5).to_dict("records")
        raise MergeConflictError(
            f"duplicate natural keys {context or ''} on {key}: {sample}")
    return key


def _rows_equal(a: pd.Series, b: pd.Series) -> bool:
    for col in a.index:
        va, vb = a[col], b[col]
        if pd.isna(va) and pd.isna(vb):
            continue
        if isinstance(va, float) or isinstance(vb, float):
            try:
                if np.isclose(float(va), float(vb), rtol=1e-9, atol=1e-12,
                              equal_nan=True):
                    continue
                return False
            except (TypeError, ValueError):
                pass
        if va != vb:
            return False
    return True


def merge_dataframe(existing: pd.DataFrame, new: pd.DataFrame, key):
    """Merge new rows into existing by natural key.

    - rows only in `existing` are preserved untouched;
    - rows only in `new` are appended;
    - identical duplicate rows (same key, same values) collapse to one;
    - same key with different values raises MergeConflictError.
    """
    all_cols = list(dict.fromkeys(list(existing.columns) + list(new.columns)))
    existing = existing.reindex(columns=all_cols)
    new = new.reindex(columns=all_cols)
    key = [k for k in key if k in all_cols]
    if not key:
        raise ValueError("merge key has no columns in common with the data")

    def _key_of(row):
        return tuple("<NA>" if pd.isna(row[k]) else str(row[k]) for k in key)

    merged = {}
    order = []
    for _, row in existing.iterrows():
        k = _key_of(row)
        merged[k] = row
        order.append(k)
    for _, row in new.iterrows():
        k = _key_of(row)
        if k in merged:
            if not _rows_equal(merged[k], row):
                raise MergeConflictError(
                    f"conflicting rows for key {dict(zip(key, k))}: "
                    f"existing and new values differ")
            # identical duplicate: keep the existing row
        else:
            merged[k] = row
            order.append(k)
    out = pd.DataFrame([merged[k] for k in order], columns=all_cols)
    return out.reset_index(drop=True)


def write_dataframe_atomic(df: pd.DataFrame, path, mode: str = "fail_if_exists",
                           key=None, sort_by=None):
    """Atomically write a DataFrame as CSV under the given write mode.

    key:     natural-key columns (required for merge; also used to check
             uniqueness in every mode when provided)
    sort_by: columns for deterministic output ordering (defaults to key)
    """
    mode = resolve_write_mode(mode)
    path = Path(path)

    if key is not None:
        validate_unique_keys(df, key, context=f"in new rows for {path.name}")

    if path.exists():
        if mode == "fail_if_exists":
            raise ResultExistsError(
                f"refusing to overwrite existing evidence: {path} "
                f"(use --write_mode replace or merge)")
        if mode == "merge":
            if key is None:
                raise ValueError(f"merge mode requires a natural key ({path})")
            existing = pd.read_csv(path)
            df = merge_dataframe(existing, df, key)

    sort_cols = [c for c in (sort_by or key or []) if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    _atomic_write_text(df.to_csv(index=False), path)
    return path


def preflight_output(path, mode: str):
    """Fail fast BEFORE any expensive work when the target already exists
    and the mode forbids touching it. Call at script start."""
    mode = resolve_write_mode(mode)
    path = Path(path)
    if mode == "fail_if_exists" and path.exists():
        raise ResultExistsError(
            f"output already exists: {path}\n"
            f"  refusing to start (write mode fail_if_exists). Use "
            f"--write_mode replace to regenerate or merge to extend.")
    return mode
