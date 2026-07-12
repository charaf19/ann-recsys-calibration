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
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

WRITE_MODES = ("fail_if_exists", "replace", "merge")

import random
import time


def _replace_with_retry(
    source: Path,
    destination: Path,
    *,
    attempts: int = 8,
    initial_delay_s: float = 0.05,
) -> None:
    """Atomically replace destination, retrying transient Windows locks."""

    delay = initial_delay_s
    last_error: PermissionError | None = None

    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            last_error = exc

            if attempt == attempts:
                break

            # Small jitter prevents repeated collisions with scanners/watchers.
            time.sleep(delay + random.uniform(0.0, delay * 0.2))
            delay = min(delay * 2.0, 1.0)

    assert last_error is not None
    raise PermissionError(
        f"Unable to atomically replace {destination} after "
        f"{attempts} attempts. Temporary file remains at {source}. "
        f"Another process may be holding the destination open."
    ) from last_error
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


def _fsync_written_file(path) -> None:
    """Flush an already-written regular file to disk before atomic publish.

    The file must already exist and hold its final contents. It is reopened
    for update (``"r+b"``) so the descriptor passed to ``os.fsync`` is
    writable: on Windows ``os.fsync`` on a read-only (``"rb"``) descriptor
    raises ``OSError: [Errno 9] Bad file descriptor``. Opening ``"r+b"``
    (rather than ``"wb"``/``"ab"``) never truncates or appends, so the bytes
    just written by the caller are preserved untouched.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"cannot fsync a temporary artifact that was never created: {path}")
    with open(path, "r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        _replace_with_retry(tmp, path)
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


def write_text_atomic(text: str, path, mode: str = "fail_if_exists"):
    """Atomically write UTF-8 text under the standard write modes.

    Text has no natural-key merge contract, so ``merge`` is accepted only
    when the existing content is byte-for-byte identical.
    """
    mode = resolve_write_mode(mode)
    path = Path(path)
    if path.exists() and mode == "fail_if_exists":
        raise ResultExistsError(f"refusing to overwrite existing file: {path}")
    if path.exists() and mode == "merge":
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise MergeConflictError(
                f"cannot merge conflicting text documents in {path}")
        return path
    _atomic_write_text(text, path)
    return path


def validate_unique_keys(df: pd.DataFrame, key, context: str = ""):
    """Raise if the natural key does not uniquely identify rows."""
    requested = list(key)
    missing = [k for k in requested if k not in df.columns]
    if missing:
        raise ValueError(
            f"missing natural-key columns {missing} in frame {context}; "
            f"required key is {requested}")
    key = requested
    null_rows = df[key].isna().any(axis=1)
    if null_rows.any():
        sample = df.loc[null_rows, key].head(5).to_dict("records")
        raise MergeConflictError(
            f"null values in natural keys {context or ''} on {key}: {sample}")
    dup = df.duplicated(subset=key, keep=False)
    if dup.any():
        sample = df.loc[dup, key].drop_duplicates().head(5).to_dict("records")
        raise MergeConflictError(
            f"duplicate natural keys {context or ''} on {key}: {sample}")
    return key


def _key_of(row: pd.Series, key) -> tuple:
    return tuple("<NA>" if pd.isna(row[k]) else str(row[k]) for k in key)


def _deduplicate_identical_keys(df: pd.DataFrame, key, context: str = ""):
    """Collapse identical duplicate-key rows; reject conflicting ones."""
    requested = list(key)
    missing = [k for k in requested if k not in df.columns]
    if missing:
        raise ValueError(
            f"missing natural-key columns {missing} in frame {context}; "
            f"required key is {requested}")
    key = requested
    null_rows = df[key].isna().any(axis=1)
    if null_rows.any():
        sample = df.loc[null_rows, key].head(5).to_dict("records")
        raise MergeConflictError(
            f"null values in natural keys {context or ''} on {key}: {sample}")
    kept = {}
    order = []
    for _, row in df.iterrows():
        row_key = _key_of(row, key)
        if row_key in kept:
            if not _rows_equal(kept[row_key], row):
                raise MergeConflictError(
                    f"conflicting duplicate natural keys {context or ''} "
                    f"{dict(zip(key, row_key))}")
            continue
        kept[row_key] = row
        order.append(row_key)
    return pd.DataFrame([kept[k] for k in order], columns=df.columns).reset_index(
        drop=True)


def _rows_equal(a: pd.Series, b: pd.Series) -> bool:
    for col in a.index:
        va, vb = a[col], b[col]
        if pd.isna(va) and pd.isna(vb):
            continue
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
    requested = list(key)
    for label, frame in (("existing", existing), ("new", new)):
        missing = [k for k in requested if k not in frame.columns]
        if missing:
            raise ValueError(
                f"{label} frame is missing natural-key columns {missing}; "
                f"required key is {requested}")
    all_cols = list(dict.fromkeys(list(existing.columns) + list(new.columns)))
    existing = existing.reindex(columns=all_cols)
    new = new.reindex(columns=all_cols)
    key = requested

    combined = pd.concat([existing, new], ignore_index=True)
    try:
        return _deduplicate_identical_keys(combined, key, context="during merge")
    except MergeConflictError as exc:
        raise MergeConflictError(
            f"conflicting rows for natural key {key}: {exc}") from exc


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
        df = _deduplicate_identical_keys(
            df, key, context=f"in new rows for {path.name}")
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


def write_npz_atomic(path, mode: str = "fail_if_exists", **arrays):
    """Atomically write a compressed NumPy archive.

    NPZ archives are indivisible per-query artifacts, so merge mode is not
    meaningful and is rejected explicitly.
    """
    mode = resolve_write_mode(mode)
    path = Path(path)
    if path.exists() and mode == "merge":
        with np.load(path, allow_pickle=True) as existing:
            if set(existing.files) != set(arrays):
                raise MergeConflictError(
                    f"NPZ merge conflict in {path}: archive keys differ")
            for key, value in arrays.items():
                old = np.asarray(existing[key])
                new = np.asarray(value)
                if old.shape != new.shape or old.dtype != new.dtype:
                    raise MergeConflictError(
                        f"NPZ merge conflict in {path} for {key}: "
                        f"shape or dtype differs")
                try:
                    equal = np.array_equal(old, new, equal_nan=True)
                except TypeError:
                    equal = np.array_equal(old, new)
                if not equal:
                    raise MergeConflictError(
                        f"NPZ merge conflict in {path} for {key}: values differ")
        return path
    if path.exists() and mode == "fail_if_exists":
        raise ResultExistsError(f"refusing to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.",
                               suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        _fsync_written_file(tmp)
        _replace_with_retry(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


@contextmanager
def atomic_output_path(path, mode: str = "replace"):
    """Yield a same-directory temporary path and atomically publish it.

    This is used for binary renderers such as Matplotlib that need to write
    directly to a filesystem path. The destination is untouched if the
    renderer raises.
    """
    mode = resolve_write_mode(mode)
    path = Path(path)
    if mode == "merge":
        raise ValueError(f"merge mode is not supported for binary output: {path}")
    if path.exists() and mode == "fail_if_exists":
        raise ResultExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.stem}.",
                               suffix=path.suffix)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        yield tmp_path
        if not tmp_path.exists():
            raise FileNotFoundError(
                f"renderer for {path} returned without creating its temporary "
                f"artifact {tmp_path}")
        _fsync_written_file(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


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
