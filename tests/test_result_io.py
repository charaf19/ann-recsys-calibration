"""Safe result writing: atomicity, write modes, merge semantics."""
import builtins
import json

import numpy as np
import pandas as pd
import pytest

from utils import result_io
from utils.result_io import (write_dataframe_atomic, write_json_atomic,
                             write_npz_atomic, atomic_output_path,
                             _fsync_written_file,
                             merge_dataframe, validate_unique_keys,
                             resolve_write_mode, preflight_output,
                             ResultExistsError, MergeConflictError)

KEY = ["dataset", "method"]


def df_of(rows):
    return pd.DataFrame(rows)


def test_resolve_write_mode_rejects_unknown():
    assert resolve_write_mode("REPLACE") == "replace"
    with pytest.raises(ValueError, match="unknown write mode"):
        resolve_write_mode("clobber")


def test_fail_if_exists_refuses_overwrite(tmp_path):
    p = tmp_path / "out.csv"
    write_dataframe_atomic(df_of([{"dataset": "a", "method": "flat", "v": 1}]),
                           p, mode="fail_if_exists", key=KEY)
    with pytest.raises(ResultExistsError):
        write_dataframe_atomic(df_of([{"dataset": "b", "method": "flat", "v": 2}]),
                               p, mode="fail_if_exists", key=KEY)


def test_preflight_fails_before_work(tmp_path):
    p = tmp_path / "out.csv"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ResultExistsError, match="refusing to start"):
        preflight_output(p, "fail_if_exists")
    assert preflight_output(p, "replace") == "replace"


def test_merge_preserves_unrelated_rows(tmp_path):
    p = tmp_path / "out.csv"
    write_dataframe_atomic(df_of([{"dataset": "ml-1m", "method": "flat", "v": 1.0},
                                  {"dataset": "ml-1m", "method": "hnsw", "v": 2.0}]),
                           p, mode="fail_if_exists", key=KEY)
    write_dataframe_atomic(df_of([{"dataset": "goodbooks", "method": "flat", "v": 3.0}]),
                           p, mode="merge", key=KEY)
    out = pd.read_csv(p)
    assert len(out) == 3
    assert set(out["dataset"]) == {"ml-1m", "goodbooks"}
    kept = out[(out["dataset"] == "ml-1m") & (out["method"] == "hnsw")]
    assert kept["v"].iloc[0] == 2.0  # unrelated row untouched


def test_merge_conflicting_values_raise(tmp_path):
    p = tmp_path / "out.csv"
    write_dataframe_atomic(df_of([{"dataset": "a", "method": "flat", "v": 1.0}]),
                           p, mode="fail_if_exists", key=KEY)
    with pytest.raises(MergeConflictError, match="conflicting rows"):
        write_dataframe_atomic(df_of([{"dataset": "a", "method": "flat", "v": 9.9}]),
                               p, mode="merge", key=KEY)
    assert pd.read_csv(p)["v"].iloc[0] == 1.0  # previous file undamaged


def test_merge_identical_duplicates_dedupe():
    a = df_of([{"dataset": "a", "method": "flat", "v": 1.0}])
    merged = merge_dataframe(a, a.copy(), KEY)
    assert len(merged) == 1


def test_identical_duplicate_new_rows_are_deduplicated(tmp_path):
    p = tmp_path / "out.csv"
    row = {"dataset": "a", "method": "flat", "v": 1.0}
    write_dataframe_atomic(df_of([row, row]), p, mode="replace", key=KEY)
    assert len(pd.read_csv(p)) == 1


def test_duplicate_keys_in_new_rows_rejected(tmp_path):
    bad = df_of([{"dataset": "a", "method": "flat", "v": 1},
                 {"dataset": "a", "method": "flat", "v": 2}])
    with pytest.raises(MergeConflictError, match="duplicate natural keys"):
        write_dataframe_atomic(bad, tmp_path / "out.csv",
                               mode="replace", key=KEY)


def test_validate_unique_keys_direct():
    ok = df_of([{"dataset": "a", "method": "flat"},
                {"dataset": "a", "method": "hnsw"}])
    validate_unique_keys(ok, KEY)


def test_complete_natural_key_is_required(tmp_path):
    bad = df_of([{"dataset": "a", "v": 1}])
    with pytest.raises(ValueError, match="missing natural-key columns"):
        write_dataframe_atomic(bad, tmp_path / "out.csv", mode="replace",
                               key=KEY)


def test_null_natural_key_is_rejected(tmp_path):
    bad = df_of([{"dataset": "a", "method": None, "v": 1}])
    with pytest.raises(MergeConflictError, match="null values"):
        write_dataframe_atomic(bad, tmp_path / "out.csv", mode="replace",
                               key=KEY)


def test_nearly_equal_values_are_conflicting():
    a = df_of([{"dataset": "a", "method": "flat", "v": 1.0}])
    b = df_of([{"dataset": "a", "method": "flat", "v": 1.0 + 1e-13}])
    with pytest.raises(MergeConflictError, match="conflicting rows"):
        merge_dataframe(a, b, KEY)


def test_conflicting_duplicates_in_existing_file_are_rejected(tmp_path):
    p = tmp_path / "out.csv"
    p.write_text("dataset,method,v\na,flat,1\na,flat,9\n", encoding="utf-8")
    new = df_of([{"dataset": "b", "method": "flat", "v": 2}])
    with pytest.raises(MergeConflictError, match="conflicting rows"):
        write_dataframe_atomic(new, p, mode="merge", key=KEY)


def test_output_ordering_deterministic(tmp_path):
    p = tmp_path / "out.csv"
    rows = [{"dataset": "b", "method": "hnsw", "v": 1},
            {"dataset": "a", "method": "flat", "v": 2},
            {"dataset": "a", "method": "hnsw", "v": 3}]
    write_dataframe_atomic(df_of(rows), p, mode="replace", key=KEY)
    first = p.read_text(encoding="utf-8")
    write_dataframe_atomic(df_of(list(reversed(rows))), p, mode="replace",
                           key=KEY)
    assert p.read_text(encoding="utf-8") == first


def test_failed_write_does_not_corrupt_previous(tmp_path, monkeypatch):
    p = tmp_path / "out.csv"
    write_dataframe_atomic(df_of([{"dataset": "a", "method": "flat", "v": 1}]),
                           p, mode="replace", key=KEY)
    before = p.read_text(encoding="utf-8")

    class Exploding(pd.DataFrame):
        def to_csv(self, *a, **k):
            raise RuntimeError("disk full (simulated)")

    with pytest.raises(RuntimeError, match="disk full"):
        write_dataframe_atomic(
            Exploding([{"dataset": "z", "method": "flat", "v": 9}]),
            p, mode="replace")
    assert p.read_text(encoding="utf-8") == before
    assert not list(tmp_path.glob("*.tmp")), "temp file must be cleaned up"


def test_json_atomic_modes(tmp_path):
    p = tmp_path / "doc.json"
    write_json_atomic({"a": 1}, p)
    with pytest.raises(ResultExistsError):
        write_json_atomic({"a": 2}, p, mode="fail_if_exists")
    write_json_atomic({"b": 2}, p, mode="merge")
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    with pytest.raises(MergeConflictError):
        write_json_atomic({"a": 999}, p, mode="merge")
    write_json_atomic({"c": 3}, p, mode="replace")
    assert json.loads(p.read_text(encoding="utf-8")) == {"c": 3}


# ---------------------------------------------------------------------------
# Atomic NPZ / binary publication (regression for OSError [Errno 9] on
# Windows: os.fsync() was called on a read-only "rb" descriptor).
# ---------------------------------------------------------------------------

ARR_A = np.arange(6, dtype=np.int64).reshape(2, 3)
ARR_B = np.linspace(0, 1, 4, dtype=np.float32)


def _write_npz(path, mode="fail_if_exists", **arrays):
    if not arrays:
        arrays = {"a": ARR_A, "b": ARR_B}
    return write_npz_atomic(path, mode=mode, **arrays)


def test_write_npz_atomic_creates_readable_archive(tmp_path):
    p = tmp_path / "q.npz"
    _write_npz(p)
    with np.load(p) as z:
        assert set(z.files) == {"a", "b"}


def test_write_npz_atomic_roundtrips_exact_arrays(tmp_path):
    p = tmp_path / "q.npz"
    _write_npz(p, a=ARR_A, b=ARR_B)
    with np.load(p) as z:
        np.testing.assert_array_equal(z["a"], ARR_A)
        np.testing.assert_array_equal(z["b"], ARR_B)
        assert z["a"].dtype == ARR_A.dtype
        assert z["b"].dtype == ARR_B.dtype
    assert not list(tmp_path.glob(".*.npz")), "no temp file must remain"


def test_write_npz_replace_atomically_overwrites(tmp_path):
    p = tmp_path / "q.npz"
    _write_npz(p, a=ARR_A)
    _write_npz(p, mode="replace", a=ARR_A + 100)
    with np.load(p) as z:
        np.testing.assert_array_equal(z["a"], ARR_A + 100)


def test_write_npz_fail_if_exists_refuses(tmp_path):
    p = tmp_path / "q.npz"
    _write_npz(p, a=ARR_A)
    with pytest.raises(ResultExistsError):
        _write_npz(p, mode="fail_if_exists", a=ARR_A + 1)
    with np.load(p) as z:
        np.testing.assert_array_equal(z["a"], ARR_A)  # original untouched


def test_write_npz_merge_accepts_identical(tmp_path):
    p = tmp_path / "q.npz"
    _write_npz(p, a=ARR_A, b=ARR_B)
    # identical archive is accepted (idempotent republish)
    _write_npz(p, mode="merge", a=ARR_A, b=ARR_B)
    with np.load(p) as z:
        np.testing.assert_array_equal(z["a"], ARR_A)


def test_write_npz_merge_rejects_different_keys(tmp_path):
    p = tmp_path / "q.npz"
    _write_npz(p, a=ARR_A, b=ARR_B)
    with pytest.raises(MergeConflictError, match="archive keys differ"):
        _write_npz(p, mode="merge", a=ARR_A, c=ARR_B)


def test_write_npz_merge_rejects_shape_difference(tmp_path):
    p = tmp_path / "q.npz"
    _write_npz(p, a=ARR_A)
    with pytest.raises(MergeConflictError, match="shape or dtype differs"):
        _write_npz(p, mode="merge", a=ARR_A.reshape(3, 2))


def test_write_npz_merge_rejects_dtype_difference(tmp_path):
    p = tmp_path / "q.npz"
    _write_npz(p, a=ARR_A)
    with pytest.raises(MergeConflictError, match="shape or dtype differs"):
        _write_npz(p, mode="merge", a=ARR_A.astype(np.float64))


def test_write_npz_merge_rejects_value_difference(tmp_path):
    p = tmp_path / "q.npz"
    _write_npz(p, a=ARR_A)
    with pytest.raises(MergeConflictError, match="values differ"):
        _write_npz(p, mode="merge", a=ARR_A + 1)


def test_write_npz_savez_failure_cleans_tmp_and_preserves_dest(tmp_path,
                                                               monkeypatch):
    p = tmp_path / "q.npz"
    _write_npz(p, a=ARR_A)
    before = p.read_bytes()

    def boom(*a, **k):
        raise RuntimeError("savez exploded (simulated)")

    monkeypatch.setattr(result_io.np, "savez_compressed", boom)
    with pytest.raises(RuntimeError, match="savez exploded"):
        _write_npz(p, mode="replace", a=ARR_A + 5)
    assert p.read_bytes() == before, "destination NPZ must be untouched"
    assert not list(tmp_path.glob(".*.npz")), "temp file must be removed"


def test_write_npz_fsync_failure_cleans_tmp_and_preserves_dest(tmp_path,
                                                               monkeypatch):
    p = tmp_path / "q.npz"
    _write_npz(p, a=ARR_A)
    before = p.read_bytes()

    def boom(_path):
        raise OSError(9, "Bad file descriptor (simulated)")

    monkeypatch.setattr(result_io, "_fsync_written_file", boom)
    with pytest.raises(OSError, match="Bad file descriptor"):
        _write_npz(p, mode="replace", a=ARR_A + 7)
    assert p.read_bytes() == before, "destination NPZ must be untouched"
    assert not list(tmp_path.glob(".*.npz")), "temp file must be removed"


def test_atomic_output_path_publishes_binary(tmp_path):
    p = tmp_path / "fig.png"
    payload = b"\x89PNG\r\n\x1a\n binary payload"
    with atomic_output_path(p, mode="replace") as tmp:
        tmp.write_bytes(payload)
    assert p.read_bytes() == payload
    assert not list(tmp_path.glob(".*")), "no temp file must remain"


def test_atomic_output_path_cleans_tmp_on_error(tmp_path):
    p = tmp_path / "fig.png"
    with pytest.raises(RuntimeError, match="render failed"):
        with atomic_output_path(p, mode="replace") as tmp:
            tmp.write_bytes(b"partial")
            raise RuntimeError("render failed (simulated)")
    assert not p.exists(), "destination must not be created on failure"
    assert not list(tmp_path.glob(".*")), "temp file must be cleaned up"


def test_atomic_output_path_errors_if_renderer_produces_nothing(tmp_path):
    p = tmp_path / "fig.png"
    with pytest.raises(FileNotFoundError, match="without creating"):
        with atomic_output_path(p, mode="replace") as tmp:
            tmp.unlink()  # renderer wrote to the wrong place / nothing
    assert not p.exists()


def test_fsync_helper_uses_writable_binary_mode(tmp_path, monkeypatch):
    """Guard: the helper MUST reopen with a writable mode ("r+b").

    If someone reverts it to "rb", os.fsync() raises OSError [Errno 9] on
    Windows. This test fails for any non-writable mode regardless of platform,
    so the Windows-only regression cannot silently pass CI again.
    """
    p = tmp_path / "artifact.bin"
    p.write_bytes(b"data")
    captured = {}
    real_open = builtins.open

    def spy_open(file, mode="r", *a, **k):
        captured["mode"] = mode
        if "+" not in mode and ("w" not in mode and "a" not in mode):
            raise OSError(9, "Bad file descriptor (simulated read-only fd)")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    _fsync_written_file(p)  # must not raise with the correct "r+b" mode
    assert "+" in captured["mode"] or "w" in captured["mode"], \
        f"fsync helper opened with non-writable mode {captured['mode']!r}"


def test_fsync_helper_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="never created"):
        _fsync_written_file(tmp_path / "does_not_exist.bin")
