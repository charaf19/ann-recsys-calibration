"""Safe result writing: atomicity, write modes, merge semantics."""
import json

import pandas as pd
import pytest

from utils.result_io import (write_dataframe_atomic, write_json_atomic,
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
