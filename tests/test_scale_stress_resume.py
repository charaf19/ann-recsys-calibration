"""Pure regression tests for scale-stress checkpoint/resume behavior.

These tests never generate vectors, build an index, or run calibration.
"""
import sys

import pandas as pd
import pytest

import run_scale_stress as S
from utils.result_io import ResultExistsError


CFG_HASH = "abc123def456"
SEED = 42


def row(n_items, dim, method, config_hash=CFG_HASH):
    return {
        "n_items": n_items,
        "dim": dim,
        "method": method,
        "seed": SEED,
        "quality_measured": False,
        "config_hash": config_hash,
        "latency_p95_ms": 1.0,
    }


def test_checkpoint_is_separate_and_does_not_overwrite_final(tmp_path):
    final_path = tmp_path / S.FINAL_FILENAME
    checkpoint_path = tmp_path / S.CHECKPOINT_FILENAME
    final_path.write_text("published evidence\n", encoding="utf-8")

    S.write_checkpoint([row(10, 2, "flat")], checkpoint_path)

    assert final_path.read_text(encoding="utf-8") == "published evidence\n"
    assert checkpoint_path.is_file()
    assert set(pd.read_csv(checkpoint_path)["method"]) == {"flat"}


def test_resume_loads_checkpoint_and_skips_completed_natural_keys(tmp_path):
    checkpoint_path = tmp_path / S.CHECKPOINT_FILENAME
    S.write_checkpoint([row(10, 2, "flat")], checkpoint_path)

    loaded = S.load_resume_checkpoint(checkpoint_path, CFG_HASH)
    completed = S.completed_grid_keys(loaded)
    pending = S.pending_grid_cells(
        [10], [2], ["flat", "hnsw"], SEED, completed)

    assert completed == {(10, 2, "flat", SEED)}
    assert pending == [(10, 2, "hnsw")]


def test_resume_rejects_incompatible_configuration_hash(tmp_path):
    checkpoint_path = tmp_path / S.CHECKPOINT_FILENAME
    S.write_checkpoint(
        [row(10, 2, "flat", config_hash="old000000000")],
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="incompatible checkpoint.*hash"):
        S.load_resume_checkpoint(checkpoint_path, CFG_HASH)


def test_complete_grid_validation_rejects_missing_cell():
    frame = pd.DataFrame([row(10, 2, "flat")])

    with pytest.raises(ValueError, match="grid is incomplete"):
        S.validate_complete_grid(
            frame, [10], [2], ["flat", "hnsw"], SEED, CFG_HASH)


def test_successful_finalization_publishes_then_removes_checkpoint(tmp_path):
    final_path = tmp_path / S.FINAL_FILENAME
    checkpoint_path = tmp_path / S.CHECKPOINT_FILENAME
    rows = [row(10, 2, "flat"), row(10, 2, "hnsw")]
    S.write_checkpoint(rows, checkpoint_path)

    S.finalize_results(
        rows, final_path, checkpoint_path, "fail_if_exists",
        [10], [2], ["flat", "hnsw"], SEED, CFG_HASH,
    )

    assert final_path.is_file()
    assert set(pd.read_csv(final_path)["method"]) == {"flat", "hnsw"}
    assert not checkpoint_path.exists()


def test_failed_final_write_retains_checkpoint(tmp_path):
    final_path = tmp_path / S.FINAL_FILENAME
    checkpoint_path = tmp_path / S.CHECKPOINT_FILENAME
    rows = [row(10, 2, "flat")]
    S.write_checkpoint(rows, checkpoint_path)
    final_path.write_text("existing evidence\n", encoding="utf-8")

    with pytest.raises(ResultExistsError):
        S.finalize_results(
            rows, final_path, checkpoint_path, "fail_if_exists",
            [10], [2], ["flat"], SEED, CFG_HASH,
        )

    assert checkpoint_path.is_file()
    assert final_path.read_text(encoding="utf-8") == "existing evidence\n"


def test_cost_only_configuration_rejects_quality_measurement():
    S.require_cost_only(False)
    with pytest.raises(ValueError, match="must be false"):
        S.require_cost_only(True)


def test_dry_run_does_not_create_output_directory(tmp_path, repo_root,
                                                   monkeypatch):
    out_dir = tmp_path / "never-created"
    monkeypatch.setattr(sys, "argv", [
        "run_scale_stress.py",
        "--config", str(repo_root / "configs" / "analyses.yml"),
        "--catalog_sizes", "10",
        "--dimensions", "2",
        "--methods", "flat",
        "--out_dir", str(out_dir),
        "--dry_run",
    ])

    S.main()

    assert not out_dir.exists()

