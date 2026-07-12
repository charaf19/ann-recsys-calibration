import json
from pathlib import Path

import pandas as pd
import pytest

import run_revision_experiments as R
from utils.fingerprints import embedding_fingerprint, index_fingerprint


def _grid():
    return R.expected_main_grid(["tiny"], "bm25", 4,
                                ["u2i", "i2i"], ["flat"], 42)


def _row(modality):
    return {"dataset": "tiny", "weighting": "bm25", "dim": 4,
            "modality": modality, "method": "flat", "seed": 42,
            "recall_at_k_mean": .1}


def test_checkpoint_each_cell_and_resume_verified_rows(tmp_path):
    path = tmp_path / "checkpoint.csv"
    created = R.write_main_checkpoint([_row("u2i")], path, "fp", _grid(), {})
    assert pd.read_csv(path).shape[0] == 1
    R.write_main_checkpoint([_row("u2i"), _row("i2i")], path, "fp", _grid(), {},
                            created_at=created["created_at_utc"])
    loaded, meta = R.load_main_checkpoint(path, "fp", _grid(),
                                          verify_cell=lambda row: True)
    assert len(loaded) == 2
    assert meta["created_at_utc"] == created["created_at_utc"]


def test_invalid_cell_is_not_resumed(tmp_path):
    path = tmp_path / "checkpoint.csv"
    R.write_main_checkpoint([_row("u2i"), _row("i2i")], path, "fp", _grid(), {})
    loaded, _ = R.load_main_checkpoint(
        path, "fp", _grid(),
        verify_cell=lambda row: (_ for _ in ()).throw(ValueError("bad"))
        if row["modality"] == "i2i" else True)
    assert list(loaded["modality"]) == ["u2i"]


def test_duplicate_and_incompatible_checkpoints_rejected(tmp_path):
    path = tmp_path / "checkpoint.csv"
    R.write_main_checkpoint([_row("u2i")], path, "fp", _grid(), {})
    pd.concat([pd.read_csv(path), pd.read_csv(path)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate natural keys"):
        R.load_main_checkpoint(path, "fp", _grid())
    R.write_main_checkpoint([_row("u2i")], path, "fp", _grid(), {})
    with pytest.raises(ValueError, match="configuration fingerprint"):
        R.load_main_checkpoint(path, "different", _grid())


def test_legacy_checkpoint_is_archived_not_deleted(tmp_path):
    path = tmp_path / "summary_main_checkpoint.csv"
    path.write_text("legacy\n", encoding="utf-8")
    archived = R.archive_legacy_checkpoint(path)
    assert archived.is_file()
    assert archived.read_text(encoding="utf-8") == "legacy\n"
    assert not path.exists()


def test_stage_fingerprints_ignore_unrelated_settings():
    emb = embedding_fingerprint(
        dataset="tiny", min_user_interactions=5, weighting="bm25", dim=128,
        normalize="l2", bm25_k1=1.2, bm25_b=.75, seed=42)
    base = {"hnsw": {"M": 24}, "pq": {"m": 32}, "docs": "old"}
    changed = {"hnsw": {"M": 24}, "pq": {"m": 64}, "docs": "new"}
    assert index_fingerprint(embedding_fp=emb, method="hnsw", budget_mb=100,
                             omp_threads=1, index_config=base, seed=42) == \
        index_fingerprint(embedding_fp=emb, method="hnsw", budget_mb=100,
                          omp_threads=1, index_config=changed, seed=42)


def test_powershell_forwards_both_resume_flags(repo_root):
    text = (repo_root / "run_full_experiments.ps1").read_text(encoding="utf-8")
    assert '"--reuse_existing", "--resume"' in text
    assert '$ScaleArguments += "--resume"' in text

