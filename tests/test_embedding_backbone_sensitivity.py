"""Backbone sensitivity propagates and validates the canonical k-core filter.

All tests avoid training: Test B captures the constructed subprocess command,
Tests C/D exercise the metadata validator directly.
"""
import json

import pytest

import run_embedding_backbone_sensitivity as mod
from run_embedding_backbone_sensitivity import _require_embedding_metadata


BASE_META = {
    "embedding_backend": "bpr_matrix_factorization",
    "dim": 128, "normalize": "l2", "seed": 42,
    "epochs": 10, "lr": 0.05, "reg": 0.002, "n_negatives": 1,
    "min_user_interactions": 5,
    "n_users": 6034, "n_items": 3533, "n_interactions": 575272,
}

BASE_EXPECTED = {
    "embedding_backend": "bpr_matrix_factorization",
    "dim": 128, "normalize": "l2", "seed": 42,
    "epochs": 10, "lr": 0.05, "reg": 0.002, "n_negatives": 1,
    "min_user_interactions": 5,
}


def _write_meta(tmp_path, meta):
    (tmp_path / "embedding_meta.json").write_text(
        json.dumps(meta), encoding="utf-8")
    return str(tmp_path)


def test_command_propagates_min_user_interactions(tmp_path, monkeypatch):
    """Test B: train_neural_embeddings.py is invoked with --min_user_interactions."""
    # Run from a clean cwd so no cached emb dir short-circuits into reuse.
    monkeypatch.chdir(tmp_path)
    captured = []

    def fake_run(cmd, check=True):
        captured.append([str(c) for c in cmd])
        return 0

    monkeypatch.setattr(mod, "run", fake_run)

    es_cfg = {"bpr": {"epochs": 10, "lr": 0.05, "reg": 0.002, "n_negatives": 1}}
    emb = mod.ensure_embeddings(
        "bpr_matrix_factorization", "ml-1m", 128, "l2", 42,
        es_cfg, embedding_cfg={}, cfg_hash="deadbeef",
        min_user_interactions=5)

    assert emb is not None
    assert len(captured) == 1
    cmd = captured[0]
    assert "src/train_neural_embeddings.py" in cmd
    assert "--min_user_interactions" in cmd
    assert cmd[cmd.index("--min_user_interactions") + 1] == "5"


def test_stale_metadata_rejected_when_min_missing(tmp_path):
    """Test C: metadata lacking min_user_interactions is refused."""
    meta = {k: v for k, v in BASE_META.items() if k != "min_user_interactions"}
    emb = _write_meta(tmp_path, meta)
    with pytest.raises(RuntimeError) as exc:
        _require_embedding_metadata(emb, dict(BASE_EXPECTED),
                                    "bpr_matrix_factorization")
    assert "min_user_interactions" in str(exc.value)


def test_stale_metadata_rejected_when_min_differs(tmp_path):
    """Test C: a different min_user_interactions is refused."""
    meta = {**BASE_META, "min_user_interactions": 1}
    emb = _write_meta(tmp_path, meta)
    with pytest.raises(RuntimeError) as exc:
        _require_embedding_metadata(emb, dict(BASE_EXPECTED),
                                    "bpr_matrix_factorization")
    assert "min_user_interactions" in str(exc.value)


def test_stale_metadata_rejected_when_population_prefilter(tmp_path):
    """Test C: a stored raw (pre-filter) population is refused."""
    meta = {**BASE_META, "n_users": 6038, "n_interactions": 575281}
    emb = _write_meta(tmp_path, meta)
    expected = {**BASE_EXPECTED, "n_users": 6034, "n_interactions": 575272}
    with pytest.raises(RuntimeError) as exc:
        _require_embedding_metadata(emb, expected, "bpr_matrix_factorization")
    msg = str(exc.value)
    assert "n_users" in msg or "n_interactions" in msg


def test_valid_metadata_accepted(tmp_path):
    """Test D: correctly filtered metadata at the configured threshold passes."""
    emb = _write_meta(tmp_path, dict(BASE_META))
    expected = {**BASE_EXPECTED, "n_users": 6034, "n_items": 3533,
                "n_interactions": 575272}
    # No exception == accepted.
    _require_embedding_metadata(emb, expected, "bpr_matrix_factorization")
