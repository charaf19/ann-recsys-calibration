import json

import numpy as np
import pandas as pd
import pytest

import calibrate
from run_calibration_sensitivity import (calibration_filename,
                                         consolidate_calibration_artifacts)
from utils.modality_queries import (build_query_population, item_id_map,
                                    load_query_cache, write_query_cache)


def _fixture(tmp_path):
    csv = tmp_path / "interactions.csv"
    pd.DataFrame([
        ("u1", "a", 1), ("u1", "b", 2), ("u1", "c", 3),
        ("u2", "b", 1), ("u2", "d", 2), ("u2", "e", 3),
    ], columns=["user_id", "item_id", "timestamp"]).to_csv(csv, index=False)
    emb = tmp_path / "emb"
    emb.mkdir()
    vecs = np.arange(20, dtype=np.float32).reshape(5, 4)
    np.save(emb / "item_vecs.npy", vecs)
    np.save(emb / "item_ids.npy", np.array(list("abcde")))
    return csv, emb / "item_vecs.npy", vecs


@pytest.mark.parametrize("modality", ["u2i", "i2i"])
def test_shared_query_population_is_deterministic_and_cache_validated(tmp_path,
                                                                      modality):
    csv, vec_path, vecs = _fixture(tmp_path)
    kwargs = dict(interactions_path=csv, item_vecs=vecs,
                  id2idx=item_id_map(vec_path, len(vecs)), modality=modality,
                  max_queries="full", seed=42, min_user_interactions=1,
                  metadata={"query_fingerprint": f"fp-{modality}"})
    first = build_query_population(**kwargs)
    second = build_query_population(**kwargs)
    np.testing.assert_array_equal(first.query_vectors, second.query_vectors)
    np.testing.assert_array_equal(first.query_ids, second.query_ids)
    cache = tmp_path / f"{modality}.npz"
    write_query_cache(first, cache)
    loaded = load_query_cache(cache, {"query_fingerprint": f"fp-{modality}"})
    np.testing.assert_array_equal(loaded.query_vectors, first.query_vectors)
    with pytest.raises(ValueError, match="incompatible query cache"):
        load_query_cache(cache, {"query_fingerprint": "wrong"})


def test_u2i_and_i2i_use_different_query_pools(tmp_path):
    csv, vec_path, vecs = _fixture(tmp_path)
    common = dict(interactions_path=csv, item_vecs=vecs,
                  id2idx=item_id_map(vec_path, len(vecs)), max_queries="full",
                  seed=42, min_user_interactions=1, metadata={})
    u2i = build_query_population(modality="u2i", **common)
    i2i = build_query_population(modality="i2i", **common)
    assert not np.array_equal(u2i.query_vectors, i2i.query_vectors)
    np.testing.assert_array_equal(u2i.query_ids, i2i.query_ids)


class _Exact:
    def search(self, q, topk):
        return np.tile(np.arange(topk), (len(q), 1))


class _Ann:
    method = "hnsw"

    def __init__(self, scores):
        self.scores = scores
        self.value = None

    def set_calibration_param(self, value):
        self.value = value

    def search(self, q, topk):
        matches = int(round(self.scores[self.value] * topk))
        row = np.r_[np.arange(matches), np.arange(100, 100 + topk - matches)]
        return np.tile(row, (len(q), 1))


def test_calibration_selects_smallest_reaching_and_best_unreachable(monkeypatch):
    monkeypatch.setattr(calibrate, "build_exact_index", lambda _: _Exact())
    monkeypatch.setattr(calibrate, "measure_latency",
                        lambda *args, **kwargs: {"mean": 1, "p50": 1,
                                                "p95": 1, "p99": 1})
    corpus = np.zeros((20, 3), dtype=np.float32)
    queries = np.ones((9, 3), dtype=np.float32)
    reached = calibrate.calibrate_index(
        _Ann({1: .7, 2: .9, 4: 1.0}), corpus, .9, 10, 5, 42,
        param_grid=[4, 2, 1], query_vectors=queries, modality="u2i")
    assert reached["target_reached"] is True
    assert reached["selected_param_value"] == 2
    assert reached["n_calibration_queries"] == 5
    assert reached["modality"] == "u2i"

    missed = calibrate.calibrate_index(
        _Ann({1: .7, 2: .8, 4: .8}), corpus, .95, 10, 5, 42,
        param_grid=[1, 2, 4], query_vectors=queries, modality="i2i")
    assert missed["target_reached"] is False
    assert missed["selected_param_value"] == 2  # best recall, smallest cost
    assert missed["achieved_recall_vs_exact"] == pytest.approx(.8)


def test_calibration_query_selection_is_seeded():
    q = np.arange(60, dtype=np.float32).reshape(20, 3)
    np.testing.assert_array_equal(
        calibrate.select_calibration_queries(q, 7, 42),
        calibrate.select_calibration_queries(q, 7, 42))


def test_modality_calibration_filenames_rows_and_keys(tmp_path):
    for modality in ("u2i", "i2i"):
        name = calibration_filename("tiny", "bm25", 4, modality,
                                    "hnsw", .95)
        doc = {
            "dataset": "tiny", "weighting": "bm25", "dim": 4,
            "modality": modality, "method": "hnsw", "target_recall": .95,
            "target_reached": True, "selected_param_value": 32,
            "achieved_recall_vs_exact": .96, "n_calibration_queries": 2,
            "query_source": "shared_modality_query_cache", "seed": 42,
            "calibration_fingerprint": f"cal-{modality}", "param_name": "ef",
            "latency_ms_at_calibrated": {"mean": 1, "p50": 1,
                                          "p95": 1, "p99": 1},
        }
        (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
    rows = consolidate_calibration_artifacts(
        tmp_path, ["tiny"], ["u2i", "i2i"], ["hnsw"], [.95],
        "bm25", 4, 42)
    assert len(rows) == 2
    assert {row["modality"] for row in rows} == {"u2i", "i2i"}
    assert all(row["selected_param_value"] == 32 for row in rows)
