"""U2I / I2I query-construction semantics (eval_modalities._build_queries)."""
import numpy as np
import pytest

from eval_modalities import _build_queries, _ranked_lists
from utils.ann_io import build_exact_index


@pytest.fixture()
def tiny_vectors():
    rng = np.random.default_rng(0)
    return rng.standard_normal((8, 4)).astype("float32")


def test_u2i_query_is_mean_of_training_history(tiny_vectors):
    hist = np.array([0, 2, 5], dtype=np.int32)
    Q, exclusions, positives = _build_queries("u2i", tiny_vectors,
                                              [hist], np.array([7]))
    np.testing.assert_allclose(Q[0], tiny_vectors[[0, 2, 5]].mean(axis=0),
                               rtol=1e-6)
    assert exclusions[0] == {0, 2, 5}   # every seen training item excluded
    assert positives[0] == {7}


def test_i2i_query_is_last_training_item_and_anchor_excluded(tiny_vectors):
    hist = np.array([3, 1, 6], dtype=np.int32)  # chronological order
    Q, exclusions, positives = _build_queries("i2i", tiny_vectors,
                                              [hist], np.array([2]))
    np.testing.assert_allclose(Q[0], tiny_vectors[6], rtol=1e-6)
    assert exclusions[0] == {6}, "only the anchor is excluded in I2I"
    assert positives[0] == {2}


def test_unknown_modality_raises(tiny_vectors):
    with pytest.raises(ValueError, match="unknown modality"):
        _build_queries("x2y", tiny_vectors, [np.array([0])], np.array([1]))


def test_ranked_lists_strip_exclusions(tiny_vectors):
    index = build_exact_index(tiny_vectors)
    hist = np.array([0, 2, 5], dtype=np.int32)
    Q, exclusions, _ = _build_queries("u2i", tiny_vectors, [hist],
                                      np.array([7]))
    ranked = _ranked_lists(index, Q, exclusions, topk=5,
                           N=tiny_vectors.shape[0])
    assert not (set(ranked[0]) & {0, 2, 5}), \
        "seen training items must never appear in U2I recommendations"
    assert len(ranked[0]) == 5


def test_i2i_anchor_never_recommended(tiny_vectors):
    index = build_exact_index(tiny_vectors)
    hist = np.array([1, 4], dtype=np.int32)
    Q, exclusions, _ = _build_queries("i2i", tiny_vectors, [hist],
                                      np.array([0]))
    ranked = _ranked_lists(index, Q, exclusions, topk=5,
                           N=tiny_vectors.shape[0])
    assert 4 not in ranked[0], "the I2I anchor item must be excluded"
