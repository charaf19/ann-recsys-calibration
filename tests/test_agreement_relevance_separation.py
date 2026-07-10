"""Agreement recall (index fidelity) and recommendation relevance are two
different quantities and must be computed separately — never conflated."""
import numpy as np

from eval_modalities import evaluate_modality
from utils.ann_io import build_exact_index


def _tiny_setup(n=30, d=8, n_users=6, seed=0):
    rng = np.random.default_rng(seed)
    item_vecs = rng.standard_normal((n, d)).astype("float32")
    train_idx_list = [np.array(sorted(rng.choice(n, size=3, replace=False)),
                               dtype=np.int32) for _ in range(n_users)]
    test_idx = np.array([int(rng.integers(0, n)) for _ in range(n_users)],
                        dtype=np.int32)
    pop = rng.integers(1, 50, size=n).astype(np.int64)
    return item_vecs, train_idx_list, test_idx, pop


def test_agreement_and_relevance_are_separate_arrays():
    item_vecs, hists, test_idx, pop = _tiny_setup()
    exact = build_exact_index(item_vecs)
    agg, per_query, _, _ = evaluate_modality(
        "u2i", item_vecs, exact, exact, hists, test_idx, pop,
        topk=10, metric_topk=5, tail_frac=0.2)
    assert "recall" in per_query and "ann_recall_vs_exact" in per_query
    assert per_query["recall"].shape == per_query["ann_recall_vs_exact"].shape
    # both aggregate keys exist and are distinct fields
    assert "recall_at_k_mean" in agg
    assert "ann_recall_vs_exact_at_k_mean" in agg


def test_perfect_agreement_does_not_imply_relevance():
    """When ANN == exact, agreement recall is exactly 1.0 for every query,
    while relevance depends on the held-out item — an index can be perfectly
    faithful and still not useful (an embedding problem, not an index one)."""
    item_vecs, hists, test_idx, pop = _tiny_setup(seed=3)
    exact = build_exact_index(item_vecs)
    agg, per_query, _, _ = evaluate_modality(
        "u2i", item_vecs, exact, exact, hists, test_idx, pop,
        topk=10, metric_topk=5, tail_frac=0.2)
    np.testing.assert_allclose(per_query["ann_recall_vs_exact"], 1.0)
    assert agg["ann_recall_vs_exact_at_k_mean"] == 1.0
    assert agg["recall_at_k_mean"] < 1.0, \
        "with random embeddings the held-out item is not always retrieved"


def test_degraded_index_lowers_agreement_not_definitionally_relevance():
    """A deliberately wrong 'ANN' (shifted rankings) must lower agreement
    recall; relevance is computed against the held-out item, not against
    the exact list, so the two metrics move independently."""
    item_vecs, hists, test_idx, pop = _tiny_setup(seed=5)
    exact = build_exact_index(item_vecs)

    class ShiftedIndex:
        """Returns the exact ranking rotated by 7 — a broken index."""
        method = "broken"

        def search(self, Q, k):
            I = exact.search(Q, min(item_vecs.shape[0], k + 7))
            return np.roll(I, -7, axis=1)[:, :k]

    agg, per_query, _, _ = evaluate_modality(
        "u2i", item_vecs, ShiftedIndex(), exact, hists, test_idx, pop,
        topk=10, metric_topk=5, tail_frac=0.2)
    assert agg["ann_recall_vs_exact_at_k_mean"] < 1.0
    assert 0.0 <= agg["recall_at_k_mean"] <= 1.0
