"""End-to-end evaluator test (eval_modalities.main) and per-batch retrieval
equivalence (Phase 6).

Everything runs on a tiny in-memory catalog: a handful of items, a Flat FAISS
index, and a couple of users. No downloads, no scientific run.
"""
import json

import faiss
import numpy as np
import pytest

import eval_modalities
from eval_modalities import _build_queries, _ranked_lists
from utils.ann_io import build_exact_index
from utils.splits import temporal_leave_one_out, build_eval_cases

CONFIG = "configs/main_cpu.yml"

# Required NPZ arrays per the evidence contract (see task Phase 3.5).
REQUIRED_NPZ_KEYS = {
    "meta", "query_ids", "recall", "precision", "hr", "ndcg", "map", "mrr",
    "ann_recall_vs_exact", "recall_at_100", "ann_recall_vs_exact_at_100",
    "exposure_counts_at_k", "exposure_counts_at_100", "pop_counts",
    "recs_at_k", "hist_pop_mean", "exposure_proxy",
}
PER_QUERY_ARRAYS = {
    "recall", "precision", "hr", "ndcg", "map", "mrr",
    "ann_recall_vs_exact", "recall_at_100", "ann_recall_vs_exact_at_100",
    "recs_at_k", "hist_pop_mean",
}


@pytest.fixture()
def tiny_dataset(tmp_path):
    """A tiny normalized interactions CSV + aligned item vectors + Flat index."""
    # 8 items, distinct item ids (strings), 4 users with short histories.
    n_items, dim = 8, 6
    rng = np.random.default_rng(7)
    item_ids = np.array([f"i{j}" for j in range(n_items)], dtype=str)
    item_vecs = rng.standard_normal((n_items, dim)).astype("float32")

    rows = [
        # user, item, timestamp  (chronological within each user)
        ("u1", "i0", 1), ("u1", "i1", 2), ("u1", "i2", 3),
        ("u2", "i3", 1), ("u2", "i4", 2),
        ("u3", "i5", 1), ("u3", "i0", 2), ("u3", "i6", 3),
        ("u4", "i7", 1), ("u4", "i2", 2),
    ]
    csv = tmp_path / "tiny.csv"
    csv.write_text(
        "user_id,item_id,timestamp\n"
        + "\n".join(f"{u},{i},{t}" for u, i, t in rows) + "\n",
        encoding="utf-8")

    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    item_vecs_path = emb_dir / "item_vecs.npy"
    np.save(item_vecs_path, item_vecs)
    np.save(emb_dir / "item_ids.npy", item_ids)

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    index = faiss.IndexFlatL2(dim)
    index.add(np.ascontiguousarray(item_vecs))
    faiss.write_index(index, str(index_dir / "faiss_flat.index"))

    return {
        "csv": csv, "item_vecs": item_vecs_path, "index": index_dir,
        "item_vecs_arr": item_vecs, "item_ids": item_ids,
        "n_items": n_items, "dim": dim, "tmp": tmp_path,
    }


def _run_eval(ds, tmp_path, modality, min_user_interactions=1):
    agg_dir = tmp_path / "agg"
    pq_dir = tmp_path / "pq"
    # The tiny fixture users have <5 interactions, so the e2e run overrides the
    # canonical k-core filter to keep them evaluable; the filter itself is
    # exercised by test_min_user_interactions_* below.
    argv = [
        "--interactions", str(ds["csv"]),
        "--item_vecs", str(ds["item_vecs"]),
        "--index", str(ds["index"]),
        "--ann_method", "flat",
        "--modality", modality,
        "--queries", "full",
        "--topk", "4",
        "--metric_topk", "2",
        "--ef", "64", "--nprobe", "8",
        "--seed", "42",
        "--dataset", "tiny",
        "--weighting", "bm25",
        "--min_user_interactions", str(min_user_interactions),
        "--config", CONFIG,
        "--aggregate_dir", str(agg_dir),
        "--perquery_dir", str(pq_dir),
        "--write_mode", "replace",
    ]
    rc = eval_modalities.main(argv)
    return rc, agg_dir, pq_dir


@pytest.mark.parametrize("modality", ["u2i", "i2i"])
def test_evaluator_end_to_end(tiny_dataset, tmp_path, modality):
    ds = tiny_dataset
    rc, agg_dir, pq_dir = _run_eval(ds, tmp_path, modality)

    assert rc == 0, "evaluator must exit zero"

    stem = f"tiny__bm25__d{ds['dim']}__{modality}__flat"
    agg_path = agg_dir / f"{stem}.json"
    npz_path = pq_dir / f"{stem}.npz"
    assert agg_path.is_file(), "aggregate JSON must exist"
    assert npz_path.is_file(), "per-query NPZ must exist"

    # aggregate metadata contract
    agg = json.loads(agg_path.read_text(encoding="utf-8"))
    for field in ("dataset", "weighting", "method", "detected_method", "N",
                  "D", "dim", "seed"):
        assert field in agg, f"aggregate metadata missing {field}"
    assert agg["dataset"] == "tiny"
    assert agg["method"] == "flat" and agg["detected_method"] == "flat"
    assert agg["dim"] == ds["dim"] and agg["N"] == ds["n_items"]
    assert agg["seed"] == 42
    assert agg["min_user_interactions"] == 1  # recorded k-core filter value

    with np.load(npz_path, allow_pickle=True) as z:
        keys = set(z.files)
        assert REQUIRED_NPZ_KEYS <= keys, \
            f"missing NPZ arrays: {REQUIRED_NPZ_KEYS - keys}"

        query_ids = z["query_ids"]
        n_q = len(query_ids)
        assert n_q > 0
        # every per-query array aligns with query_ids
        for name in PER_QUERY_ARRAYS:
            assert z[name].shape[0] == n_q, \
                f"{name} not aligned with query_ids ({z[name].shape[0]} != {n_q})"

        recs = z["recs_at_k"]  # (n_q, metric_topk) of item indices

    # Reconstruct the split to check exclusions honestly.
    inter = eval_modalities._load_interactions(str(ds["csv"]))
    id2idx = eval_modalities._load_id_map(ds["item_vecs"], ds["n_items"])
    train_df, test_df = temporal_leave_one_out(inter)
    users, train_idx_list, _ = build_eval_cases(
        train_df, test_df, id2idx, max_queries="full", seed=42)
    assert list(query_ids) == [str(u) for u in users], "query id ordering"

    for i in range(n_q):
        rec_items = {int(x) for x in recs[i] if int(x) >= 0}
        hist = set(int(h) for h in train_idx_list[i])
        if modality == "u2i":
            assert not (rec_items & hist), \
                "U2I recommendations must exclude all training-history items"
        else:  # i2i excludes the anchor (chronologically last training item)
            anchor = int(train_idx_list[i][-1])
            assert anchor not in rec_items, "I2I must exclude its anchor item"

    # no temporary files anywhere under the output dirs
    for d in (agg_dir, pq_dir):
        assert not list(d.glob(".*")), f"stray temp file under {d}"


def test_evaluator_both_modalities_no_tempfiles(tiny_dataset, tmp_path):
    ds = tiny_dataset
    for modality in ("u2i", "i2i"):
        rc, agg_dir, pq_dir = _run_eval(ds, tmp_path, modality)
        assert rc == 0
    # every produced artifact is a real, loadable file with no temp residue
    for f in list((tmp_path / "pq").glob("*.npz")):
        with np.load(f, allow_pickle=True) as z:
            assert len(z["query_ids"]) > 0


# ---------------------------------------------------------------------------
# Phase 5: the evaluated population must equal the k-core-filtered population,
# and the same filter must feed the dataset-statistics table.
# ---------------------------------------------------------------------------

def test_min_user_interactions_filters_evaluated_population(tiny_dataset,
                                                            tmp_path):
    ds = tiny_dataset
    # min=3 keeps only u1 (3) and u3 (3); u2 (2) and u4 (2) are dropped.
    rc, agg_dir, pq_dir = _run_eval(ds, tmp_path, "u2i",
                                    min_user_interactions=3)
    assert rc == 0
    stem = f"tiny__bm25__d{ds['dim']}__u2i__flat"
    with np.load(pq_dir / f"{stem}.npz", allow_pickle=True) as z:
        query_ids = set(str(q) for q in z["query_ids"])
    assert query_ids == {"u1", "u3"}, \
        "only users with >= min_user_interactions may be evaluated"


def test_stats_and_eval_agree_on_population(tiny_dataset):
    """The dataset-stats k-core population must equal the evaluated one."""
    from dataset_stats import stats_for
    ds = tiny_dataset
    m = 3
    s = stats_for("tiny", str(ds["csv"]), m)
    assert s["users"] == 2  # u1, u3
    assert s["min_user_interactions_filter"] == m

    inter = eval_modalities._load_interactions(str(ds["csv"]))
    from utils.preprocessing import filter_min_user_interactions
    filtered = filter_min_user_interactions(inter, m)
    assert filtered["user_id"].nunique() == s["users"]
    assert len(filtered) == s["interactions"]


# ---------------------------------------------------------------------------
# Phase 6: per-batch retrieval depth must produce identical ranked lists to a
# simple reference that searches once at global (topk + max_exclusions) depth.
# ---------------------------------------------------------------------------

def _reference_ranked_lists(index, Q, exclusions, topk, N):
    """Reference: single global depth = topk + max over ALL exclusion sets."""
    max_excl = max((len(e) for e in exclusions), default=0)
    depth = min(N, topk + max_excl)
    ranked = []
    I = index.search(Q, depth)
    for r, row in enumerate(I):
        ex = exclusions[r]
        ranked.append([int(x) for x in row
                       if int(x) >= 0 and int(x) not in ex][:topk])
    return ranked


def _catalog(n_items, dim, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_items, dim)).astype("float32")


@pytest.mark.parametrize("modality", ["u2i", "i2i"])
@pytest.mark.parametrize("batch", [1, 3, 1024])
def test_ranked_lists_equivalent_to_reference(modality, batch):
    vecs = _catalog(40, 5)
    index = build_exact_index(vecs)
    rng = np.random.default_rng(11)
    # varying history lengths, including a single very long history
    hist_lengths = [1, 2, 3, 7, 15, 1, 4]
    train_idx_list, test_idx = [], []
    for L in hist_lengths:
        h = rng.choice(vecs.shape[0], size=L, replace=False).astype(np.int32)
        train_idx_list.append(h)
        test_idx.append(int(rng.integers(0, vecs.shape[0])))
    Q, exclusions, _ = _build_queries(modality, vecs, train_idx_list,
                                      np.array(test_idx))
    topk, N = 5, vecs.shape[0]
    opt = _ranked_lists(index, Q, exclusions, topk, N, batch=batch)
    ref = _reference_ranked_lists(index, Q, exclusions, topk, N)
    assert opt == ref


def test_ranked_lists_catalog_smaller_than_depth():
    """Catalog smaller than topk + exclusions: both must degrade identically."""
    vecs = _catalog(6, 4)
    index = build_exact_index(vecs)
    train_idx_list = [np.array([0, 1, 2, 3], dtype=np.int32),
                      np.array([5], dtype=np.int32)]
    Q, exclusions, _ = _build_queries("u2i", vecs, train_idx_list,
                                      np.array([4, 0]))
    topk, N = 5, vecs.shape[0]  # topk + 4 exclusions > N
    opt = _ranked_lists(index, Q, exclusions, topk, N, batch=1)
    ref = _reference_ranked_lists(index, Q, exclusions, topk, N)
    assert opt == ref


def test_ranked_lists_handles_negative_padding_ids():
    """FAISS returns -1 padding when depth exceeds catalog; both drop them."""
    vecs = _catalog(3, 4)
    index = build_exact_index(vecs)
    train_idx_list = [np.array([0], dtype=np.int32)]
    Q, exclusions, _ = _build_queries("i2i", vecs, train_idx_list,
                                      np.array([1]))
    topk, N = 10, vecs.shape[0]  # depth capped at N=3, one excluded -> <=2 recs
    opt = _ranked_lists(index, Q, exclusions, topk, N, batch=1)
    ref = _reference_ranked_lists(index, Q, exclusions, topk, N)
    assert opt == ref
    assert all(x >= 0 for x in opt[0]), "negative padding ids must be stripped"
