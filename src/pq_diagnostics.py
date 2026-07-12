"""PQ compression diagnostics (library).

Reviewer concern addressed: the earlier "PQ may act as implicit
regularization" statement was speculative. This module replaces it with
measurable diagnostics of what PQ actually does to the embedding geometry
and to retrieval, WITHOUT ever asserting a regularization mechanism. The
interpretation labels are deliberately conservative and evidence-bound:

    compression_hurts_quality      delta-NDCG clearly negative
    compression_preserves_quality  delta-NDCG within the negligible band
    compression_may_smooth_noise   delta-NDCG clearly positive (a *may*, not a claim)
    insufficient_evidence          quality delta unavailable

Diagnostics per (dataset, PQ method):
  1. reconstruction error (relative L2, via faiss sa_encode/sa_decode)
  2. embedding norm distortion
  3. pairwise distance distortion (seeded sample of pairs)
  4. top-10 / top-100 neighbor overlap with exact Flat
  5. retrieval score variance before/after PQ
  6. popularity-decile effects (recon error and overlap per decile)
  7. long-tail exposure shift (PQ top-10 exposure vs Flat top-10 exposure)
  8. correlation: reconstruction error vs delta-NDCG (across datasets)
  9. correlation: neighbor overlap vs delta-NDCG (across datasets)
"""
import numpy as np

from utils import metrics as M

# negligible band for the end-to-end NDCG delta (absolute)
NDCG_EPS = 0.005


def reconstruct(index, item_vecs):
    """Round-trip item vectors through the index's PQ codec."""
    codes = index.sa_encode(np.ascontiguousarray(item_vecs, dtype=np.float32))
    return index.sa_decode(codes)


def reconstruction_error(x, x_hat):
    """Mean relative L2 reconstruction error."""
    num = np.linalg.norm(x - x_hat, axis=1)
    den = np.linalg.norm(x, axis=1)
    den[den == 0] = 1.0
    return num / den


def norm_distortion(x, x_hat):
    """Per-vector relative norm change (signed)."""
    nx = np.linalg.norm(x, axis=1)
    nh = np.linalg.norm(x_hat, axis=1)
    nx_safe = np.where(nx == 0, 1.0, nx)
    return (nh - nx) / nx_safe


def pairwise_distance_distortion(x, x_hat, n_pairs=20000, seed=42):
    """Mean relative absolute error of pairwise L2 distances on sampled pairs."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d = np.linalg.norm(x[i] - x[j], axis=1)
    dh = np.linalg.norm(x_hat[i] - x_hat[j], axis=1)
    d_safe = np.where(d == 0, 1.0, d)
    return float(np.mean(np.abs(dh - d) / d_safe))


def neighbor_overlap(exact, ann, Q, k):
    """Mean |topk(ann) ∩ topk(exact)| / k over query rows."""
    I_e = exact.search(Q, k)
    I_a = ann.search(Q, k)
    ov = np.zeros(Q.shape[0], dtype=np.float64)
    for r in range(Q.shape[0]):
        ov[r] = len(set(int(x) for x in I_a[r]) & set(int(x) for x in I_e[r])) / float(k)
    return ov


def score_variance(index_like, Q, k=100):
    """Mean per-query variance of valid top-k L2 distances.

    FAISS IVF indexes can fill missing neighbors with a float32-maximum
    sentinel. These entries are not retrieval scores and must be excluded.
    Calculations are performed in float64 to prevent overflow.
    """
    import faiss  # noqa: F401  (index_like is a faiss index or AnnIndex-like)

    if hasattr(index_like, "search") and not hasattr(index_like, "d"):
        raise ValueError(
            "score_variance needs a raw faiss index (returns distances)"
        )

    distances, _ = index_like.search(
        np.ascontiguousarray(Q, dtype=np.float32), int(k)
    )
    distances = np.asarray(distances, dtype=np.float64)

    sentinel_limit = float(np.finfo(np.float32).max) / 2.0
    valid = np.isfinite(distances) & (distances < sentinel_limit)

    row_variances = []
    for row, mask in zip(distances, valid):
        values = row[mask]
        if values.size >= 2:
            row_variances.append(
                float(np.var(values, dtype=np.float64))
            )

    if not row_variances:
        return float("nan")

    return float(
        np.mean(np.asarray(row_variances, dtype=np.float64))
    )


def popularity_deciles(pop_counts):
    """Assign each item to a popularity decile (0 = least popular).

    Items with zero training popularity all land in decile 0.
    """
    pop = np.asarray(pop_counts, dtype=np.float64)
    deciles = np.zeros(pop.shape[0], dtype=np.int32)
    nz = pop > 0
    if nz.sum() >= 10:
        qs = np.quantile(pop[nz], np.linspace(0.1, 0.9, 9))
        deciles[nz] = np.searchsorted(qs, pop[nz], side="right")
    return deciles


def per_decile(values, deciles):
    """Mean of `values` per decile -> dict {decile: mean}."""
    out = {}
    for d in range(10):
        mask = deciles == d
        if mask.any():
            out[d] = float(np.mean(values[mask]))
    return out


def exposure_shift(exact, ann, Q, pop_counts, k=10, tail_frac=0.2):
    """long_tail_exposure(PQ top-k) - long_tail_exposure(Flat top-k)."""
    N = len(pop_counts)
    tail = M.tail_mask_from_popularity(pop_counts, tail_frac)
    exp_e = np.zeros(N, dtype=np.int64)
    exp_a = np.zeros(N, dtype=np.int64)
    I_e = exact.search(Q, k)
    I_a = ann.search(Q, k)
    for r in range(Q.shape[0]):
        for x in I_e[r]:
            if 0 <= int(x) < N:
                exp_e[int(x)] += 1
        for x in I_a[r]:
            if 0 <= int(x) < N:
                exp_a[int(x)] += 1
    lte_e = M.long_tail_exposure(exp_e, tail)
    lte_a = M.long_tail_exposure(exp_a, tail)
    return {"long_tail_exposure_flat": lte_e,
            "long_tail_exposure_pq": lte_a,
            "long_tail_exposure_shift": float(lte_a - lte_e)}


def interpretation_label(delta_ndcg, eps=NDCG_EPS):
    """Conservative, evidence-bound label. Never asserts regularization."""
    if delta_ndcg is None or (isinstance(delta_ndcg, float) and np.isnan(delta_ndcg)):
        return "insufficient_evidence"
    if delta_ndcg < -eps:
        return "compression_hurts_quality"
    if delta_ndcg > eps:
        return "compression_may_smooth_noise"
    return "compression_preserves_quality"


def safe_corr(x, y, min_points=3):
    """Spearman correlation with a small-sample guard; returns (rho, n)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < min_points:
        return float("nan"), int(x.size)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x, y)
    return float(rho), int(x.size)
