"""Interaction-matrix weighting schemes: none, TF-IDF, BM25.

The user-item interaction matrix is treated as a document collection where
each *user* is a document and each *item* is a term. Weights are applied to
the CSR matrix before factorization (TruncatedSVD) so that popular items do
not dominate the latent space.

All functions are deterministic (no randomness involved).
"""
import numpy as np
import scipy.sparse as sp

WEIGHTING_CHOICES = ("none", "tfidf", "bm25")


def _document_frequency(R: sp.csr_matrix) -> np.ndarray:
    """Number of users that interacted with each item (column df)."""
    return np.asarray((R > 0).sum(axis=0)).ravel().astype(np.float64)


def apply_tfidf(R: sp.csr_matrix) -> sp.csr_matrix:
    """TF-IDF weighting: w(u,i) = tf(u,i) * (ln((1+n_users)/(1+df(i))) + 1).

    Uses the smoothed idf variant (as in scikit-learn) so items seen by every
    user still get a positive weight.
    """
    R = R.tocsr().astype(np.float64)
    n_users = R.shape[0]
    df = _document_frequency(R)
    idf = np.log((1.0 + n_users) / (1.0 + df)) + 1.0
    W = R.multiply(sp.csr_matrix(idf.reshape(1, -1)))
    return W.tocsr()


def apply_bm25(R: sp.csr_matrix, k1: float = 1.2, b: float = 0.75) -> sp.csr_matrix:
    """Okapi BM25 weighting on the user-item matrix.

    idf(i)  = ln(1 + (n_users - df(i) + 0.5) / (df(i) + 0.5))
    tf-part = tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(u) / avg_len))

    where len(u) is the number of interactions of user u and avg_len the mean
    across users.
    """
    R = R.tocsr().astype(np.float64)
    n_users = R.shape[0]
    df = _document_frequency(R)
    idf = np.log(1.0 + (n_users - df + 0.5) / (df + 0.5))

    row_len = np.asarray(R.sum(axis=1)).ravel()
    avg_len = row_len.mean() if row_len.size and row_len.mean() > 0 else 1.0

    W = R.copy().tocoo()
    tf = W.data
    norm = 1.0 - b + b * (row_len[W.row] / avg_len)
    W.data = (tf * (k1 + 1.0)) / (tf + k1 * norm) * idf[W.col]
    return W.tocsr()


def apply_weighting(R: sp.csr_matrix, scheme: str = "none",
                    bm25_k1: float = 1.2, bm25_b: float = 0.75) -> sp.csr_matrix:
    """Dispatch weighting scheme. 'none' returns the matrix unchanged
    (backward-compatible default)."""
    scheme = (scheme or "none").lower()
    if scheme == "none":
        return R.tocsr()
    if scheme == "tfidf":
        return apply_tfidf(R)
    if scheme == "bm25":
        return apply_bm25(R, k1=bm25_k1, b=bm25_b)
    raise ValueError(f"Unknown weighting scheme '{scheme}'. Choices: {WEIGHTING_CHOICES}")
