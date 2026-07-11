"""Ranking metrics, long-tail exposure metrics, bootstrap CIs, effect sizes.

Terminology (used consistently across results and docs):
  - long_tail_exposure: share of recommendation-slot exposure that goes to
    long-tail items (bottom `tail_frac` of items by training popularity).
  - long_tail_uplift: long_tail_exposure minus the tail's share of training
    interactions (positive => the index surfaces the tail more than the
    historical data does).
  - exposure_proxy: normalized per-item exposure distribution over the
    top-k recommendation slots (a proxy for real impression exposure).

All stochastic routines take an explicit seed and use numpy Generator for
determinism.
"""
import numpy as np

# ----------------------------
# Per-query ranking metrics
# ----------------------------

def recall_at_k(ranked, positives, k):
    if k <= 0 or len(positives) == 0:
        return 0.0
    return float(len(set(ranked[:k]).intersection(positives))) / float(min(k, len(positives)))


def precision_at_k(ranked, positives, k):
    if k <= 0:
        return 0.0
    return float(len(set(ranked[:k]).intersection(positives))) / float(k)


def hit_rate_at_k(ranked, positives, k):
    return 1.0 if len(set(ranked[:k]).intersection(positives)) > 0 else 0.0


def ndcg_at_k(ranked, positives, k):
    if k <= 0:
        return 0.0
    pos = set(positives)
    dcg = 0.0
    for i, item in enumerate(ranked[:k]):
        if item in pos:
            dcg += 1.0 / np.log2(i + 2)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(pos))))
    return 0.0 if ideal == 0.0 else float(dcg / ideal)


def average_precision_at_k(ranked, positives, k):
    if k <= 0:
        return 0.0
    pos = set(positives)
    hits, sum_prec = 0, 0.0
    for i in range(min(k, len(ranked))):
        if ranked[i] in pos:
            hits += 1
            sum_prec += hits / float(i + 1)
    denom = min(k, len(pos))
    return 0.0 if denom == 0 else float(sum_prec / denom)


def mrr_at_k(ranked, positives, k):
    pos = set(positives)
    for i in range(min(k, len(ranked))):
        if ranked[i] in pos:
            return 1.0 / float(i + 1)
    return 0.0


PER_QUERY_METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "hr": hit_rate_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
    "mrr": mrr_at_k,
}

# ----------------------------
# Catalog-level exposure metrics
# ----------------------------

def coverage_at_k(exposure_counts):
    """Fraction of catalog items recommended at least once."""
    n = len(exposure_counts)
    return 0.0 if n == 0 else float(np.count_nonzero(exposure_counts)) / float(n)


def gini_exposure(exposure_counts):
    """Gini coefficient of the per-item exposure distribution (0 = uniform)."""
    x = np.sort(np.asarray(exposure_counts, dtype=np.float64))
    n = x.size
    if n == 0:
        return 0.0
    cumx = np.cumsum(x)
    if cumx[-1] == 0:
        return 0.0
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)


def exposure_proxy(exposure_counts):
    """Normalized exposure distribution (sums to 1); proxy for impression share."""
    x = np.asarray(exposure_counts, dtype=np.float64)
    total = x.sum()
    return x / total if total > 0 else x


def tail_mask_from_popularity(pop_counts, tail_frac=0.2):
    """Boolean mask of long-tail items: at or below the tail_frac popularity
    quantile computed over items with nonzero training popularity."""
    pop_counts = np.asarray(pop_counts)
    nz = pop_counts > 0
    if np.any(nz):
        thr = np.quantile(pop_counts[nz], tail_frac)
        return pop_counts <= thr
    return pop_counts == 0


def long_tail_exposure(exposure_counts, tail_mask):
    """Share of total exposure allocated to long-tail items."""
    total = float(np.sum(exposure_counts))
    if total == 0:
        return 0.0
    return float(np.sum(np.asarray(exposure_counts)[tail_mask])) / total


def long_tail_uplift(exposure_counts, pop_counts, tail_frac=0.2):
    """long_tail_exposure minus the tail's share of training interactions."""
    mask = tail_mask_from_popularity(pop_counts, tail_frac)
    lte = long_tail_exposure(exposure_counts, mask)
    total_pop = float(np.sum(pop_counts))
    tail_pop_share = 0.0 if total_pop == 0 else float(np.sum(np.asarray(pop_counts)[mask])) / total_pop
    return float(lte - tail_pop_share)

# ----------------------------
# Bootstrap confidence intervals
# ----------------------------

def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for the mean of per-query values.

    Returns dict with mean, ci_low, ci_high, n, n_boot.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = x[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(x.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(n), "n_boot": int(n_boot)}


def paired_bootstrap_test(a, b, n_boot=2000, seed=42):
    """Paired bootstrap for mean(a - b) on per-query metrics of two systems
    evaluated on the *same* queries (arrays aligned by query).

    Returns mean_diff, CI of the diff, and a two-sided bootstrap p-value for
    H0: mean(a - b) == 0.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must align: {a.shape} vs {b.shape}")
    d = a - b
    n = d.size
    if n == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0,
                "n": 0, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided p-value: how often the bootstrapped mean crosses zero
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(min(1.0, p)), "n": int(n), "n_boot": int(n_boot)}

# ----------------------------
# Effect sizes
# ----------------------------

def cohens_d_paired(a, b):
    """Cohen's d for paired samples: mean(a-b) / std(a-b, ddof=1)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.size < 2:
        return 0.0
    sd = d.std(ddof=1)
    return 0.0 if sd == 0 else float(d.mean() / sd)


def cliffs_delta(a, b, max_pairs=2_000_000, seed=42):
    """Cliff's delta: P(a > b) - P(a < b) over all cross pairs.

    For large samples, a seeded random subsample of pairs is used to keep
    memory bounded; results remain deterministic for a given seed.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return 0.0
    if n * m <= max_pairs:
        diff = a.reshape(-1, 1) - b.reshape(1, -1)
        return float((diff > 0).mean() - (diff < 0).mean())
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, n, size=max_pairs)
    ib = rng.integers(0, m, size=max_pairs)
    diff = a[ia] - b[ib]
    return float((diff > 0).mean() - (diff < 0).mean())


def effect_size_interpretation(d, kind="cohens_d"):
    """Conventional magnitude bins for effect sizes."""
    x = abs(d)
    if kind == "cliffs_delta":
        if x < 0.147:
            return "negligible"
        if x < 0.33:
            return "small"
        if x < 0.474:
            return "medium"
        return "large"
    # Cohen's d
    if x < 0.2:
        return "negligible"
    if x < 0.5:
        return "small"
    if x < 0.8:
        return "medium"
    return "large"
