"""Exposure-proxy analysis (library).

Reviewer concern addressed: "fairness is narrowly defined as long-tail
uplift". This module widens the *exposure* analysis while explicitly
refusing to widen the *claim*: every output row carries a `fairness_scope`
value making clear what the number is and is not.

Allowed fairness_scope values:
    long_tail_exposure_proxy_only       item-side exposure counts over top-k slots
    popularity_calibration_proxy        user-level popularity calibration error
    provider_proxy_if_metadata_available provider-side exposure (needs metadata)
    not_full_fairness_evaluation        summary rows; no full fairness audit is claimed

Computations operate on the raw arrays saved by eval_modalities.py in
results/main/perquery/*.npz (exposure counts at k and 100, popularity
counts, per-query top-k recommendations, per-query history popularity).
"""
import numpy as np

from utils import metrics as M

FAIRNESS_SCOPES = ("long_tail_exposure_proxy_only",
                   "popularity_calibration_proxy",
                   "provider_proxy_if_metadata_available",
                   "not_full_fairness_evaluation")


def popularity_deciles(pop_counts):
    """Item -> popularity decile (0 = least popular; zero-pop items in 0)."""
    pop = np.asarray(pop_counts, dtype=np.float64)
    deciles = np.zeros(pop.shape[0], dtype=np.int32)
    nz = pop > 0
    if nz.sum() >= 10:
        qs = np.quantile(pop[nz], np.linspace(0.1, 0.9, 9))
        deciles[nz] = np.searchsorted(qs, pop[nz], side="right")
    return deciles


def exposure_share_by_decile(exposure_counts, deciles):
    """Share of total exposure per popularity decile -> {decile: share}."""
    total = float(np.sum(exposure_counts))
    out = {}
    for d in range(10):
        mask = deciles == d
        out[d] = 0.0 if total == 0 else float(np.sum(exposure_counts[mask])) / total
    return out


def head_mid_tail_shares(exposure_counts, pop_counts,
                         head_frac=0.1, tail_frac=0.2):
    """Exposure share of head (top head_frac by popularity), tail (bottom
    tail_frac), and mid (rest)."""
    pop = np.asarray(pop_counts, dtype=np.float64)
    n = pop.size
    order = np.argsort(pop)  # ascending
    tail_idx = order[: max(1, int(n * tail_frac))]
    head_idx = order[n - max(1, int(n * head_frac)):]
    total = float(np.sum(exposure_counts))
    if total == 0:
        return {"head": 0.0, "mid": 0.0, "tail": 0.0}
    head = float(np.sum(exposure_counts[head_idx])) / total
    tail = float(np.sum(exposure_counts[tail_idx])) / total
    return {"head": head, "mid": max(0.0, 1.0 - head - tail), "tail": tail}


def user_popularity_calibration_error(recs_at_k, hist_pop_mean, pop_counts):
    """Mean per-user |log1p(mean rec popularity) - log1p(mean hist popularity)|.

    A proxy for popularity mis-calibration: how far the popularity level of
    what a user is *shown* deviates from the popularity level of what the
    user *consumed*. Not a full calibration curve.
    """
    pop = np.asarray(pop_counts, dtype=np.float64)
    errs = []
    for i in range(recs_at_k.shape[0]):
        rec = recs_at_k[i]
        rec = rec[rec >= 0]
        if rec.size == 0:
            continue
        rec_pop = float(np.mean(pop[rec]))
        errs.append(abs(np.log1p(rec_pop) - np.log1p(float(hist_pop_mean[i]))))
    return float(np.mean(errs)) if errs else float("nan")


def provider_exposure(exposure_counts, item_provider):
    """Provider-side exposure proxy: share and Gini across providers.

    item_provider: array of provider labels aligned with item indices
    (None entries / missing metadata are excluded).
    Returns dict with n_providers, provider_exposure_gini, top_provider_share.
    """
    providers = {}
    for idx, prov in enumerate(item_provider):
        if prov is None or (isinstance(prov, float) and np.isnan(prov)):
            continue
        providers.setdefault(str(prov), 0.0)
        providers[str(prov)] += float(exposure_counts[idx])
    if not providers:
        return None
    counts = np.array(list(providers.values()), dtype=np.float64)
    total = counts.sum()
    return {
        "n_providers": int(len(counts)),
        "provider_exposure_gini": M.gini_exposure(counts),
        "top_provider_share": float(counts.max() / total) if total > 0 else 0.0,
    }


def analyze_run(arrays, tail_frac=0.2, head_frac=0.1, item_provider=None):
    """Full exposure analysis for one (dataset, weighting, modality, method) run.

    arrays: dict-like with exposure_counts_at_k, exposure_counts_at_100,
            pop_counts, recs_at_k, hist_pop_mean.
    Returns list of metric rows: {metric, k, decile, group, value, fairness_scope}.
    """
    exp_k = np.asarray(arrays["exposure_counts_at_k"])
    exp_100 = np.asarray(arrays["exposure_counts_at_100"])
    pop = np.asarray(arrays["pop_counts"])
    tail = M.tail_mask_from_popularity(pop, tail_frac)
    deciles = popularity_deciles(pop)

    rows = []

    def add(metric, value, k=None, decile=None, group=None,
            scope="long_tail_exposure_proxy_only", notes=""):
        rows.append({"metric": metric, "k": k, "decile": decile, "group": group,
                     "value": float(value), "fairness_scope": scope, "notes": notes})

    # 1-4: shares, uplift, gini, coverage
    add("long_tail_exposure", M.long_tail_exposure(exp_k, tail), k=10)
    add("long_tail_exposure", M.long_tail_exposure(exp_100, tail), k=100)
    add("long_tail_uplift", M.long_tail_uplift(exp_k, pop, tail_frac), k=10)
    add("gini_exposure", M.gini_exposure(exp_k), k=10)
    add("gini_exposure", M.gini_exposure(exp_100), k=100)
    add("coverage", M.coverage_at_k(exp_k), k=10)
    add("coverage", M.coverage_at_k(exp_100), k=100)

    # 5: popularity-decile exposure
    for d, share in exposure_share_by_decile(exp_k, deciles).items():
        add("exposure_share_decile", share, k=10, decile=d)

    # 6: head/mid/tail
    for group, share in head_mid_tail_shares(exp_k, pop, head_frac, tail_frac).items():
        add("exposure_share_group", share, k=10, group=group)

    # 7: user-level popularity calibration error
    err = user_popularity_calibration_error(np.asarray(arrays["recs_at_k"]),
                                            np.asarray(arrays["hist_pop_mean"]), pop)
    add("user_popularity_calibration_error", err, k=10,
        scope="popularity_calibration_proxy",
        notes="mean |log1p(rec pop) - log1p(hist pop)| per user")

    # 8-9: provider proxy (only if metadata supplied)
    if item_provider is not None:
        pe = provider_exposure(exp_k, item_provider)
        if pe is not None:
            for name, value in pe.items():
                add(name, value, k=10,
                    scope="provider_proxy_if_metadata_available")
    return rows
