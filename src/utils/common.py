\
import os, time, json, math, random, psutil, numpy as np
from contextlib import contextmanager

DEFAULT_SEED = 42

def set_global_seed(seed: int = DEFAULT_SEED):
    """Seed Python and NumPy global RNGs and return a numpy Generator.

    Scripts should prefer the returned Generator for sampling; the global
    seeding covers legacy np.random.* call sites for reproducibility.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    return np.random.default_rng(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def now_ms():
    return int(time.time() * 1000)

@contextmanager
def timer():
    t0 = time.perf_counter()
    yield lambda: (time.perf_counter() - t0)

def rss_mb():
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024*1024)

def percentiles(arr, ps=(50,95,99)):
    a = np.array(arr, dtype=np.float64)
    return { f"p{p}": float(np.percentile(a, p)) for p in ps }

def ndcg_at_k(ranked_relevances, k=10):
    def dcg(rels):
        return sum((rel / math.log2(i+2) for i, rel in enumerate(rels[:k])))
    ideal = sorted(ranked_relevances, reverse=True)
    denom = dcg(ideal)
    return dcg(ranked_relevances) / denom if denom > 0 else 0.0

def hit_rate_at_k(ranked_relevances, k=10):
    return 1.0 if any(ranked_relevances[:k]) else 0.0

def precision_at_k(ranked_relevances, k=10):
    k = min(k, len(ranked_relevances))
    if k == 0: return 0.0
    return sum(ranked_relevances[:k]) / float(k)

def average_precision_at_k(ranked_relevances, k=10):
    k = min(k, len(ranked_relevances))
    if k == 0: return 0.0
    num_rel = 0; ap_sum = 0.0
    for i in range(k):
        if ranked_relevances[i]:
            num_rel += 1
            ap_sum += num_rel / float(i+1)
    return ap_sum / max(1, num_rel)

def mrr_at_k(ranked_relevances, k=10):
    k = min(k, len(ranked_relevances))
    for i in range(k):
        if ranked_relevances[i]:
            return 1.0 / float(i+1)
    return 0.0

def coverage_at_k(recommendations, n_items, k=10):
    seen = set()
    for rec in recommendations:
        for r in rec[:k]:
            seen.add(int(r))
    return len(seen) / float(n_items)

def gini_coefficient(counts):
    import numpy as np
    x = np.array(counts, dtype=np.float64)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n+1, dtype=np.float64)
    g = (np.sum((2*idx - n - 1) * x)) / (n * np.sum(x))
    return float(max(0.0, min(1.0, g)))

def long_tail_uplift(exposure, item_popularity, tail_frac=0.2):
    N = len(item_popularity)
    idx = np.argsort(item_popularity)  # ascending popularity
    tail_k = max(1, int(N * tail_frac))
    tail_idx = idx[:tail_k]
    exp_sum = exposure.sum()
    if exp_sum == 0: return 0.0
    return float(exposure[tail_idx].sum() / exp_sum)

def metric_registry():
    return {
        "recall_at_k": None,
        "precision_at_k": precision_at_k,
        "map_at_k": average_precision_at_k,
        "mrr_at_k": mrr_at_k,
        "ndcg_at_k": ndcg_at_k,
        "hr_at_k": hit_rate_at_k,
        "coverage_at_k": None,
        "gini_exposure": None,
        "long_tail_uplift": None,
    }
