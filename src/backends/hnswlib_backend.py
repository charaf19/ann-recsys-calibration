"""hnswlib backend adapter (OPTIONAL dependency; not in requirements.txt)."""
import numpy as np

NAME = "hnswlib"
DEFAULT_PARAMS = {"M": 24, "ef_construction": 200, "ef": 64}


class _HnswlibHandle:
    def __init__(self, index):
        self.index = index

    def search(self, Q, topk):
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        ids = []
        for r in range(Q.shape[0]):
            I, _ = self.index.knn_query(Q[r], k=topk)
            ids.append(I[0])
        return np.vstack(ids)


def get_backend():
    try:
        import hnswlib  # noqa: F401
        available, err = True, ""
    except ImportError:
        available, err = False, "package_not_installed"

    def build(vectors, seed=42, **params):
        import hnswlib
        p = {**DEFAULT_PARAMS, **params}
        n, d = vectors.shape
        index = hnswlib.Index(space="l2", dim=d)
        index.init_index(max_elements=n, ef_construction=int(p["ef_construction"]),
                         M=int(p["M"]), random_seed=int(seed))
        index.add_items(np.ascontiguousarray(vectors, dtype=np.float32),
                        np.arange(n))
        index.set_ef(int(p["ef"]))
        return _HnswlibHandle(index)

    return {"name": NAME, "available": available, "error_message": err,
            "build": build, "default_params": DEFAULT_PARAMS}
