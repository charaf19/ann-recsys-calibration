"""NGT backend adapter (OPTIONAL; not in requirements.txt).

Yahoo Japan's NGT (`pip install ngt`, module `ngtpy`) is mainly distributed
for Linux; when the package is missing the comparison records
backend_available=false with backend_error_message=package_not_installed.
"""
import tempfile

import numpy as np

NAME = "ngt"
DEFAULT_PARAMS = {"edge_size_for_creation": 10, "epsilon": 0.1}


class _NgtHandle:
    def __init__(self, index, epsilon):
        self.index = index
        self.epsilon = epsilon

    def search(self, Q, topk):
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        ids = []
        for r in range(Q.shape[0]):
            res = self.index.search(Q[r], size=topk, epsilon=self.epsilon)
            row = [int(oid) - 1 for oid, _ in res]  # ngt object ids are 1-based
            row += [-1] * (topk - len(row))
            ids.append(row)
        return np.asarray(ids, dtype=np.int64)


def get_backend():
    try:
        import ngtpy  # noqa: F401
        available, err = True, ""
    except ImportError:
        available, err = False, "package_not_installed"

    def build(vectors, seed=42, **params):
        import ngtpy
        p = {**DEFAULT_PARAMS, **params}
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        path = tempfile.mkdtemp(prefix="ngt_index_").encode()
        ngtpy.create(path, vectors.shape[1], distance_type="L2",
                     edge_size_for_creation=int(p["edge_size_for_creation"]))
        index = ngtpy.Index(path)
        index.batch_insert(vectors)
        index.save()
        return _NgtHandle(index, float(p["epsilon"]))

    return {"name": NAME, "available": available, "error_message": err,
            "build": build, "default_params": DEFAULT_PARAMS}
