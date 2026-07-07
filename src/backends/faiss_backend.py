"""FAISS backend adapter (required; HNSW graph index for the comparison)."""
import numpy as np

NAME = "faiss_hnsw"
DEFAULT_PARAMS = {"M": 24, "efConstruction": 200, "efSearch": 64}


class _FaissHandle:
    def __init__(self, index):
        self.index = index

    def search(self, Q, topk):
        _, I = self.index.search(np.ascontiguousarray(Q, dtype=np.float32), topk)
        return I


def get_backend():
    try:
        import faiss  # noqa: F401
        available, err = True, ""
    except ImportError:
        available, err = False, "package_not_installed"

    def build(vectors, seed=42, **params):
        import faiss
        p = {**DEFAULT_PARAMS, **params}
        d = vectors.shape[1]
        index = faiss.IndexHNSWFlat(d, int(p["M"]))
        index.hnsw.efConstruction = int(p["efConstruction"])
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        index.hnsw.efSearch = int(p["efSearch"])
        return _FaissHandle(index)

    return {"name": NAME, "available": available, "error_message": err,
            "build": build, "default_params": DEFAULT_PARAMS}
