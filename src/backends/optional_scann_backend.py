"""ScaNN backend adapter (OPTIONAL; not in requirements.txt).

ScaNN ships Linux x86_64 wheels only; on Windows/macOS it will typically be
unavailable and the comparison records backend_available=false with
backend_error_message=package_not_installed. Install manually on Linux with
`pip install scann` to enable this adapter.
"""
import numpy as np

NAME = "scann"
DEFAULT_PARAMS = {"num_leaves": 256, "num_leaves_to_search": 32,
                  "reorder_k": 200}


class _ScannHandle:
    def __init__(self, searcher):
        self.searcher = searcher

    def search(self, Q, topk):
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        ids = []
        for r in range(Q.shape[0]):
            I, _ = self.searcher.search(Q[r], final_num_neighbors=topk)
            ids.append(np.asarray(I))
        return np.vstack(ids)


def get_backend():
    try:
        import scann  # noqa: F401
        available, err = True, ""
    except ImportError:
        available, err = False, "package_not_installed"

    def build(vectors, seed=42, **params):
        import scann
        p = {**DEFAULT_PARAMS, **params}
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        num_leaves = min(int(p["num_leaves"]), max(1, vectors.shape[0] // 39))
        searcher = (scann.scann_ops_pybind.builder(vectors, 10, "squared_l2")
                    .tree(num_leaves=num_leaves,
                          num_leaves_to_search=int(p["num_leaves_to_search"]),
                          training_sample_size=min(250000, vectors.shape[0]))
                    .score_ah(2, anisotropic_quantization_threshold=0.2)
                    .reorder(int(p["reorder_k"]))
                    .build())
        return _ScannHandle(searcher)

    return {"name": NAME, "available": available, "error_message": err,
            "build": build, "default_params": DEFAULT_PARAMS}
