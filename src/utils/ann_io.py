"""Shared ANN index loading, calibration parameters, and backend adapters.

This consolidates the index-resolution/search logic previously duplicated in
run_device.py and eval_modalities.py, and exposes a small adapter registry so
additional ANN backends (e.g. Annoy, ScaNN) can be plugged in without
touching the evaluation scripts.

Built-in backends: FAISS (flat, ivfflat, ivfpq, flatpq, hnsw).
"""
from pathlib import Path

import numpy as np

KNOWN_INDEX_FILES = [
    "index.faiss",
    "faiss_hnsw.index",
    "hnsw.bin",
    "hnsw_index.bin",
    "faiss_ivfpq.index",
    "faiss_ivfflat.index",
    "faiss_flatpq.index",
    "faiss_flat.index",
]

# Which runtime parameter each method calibrates, and the ascending grid
# swept during calibration.
CALIBRATION_PARAM = {
    "hnsw": "ef",
    "ivfflat": "nprobe",
    "ivfpq": "nprobe",
    "flat": None,
    "flatpq": None,
}

DEFAULT_PARAM_GRIDS = {
    "hnsw": [8, 16, 32, 64, 128, 256, 512],
    "ivfflat": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "ivfpq": [1, 2, 4, 8, 16, 32, 64, 128, 256],
}


def resolve_index_path(p: str) -> Path:
    """Accept a file or a directory; return the concrete index file path."""
    pth = Path(p)
    if pth.is_file():
        return pth
    if not pth.is_dir():
        raise ValueError(f"Index path not found: {p}")
    for name in KNOWN_INDEX_FILES:
        q = pth / name
        if q.is_file():
            return q
    faiss_files = [f for f in pth.iterdir() if f.is_file() and f.suffix.lower() == ".faiss"]
    if faiss_files:
        faiss_files.sort(key=lambda f: (0 if f.name == "index.faiss" else 1, -f.stat().st_size))
        return faiss_files[0]
    raise ValueError(f"No known index files in directory: {p}")


class AnnIndex:
    """Uniform search interface over heterogeneous ANN backends.

    Attributes:
        method: one of flat, flatpq, ivfflat, ivfpq, hnsw (or a plugin name).
    Methods:
        search(Q, topk) -> np.ndarray[int] of shape (B, topk)
        set_calibration_param(value) -> apply ef / nprobe (no-op if untunable)
    """

    def __init__(self, method, search_fn, set_param_fn=None):
        self.method = method
        self._search = search_fn
        self._set_param = set_param_fn

    def search(self, Q, topk):
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)
        return self._search(Q, int(topk))

    @property
    def calibration_param_name(self):
        return CALIBRATION_PARAM.get(self.method)

    def set_calibration_param(self, value):
        if self._set_param is not None:
            self._set_param(int(value))


def _load_faiss(fpath: Path, nprobe=None, ef=None):
    import faiss
    index = faiss.read_index(str(fpath))
    index_dc = faiss.downcast_index(index)
    ivf = None
    try:
        ivf = faiss.extract_index_ivf(index)
    except Exception:
        ivf = None

    set_param_fn = None
    if ivf is not None:
        ivf_dc = faiss.downcast_index(ivf)
        method = "ivfpq" if isinstance(ivf_dc, faiss.IndexIVFPQ) else "ivfflat"
        if nprobe is not None:
            ivf.nprobe = int(nprobe)

        def set_param_fn(v, _ivf=ivf):
            _ivf.nprobe = int(v)
    else:
        if isinstance(index_dc, faiss.IndexHNSWFlat):
            method = "hnsw"
            if ef is not None:
                index_dc.hnsw.efSearch = int(ef)

            def set_param_fn(v, _index=index_dc):
                _index.hnsw.efSearch = int(v)
        else:
            method = "flatpq" if isinstance(index_dc, faiss.IndexPQ) else "flat"

    def search_fn(Q, topk):
        _, I = index.search(Q, topk)
        return I

    return AnnIndex(method, search_fn, set_param_fn=set_param_fn)


def load_ann_index(path: str, dim: int, N: int, ef=None, nprobe=None) -> AnnIndex:
    """Load any supported index (file or directory) as an AnnIndex."""
    fpath = resolve_index_path(path)
    if fpath.suffix.lower() == ".bin" or fpath.name in {"hnsw_index.bin", "hnsw.bin"}:
        raise ValueError(
            "Legacy hnswlib index files are no longer supported. "
            "Rebuild HNSW with src/build_index.py to create faiss_hnsw.index."
        )
    return _load_faiss(fpath, nprobe=nprobe, ef=ef)


def build_exact_index(item_vecs: np.ndarray) -> AnnIndex:
    """Exact L2 baseline (FAISS IndexFlatL2) as an AnnIndex."""
    import faiss
    item_vecs = np.ascontiguousarray(item_vecs, dtype=np.float32)
    index = faiss.IndexFlatL2(item_vecs.shape[1])
    index.add(item_vecs)

    def search_fn(Q, topk):
        _, I = index.search(Q, topk)
        return I

    return AnnIndex("flat", search_fn)


# ----------------------------
# Optional backend adapters
# ----------------------------
# Third-party backends can register a loader:
#   from utils.ann_io import register_backend
#   register_backend("annoy", loader_fn)
# where loader_fn(path, dim, N, **kwargs) -> AnnIndex.
_BACKENDS = {}


def register_backend(name: str, loader_fn):
    _BACKENDS[name.lower()] = loader_fn


def load_backend(name: str, path: str, dim: int, N: int, **kwargs) -> AnnIndex:
    name = name.lower()
    if name in _BACKENDS:
        return _BACKENDS[name](path, dim, N, **kwargs)
    return load_ann_index(path, dim, N, **kwargs)
