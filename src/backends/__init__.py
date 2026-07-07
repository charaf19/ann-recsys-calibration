"""Pluggable ANN backend adapters for the optional backend comparison.

Each backend module exposes get_backend() -> dict with:
    name                 str
    available            bool
    error_message        "" or "package_not_installed" / reason
    build(vectors, seed, **params) -> handle with .search(Q, topk) -> ids
    default_params       dict

FAISS is the only required backend; hnswlib, ScaNN, and NGT are optional and
must NOT be added to requirements.txt.
"""
from . import faiss_backend, hnswlib_backend  # noqa: F401
from . import optional_scann_backend, optional_ngt_backend  # noqa: F401

ALL_BACKENDS = [
    faiss_backend,
    hnswlib_backend,
    optional_scann_backend,
    optional_ngt_backend,
]
