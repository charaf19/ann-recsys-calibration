"""Build-index command construction from a resolved configuration.

This module is the only place where experiment orchestrators translate the
canonical ``index`` configuration into ``build_index.py`` CLI arguments.
It intentionally contains no scientific fallback values: callers must pass
the fully resolved configuration produced by :mod:`utils.config`.
"""
from pathlib import Path

from utils.config import cfg_get


CANONICAL_METHODS = ("flat", "hnsw", "ivfflat", "ivfpq", "flatpq")


def build_index_command(method, item_vecs, item_ids, out_dir, budget_mb,
                        seed, omp_threads, index_cfg, configuration_hash=None,
                        embedding_fingerprint=None, index_fingerprint=None):
    """Return a ``build_index.py`` command for one canonical method."""
    if method not in CANONICAL_METHODS:
        raise ValueError(
            f"unknown index method {method!r}; expected {CANONICAL_METHODS}")

    cmd = [
        "python", "src/build_index.py",
        "--method", method,
        "--item_vecs", str(Path(item_vecs)),
        "--item_ids", str(Path(item_ids)),
        "--out_dir", str(Path(out_dir)),
        "--budget_mb", str(int(budget_mb)),
        "--seed", str(int(seed)),
        "--omp_threads", str(int(omp_threads)),
    ]
    if configuration_hash is not None:
        cmd += ["--config_hash", str(configuration_hash)]
    if embedding_fingerprint is not None:
        cmd += ["--embedding_fingerprint", str(embedding_fingerprint)]
    if index_fingerprint is not None:
        cmd += ["--index_fingerprint", str(index_fingerprint)]
    if method == "hnsw":
        cmd += [
            "--M", str(cfg_get(index_cfg, "hnsw.M", type=int,
                               required=True)),
            "--efc", str(cfg_get(index_cfg, "hnsw.ef_construction", type=int,
                                 required=True)),
        ]
    elif method == "ivfflat":
        cmd += ["--nlist", str(cfg_get(index_cfg, "ivf.nlist",
                                        required=True))]
    elif method == "ivfpq":
        cmd += [
            "--nlist", str(cfg_get(index_cfg, "ivf.nlist", required=True)),
            "--m", str(cfg_get(index_cfg, "pq.m", type=int, required=True)),
            "--bits", str(cfg_get(index_cfg, "pq.bits", type=int,
                                  required=True)),
        ]
        if cfg_get(index_cfg, "ivfpq.use_opq", type=bool, required=True):
            # The canonical contract requires OPQ on every IVF-PQ dataset,
            # including catalogs below build_index.py's conservative
            # standalone threshold.
            cmd += ["--opq", "--force_opq"]
    elif method == "flatpq":
        cmd += [
            "--m", str(cfg_get(index_cfg, "pq.m", type=int, required=True)),
            "--bits", str(cfg_get(index_cfg, "pq.bits", type=int,
                                  required=True)),
        ]
    return cmd
