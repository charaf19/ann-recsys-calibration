import argparse
import time
import os
import json
from pathlib import Path

import numpy as np
import faiss
import psutil  # kept for env parity; rss_mb comes from utils.common
from utils.common import percentiles, rss_mb


# ---------- helpers ----------

def _resolve_index_path(p: str) -> Path:
    """
    Accepts either a file or a directory.
    If directory, prefers known names; else falls back to any *.faiss (preferring index.faiss, else largest).
    """
    pth = Path(p)
    if pth.is_file():
        return pth

    if not pth.is_dir():
        raise ValueError(f"Index path not found: {p}")

    # Known names first
    known = ["index.faiss", "faiss_hnsw.index", "hnsw.bin", "hnsw_index.bin",
             "faiss_ivfpq.index", "faiss_ivfflat.index",
             "faiss_flatpq.index", "faiss_flat.index"]
    for name in known:
        q = pth / name
        if q.is_file():
            return q

    # Fallback: any .faiss, prefer 'index.faiss', else largest
    faiss_files = [f for f in pth.iterdir() if f.is_file() and f.suffix.lower() == ".faiss"]
    if faiss_files:
        faiss_files.sort(key=lambda f: (0 if f.name == "index.faiss" else 1, -f.stat().st_size))
        return faiss_files[0]

    raise ValueError("No known index files in the directory.")


def _detect_method_and_prepare(fpath: Path, dim: int, N: int, args: argparse.Namespace):
    """
    Returns:
      method: str in {"hnsw","ivfflat","ivfpq","flatpq","flat"}
      search_fn: callable (Q: np.ndarray[B,D], topk:int) -> (I, D)
      warmup_fn: callable () -> None
    Applies nprobe to IVF if present, ef to HNSW if present.
    """
    index = faiss.read_index(str(fpath))
    index_dc = faiss.downcast_index(index)

    # Try to detect IVF even through pretransform (e.g., OPQ + IVF)
    ivf = None
    try:
        ivf = faiss.extract_index_ivf(index)
    except Exception:
        ivf = None

    if ivf is not None:
        # Identify IVFFlat vs IVFPQ
        ivf_dc = faiss.downcast_index(ivf)
        if isinstance(ivf_dc, faiss.IndexIVFPQ):
            method = "ivfpq"
        else:
            method = "ivfflat"

        # Apply nprobe
        if getattr(args, "nprobe", None) is not None:
            ivf.nprobe = int(args.nprobe)
    elif isinstance(index_dc, faiss.IndexHNSWFlat):
        method = "hnsw"
        index_dc.hnsw.efSearch = int(getattr(args, "ef", 64))
    else:
        # Non-IVF: flat exact or flat-PQ
        if isinstance(index_dc, faiss.IndexPQ):
            method = "flatpq"
        else:
            method = "flat"

    def warmup():
        for _ in range(200):
            q = np.random.randint(0, N)
            _ = index.search(item_vecs[q].reshape(1, -1), args.topk)

    def search_fn(Q, topk: int):
        D, I = index.search(Q, topk)
        return I, D

    return method, search_fn, warmup


# ---------- main ----------

def main():
    print("[run_device] starting...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--item_vecs", required=True)
    ap.add_argument("--queries", type=int, default=10000)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--ef", type=int, default=64)       # for HNSW
    ap.add_argument("--nprobe", type=int, default=16)   # for IVF
    args = ap.parse_args()

    index_arg = Path(args.index)
    print(f"[run_device] index={index_arg} item_vecs={args.item_vecs} queries={args.queries} topk={args.topk}")

    # Load item vectors
    global item_vecs  # used in closures above for warmup/search convenience
    item_vecs = np.load(args.item_vecs).astype("float32")
    N, D = item_vecs.shape

    # Resolve index file path and prepare search
    fpath = _resolve_index_path(str(index_arg))
    method, search_fn, warmup_fn = _detect_method_and_prepare(fpath, D, N, args)

    # Decide output directory (directory if args.index is a dir; else file's parent)
    out_dir = index_arg if index_arg.is_dir() else index_arg.parent

    # Warmup + measurements
    lat_ms = []
    rss_before = rss_mb()

    # Warmup (cache pages, JITs, etc.)
    warmup_fn()

    # Timed queries
    for _ in range(args.queries):
        q = item_vecs[np.random.randint(0, N)].reshape(1, -1)
        t0 = time.perf_counter()
        _ = search_fn(q, args.topk)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    rss_after = rss_mb()

    stats = {
        "method": method,
        "N": int(N),
        "D": int(D),
        "queries": int(args.queries),
        "topk": int(args.topk),
        "latency_ms": {**percentiles(lat_ms), "mean": float(np.mean(lat_ms))},
        "rss_mb_delta": float(rss_after - rss_before),
        "rss_mb_after": float(rss_after),
    }

    out = out_dir / f"latency_{method}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
