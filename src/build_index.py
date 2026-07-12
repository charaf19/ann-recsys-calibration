"""Build a FAISS index (flat / hnsw / ivfflat / ivfpq / flatpq) over item vectors.

Reproducibility contract:
  - `--omp_threads` (default 1) pins FAISS OpenMP threads BEFORE any index
    construction. Single-threaded construction makes HNSW graphs and IVF
    k-means bit-reproducible for a given seed; more threads are faster but
    may introduce small run-to-run variation (documented in
    docs/limitations_code_level.md).
  - `--seed` (default 42) drives every training-sample draw and the FAISS
    clustering seed.
  - Every build writes `index_meta.json` next to the index recording the
    method, hyperparameters, omp_threads, and seed.

IndexWise-Recsys is evaluated as a CPU-only framework. GPU-specific
acceleration is outside the present scope.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import faiss

SCRIPT = "build_index"


def human(n):
    return f"{n/1024/1024:.1f} MB"


def est_pq_bytes(N, m, bits):
    return int(N * m * (bits / 8.0))


def auto_nlist(N):
    return max(8, int(math.sqrt(N)))


def _write_meta(out_dir, meta):
    meta_path = Path(out_dir) / "index_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[{SCRIPT}] output path: {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["hnsw", "ivfpq", "ivfflat", "flatpq", "flat"],
                    required=True)
    ap.add_argument("--item_vecs", required=True)
    ap.add_argument("--item_ids", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--budget_mb", type=int, default=100)
    # HNSW
    ap.add_argument("--M", type=int, default=24)
    ap.add_argument("--efc", type=int, default=200)
    # IVF-PQ/Flat
    ap.add_argument("--nlist", default="auto")
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--opq", action="store_true")
    ap.add_argument("--force_opq", action="store_true",
                    help="force OPQ even on tiny catalogs")
    # reproducibility / hardware
    ap.add_argument("--omp_threads", type=int, default=1,
                    help="FAISS OpenMP threads (1 = bit-reproducible builds)")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for training-sample draws and FAISS clustering")
    ap.add_argument("--config_hash", default="unknown",
                    help="resolved experiment configuration hash")
    ap.add_argument("--embedding_fingerprint", default=None)
    ap.add_argument("--index_fingerprint", default=None)
    args = ap.parse_args()

    print(f"[{SCRIPT}] starting...")
    print(f"[{SCRIPT}] input path: {args.item_vecs}")
    print(f"[{SCRIPT}] input path: {args.item_ids}")
    print(f"[{SCRIPT}] output path: {args.out_dir}")

    faiss.omp_set_num_threads(int(args.omp_threads))
    if int(args.omp_threads) == 1:
        print(f"[{SCRIPT}] FAISS OMP threads set to 1 for reproducible construction.")
    else:
        print(f"[{SCRIPT}] WARN: OMP threads > 1 may introduce small "
              f"run-to-run variation for HNSW construction.")

    rng = np.random.default_rng(args.seed)

    item_vecs = np.load(args.item_vecs).astype("float32")
    item_ids = np.load(args.item_ids, allow_pickle=True)
    N, D = item_vecs.shape
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{SCRIPT}] method={args.method} N={N} D={D} "
          f"budget={args.budget_mb}MB omp_threads={args.omp_threads} "
          f"seed={args.seed} out={out}")

    meta = {
        "method": args.method,
        "N": int(N), "D": int(D),
        "budget_mb": int(args.budget_mb),
        "omp_threads": int(args.omp_threads),
        "seed": int(args.seed),
        "config_hash": str(args.config_hash),
        "embedding_fingerprint": args.embedding_fingerprint,
        "index_fingerprint": args.index_fingerprint,
        "item_vecs": str(args.item_vecs),
    }

    if args.method == "hnsw":
        index = faiss.IndexHNSWFlat(D, args.M)
        index.hnsw.efConstruction = int(args.efc)
        index.add(item_vecs)
        index_path = out / "faiss_hnsw.index"
        faiss.write_index(index, str(index_path))
        np.save(out / "item_ids.npy", item_ids)
        size = index_path.stat().st_size + (out / "item_ids.npy").stat().st_size
        meta.update({"M": int(args.M), "efConstruction": int(args.efc),
                     "index_file": index_path.name})
        print(f"[{SCRIPT}] HNSW saved. ~size={human(size)} budget={args.budget_mb}MB")

    elif args.method == "ivfpq":
        nlist = auto_nlist(N) if args.nlist == "auto" else int(args.nlist)

        use_opq = bool(args.opq)
        if use_opq and (N < 10_000) and (not args.force_opq):
            print(f"[{SCRIPT}] INFO: tiny catalog; disabling OPQ to avoid "
                  f"slow/fragile training. Use --force_opq to override.")
            use_opq = False

        m = int(args.m)
        if D % m != 0:
            m_new = math.gcd(D, m)
            print(f"[{SCRIPT}] WARN: D={D} not divisible by m={m}; using m={m_new} instead.")
            m = m_new

        bits = int(args.bits)
        k = 1 << bits
        # training sample: >=64 vectors per IVF centroid, >=39*k for PQ codebooks
        sample_size = min(N, max(2000, 64 * nlist, 39 * k))
        train_idx = rng.choice(N, size=sample_size, replace=False)
        train_sample = item_vecs[train_idx]

        print(f"[{SCRIPT}] training IVF-PQ with nlist={nlist}, m={m}, "
              f"bits={bits}, sample={sample_size}, OPQ={use_opq}")

        quantizer = faiss.IndexFlatL2(D)
        base = faiss.IndexIVFPQ(quantizer, D, nlist, m, bits)
        base.cp.niter = 20
        base.cp.seed = int(args.seed)  # deterministic k-means given 1 thread
        base.pq.cp.niter = 20
        base.pq.cp.seed = int(args.seed)

        if use_opq:
            opq = faiss.OPQMatrix(D, m)
            opq.niter = 12
            index = faiss.IndexPreTransform(opq, base)
        else:
            index = base
        index.train(train_sample)

        print(f"[{SCRIPT}] adding vectors...")
        index.add(item_vecs)

        out_path = os.path.join(args.out_dir, "index.faiss")
        faiss.write_index(index, out_path)
        np.save(out / "item_ids.npy", item_ids)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        est_codes_mb = (N * (m * bits // 8)) / (1024 * 1024)
        meta.update({"nlist": int(nlist), "m": int(m), "bits": int(bits),
                     "opq": bool(use_opq), "train_sample_size": int(sample_size),
                     "index_file": "index.faiss"})
        print(f"[{SCRIPT}] IVF-PQ saved. file={size_mb:.1f} MB "
              f"est_codes={est_codes_mb:.1f} MB")

    elif args.method == "ivfflat":
        nlist = auto_nlist(N) if args.nlist == "auto" else int(args.nlist)
        quantizer = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist)
        if hasattr(index, "cp"):
            try:
                index.cp.niter = 10
                index.cp.seed = int(args.seed)
            except Exception:
                pass
        sample_size = min(max(2000, 64 * nlist), N)
        train_idx = rng.choice(N, size=sample_size, replace=False)
        train_sample = item_vecs[train_idx]
        print(f"[{SCRIPT}] training IVF-Flat with nlist={nlist} (sample={sample_size})")
        index.train(train_sample)
        print(f"[{SCRIPT}] adding vectors...")
        index.add(item_vecs)
        faiss.write_index(index, str(out / "faiss_ivfflat.index"))
        np.save(out / "item_ids.npy", item_ids)
        size = (out / "faiss_ivfflat.index").stat().st_size + (out / "item_ids.npy").stat().st_size
        meta.update({"nlist": int(nlist), "train_sample_size": int(sample_size),
                     "index_file": "faiss_ivfflat.index"})
        print(f"[{SCRIPT}] IVF-Flat saved. file={human(size)}")

    elif args.method == "flatpq":
        pq = faiss.IndexPQ(D, args.m, args.bits)
        try:
            pq.pq.cp.seed = int(args.seed)
        except Exception:
            pass
        sample_size = min(max(2000, 39 * (1 << int(args.bits))), N)
        train_idx = rng.choice(N, size=sample_size, replace=False)
        train_sample = item_vecs[train_idx]
        print(f"[{SCRIPT}] training Flat-PQ m={args.m}, bits={args.bits} "
              f"(sample={sample_size})")
        pq.train(train_sample)  # IndexPQ training is CPU-only in FAISS
        print(f"[{SCRIPT}] adding vectors...")
        pq.add(item_vecs)
        faiss.write_index(pq, str(out / "faiss_flatpq.index"))
        np.save(out / "item_ids.npy", item_ids)
        size = (out / "faiss_flatpq.index").stat().st_size + (out / "item_ids.npy").stat().st_size
        meta.update({"m": int(args.m), "bits": int(args.bits),
                     "train_sample_size": int(sample_size),
                     "index_file": "faiss_flatpq.index"})
        print(f"[{SCRIPT}] Flat-PQ saved. file={human(size)}")

    elif args.method == "flat":
        index = faiss.IndexFlatL2(D)
        index.add(item_vecs)
        faiss.write_index(index, str(out / "faiss_flat.index"))
        np.save(out / "item_ids.npy", item_ids)
        size = (out / "faiss_flat.index").stat().st_size + (out / "item_ids.npy").stat().st_size
        meta.update({"index_file": "faiss_flat.index"})
        print(f"[{SCRIPT}] Flat exact saved. file={human(size)}")

    _write_meta(out, meta)
    print(f"[{SCRIPT}] completed.")


if __name__ == "__main__":
    main()
