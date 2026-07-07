import argparse, os, numpy as np, faiss, math, json
from pathlib import Path

def human(n):
    return f"{n/1024/1024:.1f} MB"

def est_pq_bytes(N, m, bits):
    return int(N * m * (bits/8.0))

def auto_nlist(N): return max(8, int(math.sqrt(N)))

def main():
    print("[build_index] starting...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["hnsw","ivfpq","ivfflat","flatpq","flat"], required=True)
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
    ap.add_argument("--force_opq", action="store_true", help="force OPQ even on tiny catalogs")
    args = ap.parse_args()

    print(f"[build_index] loading item_vecs: {args.item_vecs}")
    item_vecs = np.load(args.item_vecs).astype("float32")
    print(f"[build_index] loading item_ids: {args.item_ids}")
    item_ids = np.load(args.item_ids, allow_pickle=True)
    N, D = item_vecs.shape
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[build_index] method={args.method} N={N} D={D} budget={args.budget_mb}MB out={out}")

    if args.method == "hnsw":
        index = faiss.IndexHNSWFlat(D, args.M)
        index.hnsw.efConstruction = int(args.efc)
        index.add(item_vecs)
        index_path = out / "faiss_hnsw.index"
        faiss.write_index(index, str(index_path))
        np.save(out / "item_ids.npy", item_ids)
        size = index_path.stat().st_size + (out / "item_ids.npy").stat().st_size
        print(f"[build_index] HNSW saved. ~size={human(size)} budget={args.budget_mb}MB")

    elif args.method == "ivfpq":
        # nlist auto: ~sqrt(N), minimum 8
        if args.nlist == "auto":
            nlist = max(8, int(round(math.sqrt(N))))
        else:
            nlist = int(args.nlist)

        # guard OPQ on small catalogs
        use_opq = bool(args.opq)
        if use_opq and (N < 10_000) and (not getattr(args, "force_opq", False)):
            print("[build_index] INFO: tiny catalog; disabling OPQ to avoid slow/fragile training. Use --force_opq to override.")
            use_opq = False

        # ensure m divides D; if not, fall back to gcd(D, m)
        m = int(args.m)
        if D % m != 0:
            m_new = math.gcd(D, m)
            print(f"[build_index] WARN: D={D} not divisible by m={m}; using m={m_new} instead.")
            m = m_new

        bits = int(args.bits)
        k = 1 << bits  # codewords per sub-quantizer (256 for 8 bits)

        # choose a stronger training sample for IVF/PQ
        # - at least 64 vectors per IVF centroid
        # - at least ~39*k vectors for PQ codebooks (FAISS heuristic)
        min_for_ivf = 64 * nlist
        min_for_pq  = 39 * k
        sample_size = min(N, max(2000, min_for_ivf, min_for_pq))
        rng = np.random.default_rng(123)
        train_idx = rng.choice(N, size=sample_size, replace=False)
        train_sample = item_vecs[train_idx]

        print(f"[build_index] training IVF-PQ with nlist={nlist}, m={m}, bits={bits}, sample={sample_size}, OPQ={use_opq}")

        # build base IVFPQ index
        quantizer = faiss.IndexFlatL2(D)
        base = faiss.IndexIVFPQ(quantizer, D, nlist, m, bits)
        base.cp.niter = 20
        base.pq.cp.niter = 20

        if use_opq:
            # OPQ transform + pretransform wrapper
            opq = faiss.OPQMatrix(D, m)
            opq.niter = 12  # lighter than default to keep CPU training reasonable
            index = faiss.IndexPreTransform(opq, base)
            # Single call trains both OPQ and IVFPQ internals
            index.train(train_sample)
        else:
            # No OPQ: train IVFPQ directly
            base.train(train_sample)
            index = base

        print("[build_index] adding vectors...")
        index.add(item_vecs)

        out_path = os.path.join(args.out_dir, "index.faiss")
        faiss.write_index(index, out_path)
        np.save(out / "item_ids.npy", item_ids)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        # rough code size estimate for PQ
        code_bytes = (m * bits // 8)
        est_codes_mb = (N * code_bytes) / (1024 * 1024)
        print(f"[build_index] IVF-PQ saved. file={size_mb:.1f} MB est_codes={est_codes_mb:.1f} MB")

    elif args.method == "ivfflat":
        nlist = auto_nlist(N) if args.nlist == "auto" else int(args.nlist)
        quantizer = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFFlat(quantizer, D, nlist)
        if hasattr(index, "cp"):
            try: index.cp.niter = 10
            except Exception: pass
        train_sample = item_vecs[np.random.choice(N, min(2000, N), replace=False)]
        print(f"[build_index] training IVF-Flat with nlist={nlist} (sample={len(train_sample)})")
        index.train(train_sample); print("[build_index] adding vectors..."); index.add(item_vecs)
        faiss.write_index(index, str(out / "faiss_ivfflat.index"))
        np.save(out / "item_ids.npy", item_ids)
        size = (out / "faiss_ivfflat.index").stat().st_size + (out / "item_ids.npy").stat().st_size
        print(f"[build_index] IVF-Flat saved. file={human(size)}")

    elif args.method == "flatpq":
        pq = faiss.IndexPQ(D, args.m, args.bits)
        train_sample = item_vecs[np.random.choice(N, min(2000, N), replace=False)]
        print(f"[build_index] training Flat-PQ m={args.m}, bits={args.bits} (sample={len(train_sample)})")
        pq.train(train_sample); print("[build_index] adding vectors..."); pq.add(item_vecs)
        faiss.write_index(pq, str(out / "faiss_flatpq.index"))
        np.save(out / "item_ids.npy", item_ids)
        size = (out / "faiss_flatpq.index").stat().st_size + (out / "item_ids.npy").stat().st_size
        print(f"[build_index] Flat-PQ saved. file={human(size)}")

    elif args.method == "flat":
        index = faiss.IndexFlatL2(D); index.add(item_vecs)
        faiss.write_index(index, str(out / "faiss_flat.index"))
        np.save(out / "item_ids.npy", item_ids)
        size = (out / "faiss_flat.index").stat().st_size + (out / "item_ids.npy").stat().st_size
        print(f"[build_index] Flat exact saved. file={human(size)}")

if __name__ == "__main__":
    main()
