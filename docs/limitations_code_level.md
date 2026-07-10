# Code-level limitations

Honest constraints of the current implementation (distinct from any
scientific limitations discussed in the paper).

1. **Embedding model.** Item embeddings come from TruncatedSVD over the
   (optionally BM25/TF-IDF-weighted) interaction matrix. No neural two-tower
   or sequence models are included; conclusions are about *index structures
   given fixed embeddings*, not about state-of-the-art recommendation quality.
2. **U2I user vectors are aggregates.** The U2I query is the unweighted mean
   of the user's training item vectors (there is no jointly trained user
   tower). This is a standard proxy but underestimates personalized models.
3. **Single held-out positive.** The temporal leave-one-out protocol yields
   one positive per test user, so recall@k and hit-rate@k coincide per query
   and per-query metric distributions are heavily zero-inflated. This is why
   Cliff's delta is reported next to Cohen's d.
4. **HNSW / k-means build determinism is thread-bound.** With the default
   `build_index.py --omp_threads 1`, FAISS HNSW construction and seeded IVF/PQ
   k-means are bit-reproducible. Enabling `--omp_threads > 1` trades that for
   speed: thread-order-dependent HNSW insertion and reordered floating-point
   reductions can shift individual neighbors, so exact numeric agreement
   across runs may vary slightly (aggregate metrics move negligibly, and
   per-build recalibration absorbs most of it). The thread count used is
   recorded in each index's `index_meta.json` and in `summary_main.csv`.
5. **Latency is machine-relative.** Absolute milliseconds depend on CPU,
   cache, and thread settings (see `docs/hardware_protocol.md`). Only
   relative orderings and calibrated-parameter trends transfer.
6. **Exposure is a proxy.** `long_tail_exposure` counts top-k recommendation
   slots over the evaluated query sample (`exposure_proxy`), not real user
   impressions; no position-weighting is applied within the top-k.
7. **Bootstrap p-values are resampling-based.** With `--n_boot 1000` the
   smallest resolvable two-sided p-value is 0.002; increase `--n_boot` if
   smaller granularity is needed.
8. **Single-query timing protocol.** Latency is measured one query at a
   time (the recommender serving pattern), which understates FAISS's batched
   throughput; QPS derived from p50 is a single-stream figure, not a
   saturated-throughput figure.
9. **IVF-PQ training subsamples.** Codebook/centroid training uses a seeded
   subsample (FAISS heuristics), so index contents are deterministic per
   seed but depend on that heuristic sample size.
10. **Memory accounting is coarse.** RSS deltas include allocator noise;
    on-disk index size is exact, resident working set is approximate.
11. **Amazon-Books is optional, not part of the default grid** (very slow to
    prepare on some connections); it is listed under `datasets.optional` in
    `configs/main_cpu.yml` and runs only when passed explicitly via
    `--datasets amazon-books`.
12. **GPU acceleration is out of scope.** IndexWise-Recsys is evaluated as a
    CPU-only framework; GPU latency and transfer (host↔device copy,
    cross-device search) behavior are not evaluated anywhere in the active
    pipeline. `capture_hardware.py` records GPU *presence* on the machine
    passively (`cuda_available`, `faiss_gpu_available`), strictly for
    environment-capture honesty — this is never GPU *usage* and no
    canonical workflow depends on a GPU, `faiss-gpu`, CUDA, or PyNVML. An
    earlier exploratory GPU-experimentation layer (GPU-cloned FAISS
    indexes, agreement/speedup checks vs CPU search) has been archived
    under `legacy/experimental_gpu/` and is unmaintained.
