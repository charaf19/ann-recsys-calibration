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
4. **HNSW build nondeterminism.** hnswlib graph construction with multiple
   threads is not bit-reproducible across runs/machines; search parameters
   are recalibrated per build, which absorbs most of the variation, but exact
   per-query results can differ slightly between builds.
5. **Latency is machine-relative.** Absolute milliseconds depend on CPU,
   cache, and thread settings (see `docs/hardware_protocol.md`). Only
   relative orderings and calibrated-parameter trends transfer.
6. **Exposure is a proxy.** `long_tail_exposure` counts top-k recommendation
   slots over the evaluated query sample (`exposure_proxy`), not real user
   impressions; no position-weighting is applied within the top-k.
7. **Bootstrap p-values are resampling-based.** With `--n_boot 1000` the
   smallest resolvable two-sided p-value is 0.002; increase `--n_boot` if
   smaller granularity is needed.
8. **hnswlib query path is row-wise.** Batch queries are looped per row for
   determinism and per-query timing, which slightly understates hnswlib's
   batched throughput (irrelevant for the single-query serving protocol).
9. **IVF-PQ training subsamples.** Codebook/centroid training uses a seeded
   subsample (FAISS heuristics), so index contents are deterministic per
   seed but depend on that heuristic sample size.
10. **Memory accounting is coarse.** RSS deltas include allocator noise;
    on-disk index size is exact, resident working set is approximate.
11. **Amazon-Books is plumbed but not part of the default grid** (very slow
    to prepare on some connections); pass it explicitly via `--datasets` if
    desired.
