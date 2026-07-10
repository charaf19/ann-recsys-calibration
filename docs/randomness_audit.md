# Randomness audit

Every stochastic operation in the pipeline, its seed source, and whether it
is fully deterministic. Global policy: every script exposes `--seed`
(default 42) and calls `utils.common.set_global_seed` (seeds Python
`random`, legacy `np.random.*`, and returns a `np.random.default_rng(seed)`
Generator); sampling uses explicit `default_rng(seed)` Generators.

| File | Random operation | Seed source | Deterministic? | Notes |
| --- | --- | --- | --- | --- |
| `src/train_embeddings.py` | `TruncatedSVD(random_state=args.seed)` | `--seed` (42) | Yes | randomized SVD solver seeded explicitly |
| `src/train_embeddings.py` | global seeding | `set_global_seed(args.seed)` | Yes | belt-and-braces for library internals |
| `src/build_index.py` | IVF/PQ training-sample draw (`rng.choice`) | `np.random.default_rng(args.seed)` | Yes | previously unseeded `np.random.choice` for ivfflat/flatpq — fixed |
| `src/build_index.py` | FAISS k-means clustering (IVF centroids, PQ codebooks) | `cp.seed = args.seed` + `--omp_threads 1` | Yes at 1 thread | k-means is seeded; >1 OMP thread can reorder float reductions → tiny centroid drift (documented limitation) |
| `src/build_index.py` | HNSW graph construction | insertion order (fixed) + `--omp_threads` | Yes at 1 thread | multithreaded insertion is non-reproducible; default is 1 |
| `src/prepare_dataset.py` | none | — | Yes | pure download + deterministic normalization/sort |
| `src/run_revision_experiments.py` | global seeding; all sampling delegated to child scripts | `--seed` (42), forwarded to every subprocess | Yes | orchestration only |
| `src/run_device.py` | warmup + timed query index draws | `np.random.default_rng(args.seed)` | Yes (stream); latency values machine-dependent | previously unseeded `np.random.randint` — fixed |
| `src/calibrate.py` | calibration query sampling (`sample_queries`) | `default_rng(seed)` | Yes | same query set for every grid point and method |
| `src/run_calibration_sensitivity.py` | delegates to `calibrate_index` | config/CLI seed (42) | Yes | fresh index load per target, same seed |
| `src/eval_modalities.py` | test-user subsampling (`build_eval_cases`) | `default_rng(seed)` | Yes | temporal LOO split itself is sort-based, no RNG |
| `src/bootstrap_significance.py` | bootstrap resampling indices | `default_rng(args.seed)` per call | Yes | same seed → same CIs/p-values |
| `src/effect_size_tables.py` | Cliff's-delta pair subsampling (large n) | `default_rng(args.seed)` | Yes | exact (no RNG) below the pair cap |
| `src/run_embedding_backbone_sensitivity.py` | delegates training/calibration/eval | config/CLI seed | Yes* | *two_tower_mlp: see train_neural_embeddings |
| `src/train_neural_embeddings.py` (BPR) | init + SGD order + negative sampling | `default_rng(seed)` | Yes | pure NumPy, single-threaded |
| `src/train_neural_embeddings.py` (two-tower) | init + batch shuffling | `torch.manual_seed(seed)` + seeded `torch.randperm` generator | Mostly | CPU-only training; PyTorch op determinism is high on CPU but not contractually bit-stable across torch versions (optional module; documented) |
| `src/run_scale_stress.py` | synthetic vector generation (Gaussian mixture) | `default_rng(seed)` | Yes | plus seeded builds via build_index `--seed`/`--omp_threads` |
| `src/synthetic_scaling.py` | synthetic interactions | `default_rng(42)` in generate_synth + `--seed` | Yes | |
| `src/run_optional_ann_backend_comparison.py` | query/vector sampling | `default_rng(args.seed)` | Yes (FAISS/hnswlib); ScaNN/NGT builds not guaranteed | third-party backend internals may thread — comparison is exploratory |
| `src/run_energy_measurement.py` | query stream construction | `default_rng(args.seed)` | Yes (stream); energy/time machine-dependent | |

## Known non-deterministic residuals

1. **Multithreaded FAISS construction** (`--omp_threads > 1`): HNSW graphs
   and k-means reductions vary slightly run-to-run. Default is 1 thread.
2. **Latency / energy values**: deterministic *protocol*, machine-dependent
   *numbers* — only relative comparisons transfer (docs/hardware_protocol.md).
3. **Optional third-party backends / torch**: seeded, but bit-stability
   across library versions is not guaranteed; both are optional modules.

GPU-specific acceleration is outside the present scope: no canonical script
contains a GPU execution path, so GPU reduction-order nondeterminism is not
a residual of this benchmark. An earlier exploratory GPU layer has been
archived under `legacy/experimental_gpu/`.
