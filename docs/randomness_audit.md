# Randomness audit

The canonical seed is 42, defined once in `configs/defaults.yml`
(`reproducibility.seed`). Dataset splitting is deterministic temporal
leave-one-out. TruncatedSVD, index construction/training, calibration query
sampling, evaluation sampling, bootstrap resampling, embedding sensitivity,
PQ diagnostics, exposure analysis, and scale stress all receive that seed
through the resolved configuration or an explicit CLI override.

CPU thread settings are captured in `results/_meta/hardware.json`;
canonical index builds use one OpenMP thread
(`reproducibility.omp_threads: 1`) for bit-reproducible HNSW/IVF builds.
GPU nondeterminism is inapplicable because no GPU execution path exists.
