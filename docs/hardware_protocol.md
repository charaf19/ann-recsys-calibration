# CPU hardware protocol

Run `python src/capture_hardware.py` before a fresh canonical run. It
writes platform, CPU, memory, Python, package, and thread metadata to
`results/_meta/hardware.json` / `hardware.md`, plus a `pip freeze` snapshot
to `results/_meta/environment.txt`.

The benchmark uses `faiss-cpu` exclusively. Accelerator presence is
disclosed passively (`accelerator_present`, detected as `nvidia-smi` on
PATH without importing any GPU package); `main_experiments_accelerator_used`
is fixed to `false`. The presence of an accelerator is environment metadata
only — no GPU was used by the canonical experiments, and no workflow
requires `faiss-gpu`, CUDA, or PyNVML.

Protocol notes:

- Single-query (batch size 1) latency; warmup before timing; report
  p50/p95 percentiles, never only means.
- Fix `OMP_NUM_THREADS` / BLAS thread variables before launching Python;
  `capture_hardware.py` records them.
- Canonical index builds use one OpenMP thread
  (`reproducibility.omp_threads: 1`) for bit-reproducible HNSW/IVF builds.
- Absolute latencies are machine-relative; only relative orderings and
  calibrated-parameter trends transfer across machines.
