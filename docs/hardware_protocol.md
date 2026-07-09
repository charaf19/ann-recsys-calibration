# Hardware and measurement protocol (CPU canonical, GPU optional)

## Scope

**The canonical, reproducible benchmark is CPU-only.** `faiss-cpu` is pinned
in `requirements-cpu.txt`; `run_revision_experiments.py --cpu_only`
additionally clears `CUDA_VISIBLE_DEVICES` for every subprocess. The
`--main_experiments_gpu_used false` flag passed to `capture_hardware.py`
records this in `results/hardware/hardware.json`, which also distinguishes
GPU *presence* on the machine (`cuda_available`, `faiss_gpu_available`,
probed) from GPU *usage* in experiments (`main_experiments_gpu_used`,
declared).

## Index-construction determinism

`build_index.py` pins FAISS OpenMP threads with `--omp_threads` (default
**1**) before any construction and records the value in the per-index
`index_meta.json`. At 1 thread, HNSW graph construction and IVF/PQ k-means
(seeded via `--seed`) are bit-reproducible. With `--omp_threads > 1` builds
are faster but exact numeric agreement across runs may vary slightly
(reordered floating-point reductions, thread-order-dependent HNSW insertion);
the script prints an explicit warning in that case. Choose one setting per
results set and keep it fixed.

## GPU experiments (optional, exploratory)

`build_index.py`, `run_device.py`, and `calibrate.py` accept
`--use_gpu true` (default **false**). This requires a `faiss-gpu`
installation, is not supported for HNSW/Flat-PQ search, and may introduce
nondeterminism (GPU reduction order). GPU runs:

- never overwrite CPU results — their outputs go to `results/gpu_experiments/`;
- record `gpu_used: true` in every JSON they write;
- are exploratory extensions, NOT part of the canonical reproducible
  CPU benchmark, and must not be mixed with CPU numbers in one table.

## Environment capture (mandatory first step)

```bash
python src/capture_hardware.py --out_dir results/hardware --main_experiments_gpu_used false --label workstation_cpu
```

Captures OS, CPU model, core counts, RAM, Python and pinned package versions,
BLAS/OpenMP thread environment variables, FAISS's OpenMP thread count, and a
`pip freeze` snapshot. Re-run it on every machine used; the `--label` should
match a profile in `configs/hardware_profiles.yml`.

## Thread control

Latency is sensitive to thread oversubscription. Fix the thread environment
*before* launching Python, matching the chosen profile, e.g. (PowerShell):

```powershell
$env:OMP_NUM_THREADS = "8"; $env:OPENBLAS_NUM_THREADS = "8"; $env:MKL_NUM_THREADS = "8"
```

or (bash):

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
```

`capture_hardware.py` records whatever is set, so a mismatch is auditable.

## Latency measurement rules

- Single-query searches (batch size 1) — the recommender serving pattern.
- Warmup of 200 queries before any timing (`run_device.py`) or 50 queries
  (`calibrate.py` sweeps).
- Report percentiles (p50/p95, plus mean); never rely on the mean alone.
- Query vectors are catalog item vectors sampled with a fixed seed, so every
  method sees the same query stream.
- Memory is reported as process RSS before/after index load + query loop
  (`psutil`), plus the on-disk index size printed at build time.

## What to keep constant within one results set

- One machine, one hardware profile, one thread setting.
- Package versions from `requirements.txt` (verify against
  `results/hardware/env_freeze.txt`).
- Seeds (default 42 everywhere).

Cross-machine comparisons are valid for *relative* orderings and calibration
parameters, not for absolute milliseconds; see
`docs/limitations_code_level.md`.
