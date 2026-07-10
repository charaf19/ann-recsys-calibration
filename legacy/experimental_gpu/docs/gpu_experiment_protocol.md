# GPU experiment protocol (optional, exploratory)

## Scope and status

1. **GPU experiments are optional.** Nothing in the repository requires a
   GPU; the GPU layer is disabled by default (`configs/gpu_experiments.yml`
   has `enabled: false`, all `--use_gpu` flags default to false).
2. **CPU results are the canonical reproducible baseline.** Every paper
   claim rests on the CPU pipeline (`run_revision_experiments.py` and its
   downstream statistics). GPU numbers never replace CPU evidence.
3. **GPU outputs are stored separately** under `results/gpu_experiments/`
   (`latency/`, `calibration/`, `comparison/`, `hardware/`, `tables/`,
   `figures/`). No GPU script writes into `results/main/`,
   `results/calibration_sensitivity/`, `results/bootstrap/`,
   `results/effect_sizes/`, or `results/deployment_guidance/`.
4. **Do not mix GPU and CPU results** in one table or figure; GPU tables and
   figures live in `results/gpu_experiments/tables|figures/` only.
5. **GPU experiments may be nondeterministic.** GPU kernels do not guarantee
   floating-point reduction order; identical queries can return slightly
   different neighbor sets across runs, drivers, and devices. The query
   streams themselves are seeded and deterministic.
6. **FAISS GPU support depends on installation and platform.** A faiss-cpu
   build reports `faiss.get_num_gpus() == 0`; the runner then records
   skipped rows (`status=skipped_gpu_unavailable`) or exits, per
   `--allow_cpu_fallback`. FAISS GPU does not support HNSW or IndexPQ
   (Flat-PQ) — those combinations are recorded as
   `skipped_gpu_unsupported_method`.
7. **Never install `faiss-cpu` and `faiss-gpu` in the same environment** —
   they shadow each other unpredictably. Use a separate venv/conda env for
   GPU work.
8. **Windows:** faiss-gpu wheels are generally unavailable via pip; use
   conda, Linux, or WSL2 for GPU experiments if pip installation fails.
9. **What GPU numbers mean:** latency and *agreement* (overlap with CPU
   search) checks only. `agreement_recall_vs_cpu_flat_at_100` = overlap of
   GPU retrieval with exact CPU Flat; `agreement_recall_vs_cpu_method_at_100`
   = overlap with the CPU version of the same index at the same runtime
   parameters. These are index-fidelity measurements — **not**
   recommendation quality, and not a replacement for the CPU evidence.

## Protocol

1. Run the CPU pipeline first — the GPU runner reuses CPU-trained embeddings
   and CPU-built indexes and refuses to build its own (missing artifacts
   produce `skipped_missing_cpu_artifacts` rows).
2. Capture the GPU environment: `python src/capture_gpu_hardware.py`
   → `results/gpu_experiments/hardware/gpu_hardware.{json,md}` (driver/CUDA
   versions, FAISS/torch GPU availability — probed honestly, never faked).
3. Dry-run the plan, then run:

```bash
python src/run_gpu_experiments.py --config configs/gpu_experiments.yml --dry_run
python src/run_gpu_experiments.py --datasets ml-1m --modalities u2i i2i --methods flat ivfflat ivfpq --weighting bm25 --dim 128 --queries 5000 --topk 100 --metric_topk 10 --gpu_device 0 --seed 42
```

4. Runtime parameters (`nprobe`) are applied on the CPU index **before**
   GPU cloning and are frozen afterwards; ideally pass the CPU-calibrated
   values from `results/main/calibration/`.
5. Missing measurements are written as `NA` — never as zero.
6. Validate: `python src/validate_results.py --require_gpu_outputs`
   (without the flag, missing GPU outputs never fail validation).
7. Tables/figures: `python src/tables_paper.py && python src/figures_paper.py`
   additionally emit `results/gpu_experiments/tables/gpu_summary_table.*`
   and `results/gpu_experiments/figures/gpu_*.{pdf,png}` when the summary
   exists (300 dpi PNG + vector PDF).

## Interpretation limits

- Speedups are single-query (batch size 1) figures against the CPU version
  of the same method on the same machine; batched GPU throughput would look
  very different and is out of scope.
- Cross-machine GPU comparisons are invalid (driver, clocks, PCIe/NVLink).
- `gpu_memory_allocated_mb/reserved_mb` come from torch.cuda when available
  and describe the process, not the FAISS index alone; `NA` otherwise.
