# Archived: optional GPU experimentation layer

This directory holds the exploratory GPU-acceleration layer that was removed
from the active IndexWise-Recsys pipeline. **IndexWise-Recsys is evaluated as
a CPU-only framework; GPU-specific acceleration is outside the present
scope.**

None of this code was ever required by the canonical CPU pipeline: it was
disabled by default (`enabled: false` in `configs/gpu_experiments.yml`, every
`--use_gpu` flag defaulted to `false`), wrote only to its own
`results/gpu_experiments/` tree, and no canonical script imported from it —
only the reverse (these scripts imported `calibrate.measure_latency` and
`utils.ann_io` helpers from the canonical package). It is kept here for
reference in case GPU experimentation is revisited; it is not maintained,
not tested, and not part of any reproduction recipe.

Contents:
- `src/run_gpu_experiments.py` — GPU latency/agreement experiment runner.
- `src/capture_gpu_hardware.py` — GPU driver/CUDA/torch environment capture.
- `configs/gpu_experiments.yml` — experiment plan consumed by the runner above.
- `docs/gpu_experiment_protocol.md` — protocol notes for the runner above.
- `results/gpu_experiments/hardware/` — a previously captured (non-CPU)
  hardware snapshot, kept only as an example of the runner's output shape.

The canonical pipeline's passive GPU-*presence* reporting (whether a GPU
happens to exist on the machine, distinct from whether one was *used*) still
lives in `src/capture_hardware.py` and is unaffected by this archival.

**Not runnable as-is.** As part of this archival, the `--use_gpu` /
`gpu_device` plumbing and `GPU_UNSUPPORTED_METHODS` / `try_gpu_clone` were
removed from `src/utils/ann_io.py`, `src/build_index.py`,
`src/run_device.py`, and `src/calibrate.py` to keep the canonical CPU
pipeline free of GPU-execution branches. `run_gpu_experiments.py` in this
directory imports those functions and will need them restored (or
reimplemented locally) before it can run again.
