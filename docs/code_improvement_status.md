# Code-improvement status matrix

Verification of previously requested code-level improvements, checked
against the actual files (not from memory). Rows 11/13/14/15 originally
described an exploratory GPU-experimentation layer that was later archived
under `legacy/experimental_gpu/`; IndexWise-Recsys is evaluated as a
CPU-only framework and GPU-specific acceleration is outside the present
scope. Those rows are kept for historical traceability, with an updated
"Action needed" noting the archival.
Statuses: `done` / `partial` / `missing` / `not_applicable`.

| # | Concern | Expected implementation | Current status | Evidence file | Action needed |
| --- | --- | --- | --- | --- | --- |
| 1 | HNSW reproducibility with `--omp_threads` | CLI flag default 1, `faiss.omp_set_num_threads()` before construction, metadata record, log + warning messages | done | `src/build_index.py` (flag L93, set L108, meta L133; exact required log strings present) | none |
| 2 | Random-seed audit and deterministic random ops | Audit doc + seeded `default_rng` everywhere; `TruncatedSVD(random_state=seed)` | done | `docs/randomness_audit.md`; `src/train_embeddings.py`; seeded sampling in `src/build_index.py`, `src/run_device.py`, `src/calibrate.py`, `src/utils/splits.py` | none |
| 3 | Amazon Books optional dataset integration | `datasets: {main, optional}` config; `prepare_dataset.py` accepts `amazon-books`; docs | done | `configs/main_cpu.yml` (optional: amazon-books), `src/prepare_dataset.py` (choices include amazon-books), `README.md`, `reproduction.md`, `src/run_revision_experiments.py::resolve_datasets` | none |
| 4 | Embedding-backbone sensitivity robustness | No crash without torch; skipped rows with `backend_available/status/error_message` | done | `src/run_embedding_backbone_sensitivity.py` (skip row with `skipped_missing_dependency` / `torch_not_installed`; ok rows carry the same columns) | none |
| 5 | Scale-stress `quality_measured=false` clarity | `--measure_quality` (default false, raises if true) + `quality_measured/quality_metric/quality_notes` columns | done | `src/run_scale_stress.py` (NotImplementedError path + three columns) | none |
| 6 | Deployment decision formula documented | Formula in docstring + config; component scores in output CSV | done | `src/ann_decision_framework.py` (formula block; `quality_retention/latency_score/memory_score/exposure_score` + `w_*` columns), `configs/decision_framework.yml` | none |
| 7 | Lowercase `u2i`/`i2i` schema consistency | `normalize_modality_label()` applied before writes; validator rejects drift | done | `src/utils/common.py` (utility), applied in `eval_modalities.py`, `run_revision_experiments.py`, `run_energy_measurement.py`, `ann_decision_framework.py`; checked in `src/validate_results.py` (MODALITY_VALUES) | none |
| 8 | Publication figure DPI | `dpi=300, bbox_inches="tight"` on all savefig calls | done | `src/utils/figures_ext.py:25`, `src/figures_paper.py:44`, `src/figures.py:13,21,29` (grep: no dpi=200 remains) | none |
| 9 | Energy fallback uses `NA`, not fake zeros | `NA` constants + explanatory note when unavailable | done | `src/energy_measurement.py` (`NA`, `FALLBACK_NOTE`), `src/run_energy_measurement.py` | none |
| 10 | Split requirements files | `requirements-cpu.txt` + `requirements-optional.txt` | done | both files at repo root (`requirements.txt` kept as backward-compatible copy); optional file documents faiss-gpu / torch / pynvml / backends | none |
| 11 | Hardware capture: GPU presence vs GPU use | Probed `cuda_available`/`faiss_gpu_available` distinct from declared `main_experiments_gpu_used` | done | `src/capture_hardware.py` (`_gpu_presence()`, top-level keys, md wording "present on machine" vs "used in main experiments") | none — this passive reporting is preserved; the driver/CUDA/torch probe (`src/capture_gpu_hardware.py`) has been archived to `legacy/experimental_gpu/` |
| 12 | CPU pipeline status tracking | `results/status/status_{dataset}_{method}.json` with per-step ok/failed/skipped and continue-on-failure | done | `src/run_revision_experiments.py` (`StatusTracker`, `StepError`) | none |
| 13 | README updated to IndexWise-Recsys | Project name, framing, Quick Start, CPU-only scope | done | `README.md` (title `# IndexWise-Recsys`, Quick Start block, do-not-execute note) | Quick Start no longer includes a GPU section; README states the CPU-only scope directly |
| 14 | `validate_results.py` existence and schema checks | Artifact registry, required columns, report CSV+MD, exit code | done | `src/validate_results.py` (`ARTIFACTS`, `check_artifact`, `validation_report.{csv,md}`) | GPU artifact checks and `--require_gpu_outputs` removed — no canonical validation requires GPU outputs |
| 15 | GPU hooks in `build_index.py` / `run_device.py` / `utils/ann_io.py` | `--use_gpu` flags, GPU clone with fallback, output separation | **not_applicable** | `--use_gpu`/`gpu_device`, `try_gpu_clone`, and `GPU_UNSUPPORTED_METHODS` have been removed from `src/utils/ann_io.py`, `src/build_index.py`, `src/run_device.py`, and `src/calibrate.py` — the canonical CPU pipeline has no GPU execution branches. The removed runner/config (`src/run_gpu_experiments.py`, `configs/gpu_experiments.yml`, `src/capture_gpu_hardware.py`) are archived under `legacy/experimental_gpu/` | none — GPU acceleration is out of scope by design |

## Verification method

Every `done` above was confirmed by inspecting the named file in this
session (grep for the required flags/strings/columns), not by assuming
earlier work landed. Item 15's GPU hooks were subsequently removed
entirely (not merely left partial) once the project scope was fixed as
CPU-only; items 11, 13, 14 were re-verified against the current CPU-only
scope.
