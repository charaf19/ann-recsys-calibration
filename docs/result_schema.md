# Canonical result schema and write policy

Only five top-level result directories are valid: `results/_meta/`, `results/main/`, `results/analyses/`, `results/paper/`, and `results/archive/`. `archive/` is reserved and is never read by canonical code.

## Metadata

- `_meta/hardware/hardware.json`: CPU/platform/package/thread metadata, `gpu_present` (passive disclosure), and `gpu_used_in_main_experiments=false`.
- `_meta/status/status_{dataset}_{method}.json`: step status for the current main run.
- `_meta/validation/`: schema and paper-evidence validation reports.

## Main evidence

- `main/summary_main.csv`: exactly one row per dataset × method × modality.
- `main/{dataset}__{weighting}__{modality}__{method}.json`: aggregate metrics.
- `main/perquery/*.npz`: aligned per-query metrics used for paired inference.
- `main/calibration/*.json`: exact-Flat agreement calibration records.

Agreement recall and held-out recommendation relevance are distinct fields.

## Analyses and paper evidence

Named subdirectories under `analyses/` contain calibration sensitivity, bootstrap confidence intervals and paired tests, effect sizes, embedding sensitivity, exposure analysis, PQ diagnostics, scale stress, optional CPU backend comparisons, CPU energy measurement, and the ANN decision framework.

`paper/tables/` contains CSV/Markdown/LaTeX tables and the claim-support audit. `paper/figures/` contains PNG/PDF figures generated only from current-run evidence.

## Validation and safe writes

Consolidated evidence must be complete, parseable, non-empty, use canonical modality labels, and have unique scientific keys. Writers must write through a temporary sibling and atomically replace a destination only when overwrite was explicitly authorized. A fresh canonical run starts from the empty directory skeleton so evidence from different runs cannot be combined.

`src/validate_paper_evidence.py` checks all 4 × 5 × 2 dataset/method/modality rows, canonical configuration values, and every required analysis.

IndexWise-Recsys is evaluated as a CPU-only framework. GPU acceleration and GPU-specific latency behavior are outside the current scope. GPU output is not part of any result schema or evidence manifest.
