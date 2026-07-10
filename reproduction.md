# Reproduction protocol

IndexWise-Recsys is evaluated as a CPU-only framework. GPU acceleration and GPU-specific latency behavior are outside the current scope. GPU latency, transfer behavior, and GPU memory behavior are not evaluated.

## Environment and inputs

Use Python 3.10+ and install `requirements-cpu.txt`. Optional embedding/back-end modules may be installed selectively from `requirements-optional.txt`. No canonical step requires `faiss-gpu`, CUDA, or PyNVML.

Obtain datasets independently and keep raw files outside generated results. Normalize each supported dataset using `src/prepare_dataset.py`; no reproduction command downloads data.

Capture the environment with:

```bash
python src/capture_hardware.py
```

The capture records GPU presence only as an environmental disclosure and always records `gpu_used_in_main_experiments=false`.

## Canonical order

1. Prepare `ml-1m`, `ml-20m`, `goodbooks`, and `amazon-books` with `src/prepare_dataset.py`.
2. Generate descriptive statistics with `src/dataset_stats.py`.
3. Run `src/run_revision_experiments.py --config configs/main_cpu.yml`.
4. Run calibration sensitivity, bootstrap significance, effect sizes, embedding sensitivity, exposure analysis, PQ diagnostics, and scale stress with their canonical `src/run_*.py` entry points.
5. Generate decision evidence with `src/ann_decision_framework.py` and `src/claim_support_audit.py`.
6. Generate `src/tables_paper.py` and `src/figures_paper.py` outputs.
7. Finish with `src/validate_paper_evidence.py`.

The exact scientific invariants are recorded in `docs/critical_experiment_contract.md`. Default configuration is centralized in `configs/main_cpu.yml`; analysis-specific grids remain in their named configuration files.

## Fresh-run policy

Canonical scripts read only `results/main/` and `results/analyses/` evidence from the current run. `results/archive/` is never an input. Start reproduction from the empty five-directory skeleton and do not combine evidence from different runs.

Outputs are written to `results/_meta/`, `results/main/`, `results/analyses/`, and `results/paper/`. A valid paper evidence set must pass `src/validate_paper_evidence.py` without missing dataset/method/modality combinations or contract mismatches.
