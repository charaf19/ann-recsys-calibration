# Artifact description

**Title:** Effect-size-aware, modality-separated ANN calibration and
deployment evaluation for recommender retrieval.

**What this artifact is.** A CPU-only, fully local benchmark of FAISS/hnswlib
index structures (Flat, HNSW, IVF-Flat, IVF-PQ, Flat-PQ) for recommender
retrieval over TruncatedSVD item embeddings with optional BM25/TF-IDF
interaction weighting. It measures (a) ANN agreement recall vs exact search
under explicit calibration targets, (b) end-to-end U2I and I2I ranking
quality under a temporal leave-one-out protocol, (c) latency/memory, and
(d) long-tail exposure metrics — with bootstrap confidence intervals and
effect sizes rather than significance stars alone.

## Claims supported by the artifact

1. ANN methods can be calibrated to fixed agreement-recall targets (0.90,
   0.95, 0.98) against exact Flat search; the calibrated parameter and its
   latency cost are reported per method/dataset (`run_calibration_sensitivity.py`).
2. At calibrated operating points, end-to-end quality differences vs Flat are
   quantified with paired bootstrap CIs *and* effect sizes (Cohen's d,
   Cliff's delta), separately for U2I and I2I retrieval.
3. Index choice shifts long-tail exposure (`long_tail_exposure`,
   `long_tail_uplift`, `exposure_proxy`) — reported per method, not as a
   generic "fairness" score.
4. Rule-based deployment guidance follows mechanically from the measured
   CSVs (`deployment_guidance.py`).

## Components

| Stage | Script | Output |
| --- | --- | --- |
| Environment capture | `src/capture_hardware.py` | `results/hardware/` |
| Dataset prep | `src/prepare_dataset.py`, `src/download_datasets.py` | `data/*.csv` |
| Dataset stats | `src/dataset_stats.py` | `results/paper_tables/dataset_stats.*` |
| Embeddings (none/tfidf/bm25) | `src/train_embeddings.py` | `data/emb_*` |
| Index build | `src/build_index.py` | `data/index_*` |
| Orchestrated grid | `src/run_revision_experiments.py` | `results/main/` |
| Calibration | `src/calibrate.py`, `src/run_calibration_sensitivity.py` | `results/main/calibration/`, `results/calibration_sensitivity/` |
| U2I / I2I eval | `src/eval_modalities.py` | `results/main/`, `results/main/perquery/` |
| Statistics | `src/bootstrap_significance.py`, `src/effect_size_tables.py` | `results/bootstrap/`, `results/effect_sizes/` |
| Guidance | `src/deployment_guidance.py` | `results/deployment_guidance/` |
| Tables / figures | `src/tables_paper.py`, `src/figures_paper.py` | `results/paper_tables/`, `results/figures_paper/` |
| Scaling (optional) | `src/synthetic_scaling.py` | `results/scaling/` |

Shared library code lives in `src/utils/` (weighting, splits, metrics,
ann_io backends, reporting, path conventions).

## Requirements

CPU-only. Python 3.10+, packages pinned in `requirements.txt` (faiss-cpu,
hnswlib, numpy, scipy, pandas, scikit-learn, psutil, matplotlib, PyYAML).
Disk: ~15 GB for ml-20m artifacts. No GPU, no network access after dataset
preparation.

## Reproduction

See `reproduction.md` for the exact Python command sequence, and
`docs/result_schema.md` for the schema of every output file.
