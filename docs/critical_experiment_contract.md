# Critical experiment contract

Repository cleanup may change organization, names, validation, and write safety. It must not change the following numerical semantics.

## Scope

- Datasets: `ml-1m`, `ml-20m`, `goodbooks`, `amazon-books`.
- Methods: `flat`, `hnsw`, `ivfflat`, `ivfpq`, `flatpq`.
- Modalities: `u2i`, `i2i`.
- Primary embedding: TruncatedSVD, BM25 weighting, dimension 128, L2 normalization, `bm25_k1=1.2`, `bm25_b=0.75`, seed 42.

## Evaluation semantics

- Use deterministic temporal leave-one-out; the last chronological interaction is the held-out positive.
- U2I queries are the mean of training-history item vectors, with seen training items excluded from recommendations.
- I2I queries are the last training-history item, with that anchor excluded.
- Flat is the exact reference.
- ANN agreement recall measures index fidelity and remains distinct from recommendation relevance.

## Calibration and statistics

- Calibration targets are 0.90, 0.95, and 0.98; 0.95 is the primary operating point.
- Bootstrap iterations are 2000 with seed 42.
- Comparisons are paired against Flat and report paired Cohen’s d and Cliff’s delta.

## Required analyses and artifacts

The canonical pipeline must retain main U2I/I2I evaluation, calibration sensitivity, bootstrap confidence intervals, paired significance, effect sizes, embedding-backbone sensitivity, exposure analysis, PQ diagnostics, synthetic scale stress, the ANN decision framework, claim-support audit, paper tables, paper figures, and result validation.

The sole canonical entry-point sequence is:

1. `src/prepare_dataset.py`
2. `src/dataset_stats.py`
3. `src/train_embeddings.py`
4. `src/build_index.py`
5. `src/calibrate.py`
6. `src/eval_modalities.py`
7. `src/run_revision_experiments.py`
8. `src/run_calibration_sensitivity.py`
9. `src/bootstrap_significance.py`
10. `src/effect_size_tables.py`
11. `src/run_embedding_backbone_sensitivity.py`
12. `src/run_exposure_analysis.py`
13. `src/run_pq_diagnostics.py`
14. `src/run_scale_stress.py`
15. `src/ann_decision_framework.py`
16. `src/claim_support_audit.py`
17. `src/tables_paper.py`
18. `src/figures_paper.py`
19. `src/validate_paper_evidence.py`

Supporting canonical packages are `src/datasets/`, `src/utils/`, and `src/backends/`.

## CPU-only boundary

IndexWise-Recsys is evaluated as a CPU-only framework. GPU acceleration and GPU-specific latency behavior are outside the current scope. GPU latency, host-to-device/device-to-host transfer, GPU memory behavior, and CPU/GPU parity are not evaluated. Passive reporting that a GPU is present on the host does not imply its use.

## Preservation assertion

Cleanup must not alter embedding training, CPU index construction, calibration, U2I evaluation, I2I evaluation, bootstrap resampling, effect-size definitions, exposure analysis, PQ diagnostics, scale-stress semantics, or deployment guidance logic. Existing numerical result files are never edited to satisfy this contract; all post-cleanup results must be regenerated from the canonical pipeline.
