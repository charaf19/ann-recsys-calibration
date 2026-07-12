# Critical experiment contract

This document is the regression contract for the repository: cleanup and
refactoring may change organization, names, validation, and write safety —
they must NOT change the numerical semantics recorded here. Every component
below lists its canonical implementation, I/O, configuration source, and
whether its numerical semantics are protected.

Configuration precedence everywhere:
`CLI argument > experiment YAML > configs/defaults.yml > code fallback`
(the code fallback always equals the paper value).

## Protected scope

- Datasets: `ml-1m`, `ml-20m`, `goodbooks`, `amazon-books` (all four are
  required in the canonical paper run).
- Methods: `flat`, `hnsw`, `ivfflat`, `ivfpq`, `flatpq`.
- Modalities: `u2i`, `i2i`.
- Main embedding: TruncatedSVD, `weighting=bm25`, `dim=128`,
  `normalize=l2`, `bm25_k1=1.2`, `bm25_b=0.75`, `seed=42`.
- Calibration: targets `0.90 / 0.95 / 0.98`, primary `0.95`,
  `calibration.queries=1000`.
- Statistics: `bootstrap_iterations=2000`, `seed=42`, paired vs Flat,
  paired Cohen's d, Cliff's delta.
- Scale stress: catalog sizes `10k/50k/100k/500k/1M`, dims `64/128/256`,
  all five methods, `quality_measured=false` on every row.
- Agreement recall (index fidelity vs exact Flat) and recommendation
  relevance (vs the held-out interaction) are distinct notions and are
  never conflated.

## Component contract

| Component | Canonical file | Canonical function/class | Input | Output | Config source | Required defaults | Paper section | Numerics protected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dataset preparation | `src/prepare_dataset.py` (+ `src/datasets/`) | `prepare_movielens` / `prepare_goodbooks` / `prepare_amazon_books` | raw public dataset files | normalized CSV `user_id,item_id,timestamp` in `data/` | CLI (`--dataset`, `--out`) | canonical dataset names | Datasets | yes |
| dataset statistics | `src/dataset_stats.py` | `stats_for()` | normalized CSVs | `results/paper/tables/dataset_stats.*` | CLI | `--min_user_interactions 5` for the paper table | Datasets | yes |
| temporal split | `src/utils/splits.py` | `temporal_leave_one_out()`, `build_eval_cases()` | interactions DataFrame | (train_df, test_df); eval cases | none (deterministic) | last chronological interaction held out; users with ≥2 interactions; stable-mergesort tie-break | Protocol | yes |
| BM25 weighting | `src/utils/weighting.py` (via `train_embeddings.py`) | BM25/TF-IDF weighting of the interaction matrix | interaction matrix | weighted sparse matrix | `embedding.bm25_k1/bm25_b` | k1=1.2, b=0.75 | Embeddings | yes |
| SVD embedding | `src/train_embeddings.py` | TruncatedSVD factorization | weighted matrix | `data/emb_*/item_vecs.npy`, `item_ids.npy` | `embedding.*` | dim=128, l2, seed=42 | Embeddings | yes |
| index construction | `src/build_index.py` | FAISS index builders | item vectors | `data/index_*/` FAISS index + `index_meta.json` | `index.*` via orchestrator `build_index_cmd()` | HNSW M=24 efC=200; IVF nlist=auto; PQ m=32 bits=8; IVF-PQ OPQ on; omp_threads=1 | Methods | yes |
| exact Flat reference | `src/utils/ann_io.py` | `build_exact_index()` (IndexFlatL2) | item vectors | exact top-k | none | L2 exact search is the reference | Methods | yes |
| HNSW / IVF-Flat / IVF-PQ / Flat-PQ | `src/build_index.py` + `src/utils/ann_io.py` | `load_ann_index()`, `CALIBRATION_PARAM` | index dir | `AnnIndex` handle | `index.*` | calibrated param: ef (HNSW), nprobe (IVF-*); flat/flatpq untunable | Methods | yes |
| calibration | `src/calibrate.py` | `calibrate_index()` | index + corpus vectors + shared modality queries | `results/main/calibration/*.json` | `calibration.*` | separate U2I/I2I sweeps; smallest param meeting target agreement recall@topk; 1000 queries | Calibration | yes |
| U2I evaluation | `src/eval_modalities.py` | `_build_queries("u2i")`, `evaluate_modality()` | split + vectors + index | aggregates + perquery npz | `retrieval.*`, `evaluation.*` | query = mean of training history; seen items excluded | Main results | yes |
| I2I evaluation | `src/eval_modalities.py` | `_build_queries("i2i")`, `evaluate_modality()` | split + vectors + index | aggregates + perquery npz | `retrieval.*`, `evaluation.*` | query = last chronological training item; anchor excluded | Main results | yes |
| latency measurement | `src/run_device.py` | single-query timing loop | index + shared modality queries | `latency_*.json` (mean/p50/p95/p99) | `evaluation.latency_queries` | 2000 timed queries at each modality-calibrated operating point | Efficiency | yes |
| bootstrap CIs | `src/bootstrap_significance.py` | `utils.metrics.bootstrap_ci()` | `results/main/perquery/*.npz` | `results/analyses/bootstrap/bootstrap_cis.csv` | `statistics.bootstrap_iterations` | n_boot=2000, seed=42 | Statistics | yes |
| paired significance | `src/bootstrap_significance.py` | `utils.metrics.paired_bootstrap_test()` | aligned perquery arrays | `results/analyses/bootstrap/paired_tests.csv` | `statistics.bootstrap_iterations` | paired vs flat baseline; two-sided bootstrap p | Statistics | yes |
| effect sizes | `src/effect_size_tables.py` | `cohens_d_paired()`, `cliffs_delta()` | aligned perquery arrays | `results/analyses/effect_sizes/effect_sizes.csv` | seed via CLI/meta | paired Cohen's d + Cliff's delta vs flat | Statistics | yes |
| embedding sensitivity | `src/run_embedding_backbone_sensitivity.py` | `ann_ranking_stability` (Spearman vs `svd_bm25`) | ml-1m + 4 required backbones (+ optional two-tower) | `results/analyses/embedding_sensitivity/embedding_backbone_sensitivity_all.csv` | `embedding_sensitivity.*` in `configs/analyses.yml` | U2I, 10000 queries, 5 methods; stability reported, never required | Sensitivity | yes |
| exposure analysis | `src/run_exposure_analysis.py` + `src/exposure_analysis.py` | `analyze_run()` | perquery npz extras | `results/analyses/exposure/exposure_analysis_all.csv` | CLI (`tail_frac=0.2`, `head_frac=0.1`) | every row carries `fairness_scope` (proxies only) | Exposure | yes |
| PQ diagnostics | `src/run_pq_diagnostics.py` + `src/pq_diagnostics.py` | reconstruction/overlap/variance/decile diagnostics | PQ indexes + embeddings | `results/analyses/pq_diagnostics/pq_diagnostics_{all,summary}.csv` | main config (datasets/weighting/dim/seed) | flatpq+ivfpq; conservative evidence-bound labels | PQ behavior | yes |
| scale stress | `src/run_scale_stress.py` | `synth_vectors()` + cost measurement | seeded synthetic vectors (internal) | `results/analyses/scale_stress/scale_stress_all.csv` | `scale_stress.*` in `configs/analyses.yml` | 5 sizes × 1 dim × 4 methods = 20 cells; target 0.95; `quality_measured=false` | Scale | yes |
| decision framework | `src/ann_decision_framework.py` | deployment scoring + `use_case_label()` | main summary + effect sizes + calibration | `results/analyses/decision_framework/ann_decision_framework_scores.csv` | `decision_framework.*` in `configs/analyses.yml` | weights 0.45/0.30/0.15/0.10; `allow_flatpq_online=false` | Guidance | yes (logic) |
| claim-support audit | `src/claim_support_audit.py` | `CLAIMS` × validation report | `results/_meta/validation_report.json` | `results/paper/tables/claim_support_audit.*` | fixed claim list | supported only when required validator sections pass | All | n/a (meta) |
| table generation | `src/tables_paper.py` | `TableEmitter` | canonical `results/main` + `results/analyses` | `results/paper/tables/*` + `.sources.json` sidecars | CLI (`--write_mode replace`) | no legacy inputs; sidecars mandatory | All | n/a (presentation) |
| figure generation | `src/figures_paper.py` + `src/utils/figures_ext.py` | figure functions | canonical results CSVs | `results/paper/figures/*.{png,pdf}` + sidecars | CLI (`--write_mode replace`) | 300 dpi, tight bbox, PNG+PDF | All | n/a (presentation) |
| paper-evidence validation | `src/validate_paper_evidence.py` | `SECTION_VALIDATORS` | `configs/paper_evidence_manifest.yml` + results | `results/_meta/validation_report.{csv,md,json}` | the manifest | 40 main / 72 calibration / 20 embedding / 20 scale rows; n_boot=2000; CPU-only | All | n/a (meta) |

## Canonical entry-point sequence

1. `src/capture_hardware.py`
2. `src/prepare_dataset.py` (×4 datasets)
3. `src/dataset_stats.py`
4. `src/run_revision_experiments.py` (drives `train_embeddings.py`,
   `build_index.py`, `calibrate.py`, `run_device.py`, `eval_modalities.py`)
5. `src/run_calibration_sensitivity.py`
6. `src/bootstrap_significance.py`
7. `src/effect_size_tables.py`
8. `src/run_embedding_backbone_sensitivity.py`
9. `src/run_exposure_analysis.py`
10. `src/run_pq_diagnostics.py`
11. `src/run_scale_stress.py`
12. `src/ann_decision_framework.py`
13. `src/tables_paper.py`
14. `src/figures_paper.py`
15. `src/validate_paper_evidence.py`
16. `src/claim_support_audit.py`

Supporting canonical packages: `src/datasets/`, `src/utils/`,
`src/backends/`. Optional modules (not required for the paper contract):
`src/run_energy_measurement.py`, `src/run_optional_ann_backend_comparison.py`.

## Expected evidence counts

- main rows = 40 (4 datasets × 2 modalities × 5 methods)
- calibration-sensitivity rows = 72 (4 datasets × 2 modalities × 3 tunable methods × 3 targets)
- embedding-sensitivity rows = 20 (4 required backbones × 5 methods)
- scale-stress rows = 20 (5 sizes × 1 dim × 4 methods)
- bootstrap iterations = 2000 on every row

## CPU-only boundary

IndexWise-Recsys is evaluated as a CPU-only framework. GPU acceleration and
GPU-specific latency behavior are outside the scope. Passive reporting that
an accelerator is present on the host (`accelerator_present`) is
environment metadata only; `main_experiments_accelerator_used` is always
`false`.

## Preservation assertion

Cleanup must not alter embedding training, CPU index construction,
calibration, U2I/I2I evaluation, bootstrap resampling, effect-size
definitions, exposure analysis, PQ diagnostics, scale-stress semantics, or
decision-framework logic. Existing numerical result files are never edited
to satisfy this contract; all post-cleanup results must be regenerated from
the canonical pipeline. Regression tests in `tests/` enforce the values in
this document.
