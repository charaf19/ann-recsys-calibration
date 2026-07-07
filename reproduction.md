# Reproduction guide (CPU-only, Python-only)

Every step below is a plain Python command run from the repository root.
No GPU is used anywhere. All randomness is seeded (`--seed 42` everywhere);
splits and query sampling are deterministic. All outputs land under
`results/`; large artifacts (embeddings, indexes, per-query `.npz`) are
git-ignored.

Approximate wall-clock on an 8-core CPU: ml-1m minutes, ml-20m a few hours
(dominated by SVD and index builds).

## 0. Environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Capture hardware / software environment

```bash
python src/capture_hardware.py --out_dir results/hardware --main_experiments_gpu_used false
```

Outputs: `results/hardware/hardware.json`, `hardware.md`, `env_freeze.txt`.

## 2. Prepare datasets (downloads raw archives, normalizes to user_id,item_id,timestamp)

```bash
python src/prepare_dataset.py --dataset ml-1m --out data/ml1m.csv
python src/prepare_dataset.py --dataset ml-20m --out data/ml20m.csv
python src/prepare_dataset.py --dataset goodbooks --out data/goodbooks.csv
```

## 3. Dataset statistics table

```bash
python src/dataset_stats.py --datasets ml-1m:data/ml1m.csv ml-20m:data/ml20m.csv goodbooks:data/goodbooks.csv --min_user_interactions 5 --out_dir results/paper_tables
```

Outputs: `results/paper_tables/dataset_stats.{csv,md,tex}`.

## 4. (Optional, standalone) Train embeddings

The orchestrator in step 5 trains embeddings itself; run these only if you
want the artifacts without the full grid.

```bash
python src/train_embeddings.py --interactions data/ml1m.csv --dim 128 --weighting bm25 --bm25_k1 1.2 --bm25_b 0.75 --normalize l2 --seed 42 --out_dir data/emb_ml1m_bm25_d128
python src/train_embeddings.py --interactions data/ml20m.csv --dim 128 --weighting bm25 --bm25_k1 1.2 --bm25_b 0.75 --normalize l2 --seed 42 --out_dir data/emb_ml20m_bm25_d128
python src/train_embeddings.py --interactions data/goodbooks.csv --dim 128 --weighting bm25 --bm25_k1 1.2 --bm25_b 0.75 --normalize l2 --seed 42 --out_dir data/emb_goodbooks_bm25_d128
```

`--weighting` accepts `none`, `tfidf`, `bm25` (defaults preserve the original
unweighted behavior).

## 5. Main experiments (embeddings → indexes → calibration → latency → U2I/I2I eval)

```bash
python src/run_revision_experiments.py --datasets ml-1m ml-20m goodbooks --modalities u2i i2i --methods flat hnsw ivfflat ivfpq flatpq --weighting bm25 --dim 128 --budget_mb 100 --calibration_targets 0.90 0.95 0.98 --queries_large 10000 --queries_ml1m full --cpu_only --seed 42
```

Defaults come from `configs/main_cpu.yml`; CLI flags override. Add
`--reuse_existing` to resume without rebuilding embeddings/indexes.

Outputs:
- `results/main/summary_main.csv` — one row per dataset × modality × method
- `results/main/*.json` — aggregate metrics per combination
- `results/main/perquery/*.npz` — per-query metrics (bootstrap input)
- `results/main/calibration/*.json` — calibration records per target

## 6. Calibration sensitivity (0.90 / 0.95 / 0.98)

```bash
python src/run_calibration_sensitivity.py
```

Reads `configs/calibration_thresholds.yml`; requires the indexes from step 5.
Output: `results/calibration_sensitivity/calibration_sensitivity.csv` (+ JSON
sweeps).

## 7. Bootstrap confidence intervals and paired significance

```bash
python src/bootstrap_significance.py
```

Outputs: `results/bootstrap/bootstrap_cis.csv`, `results/bootstrap/paired_tests.csv`.

## 8. Effect sizes (Cohen's d, Cliff's delta vs Flat)

```bash
python src/effect_size_tables.py
```

Outputs: `results/effect_sizes/effect_sizes.{csv,md,tex}`.

## 9. Deployment guidance

```bash
python src/deployment_guidance.py
```

Outputs: `results/deployment_guidance/deployment_guidance.{csv,md,tex}`,
`deployment_notes.md`.

## 10. Paper tables and figures

```bash
python src/tables_paper.py
python src/figures_paper.py
```

Outputs: `results/paper_tables/table_*.{csv,md,tex}`,
`results/figures_paper/fig_*.{png,pdf}`.

## 11. (Optional) Synthetic scaling study

```bash
python src/synthetic_scaling.py --items_list 5000 20000 50000 100000 --seed 42
```

Output: `results/scaling/scaling.csv` (picked up by `figures_paper.py` as
`fig_scaling`). Use `--dry_run` to print the command plan without running.

## Single-command order (copy-paste)

```bash
python src/capture_hardware.py --out_dir results/hardware --main_experiments_gpu_used false
python src/prepare_dataset.py --dataset ml-1m --out data/ml1m.csv
python src/prepare_dataset.py --dataset ml-20m --out data/ml20m.csv
python src/prepare_dataset.py --dataset goodbooks --out data/goodbooks.csv
python src/dataset_stats.py --datasets ml-1m:data/ml1m.csv ml-20m:data/ml20m.csv goodbooks:data/goodbooks.csv --min_user_interactions 5 --out_dir results/paper_tables
python src/run_revision_experiments.py --datasets ml-1m ml-20m goodbooks --modalities u2i i2i --methods flat hnsw ivfflat ivfpq flatpq --weighting bm25 --dim 128 --budget_mb 100 --calibration_targets 0.90 0.95 0.98 --queries_large 10000 --queries_ml1m full --cpu_only --seed 42
python src/run_calibration_sensitivity.py
python src/bootstrap_significance.py
python src/effect_size_tables.py
python src/deployment_guidance.py
python src/tables_paper.py
python src/figures_paper.py
```

## 12. Reviewer-limitation modules (run after steps 5–8)

All of these are optional analyses layered on the main results; each skips
gracefully when its inputs are missing. Mapping to reviewer concerns:
`docs/reviewer_limitation_to_code_map.md`.

```bash
# ANN decision framework (scores + use-case labels from measured CSVs)
python src/ann_decision_framework.py

# PQ diagnostics (replaces the speculative regularization claim with evidence)
python src/run_pq_diagnostics.py --datasets ml-1m ml-20m goodbooks --seed 42

# Exposure-proxy analysis (deciles, head/mid/tail, popularity calibration)
python src/run_exposure_analysis.py

# Embedding backbone sensitivity (ann_ranking_stability vs svd_bm25)
python src/run_embedding_backbone_sensitivity.py

# Production-scale synthetic stress test (cost only; quality_measured=false)
python src/run_scale_stress.py

# Optional ANN backend comparison (ScaNN/NGT recorded as unavailable if not installed)
python src/run_optional_ann_backend_comparison.py --item_vecs data/emb_ml1m_bm25_d128/item_vecs.npy --dataset_label ml-1m

# Energy measurement (RAPL on Linux; honest NA fallback on Windows)
python src/run_energy_measurement.py --datasets ml-1m ml-20m goodbooks --queries 5000 --seed 42

# Claim-support audit (regenerate after every stage)
python src/claim_support_audit.py

# refresh consolidated tables/figures (now also covers the modules above)
python src/tables_paper.py
python src/figures_paper.py
```

Optional dependencies (not in requirements.txt, install only if wanted):
`torch` (two_tower_mlp backbone), `hnswlib` / `scann` (Linux) / `ngt`
(backend comparison), `pynvml` (supplementary GPU energy).

## Determinism notes

- Temporal leave-one-out split: stable mergesort on (user_id, timestamp);
  ties break by original row order (`src/utils/splits.py`).
- Query subsampling, calibration query sampling, and all bootstrap resampling
  use `numpy.random.default_rng(seed)`.
- HNSW graph construction is the one source of run-to-run nondeterminism when
  built with multiple threads; latency numbers additionally depend on the
  machine (see `docs/hardware_protocol.md` and `docs/limitations_code_level.md`).
