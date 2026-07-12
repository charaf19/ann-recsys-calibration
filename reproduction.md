# Reproduction protocol

This is the detailed version of the single canonical workflow in
`README.md`. There is no second workflow. IndexWise-Recsys is CPU-only:
no step uses a GPU, `faiss-gpu`, CUDA, or PyNVML.

## 1. Environment

Use Python 3.10+ and install the pinned canonical environment:

```powershell
python -m pip install -r requirements-cpu.txt
```

Optional modules (PyTorch for the `two_tower_mlp` sensitivity backbone,
alternative CPU ANN backends) install selectively from
`requirements-optional.txt`; every consumer degrades gracefully and
records unavailability honestly when they are absent.

Fix thread environment variables before launching Python if you want them
recorded (see `docs/hardware_protocol.md`), then capture the environment:

```powershell
python src\capture_hardware.py
```

This writes `results/_meta/hardware.{json,md}` (with
`accelerator_present` as passive metadata and
`main_experiments_accelerator_used=false`) and
`results/_meta/environment.txt`.

## 2. Datasets

Prepare all four canonical datasets (raw sources are fetched/normalized by
the preparers; the experiment pipeline itself never downloads):

```powershell
python src\prepare_dataset.py --dataset ml-1m --out data\ml1m.csv
python src\prepare_dataset.py --dataset ml-20m --out data\ml20m.csv
python src\prepare_dataset.py --dataset goodbooks --out data\goodbooks.csv
python src\prepare_dataset.py --dataset amazon-books --out data\amazon_books.csv
```

Then generate the dataset-statistics table:

```powershell
python src\dataset_stats.py `
    --datasets `
        ml-1m:data/ml1m.csv `
        ml-20m:data/ml20m.csv `
        goodbooks:data/goodbooks.csv `
        amazon-books:data/amazon_books.csv `
    --min_user_interactions 5 `
    --out_dir results\paper\tables
```

## 3. Main experiment grid

```powershell
python src\run_revision_experiments.py `
    --config configs\main_cpu.yml `
    --write_mode fail_if_exists
```

For every dataset × method this trains the SVD/BM25 embeddings, builds the
index (parameters resolved from `configs/defaults.yml`: HNSW M=24
efConstruction=200, IVF nlist=auto, PQ m=32 bits=8, IVF-PQ with OPQ),
calibrates ef/nprobe against exact Flat at targets 0.90/0.95/0.98 with
1000 deterministic queries from each modality population, measures
single-query latency (2000 queries) at each modality's 0.95 operating point,
and runs the U2I and I2I evaluations
(ml-1m: all eligible users; other datasets: 10000 seeded users).

The run fails before any expensive work if a configured dataset CSV is
missing (pass `--allow_missing_datasets` only for deliberate partial runs —
the validator will then fail on the gap). Outputs land under
`results/main/` with per-combination status files and a provenance
manifest in `results/_meta/run_manifest.json`.

## 4. Statistics and analyses

```powershell
python src\run_calibration_sensitivity.py `
    --config configs\main_cpu.yml `
    --write_mode fail_if_exists

python src\bootstrap_significance.py `
    --config configs\main_cpu.yml `
    --n_boot 2000 `
    --seed 42 `
    --write_mode fail_if_exists

python src\effect_size_tables.py `
    --write_mode fail_if_exists

python src\run_embedding_backbone_sensitivity.py `
    --config configs\analyses.yml `
    --write_mode fail_if_exists

python src\run_exposure_analysis.py `
    --write_mode fail_if_exists

python src\run_pq_diagnostics.py `
    --write_mode fail_if_exists

python src\run_scale_stress.py `
    --config configs\analyses.yml `
    --write_mode fail_if_exists

python src\ann_decision_framework.py `
    --config configs\analyses.yml `
    --write_mode fail_if_exists
```

Notes:

- `bootstrap_significance.py` reads `statistics.bootstrap_iterations`
  (2000) from the config; `--n_boot` is an explicit override and
  non-positive values are rejected. There is no `--iterations` flag.
- The embedding-sensitivity study requires the four backbones
  `svd_bm25`, `svd_tfidf`, `svd_none`, `bpr_matrix_factorization`
  (U2I, ml-1m, 10000 queries); `two_tower_mlp` is optional and is recorded
  as unavailable when PyTorch is not installed. Ranking stability is
  reported, never required.
- The scale stress test is cost-only: every one of its 20 rows carries
  `quality_measured=false` and no recommendation-quality metric.

## 5. Paper artifacts

```powershell
python src\tables_paper.py `
    --write_mode replace

python src\figures_paper.py `
    --write_mode replace
```

Tables (CSV/MD/LaTeX) go to `results/paper/tables/`, figures (PNG+PDF,
300 dpi) to `results/paper/figures/`; every artifact gets a
`<name>.sources.json` sidecar with source hashes, config hash, Git commit,
and timestamp. Inputs come only from `results/main/` and
`results/analyses/`.

## 6. Validation and claim audit

```powershell
python src\validate_paper_evidence.py
python src\claim_support_audit.py
```

The validator checks the full contract in
`configs/paper_evidence_manifest.yml`:

| evidence | expectation |
| --- | --- |
| main rows | 40 (4 datasets × 2 modalities × 5 methods), weighting=bm25, dim=128, seed=42 |
| calibration-sensitivity rows | 72 (4 datasets × 2 modalities × hnsw/ivfflat/ivfpq × targets 0.90/0.95/0.98) |
| embedding-sensitivity rows | 20 required (4 backbones × 5 methods), stability reported |
| scale-stress rows | 20, all `quality_measured=false` |
| bootstrap | `n_boot=2000` on every row, paired vs flat |
| effect sizes | Cohen's d + Cliff's delta vs flat |
| exposure | all datasets/modalities/methods with a `fairness_scope` field |
| CPU scope | `main_experiments_accelerator_used=false`, no GPU result paths |

It writes `results/_meta/validation_report.{csv,md,json}` and exits
non-zero on any failure. The claim audit then marks each paper claim
supported only when its required validation sections pass.

## 7. Fresh-run policy

Canonical scripts read only canonical filenames under `results/main/` and
`results/analyses/`; there are no legacy-path fallbacks. Start from the
empty results skeleton, do not combine evidence from different runs, and
rely on the default `fail_if_exists` write mode to protect completed
evidence (use `merge` only for deliberate incremental additions — rows are
merged by natural key and conflicting values raise an error).
