# IndexWise-Recsys

Reproducible, CPU-only benchmark of approximate-nearest-neighbor (ANN)
index choices for recommender-system retrieval, with calibrated operating
points, paired statistics, and a strict paper-evidence contract.

## 1. Project overview

IndexWise-Recsys measures how FAISS index structures (`flat`, `hnsw`,
`ivfflat`, `ivfpq`, `flatpq`) trade recommendation quality, latency,
memory, and long-tail exposure when serving user-to-item (U2I) and
item-to-item (I2I) retrieval over four public interaction datasets. Every
ANN method is calibrated against exact Flat search to a target agreement
recall before it is measured, so methods are compared at comparable
fidelity rather than at arbitrary parameter settings.

## 2. Scientific scope

- Datasets: `ml-1m`, `ml-20m`, `goodbooks`, `amazon-books` (all four are
  required for the revised paper).
- Embedding: TruncatedSVD over the BM25-weighted interaction matrix
  (dim 128, L2-normalized, `bm25_k1=1.2`, `bm25_b=0.75`, seed 42).
- Protocol: deterministic temporal leave-one-out; the chronologically last
  interaction per user is the held-out positive.
- Two metric families that are never conflated: **agreement recall**
  (`ann_recall_vs_exact_*`, index fidelity vs exact Flat, used for
  calibration) and **recommendation relevance** (`recall/ndcg/...` vs the
  held-out interaction).

## 3. Main contribution

A calibrated, modality-aware, effect-size-aware decision procedure for ANN
index selection: methods are compared at calibrated agreement-recall
operating points (targets 0.90/0.95/0.98, primary 0.95), differences are
tested with paired bootstrap (2000 iterations) plus paired Cohen's d and
Cliff's delta, and the measurements feed a reproducible deployment-scoring
framework (`src/ann_decision_framework.py`).

## 4. CPU-only boundary

All canonical experiments run on CPU with `faiss-cpu`. GPU acceleration and
GPU-specific latency behavior are outside the scope. Hardware capture
discloses accelerator presence passively (`accelerator_present`) as
environment metadata only; `main_experiments_accelerator_used` is always
`false`. No workflow requires `faiss-gpu`, CUDA, or PyNVML.

## 5. Repository structure

```text
configs/            defaults.yml, main_cpu.yml, analyses.yml,
                    paper_evidence_manifest.yml
src/                pipeline scripts (one canonical entry point per stage)
src/utils/          config loader, canonical paths, atomic result I/O,
                    provenance, metrics, splits
src/datasets/       dataset normalizers
src/backends/       optional CPU ANN backend adapters
tests/              lightweight regression tests (tiny fixtures only)
docs/               scientific contract, schemas, protocols, cleanup report
results/            generated evidence (starts empty; see section 10)
data/               datasets, embeddings, indexes (gitignored)
```

## 6. Installation

```powershell
python -m pip install -r requirements-cpu.txt
```

`requirements-cpu.txt` is the only file needed for every canonical result.
Optional CPU modules (PyTorch for the two-tower sensitivity backbone,
alternative ANN backends) are listed in `requirements-optional.txt`.

## 7. Dataset preparation

Datasets are prepared locally with one command each; the experiment
pipeline itself never downloads anything.

```powershell
python src\prepare_dataset.py --dataset ml-1m --out data\ml1m.csv
python src\prepare_dataset.py --dataset ml-20m --out data\ml20m.csv
python src\prepare_dataset.py --dataset goodbooks --out data\goodbooks.csv
python src\prepare_dataset.py --dataset amazon-books --out data\amazon_books.csv
```

Dataset statistics for the paper:

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

## 8. Canonical revised-paper workflow

There is exactly one workflow. Experiment producers default to
`--write_mode fail_if_exists` (they refuse to overwrite existing evidence);
regenerated paper tables/figures default to `replace`. The main run fails
before any expensive work if a configured dataset is missing, unless
`--allow_missing_datasets` is passed explicitly.

```powershell
python src\capture_hardware.py

python src\run_revision_experiments.py `
    --config configs\main_cpu.yml `
    --write_mode fail_if_exists

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

python src\tables_paper.py `
    --write_mode replace

python src\figures_paper.py `
    --write_mode replace

python src\validate_paper_evidence.py
python src\claim_support_audit.py
```

This list describes execution order; the main run and several analyses are
deliberately expensive — do not launch them casually. For a quick
structural check use `tests\fixtures\test_tiny.yml` (never for paper
reproduction).

## 9. Critical experiments and expected evidence counts

| evidence | count |
| --- | --- |
| main rows (4 datasets × 2 modalities × 5 methods) | 40 |
| calibration-sensitivity rows (4 × 3 tunable methods × 3 targets) | 36 |
| embedding-sensitivity rows (4 required backbones × 5 methods) | 20 |
| scale-stress rows (5 sizes × 3 dims × 5 methods, cost only) | 75 |
| bootstrap iterations (every row) | 2000 |

The full component contract is in
[docs/critical_experiment_contract.md](docs/critical_experiment_contract.md).

## 10. Results directory

```text
results/
├── _meta/       hardware, environment, run manifest, validation reports
├── main/        summary_main.csv, aggregates/, perquery/, calibration/, status/
├── analyses/    one named subdirectory per analysis
└── paper/       tables/, figures/, supplementary/ (with .sources.json sidecars)
```

The repository intentionally starts with **empty** result directories:
fresh results must be produced by the canonical workflow above. Canonical
scripts never read legacy paths or alternate filenames, and all
consolidated evidence is written atomically through
`src/utils/result_io.py` (modes: `fail_if_exists` / `replace` / `merge`).

## 11. Validation

`python src\validate_paper_evidence.py` strictly validates the evidence set
against `configs/paper_evidence_manifest.yml` — completeness of every
dataset/method/modality combination, expected row counts, `n_boot=2000`,
cost-only scale rows, and the CPU-only scope — and writes
`results/_meta/validation_report.{csv,md,json}`, returning a non-zero exit
status on any failure. While `results/` is empty the validator fails with
missing-evidence messages by design. `claim_support_audit.py` then marks
each paper claim supported only when its required validation sections pass.

## 12. Reproducibility guarantees

- One seed (42) defined once in `configs/defaults.yml` and threaded through
  every stage; index builds use one OpenMP thread for bit-reproducible
  HNSW/IVF construction.
- Deterministic temporal splits and seeded query subsampling.
- Full run provenance in `results/_meta/run_manifest.json` (resolved
  config + hash, Git commit, package versions, hardware) and per-artifact
  `.sources.json` sidecars with file hashes.
- Regression tests: `python -m pytest tests -q` (tiny fixtures only; no
  experiments, no downloads).

## 13. Limitations

See [docs/limitations_code_level.md](docs/limitations_code_level.md) —
SVD-embedding scope, single held-out positive, machine-relative latency,
exposure proxies (not a fairness audit), cost-only synthetic scale testing,
and the CPU-only boundary.

## 14. Citation and license

Citation and license information will be added upon publication.
