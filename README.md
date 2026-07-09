# IndexWise-Recsys

**IndexWise-Recsys** is a reproducible recommender-retrieval framework for
calibrated approximate-nearest-neighbor (ANN) evaluation: modality-separated
U2I / I2I retrieval, effect-size-aware statistics, long-tail exposure
analysis, PQ diagnostics, synthetic scaling, deployment guidance, and
optional GPU-aware experimentation.

Everything runs **locally**. The **canonical, reproducible benchmark is
CPU-only** (deterministic seeds everywhere, single-threaded FAISS index
construction by default). GPU support exists as an **optional, exploratory
extension** (`--use_gpu`, outputs isolated under `results/gpu_experiments/`)
and is not part of the main results — GPU kernels may introduce
nondeterminism (see [docs/hardware_protocol.md](docs/hardware_protocol.md)).

## Quick Start (CPU baseline)

> These commands are provided for **users to run manually**. They download a
> dataset and run experiments — they must NOT be executed by any automated
> agent working on this repository.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-cpu.txt

python src/capture_hardware.py --out_dir results/hardware --main_experiments_gpu_used false
python src/prepare_dataset.py --dataset ml-1m --out data/ml1m.csv
python src/run_revision_experiments.py --datasets ml-1m --modalities u2i i2i --methods flat hnsw ivfflat ivfpq flatpq --weighting bm25 --dim 128 --budget_mb 100 --calibration_targets 0.90 0.95 0.98 --queries_ml1m full --cpu_only --seed 42
```

Then validate what was produced:

```bash
python src/validate_results.py --allow_missing_optional
```

Optional extras (torch backbone, extra ANN backends, GPU/NVML tooling) are in
`requirements-optional.txt` — none are needed for the main results. The full
pipeline recipe is in [reproduction.md](reproduction.md).

**Revision pipeline** (effect-size-aware, modality-separated ANN calibration and deployment evaluation): see the [Revision pipeline](#revision-pipeline-calibration-modalities-statistics) section below and the full step-by-step recipe in [reproduction.md](reproduction.md).

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Download datasets (auto, normalized to `user_id,item_id,timestamp`)
```bash
python src/download_datasets.py
```

By default this prepares the three benchmark datasets: `ml-1m`, `ml-20m`, and
`goodbooks`. You can still prepare a single dataset directly:

```bash
python src/prepare_dataset.py --dataset ml-1m --out data/ml1m.csv
```

Amazon Books is supported as an **optional** dataset (large; listed under
`datasets.optional` in `configs/main_cpu.yml`, so the orchestrator only runs
it when you name it explicitly via `--datasets amazon-books`):

```bash
python src/prepare_dataset.py --dataset amazon-books --out data/amazon_books.csv
python src/download_datasets.py --datasets amazon-books
```

*(Alternatively)* run the default dataset downloader:
```bash
./prepare_all_datasets.sh     # or:  ./prepare_all_datasets.ps1  (PowerShell)
```

## Train embeddings (TruncatedSVD, float32)
```bash
python src/train_embeddings.py --interactions data/ml1m.csv --dim 128 --out_dir data/emb_ml1m
```

## Build indices (verbose)
- HNSW
```bash
python src/build_index.py --method hnsw --item_vecs data/emb_ml1m/item_vecs.npy --item_ids data/emb_ml1m/item_ids.npy   --out_dir data/index_ml1m_hnsw --budget_mb 100 --M 24 --efc 200
```
- IVF-PQ (+OPQ) **auto-safeguards for small N** (OPQ auto-disabled <10k items unless `--force_opq`)
```bash
python src/build_index.py --method ivfpq --item_vecs data/emb_ml1m/item_vecs.npy --item_ids data/emb_ml1m/item_ids.npy   --out_dir data/index_ml1m_ivfpq --budget_mb 100 --nlist auto --m 32 --bits 8 --opq
```
- IVF-Flat
```bash
python src/build_index.py --method ivfflat --item_vecs data/emb_ml1m/item_vecs.npy --item_ids data/emb_ml1m/item_ids.npy   --out_dir data/index_ml1m_ivfflat --budget_mb 100 --nlist auto
```
- Flat-PQ
```bash
python src/build_index.py --method flatpq --item_vecs data/emb_ml1m/item_vecs.npy --item_ids data/emb_ml1m/item_ids.npy   --out_dir data/index_ml1m_flatpq --budget_mb 100 --m 32 --bits 8
```
- Flat (exact)
```bash
python src/build_index.py --method flat --item_vecs data/emb_ml1m/item_vecs.npy --item_ids data/emb_ml1m/item_ids.npy   --out_dir data/index_ml1m_flat --budget_mb 100
```

## Measure latency & memory
```bash
python src/run_device.py --index data/index_ml1m_hnsw --item_vecs data/emb_ml1m/item_vecs.npy --queries 10000 --topk 100 --ef 64
```

## End-to-end metrics (Recall, Precision, MAP, MRR, NDCG, HR; + Coverage, Gini, Long-tail uplift)
```bash
python src/eval_end2end.py --item_vecs data/emb_ml1m/item_vecs.npy --index data/index_ml1m_hnsw   --ann_method hnsw --queries 10000 --topk 100 --metric_topk 10 --ef 64
```

## Full grid (multi-dataset × methods) and report
```bash
python src/run_grid.py --emb_dim 128 --queries 2000 --topk 100 --metric_topk 10 --budget_mb 100
python src/figures.py --summary results/summary_all.csv --out_dir results/figures
python src/report.py  --summary results/summary_all.csv --out results/report.md --fig_dir results/figures
```

## One-shot (everything: 50/100/150 MB, all datasets)
```bash
./run_all.sh         # or:  ./run_all.ps1    (PowerShell)
```
Outputs to `results/`: per-dataset CSVs, `summary_all.csv`, figures, and `report.md`.

## Smoke test
```bash
./tasks.sh smoke
```

## Ablation sweeps
```bash
python src/sweep.py --dataset ml-1m --method hnsw --budget_mb 100 --ef_list 32 64 128
python src/sweep.py --dataset ml-1m --method ivfpq --budget_mb 100 --nprobe_list 4 8 16 32 --m_list 16 32 --with_opq --without_opq
```

---

## Revision pipeline (calibration, modalities, statistics)

New, backward-compatible layer on top of the original benchmark:

- **BM25 / TF-IDF / none weighting** before SVD: `train_embeddings.py --weighting bm25 --bm25_k1 1.2 --bm25_b 0.75 --normalize l2 --seed 42` (`src/utils/weighting.py`). Defaults reproduce the original unweighted embeddings.
- **Modality-separated evaluation** (`src/eval_modalities.py`): user-to-item (U2I; user = mean of training item vectors, training items excluded) and item-to-item (I2I; anchor = last training item) under a deterministic **temporal leave-one-out** split (`src/utils/splits.py`). Per-query metrics are saved for statistics.
- **ANN calibration vs exact Flat** (`src/calibrate.py`): smallest `ef`/`nprobe` reaching a target agreement recall@k, with latency at each grid point. **Sensitivity across targets 0.90 / 0.95 / 0.98**: `src/run_calibration_sensitivity.py` + `configs/calibration_thresholds.yml`.
- **Bootstrap CIs + paired significance** (`src/bootstrap_significance.py`) and **effect sizes** — paired Cohen's d and Cliff's delta vs Flat (`src/effect_size_tables.py`).
- **Long-tail exposure metrics** (`long_tail_exposure`, `long_tail_uplift`, `exposure_proxy` in `src/utils/metrics.py`) instead of generic fairness terms.
- **Deployment guidance** (`src/deployment_guidance.py`): constraint-based index recommendations derived from the measured CSVs.
- **CPU-only reproducibility**: `src/capture_hardware.py`, `configs/hardware_profiles.yml`, `docs/hardware_protocol.md`; deterministic seeds everywhere (`--seed`, default 42).
- **Paper tables & figures**: `src/tables_paper.py`, `src/figures_paper.py` (post-processing only; skip gracefully when inputs are missing).
- **Synthetic scaling** (optional): `src/synthetic_scaling.py`.

Quick start (after dataset prep, see above):

```bash
python src/capture_hardware.py --out_dir results/hardware --main_experiments_gpu_used false
python src/run_revision_experiments.py --datasets ml-1m ml-20m goodbooks --modalities u2i i2i --methods flat hnsw ivfflat ivfpq flatpq --weighting bm25 --dim 128 --budget_mb 100 --calibration_targets 0.90 0.95 0.98 --queries_large 10000 --queries_ml1m full --cpu_only --seed 42
python src/run_calibration_sensitivity.py
python src/bootstrap_significance.py
python src/effect_size_tables.py
python src/deployment_guidance.py
python src/tables_paper.py
python src/figures_paper.py
```

All outputs land under `results/` (see `docs/result_schema.md`). Docs: `docs/artifact_description.md`, `docs/reviewer_revision_map.md`, `docs/hardware_protocol.md`, `docs/limitations_code_level.md`. Full recipe: [reproduction.md](reproduction.md).
