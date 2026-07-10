# IndexWise-Recsys

IndexWise-Recsys is the reproducible artifact for evaluating CPU approximate-nearest-neighbor index choices in recommender systems.

IndexWise-Recsys is evaluated as a CPU-only framework. GPU acceleration and GPU-specific latency behavior are outside the current scope.

## Canonical pipeline

Install the canonical environment with:

```bash
python -m pip install -r requirements-cpu.txt
```

Datasets must be supplied locally; the canonical experiment pipeline does not download them. Prepare each dataset with `src/prepare_dataset.py`, then run:

```bash
python src/capture_hardware.py
python src/dataset_stats.py
python src/run_revision_experiments.py --config configs/main_cpu.yml
python src/run_calibration_sensitivity.py
python src/bootstrap_significance.py --iterations 2000 --seed 42
python src/effect_size_tables.py
python src/run_embedding_backbone_sensitivity.py
python src/run_exposure_analysis.py
python src/run_pq_diagnostics.py
python src/run_scale_stress.py
python src/ann_decision_framework.py
python src/claim_support_audit.py
python src/tables_paper.py
python src/figures_paper.py
python src/validate_paper_evidence.py
```

This list describes execution order; do not launch it blindly. Dataset preparation, main experiments, and analyses are intentionally explicit and can be expensive.

The main configuration fixes TruncatedSVD/BM25 at dimension 128 with L2 normalization, seed 42, methods `flat`, `hnsw`, `ivfflat`, `ivfpq`, and `flatpq`, and modalities `u2i` and `i2i`. Calibration targets are 0.90, 0.95, and 0.98, with 0.95 primary.

## Results layout

All generated evidence belongs under:

```text
results/
├── _meta/      # hardware, run status, validation
├── main/       # canonical evaluation and per-query evidence
├── analyses/   # named secondary analyses
├── paper/      # generated tables and figures
└── archive/    # reserved; never read by canonical scripts
```

The repository intentionally starts with empty result directories. Existing results are not required inputs and are never silently migrated.

Optional CPU modules are listed in `requirements-optional.txt`. PyTorch remains optional for embedding-backbone sensitivity; no workflow requires `faiss-gpu`, CUDA, or PyNVML.

See `reproduction.md`, `docs/critical_experiment_contract.md`, and `docs/result_schema.md` for the detailed protocol and evidence contract.
