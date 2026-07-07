# Reviewer revision map

Maps each revision requirement to the concrete code, configs, and outputs
that address it.

| # | Requirement | Where addressed | Output |
| --- | --- | --- | --- |
| 1 | BM25 / TF-IDF weighting for embeddings | `src/utils/weighting.py`; `--weighting {none,tfidf,bm25}` in `src/train_embeddings.py` (with `--bm25_k1`, `--bm25_b`, `--normalize`) | `data/emb_*/embedding_meta.json` records the scheme |
| 2 | Clean U2I vs I2I separation | `src/eval_modalities.py` (`--modality u2i\|i2i\|both`); query construction documented in the module docstring | `results/main/*__{u2i,i2i}__*.json`, `modality` column in `summary_main.csv` |
| 3 | ANN calibration vs Flat | `src/calibrate.py` (agreement recall@k vs exact, ascending ef/nprobe sweep) | `results/main/calibration/*.json` |
| 4 | Calibration sensitivity across thresholds (0.90/0.95/0.98) | `src/run_calibration_sensitivity.py` + `configs/calibration_thresholds.yml` | `results/calibration_sensitivity/calibration_sensitivity.csv` |
| 5 | Bootstrap confidence intervals | `src/bootstrap_significance.py`; per-query arrays from `eval_modalities.py` enable *paired* comparisons | `results/bootstrap/bootstrap_cis.csv`, `paired_tests.csv` |
| 6 | Effect sizes alongside significance | `src/effect_size_tables.py` (paired Cohen's d, Cliff's delta, magnitude bins) | `results/effect_sizes/effect_sizes.*`, merged table `table_significance_effect_sizes.*` |
| 7 | Precise long-tail terminology | `src/utils/metrics.py`: `long_tail_exposure`, `long_tail_uplift`, `exposure_proxy` (no generic "fairness" columns) | dedicated columns in `summary_main.csv`, `table_long_tail_exposure.*` |
| 8 | Deployment guidance | `src/deployment_guidance.py` (constraint-based recommendation + generated notes) | `results/deployment_guidance/` |
| 9 | CPU-only reproducibility | `src/capture_hardware.py`, `configs/hardware_profiles.yml`, `docs/hardware_protocol.md`, `--cpu_only` flag, faiss-cpu pin | `results/hardware/` |
| 10 | Deterministic seeds | `set_global_seed` in `src/utils/common.py`; `--seed` on every script; deterministic temporal split (`src/utils/splits.py`) | `seed` column recorded in all outputs |
| 11 | Reproducible command recipe | `reproduction.md` (Python-only, Windows-friendly) | — |
| 12 | Synthetic scaling | `src/synthetic_scaling.py` | `results/scaling/scaling.csv` |
| 13 | Optional ANN backend adapters | `register_backend()` in `src/utils/ann_io.py` | — |
| 14 | Result schema documentation | `docs/result_schema.md` | — |
| 15 | Code-level limitations stated | `docs/limitations_code_level.md` | — |

Backward compatibility: the original scripts (`run_grid.py`, `eval_end2end.py`,
`run_device.py`, `sweep.py`, `figures.py`, `report.py`) are unchanged in
behavior; `train_embeddings.py` defaults (`--weighting none --normalize none
--seed 42`) reproduce the original embeddings exactly.
