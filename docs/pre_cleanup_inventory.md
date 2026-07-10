# Pre-cleanup repository inventory

Inventory captured before the CPU-only cleanup. “Imported or called by” is based on static imports, subprocess calls, shell entry points, and documentation commands. Generated datasets and model artifacts under `data/` are outside this cleanup.

| Path | Type | Purpose | Imported or called by | Produces | Canonical or legacy | Safe to delete? | Replacement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `src/prepare_dataset.py` | CLI | Normalize supported datasets | canonical workflow | normalized interaction CSV | canonical | no | — |
| `src/dataset_stats.py` | CLI | Dataset descriptive statistics | paper workflow | `results/paper/dataset_stats.*` | canonical | no | — |
| `dataset_stats.py` | CLI | Duplicate dataset statistics | old docs/scripts | legacy tables | legacy | yes | `src/dataset_stats.py` |
| `src/train_embeddings.py` | CLI | BM25/TF-IDF/none TruncatedSVD embeddings | revision orchestrator | item/user vectors | canonical | no | — |
| `src/train_neural_embeddings.py` | library/CLI | Optional embedding backbones | embedding sensitivity | optional vectors | supporting | no | — |
| `src/build_index.py` | CLI | CPU FAISS index construction | revision and analysis runners | CPU indexes | canonical | no | — |
| `src/calibrate.py` | CLI/library | Exact-Flat agreement calibration | revision and sensitivity runners | calibration JSON | canonical | no | — |
| `src/eval_modalities.py` | CLI | U2I/I2I temporal evaluation | revision and sensitivity runners | aggregate JSON and per-query NPZ | canonical | no | — |
| `src/eval_end2end.py` | CLI | Older combined evaluator | `run_grid.py`, `sweep.py`, smoke script | old summaries | legacy | yes | `src/eval_modalities.py` |
| `src/run_revision_experiments.py` | CLI | Canonical main experiment orchestrator | README/reproduction | `results/main/`, metadata | canonical | no | — |
| `src/run_grid.py` | CLI | Older grid orchestrator | old README | old summary CSV | legacy | yes | `src/run_revision_experiments.py` |
| `src/sweep.py` | CLI | Older parameter sweep using old evaluator | no canonical caller | old sweep results | legacy | yes | calibration plus revision pipeline |
| `src/smoke_tests.py` | CLI | Smoke runner tied to old evaluator | no canonical caller | temporary smoke outputs | legacy | yes | lightweight unit/CLI checks |
| `src/run_calibration_sensitivity.py` | CLI | Calibration-target analysis | paper workflow | sensitivity CSV/JSON | canonical | no | — |
| `src/bootstrap_significance.py` | CLI | Paired bootstrap and tests | paper workflow | bootstrap CSVs | canonical | no | — |
| `src/effect_size_tables.py` | CLI | Paired Cohen’s d and Cliff’s delta | paper workflow | effect-size tables | canonical | no | — |
| `src/run_embedding_backbone_sensitivity.py` | CLI | Embedding-backbone sensitivity | paper workflow | sensitivity results | canonical | no | — |
| `src/run_exposure_analysis.py` | CLI | Exposure analysis | paper workflow | exposure results | canonical | no | — |
| `src/exposure_analysis.py` | library | Exposure metrics | exposure runner | in-memory metrics | supporting | no | — |
| `src/run_pq_diagnostics.py` | CLI | PQ diagnostics | paper workflow | PQ results | canonical | no | — |
| `src/pq_diagnostics.py` | library | PQ diagnostic functions | PQ runner | in-memory metrics | supporting | no | — |
| `src/run_scale_stress.py` | CLI | Canonical synthetic scale stress | paper workflow | scale-stress results | canonical | no | — |
| `src/synthetic_scaling.py` | CLI | Older synthetic scaling implementation | old docs | old scaling CSV | legacy | yes | `src/run_scale_stress.py` |
| `src/generate_synth.py` | CLI | Synthetic data helper | legacy scaling/smoke paths | synthetic CSV | legacy | yes | internal generation in scale stress |
| `src/ann_decision_framework.py` | CLI | ANN decision framework | paper workflow | decision tables | canonical | no | — |
| `src/deployment_guidance.py` | CLI | Older deployment recommendation table | old docs | old guidance tables | legacy | yes | `src/ann_decision_framework.py` |
| `src/claim_support_audit.py` | CLI | Claim-to-evidence audit | paper workflow | audit tables | canonical | no | — |
| `src/tables_paper.py` | CLI | Paper tables | paper workflow | tables | canonical | no | — |
| `src/figures_paper.py` | CLI | Paper figures | paper workflow | figures | canonical | no | — |
| `src/figures.py` | CLI | Old figures from old summary | old README | old figures | legacy | yes | `src/figures_paper.py`, `src/utils/figures_ext.py` |
| `src/report.py` | CLI | Old markdown report | old README | old report | legacy | yes | `src/tables_paper.py` |
| `src/validate_paper_evidence.py` | CLI | Paper-evidence manifest validation | paper workflow | validation report | canonical | no | — |
| `src/validate_results.py` | CLI | Result schema/existence validation | validation workflow | result validation report | supporting canonical | no | — |
| `src/capture_hardware.py` | CLI | CPU environment plus passive GPU-presence disclosure | reproduction workflow | hardware metadata | supporting canonical | no | — |
| `src/run_device.py` | CLI | CPU index latency measurement | revision pipeline | latency JSON | supporting canonical | no | — |
| `src/energy_measurement.py`, `src/run_energy_measurement.py` | library/CLI | Optional CPU energy measurement | optional analysis | energy CSV | supporting | no | remove GPU/NVML fields only |
| `src/run_optional_ann_backend_comparison.py` | CLI | Optional CPU ANN backends | optional analysis | backend comparison | supporting | no | — |
| `src/backends/` | package | CPU FAISS and optional ANN adapters | runners | in-memory indexes | supporting canonical | no | — |
| `src/datasets/` | package | Dataset loaders and normalization | preparation | normalized frames | supporting canonical | no | — |
| `src/utils/` | package | Shared paths, metrics, split, reporting and ANN I/O | canonical scripts | shared behavior | supporting canonical | no | — |
| `configs/main_cpu.yml` | config | Canonical scientific defaults | revision orchestrator | — | canonical | no | — |
| `configs/calibration_thresholds.yml`, `metrics.yml`, `budgets.yml` | config | Analysis/metric/budget policy | canonical analyses | — | canonical | no | — |
| `configs/embedding_backbones.yml`, `scale_stress.yml`, `decision_framework.yml`, `hardware_profiles.yml` | config | Paper-analysis settings | corresponding canonical scripts | — | canonical | no | — |
| `legacy/experimental_gpu/` | code/config/results | Incomplete exploratory FAISS-GPU runner and hardware probe | no canonical import/caller | obsolete GPU outputs | experimental, unused | yes | none; GPU is out of scope |
| `README.md`, `reproduction.md`, `docs/` | documentation | Artifact usage and audit record | users/reviewers | — | canonical after cleanup | no | remove stale commands |
| `docs.md` | documentation | Old pipeline notes | no canonical caller | — | legacy | yes | README, reproduction, `docs/` |
| `run_all.sh`, `run_all.ps1`, `tasks.sh` | shell entry points | Duplicated old orchestration | users following old docs | mixed old outputs | legacy | yes | documented canonical Python sequence |
| `prepare_all_datasets.sh`, `prepare_all_datasets.ps1` | shell entry points | Dataset preparation convenience | manual use | normalized datasets | supporting | no | — |
| `results/` existing contents | generated evidence | Outputs from earlier runs in inconsistent folders | paper consumers | tables, figures, JSON, CSV, metadata | stale generated data | yes, explicitly required | fresh canonical directory skeleton |
| `results_after_repo_changes.zip`, `exposure_log.txt` | generated artifact/log | Snapshot/log from earlier work | no canonical caller | stale copies/log | legacy | yes | fresh canonical runs |
| `requirements-cpu.txt` | dependency manifest | Canonical dependencies | installation | environment | canonical | no | — |
| `requirements-optional.txt` | dependency manifest | Optional CPU backends/embeddings | optional modules | environment | supporting | no | remove PyNVML/GPU packages |
| `requirements.txt` | dependency manifest | Historical combined install | old documentation | environment | legacy compatibility | yes | `requirements-cpu.txt` |

## GPU-specific classification

| Path or code path | Classification | Decision |
| --- | --- | --- |
| `legacy/experimental_gpu/src/run_gpu_experiments.py` | experimental, partial, unused | delete |
| `legacy/experimental_gpu/src/capture_gpu_hardware.py` | experimental, unused | delete |
| `legacy/experimental_gpu/configs/gpu_experiments.yml` | experimental, unused | delete |
| `legacy/experimental_gpu/docs/gpu_experiment_protocol.md` | stale experimental documentation | delete |
| `legacy/experimental_gpu/results/gpu_experiments/` | stale generated output | delete with all prior results |
| GPU branches formerly in `build_index.py`, `run_device.py`, `calibrate.py`, `utils/ann_io.py` | experimental, unused | already absent from active CPU behavior; verify no flags/imports remain |
| GPU/NVML energy fields in `energy_measurement.py` | partial optional feature, not required by CPU energy analysis | remove; retain CPU energy measurement |
| passive presence reporting in `capture_hardware.py` | required by environment disclosure | retain without a required GPU dependency |

