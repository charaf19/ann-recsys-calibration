# Repository cleanup report — final stabilization and professionalization

## Starting point

- Starting Git commit: `7621e85ec87601034651f859add818c1d1f90038`
- Branch: `cleanup/final-professionalization` (default branch untouched)
- Intermediate commit `b7b9a033019e` recorded the deletion/tree-restructure
  batch; the refactor batch follows it on the same branch.
- Initial known issues (from the pre-cleanup audit):
  - documentation contained invalid commands (`bootstrap_significance.py
    --iterations`, bare commands missing required arguments);
  - bootstrap defaulted to 1000 instead of the paper-required 2000;
  - Amazon Books was "optional" in `configs/main_cpu.yml` despite being
    required for the revised paper;
  - scientific defaults were duplicated/hardcoded (index parameters, BM25
    constants, calibration query count) across scripts;
  - result writing was non-atomic and unsafe for partial/repeated runs;
  - the result-directory migration was incomplete (legacy `RESULTS` keys,
    `results/archive/`, status files under `results/_meta/status/`);
  - two validators were active (`validate_results.py` +
    `validate_paper_evidence.py`);
  - the strict validator checked file existence more than evidence
    completeness;
  - a generated data artifact (`data/synth_smoke.csv`) was still committed;
  - `first_existing()` silently fell back to legacy result filenames;
  - required configuration/provenance/result-I/O/report files were missing.

## Files deleted

| path | reason | canonical replacement |
| --- | --- | --- |
| `configs/budgets.yml` | budget presets unused by canonical scripts; index params centralized | `configs/defaults.yml` (`index:`, `retrieval.budget_mb`) |
| `configs/metrics.yml` | metric registry duplicated code constants | `src/utils/metrics.py` (single source) |
| `configs/calibration_thresholds.yml` | duplicated the main experiment contract with only 3 datasets | `configs/main_cpu.yml` (`calibration:` inherited from defaults) |
| `configs/hardware_profiles.yml` | descriptive profiles, not configuration | `docs/hardware_protocol.md` |
| `configs/decision_framework.yml` | separate experiment config | `configs/analyses.yml` (`decision_framework:`) |
| `configs/embedding_backbones.yml` | separate experiment config | `configs/analyses.yml` (`embedding_sensitivity:`) |
| `configs/scale_stress.yml` | separate experiment config | `configs/analyses.yml` (`scale_stress:`) |
| `prepare_all_datasets.ps1` / `.sh` | second dataset workflow | documented `prepare_dataset.py` commands (README §7) |
| `src/download_datasets.py` | bulk downloader duplicating `prepare_dataset.py` | `src/prepare_dataset.py` |
| `src/clean_data_artifacts.py` | ad-hoc deletion utility, no canonical caller | none (out of scope) |
| `docs/pre_cleanup_inventory.md` | superseded historical inventory | this report |
| `data/synth_smoke.csv` | committed generated artifact of the removed legacy smoke tests | none (data/ is fully gitignored) |
| `results/archive/` | not part of the canonical structure; never a valid input | removed; four top-level result dirs only |

Pre-deletion checks per A.4: import search, subprocess-call search, and
documentation-reference search found no canonical consumer for any deleted
file (only mutual references among the deleted files themselves and the
historical inventory).

Note: the legacy pipeline files named in the audit (`run_grid.py`,
`eval_end2end.py`, `figures.py`, `report.py`, `deployment_guidance.py`,
`synthetic_scaling.py`, `sweep.py`, `smoke_tests.py`, `generate_synth.py`,
`run_all.*`, `tasks.sh`, `legacy/experimental_gpu/`, GPU configs/scripts)
had already been deleted in earlier cleanup commits; this pass verified
their absence and removed the remaining documentation references
(`tests/test_no_legacy_references.py` now guards against reintroduction).

## Files created

| path | purpose |
| --- | --- |
| `configs/defaults.yml` | single source of every scientific default |
| `configs/analyses.yml` | embedding-sensitivity / scale-stress / decision-framework protocol |
| `configs/paper_evidence_manifest.yml` | the strict evidence contract (validator input) |
| `configs/test_tiny.yml` | tiny structural test config (explicitly NOT for paper reproduction) |
| `src/utils/config.py` | shared loader: `inherits`, deep merge, typed access, hashing, CLI overrides |
| `src/utils/result_io.py` | atomic writes, `fail_if_exists`/`replace`/`merge`, key validation, conflict detection |
| `src/utils/provenance.py` | run manifest, hardware/environment capture, `.sources.json` sidecars |
| `tests/conftest.py` + 11 `tests/test_*.py` | regression suite (config, paths, I/O, split, modalities, agreement/relevance separation, validator, CPU scope, legacy scan, CLI contracts, scientific defaults) |
| `docs/repository_cleanup_report.md` | this report |

## Files modified

| path | change | scientific semantics affected? |
| --- | --- | --- |
| `configs/main_cpu.yml` | rewritten: inherits defaults; **amazon-books now required** (flat list, no `optional:` tier) | no (grid completion, not protocol change) |
| `src/utils/paths.py` | canonical `RESULTS` mapping per contract; removed `first_existing()` and legacy keys; added `results_path()`/`require_input()` | no |
| `src/run_revision_experiments.py` | config-driven (no hardcoded index/BM25/calibration values), `--write_mode`, `--allow_missing_datasets` fail-fast, new output layout with `d{dim}` filenames, provenance columns + run manifest, non-zero exit on failures | no (identical values now sourced from config) |
| `src/eval_modalities.py` | aggregates to `results/main/aggregates/`, `d{dim}` in filenames, `dim` in npz metadata | no |
| `src/calibrate.py` | canonical default output filename (`__d{dim}__…__target_…`) | no |
| `src/bootstrap_significance.py` | **default n_boot 1000 → 2000 (bug fix)**, `--config` reads `statistics.bootstrap_iterations`, rejects non-positive counts, `--write_mode`, dim-aware keys, `seed` column | yes — bug fix; full bootstrap rerun required (perquery inputs unchanged) |
| `src/effect_size_tables.py` | shared loader, `--write_mode`, canonical output only (presentation moved to tables_paper) | no |
| `src/run_calibration_sensitivity.py` | reads `main_cpu.yml` (all 4 datasets → 36 rows), fail-fast on missing inputs, `dim` column, `--write_mode` | no (protocol identical; grid completed) |
| `src/run_embedding_backbone_sensitivity.py` | reads `analyses.yml`, config-resolved index params, queries 2000 → **10000 per the revised-paper protocol**, `--write_mode`, no paper-artifact writes | yes — protocol value now matches the revised-paper contract; rerun required |
| `src/run_scale_stress.py` | reads `analyses.yml` (`dimensions:` key), config-resolved index params (**IVF-PQ now built with OPQ, consistent with the main pipeline**), seeded/omp-threaded builds, atomic checkpoints, `--write_mode` | yes — consistency fix for ivfpq cells; scale-stress rerun required (all results were deleted anyway) |
| `src/run_exposure_analysis.py` | output to `results/analyses/exposure/`, `dim`/`seed` columns, `--write_mode`, no paper-artifact writes | no |
| `src/run_pq_diagnostics.py` | `--config`, canonical summary input (no fallback), `--write_mode`, writes long+summary CSVs, no paper-artifact writes | no |
| `src/ann_decision_framework.py` | reads `analyses.yml` section, canonical inputs only (no `first_existing`, no `@k` fallbacks — missing columns are an error), `--write_mode` | no (scoring formula unchanged) |
| `src/run_energy_measurement.py` / `src/run_optional_ann_backend_comparison.py` | result_io writes, `--write_mode`, key columns; no paper-artifact writes | no |
| `src/capture_hardware.py` | rewritten over provenance module; outputs to `results/_meta/`; canonical `accelerator_present` / `main_experiments_accelerator_used=false` fields | no |
| `src/validate_paper_evidence.py` | rewritten as the single strict manifest-driven validator (completeness, counts, uniform values, n_boot, cost-only scale, CPU scope); reports to `results/_meta/validation_report.{csv,md,json}` | n/a (meta) |
| `src/validate_results.py` | reduced to a deprecated wrapper delegating to the canonical validator | n/a |
| `src/claim_support_audit.py` | consumes the validation report (sections must PASS; file existence is not evidence) | n/a |
| `src/tables_paper.py` / `src/figures_paper.py` | canonical inputs only, `.sources.json` sidecars, `--write_mode`, critical-vs-optional input handling | no (presentation) |
| `src/dataset_stats.py` | RESULTS-based default, source sidecar | no |
| `src/utils/common.py` | fixed malformed first line (stray `\`) into a docstring | no |
| `README.md` / `reproduction.md` | rewritten around the single canonical workflow; removed the invalid `--iterations` command | n/a |
| `docs/*` (contract, schemas, protocols, limitations, randomness) | updated to the new paths, fields, and required-Amazon-Books policy | n/a |
| `.gitignore` | explicit rules (no global `*.csv` + negation pattern); ignores `data/`, perquery arrays, rendered figures; retains .gitkeep/CSV/JSON/MD/TeX evidence | n/a |

## Config migration

| old config | new canonical location |
| --- | --- |
| `main_cpu.yml` (flat, optional-tier datasets) | `main_cpu.yml` (inherits `defaults.yml`; 4 required datasets) |
| `calibration_thresholds.yml` | `defaults.yml` `calibration:` + `main_cpu.yml` datasets (param grids stay in `utils/ann_io.py DEFAULT_PARAM_GRIDS`) |
| `embedding_backbones.yml` | `analyses.yml` `embedding_sensitivity:` |
| `scale_stress.yml` | `analyses.yml` `scale_stress:` |
| `decision_framework.yml` | `analyses.yml` `decision_framework:` |
| `budgets.yml` | `defaults.yml` `index:` / `retrieval.budget_mb` |
| `metrics.yml` | `src/utils/metrics.py` |
| `hardware_profiles.yml` | `docs/hardware_protocol.md` |

## Result structure migration

| old path | new path |
| --- | --- |
| `results/_meta/hardware/hardware.{json,md}` | `results/_meta/hardware.{json,md}` (+ `environment.txt`, `run_manifest.json`) |
| `results/_meta/status/status_{ds}_{m}.json` | `results/main/status/{ds}__{w}__d{dim}__{m}.status.json` |
| `results/_meta/validation/` | `results/_meta/validation_report.{csv,md,json}` |
| `results/main/{ds}__{w}__{mod}__{m}.json` | `results/main/aggregates/{ds}__{w}__d{dim}__{mod}__{m}.json` |
| `results/main/perquery/{ds}__{w}__{mod}__{m}.npz` | `results/main/perquery/{ds}__{w}__d{dim}__{mod}__{m}.npz` |
| `results/main/calibration/…__t{t}.json` | `results/main/calibration/{ds}__{w}__d{dim}__{m}__target_{t:.2f}.json` |
| `results/analyses/exposure_analysis/` | `results/analyses/exposure/` |
| `results/analyses/ann_decision_framework/` | `results/analyses/decision_framework/` |
| `results/archive/` | removed |

No result values were migrated: the results tree was already empty and is
recreated as the empty canonical skeleton (`.gitkeep` files only).

## Scientific integrity

| component | canonical implementation | numerics changed? | output path changed? | rerun required? | test covering it |
| --- | --- | --- | --- | --- | --- |
| dataset preparation | `prepare_dataset.py` + `src/datasets/` | no | no | as part of fresh run | `test_cli_contracts` (indirect) |
| dataset statistics | `dataset_stats.py` | no | no | yes (fresh evidence) | — |
| temporal split | `utils/splits.py` | no | n/a | n/a | `test_temporal_split` |
| BM25 weighting / SVD embedding | `train_embeddings.py` (untouched; params now config-fed) | no | no | yes (fresh evidence) | `test_scientific_defaults` |
| index construction | `build_index.py` (untouched); orchestrator `build_index_cmd()` config-driven | no (identical parameters) | no | yes | `test_scientific_defaults::test_index_construction_contract` |
| exact Flat reference | `utils/ann_io.build_exact_index` | no | n/a | n/a | `test_agreement_relevance_separation` |
| calibration | `calibrate.py` | no | detail filename | yes | `test_scientific_defaults::test_calibration_*` |
| U2I / I2I evaluation | `eval_modalities.py` | no | aggregates dir + `d{dim}` names | yes | `test_modalities` |
| latency measurement | `run_device.py` (untouched) | no | no | yes | — |
| bootstrap CIs / paired tests | `bootstrap_significance.py` | **yes: default 1000→2000 (bug fix)** | dim-aware keys, same files | **yes** | `test_config_loading::test_bootstrap_default_resolves_to_2000`, `test_cli_contracts` |
| effect sizes | `effect_size_tables.py` | no | same file | yes | `test_paper_evidence_validator` |
| embedding sensitivity | `run_embedding_backbone_sensitivity.py` | **yes: queries 2000→10000 (revised-paper protocol)** | same file | **yes** | `test_scientific_defaults::test_embedding_sensitivity_protocol` |
| exposure analysis | `run_exposure_analysis.py` | no | `analyses/exposure/` | yes | `test_paper_evidence_validator` |
| PQ diagnostics | `run_pq_diagnostics.py` | no | + summary CSV in analyses | yes | `test_paper_evidence_validator` |
| scale stress | `run_scale_stress.py` | **ivfpq cells: OPQ now applied (consistency fix)** | same file | **yes** | `test_scientific_defaults::test_scale_stress_grid_is_75_cost_only_cells` |
| decision framework | `ann_decision_framework.py` | no (formula unchanged) | `analyses/decision_framework/` | yes | `test_scientific_defaults::test_decision_framework_weights` |
| claim-support audit | `claim_support_audit.py` | n/a (meta) | same table | after validation | `test_cli_contracts` |
| tables / figures | `tables_paper.py` / `figures_paper.py` | n/a (presentation) | sidecars added | after evidence | `test_cli_contracts` |
| paper-evidence validation | `validate_paper_evidence.py` | n/a (meta) | `_meta/validation_report.*` | run last | `test_paper_evidence_validator` |

Every "yes" in the numerics column is one of the three explicitly
identified fixes (bootstrap 2000, embedding-sensitivity 10000 queries,
scale-stress OPQ consistency); all other components produce numerically
identical results from the same inputs. Because ALL old results were
deleted intentionally, every experiment requires a fresh run regardless.

## Verification

| check | result |
| --- | --- |
| `python -m py_compile` over all 62 active src/ + tests/ files | 0 failures |
| `python -m pytest tests -q` | **83 passed** |
| YAML parsing of all 5 configs | all parse as mappings |
| CLI `--help` (run_revision_experiments, run_calibration_sensitivity, bootstrap_significance, run_embedding_backbone_sensitivity, run_scale_stress, validate_paper_evidence, + 7 more via tests) | all exit 0 with canonical flags advertised |
| legacy pipeline reference scan (README, reproduction, docs, configs, src) | none (guarded by `test_no_legacy_references`) |
| GPU execution reference scan (`--use_gpu`, `faiss-gpu`, `index_cpu_to_gpu`, `StandardGpuResources`, `pynvml`, `gpu_energy_joules`) | none (guarded by `test_cpu_only_scope`) |
| `first_existing` / legacy result-path scan | none |
| strict validator vs deliberately empty `results/` | FAIL with 11 missing-evidence checks, exit 1 — **the expected pre-rerun state**, not a cleanup failure; no placeholder evidence was created |
| deprecated `validate_results.py` | emits DeprecationWarning and delegates to the canonical validator |

## Paper evidence status

**READY WITH RERUN**

- repository structure, configuration, validation, and provenance are ready;
- scientific semantics are preserved (three explicit bug/protocol fixes
  documented above);
- result directories are intentionally empty;
- the final experiments must now be executed via the canonical workflow in
  `README.md` §8 / `reproduction.md`.

No scientific experiment was executed during this cleanup: no dataset was
downloaded or processed, no embedding trained, no index built, no
calibration/evaluation/bootstrap/sensitivity/stress run, and no numerical
output was generated or edited. The only pipeline invocations were
`--help` checks, tiny-fixture tests in pytest tmp directories, and one
strict-validator run against the empty results tree (whose report files
were removed afterwards to keep the tree pristine).
