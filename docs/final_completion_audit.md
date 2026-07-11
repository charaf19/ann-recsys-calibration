# Final completion audit — producer vs evidence-manifest contract

Static contract review (no experiment executed): for every critical
artifact, the producer's output path, columns, and natural key were compared
against `configs/paper_evidence_manifest.yml` (contract_version 3) and
against how `src/validate_paper_evidence.py` interprets that section.

"Schema match" means the producer emits **every manifest-required column**
(producers may add extra provenance/diagnostic columns; the validator only
requires, never forbids, except for the GPU column blacklist). "Key match"
means the producer's merge/uniqueness key equals the manifest key as a set.

| Artifact | Producer | Producer path | Manifest path | Path match? | Producer columns (vs required) | Schema match? | Natural key match? | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main summary | `run_revision_experiments.py` | `results/main/summary_main.csv` | same | yes | all 21 required + `budget_mb`, `calibration_target`, `calibrated_param_*`, `rss_mb_after`, `omp_threads`, provenance cols | yes | `MAIN_KEY` = manifest `main.key` | OK |
| run_config | `run_revision_experiments.run_config_path()` | `results/main/run_config.json` | `results/main/run_config.json` | **yes — fixed this pass** (was `results/_meta/run_config.json`) | `datasets`, `modalities`, `methods`, `weighting`, `dim`, `seed` + full resolved values | yes | n/a (JSON) | FIXED |
| calibration sensitivity | `run_calibration_sensitivity.py` | `results/analyses/calibration_sensitivity/calibration_sensitivity.csv` | same | yes | all 13 required + `latency_mean_ms`, `topk` | yes | `KEY` = manifest key | OK |
| bootstrap CIs | `bootstrap_significance.py` | `results/analyses/bootstrap/bootstrap_cis.csv` | same | yes | all 12 required (`**bootstrap_ci` supplies mean/ci_low/ci_high/n/n_boot) + `evaluation_seed` | yes | **fixed this pass**: `CI_KEY` was missing `dim` | FIXED |
| paired tests | `bootstrap_significance.py` | `results/analyses/bootstrap/paired_tests.csv` | same | yes | all 14 required + `significant_at_0.05`, `evaluation_seed` | yes | **fixed this pass**: `TEST_KEY` was missing `dim` | FIXED |
| effect sizes | `effect_size_tables.py` | `results/analyses/effect_sizes/effect_sizes.csv` | same | yes | all 14 required + `evaluation_seed` | yes | `KEY` = manifest key | OK |
| embedding sensitivity | `run_embedding_backbone_sensitivity.py` | `results/analyses/embedding_sensitivity/embedding_backbone_sensitivity_all.csv` | same | yes | all 13 required + `embedding_backend`, `long_tail_uplift` | yes | `KEY` = manifest key | OK |
| exposure analysis | `run_exposure_analysis.py` + `exposure_analysis.analyze_run` | `results/analyses/exposure/exposure_analysis_all.csv` | same | yes | all 12 required; library emits all 7 required metrics (`long_tail_exposure`, `long_tail_uplift`, `gini_exposure`, `coverage`, `exposure_share_decile`, `exposure_share_group`, `user_popularity_calibration_error`) | yes | **fixed this pass**: `KEY` was missing `dim` | FIXED |
| PQ diagnostics | `run_pq_diagnostics.py` | `results/analyses/pq_diagnostics/pq_diagnostics_all.csv` | same | yes | all 8 required; emits the 3 required metrics per dataset×method | yes | `KEY` = manifest key | OK |
| scale stress | `run_scale_stress.py` | `results/analyses/scale_stress/scale_stress_all.csv` | same | yes | all 13 required + `rss_mb_delta`, `param_name`, `calibrated_param_value`, `quality_metric`, `quality_notes`, `config_hash`; no forbidden quality columns | yes | `KEY` = manifest key | OK |
| decision framework | `ann_decision_framework.py` | `results/analyses/decision_framework/ann_decision_framework_scores.csv` | same | yes | all 18 required (incl. exact weights as columns) + audit columns | yes | **fixed this pass**: `KEY` was missing `seed` | FIXED |
| dataset statistics | `dataset_stats.py` | `results/paper/tables/dataset_stats.csv` | same | yes | exactly the 11 required columns | yes | `dataset` | OK |
| hardware | `utils/provenance.write_hardware_report` | `results/_meta/hardware.json` | same | yes | all 10 required fields; `main_experiments_accelerator_used=false`; no forbidden GPU fields | yes | n/a (JSON) | OK |
| run manifest | `utils/provenance.RunManifest` | `results/_meta/run_manifest.json` | same | yes | all 17 required fields (`git_dirty_state`, `dependency_versions`, `resolved_configuration`, `configuration_hash`, `outputs`, `failed_combinations`, …); `configuration_hash` computed by the same `utils.config.config_hash` the validator recomputes; `run_id` ends with the hash | yes | n/a (JSON) | OK |
| claim-support audit | `claim_support_audit.py` | `results/paper/tables/claim_support_audit.csv` | same | yes (manifest `status: optional`) | 7 required columns emitted verbatim | yes | `claim_area` | OK (optional; absence never flips the critical verdict — it consumes the validation report, so requiring it would be circular) |

## Fixes applied in this pass

1. `run_config.json` producer path corrected from `results/_meta/` to the
   canonical `results/main/run_config.json`, exposed through the shared
   helper `run_revision_experiments.run_config_path()`; a regression test
   (`tests/test_result_paths.py::test_run_config_canonical_path`) asserts
   that the helper and `main.run_config_file` in the manifest agree, and a
   validator test asserts that a run_config left at the legacy `_meta`
   location fails validation.
2. Natural keys aligned to the manifest: `dim` added to
   `bootstrap_significance.CI_KEY` / `TEST_KEY` and
   `run_exposure_analysis.KEY`; `seed` added to
   `ann_decision_framework.KEY`. These are merge/uniqueness keys only —
   no numerical semantics changed.
3. Validator passing fixture rebuilt **from the manifest** (datasets,
   methods, modalities, metrics, grids, counts, required columns and
   required values are read from `configs/paper_evidence_manifest.yml`,
   not duplicated in the test), covering every critical artifact including
   hardware, run manifest, decision framework, dataset statistics, the
   280-row CI / 192-row paired-test bootstrap files, and a ≥840-row
   exposure fixture.
4. Stale test references corrected: `sections["cpu_scope"]` →
   `sections["hardware"]` / `sections["scope"]`;
   `scope.require_fields` → `hardware.required_values` in the CPU-scope
   test; `cfg_get` boolean-coercion errors now surface the underlying
   "boolean token" reason the test asserts.
