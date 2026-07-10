# Validation report

- Artifacts checked: 16
- OK: 13
- Missing (required): 0
- Missing (optional): 0 (allowed)
- Schema mismatches / empty / unreadable: 3

| artifact | stage | required | status | rows | missing_fields | detail |
| --- | --- | --- | --- | --- | --- | --- |
| results/hardware/hardware.json | capture_hardware.py | True | schema_mismatch | nan | cuda_available;faiss_gpu_available |  |
| results/main/summary_main.csv | run_revision_experiments.py | True | ok | 40.0000 |  |  |
| results/main/run_config.json | run_revision_experiments.py | True | ok | nan |  |  |
| results/calibration_sensitivity/calibration_sensitivity.csv | run_calibration_sensitivity.py | True | ok | 27.0000 |  |  |
| results/bootstrap/bootstrap_cis.csv | bootstrap_significance.py | True | ok | 280.0000 |  |  |
| results/bootstrap/paired_tests.csv | bootstrap_significance.py | True | ok | 192.0000 |  |  |
| results/effect_sizes/effect_sizes.csv | effect_size_tables.py | True | ok | 192.0000 |  |  |
| results/deployment_guidance/ann_decision_framework_scores.csv | ann_decision_framework.py | True | schema_mismatch | 30.0000 | quality_retention;latency_score;memory_score;exposure_score |  |
| results/paper_tables/claim_support_audit.csv | claim_support_audit.py | True | ok | 11.0000 |  |  |
| results/pq_diagnostics/pq_diagnostics_all.csv | run_pq_diagnostics.py | False | ok | 188.0000 |  |  |
| results/exposure_analysis/exposure_analysis_all.csv | run_exposure_analysis.py | False | ok | 630.0000 |  |  |
| results/embedding_sensitivity/embedding_backbone_sensitivity_all.csv | run_embedding_backbone_sensitivity.py | False | ok | 20.0000 |  |  |
| results/scale_stress/scale_stress_all.csv | run_scale_stress.py | False | schema_mismatch | 75.0000 | quality_metric;quality_notes |  |
| results/optional_backends/optional_ann_backend_comparison.csv | run_optional_ann_backend_comparison.py | False | ok | 4.0000 |  |  |
| results/energy/energy_measurement_all.csv | run_energy_measurement.py | False | ok | 30.0000 |  |  |
| results/scaling/scaling.csv | synthetic_scaling.py | False | ok | 8.0000 |  |  |
