# Paper evidence validation report

- Manifest: `configs\paper_evidence_manifest.yml`
- Checks: 245
- Critical failures: **0**
- Optional failures: **0**
- Verdict: **PASS**

| section | severity | associated_paper_evidence | check | status | detail |
| --- | --- | --- | --- | --- | --- |
| manifest_contract | critical | Evidence-contract integrity | manifest_is_mapping | pass |  |
| manifest_contract | critical | Evidence-contract integrity | section_scope_status | pass | status='critical' |
| manifest_contract | critical | Evidence-contract integrity | section_scope_paper_evidence | pass | associated_paper_evidence=['CPU-only scope'] |
| manifest_contract | critical | Evidence-contract integrity | section_main_status | pass | status='critical' |
| manifest_contract | critical | Evidence-contract integrity | section_main_paper_evidence | pass | associated_paper_evidence=['Main quality and efficiency results', 'Larger-catalog generalization', 'U2I versus I2I comparison'] |
| manifest_contract | critical | Evidence-contract integrity | section_calibration_sensitivity_status | pass | status='critical' |
| manifest_contract | critical | Evidence-contract integrity | section_calibration_sensitivity_paper_evidence | pass | associated_paper_evidence=['Calibration-sensitivity analysis'] |
| manifest_contract | critical | Evidence-contract integrity | section_embedding_sensitivity_status | pass | status='critical' |
| manifest_contract | critical | Evidence-contract integrity | section_embedding_sensitivity_paper_evidence | pass | associated_paper_evidence=['Embedding-backbone sensitivity'] |
| manifest_contract | critical | Evidence-contract integrity | section_scale_stress_status | pass | status='critical' |
| manifest_contract | critical | Evidence-contract integrity | section_scale_stress_paper_evidence | pass | associated_paper_evidence=['Production-scale cost analysis'] |
| manifest_contract | critical | Evidence-contract integrity | section_bootstrap_status | pass | status='critical' |
| manifest_contract | critical | Evidence-contract integrity | section_bootstrap_paper_evidence | pass | associated_paper_evidence=['Paired statistical significance'] |
| manifest_contract | critical | Evidence-contract integrity | section_effect_sizes_status | pass | status='critical' |
| manifest_contract | critical | Evidence-contract integrity | section_effect_sizes_paper_evidence | pass | associated_paper_evidence=['Paired effect-size analysis'] |
| manifest_contract | critical | Evidence-contract integrity | section_exposure_status | pass | status='critical' |
| manifest_contract | critical | Evidence-contract integrity | section_exposure_paper_evidence | pass | associated_paper_evidence=['Long-tail exposure and popularity-proxy analysis'] |
| manifest_contract | critical | Evidence-contract integrity | section_pq_diagnostics_status | pass | status='critical' |
| manifest_contract | critical | Evidence-contract integrity | section_pq_diagnostics_paper_evidence | pass | associated_paper_evidence=['Product-quantization diagnostics'] |
| manifest_contract | critical | Evidence-contract integrity | section_decision_framework_status | pass | status='critical' |
| manifest_contract | critical | Evidence-contract integrity | section_decision_framework_paper_evidence | pass | associated_paper_evidence=['ANN selection decision framework'] |
| manifest_contract | critical | Evidence-contract integrity | section_dataset_stats_status | pass | status='critical' |
| manifest_contract | critical | Evidence-contract integrity | section_dataset_stats_paper_evidence | pass | associated_paper_evidence=['Dataset statistics'] |
| manifest_contract | critical | Evidence-contract integrity | section_hardware_status | pass | status='critical' |
| manifest_contract | critical | Evidence-contract integrity | section_hardware_paper_evidence | pass | associated_paper_evidence=['CPU-only scope', 'Hardware provenance'] |
| manifest_contract | critical | Evidence-contract integrity | section_run_manifest_status | pass | status='critical' |
| manifest_contract | critical | Evidence-contract integrity | section_run_manifest_paper_evidence | pass | associated_paper_evidence=['Full-run provenance', 'CPU-only scope'] |
| manifest_contract | critical | Evidence-contract integrity | section_claim_support_audit_status | pass | status='optional' |
| manifest_contract | critical | Evidence-contract integrity | section_claim_support_audit_paper_evidence | pass | associated_paper_evidence=['Claim-to-evidence traceability table'] |
| scope | critical | CPU-only scope | cpu_only_contract | pass | cpu_only=True |
| scope | critical | CPU-only scope | forbidden_path_absent:results/gpu_experiments | pass | forbidden result path absent: results\gpu_experiments |
| scope | critical | CPU-only scope | forbidden_path_absent:results/analyses/gpu_experiments | pass | forbidden result path absent: results\analyses\gpu_experiments |
| scope | critical | CPU-only scope | no_gpu_columns_in_canonical_csvs | pass | none found |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | file_exists | pass | results\main\summary_main.csv |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | run_config_file_exists | pass | results\main\run_config.json |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | run_config_weighting | pass | expected weighting='bm25', found 'bm25' |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | run_config_dim | pass | expected dim=128, found 128 |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | run_config_seed | pass | expected seed=42, found 42 |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | run_config_datasets_exact | pass | expected=['ml-1m', 'ml-20m', 'goodbooks', 'amazon-books'], found=['ml-1m', 'ml-20m', 'goodbooks', 'amazon-books'] |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | run_config_modalities_exact | pass | expected=['u2i', 'i2i'], found=['u2i', 'i2i'] |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | run_config_methods_exact | pass | expected=['flat', 'hnsw', 'ivfflat', 'ivfpq', 'flatpq'], found=['flat', 'hnsw', 'ivfflat', 'ivfpq', 'flatpq'] |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | required_columns | pass | all present |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_duplicate_keys | pass | 0 rows have duplicate key ['dataset', 'weighting', 'dim', 'modality', 'method', 'seed']; sample=[] |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | uniform_weighting | pass | weighting='bm25' on every row |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | uniform_dim | pass | dim=128 on every row |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | uniform_seed | pass | seed=42 on every row |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | exact_grid | pass | expected=40, actual=40, missing=0 sample_missing=[], unexpected=0 sample_unexpected=[] |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | expected_row_count | pass | expected 40, found 40 |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_queries | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_recall_at_k_mean | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_precision_at_k_mean | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_hr_at_k_mean | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_ndcg_at_k_mean | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_map_at_k_mean | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_mrr_at_k_mean | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_ann_recall_vs_exact_at_k_mean | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_recall_at_100_mean | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_ann_recall_vs_exact_at_100_mean | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_coverage_at_k | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_gini_exposure | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_long_tail_exposure | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_long_tail_uplift | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_latency_p50_ms | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_nulls_latency_p95_ms | pass | 0 null value(s) |
| main | critical | Main quality and efficiency results; Larger-catalog generalization; U2I versus I2I comparison | no_gpu_columns | pass | forbidden columns present: [] |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | file_exists | pass | results\analyses\calibration_sensitivity\calibration_sensitivity.csv |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | required_columns | pass | all present |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | no_duplicate_keys | pass | 0 rows have duplicate key ['dataset', 'weighting', 'dim', 'modality', 'method', 'target_recall', 'seed']; sample=[] |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | uniform_weighting | pass | weighting='bm25' on every row |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | uniform_dim | pass | dim=128 on every row |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | uniform_seed | pass | seed=42 on every row |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | exact_grid | pass | expected=72, actual=72, missing=0 sample_missing=[], unexpected=0 sample_unexpected=[] |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | expected_row_count | pass | expected 72, found 72 |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | no_nulls_target_reached | pass | 0 null value(s) |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | no_nulls_param_name | pass | 0 null value(s) |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | no_nulls_selected_param_value | pass | 0 null value(s) |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | no_nulls_achieved_recall_vs_exact | pass | 0 null value(s) |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | no_nulls_latency_p50_ms | pass | 0 null value(s) |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | no_nulls_latency_p95_ms | pass | 0 null value(s) |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | no_nulls_n_calibration_queries | pass | 0 null value(s) |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | boolean_target_reached | pass | values=['false', 'true'], nulls=0 |
| calibration_sensitivity | critical | Calibration-sensitivity analysis | uniform_n_calibration_queries | pass | n_calibration_queries=1000 on every row |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | file_exists | pass | results\analyses\embedding_sensitivity\embedding_backbone_sensitivity_all.csv |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | required_columns | pass | all present |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | no_unexpected_backbones | pass | unexpected=[] |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | no_duplicate_keys | pass | 0 rows have duplicate key ['dataset', 'backbone', 'modality', 'method', 'dim', 'seed']; sample=[] |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | exact_grid | pass | expected=20, actual=20, missing=0 sample_missing=[], unexpected=0 sample_unexpected=[] |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | expected_row_count | pass | expected 20, found 20 |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | uniform_dim | pass | dim=128 on every row |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | uniform_seed | pass | seed=42 on every row |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | uniform_backend_available | pass | backend_available=True on every row |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | uniform_status | pass | status='ok' on every row |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | no_nulls_ndcg_at_10 | pass | 0 null value(s) |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | no_nulls_recall_at_10 | pass | 0 null value(s) |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | no_nulls_ann_recall_vs_exact_at_k_mean | pass | 0 null value(s) |
| embedding_sensitivity | critical | Embedding-backbone sensitivity | ranking_stability_reported | pass | bpr_matrix_factorization:5, svd_bm25:5, svd_none:5, svd_tfidf:5 (values are not judged by sign) |
| scale_stress | critical | Production-scale cost analysis | file_exists | pass | results\analyses\scale_stress\scale_stress_all.csv |
| scale_stress | critical | Production-scale cost analysis | required_columns | pass | all present |
| scale_stress | critical | Production-scale cost analysis | no_duplicate_keys | pass | 0 rows have duplicate key ['n_items', 'dim', 'method', 'seed']; sample=[] |
| scale_stress | critical | Production-scale cost analysis | uniform_seed | pass | seed=42 on every row |
| scale_stress | critical | Production-scale cost analysis | exact_grid | pass | expected=20, actual=20, missing=0 sample_missing=[], unexpected=0 sample_unexpected=[] |
| scale_stress | critical | Production-scale cost analysis | expected_row_count | pass | expected 20, found 20 |
| scale_stress | critical | Production-scale cost analysis | no_nulls_build_wall_time_sec | pass | 0 null value(s) |
| scale_stress | critical | Production-scale cost analysis | no_nulls_index_size_mb | pass | 0 null value(s) |
| scale_stress | critical | Production-scale cost analysis | no_nulls_rss_mb_after | pass | 0 null value(s) |
| scale_stress | critical | Production-scale cost analysis | no_nulls_target_reached | pass | 0 null value(s) |
| scale_stress | critical | Production-scale cost analysis | no_nulls_achieved_recall_vs_exact | pass | 0 null value(s) |
| scale_stress | critical | Production-scale cost analysis | no_nulls_latency_p50_ms | pass | 0 null value(s) |
| scale_stress | critical | Production-scale cost analysis | no_nulls_latency_p95_ms | pass | 0 null value(s) |
| scale_stress | critical | Production-scale cost analysis | boolean_target_reached | pass | values=['false', 'true'], nulls=0 |
| scale_stress | critical | Production-scale cost analysis | required_value_quality_measured | pass | quality_measured=False on every row |
| scale_stress | critical | Production-scale cost analysis | no_synthetic_recommendation_quality | pass | quality columns containing values: [] |
| bootstrap | critical | Paired statistical significance | cis_file_exists | pass | results\analyses\bootstrap\bootstrap_cis.csv |
| bootstrap | critical | Paired statistical significance | tests_file_exists | pass | results\analyses\bootstrap\paired_tests.csv |
| bootstrap | critical | Paired statistical significance | cis_required_columns | pass | all present |
| bootstrap | critical | Paired statistical significance | cis_no_duplicate_keys | pass | 0 rows have duplicate key ['dataset', 'weighting', 'dim', 'modality', 'method', 'metric', 'seed', 'n_boot']; sample=[] |
| bootstrap | critical | Paired statistical significance | cis_weighting | pass | weighting='bm25' on every row |
| bootstrap | critical | Paired statistical significance | cis_dim | pass | dim=128 on every row |
| bootstrap | critical | Paired statistical significance | cis_seed | pass | seed=42 on every row |
| bootstrap | critical | Paired statistical significance | cis_n_boot | pass | n_boot=2000 on every row |
| bootstrap | critical | Paired statistical significance | cis_exact_grid | pass | expected=280, actual=280, missing=0 sample_missing=[], unexpected=0 sample_unexpected=[] |
| bootstrap | critical | Paired statistical significance | cis_expected_row_count | pass | expected 280, found 280 |
| bootstrap | critical | Paired statistical significance | cis_no_nulls_mean | pass | 0 null value(s) |
| bootstrap | critical | Paired statistical significance | cis_no_nulls_ci_low | pass | 0 null value(s) |
| bootstrap | critical | Paired statistical significance | cis_no_nulls_ci_high | pass | 0 null value(s) |
| bootstrap | critical | Paired statistical significance | cis_no_nulls_n | pass | 0 null value(s) |
| bootstrap | critical | Paired statistical significance | cis_no_nulls_n_boot | pass | 0 null value(s) |
| bootstrap | critical | Paired statistical significance | tests_required_columns | pass | all present |
| bootstrap | critical | Paired statistical significance | tests_no_duplicate_keys | pass | 0 rows have duplicate key ['dataset', 'weighting', 'dim', 'modality', 'method', 'baseline', 'metric', 'seed', 'n_boot']; sample=[] |
| bootstrap | critical | Paired statistical significance | tests_weighting | pass | weighting='bm25' on every row |
| bootstrap | critical | Paired statistical significance | tests_dim | pass | dim=128 on every row |
| bootstrap | critical | Paired statistical significance | tests_seed | pass | seed=42 on every row |
| bootstrap | critical | Paired statistical significance | tests_n_boot | pass | n_boot=2000 on every row |
| bootstrap | critical | Paired statistical significance | tests_exact_grid | pass | expected=192, actual=192, missing=0 sample_missing=[], unexpected=0 sample_unexpected=[] |
| bootstrap | critical | Paired statistical significance | tests_expected_row_count | pass | expected 192, found 192 |
| bootstrap | critical | Paired statistical significance | tests_no_nulls_mean_diff | pass | 0 null value(s) |
| bootstrap | critical | Paired statistical significance | tests_no_nulls_ci_low | pass | 0 null value(s) |
| bootstrap | critical | Paired statistical significance | tests_no_nulls_ci_high | pass | 0 null value(s) |
| bootstrap | critical | Paired statistical significance | tests_no_nulls_p_value | pass | 0 null value(s) |
| bootstrap | critical | Paired statistical significance | tests_no_nulls_n | pass | 0 null value(s) |
| bootstrap | critical | Paired statistical significance | tests_no_nulls_n_boot | pass | 0 null value(s) |
| bootstrap | critical | Paired statistical significance | tests_flat_baseline | pass | baseline='flat' on every row |
| effect_sizes | critical | Paired effect-size analysis | file_exists | pass | results\analyses\effect_sizes\effect_sizes.csv |
| effect_sizes | critical | Paired effect-size analysis | required_columns | pass | all present |
| effect_sizes | critical | Paired effect-size analysis | no_duplicate_keys | pass | 0 rows have duplicate key ['dataset', 'weighting', 'dim', 'modality', 'method', 'baseline', 'metric', 'seed']; sample=[] |
| effect_sizes | critical | Paired effect-size analysis | uniform_weighting | pass | weighting='bm25' on every row |
| effect_sizes | critical | Paired effect-size analysis | uniform_dim | pass | dim=128 on every row |
| effect_sizes | critical | Paired effect-size analysis | uniform_seed | pass | seed=42 on every row |
| effect_sizes | critical | Paired effect-size analysis | uniform_baseline | pass | baseline='flat' on every row |
| effect_sizes | critical | Paired effect-size analysis | exact_grid | pass | expected=192, actual=192, missing=0 sample_missing=[], unexpected=0 sample_unexpected=[] |
| effect_sizes | critical | Paired effect-size analysis | expected_row_count | pass | expected 192, found 192 |
| effect_sizes | critical | Paired effect-size analysis | no_nulls_cohens_d | pass | 0 null value(s) |
| effect_sizes | critical | Paired effect-size analysis | no_nulls_cliffs_delta | pass | 0 null value(s) |
| effect_sizes | critical | Paired effect-size analysis | no_nulls_cohens_d_magnitude | pass | 0 null value(s) |
| effect_sizes | critical | Paired effect-size analysis | no_nulls_cliffs_delta_magnitude | pass | 0 null value(s) |
| effect_sizes | critical | Paired effect-size analysis | no_nulls_n | pass | 0 null value(s) |
| exposure | critical | Long-tail exposure and popularity-proxy analysis | file_exists | pass | results\analyses\exposure\exposure_analysis_all.csv |
| exposure | critical | Long-tail exposure and popularity-proxy analysis | required_columns | pass | all present |
| exposure | critical | Long-tail exposure and popularity-proxy analysis | no_duplicate_keys | pass | 0 rows have duplicate key ['dataset', 'weighting', 'dim', 'modality', 'method', 'metric', 'k', 'decile', 'group', 'seed']; sample=[] |
| exposure | critical | Long-tail exposure and popularity-proxy analysis | uniform_weighting | pass | weighting='bm25' on every row |
| exposure | critical | Long-tail exposure and popularity-proxy analysis | uniform_dim | pass | dim=128 on every row |
| exposure | critical | Long-tail exposure and popularity-proxy analysis | uniform_seed | pass | seed=42 on every row |
| exposure | critical | Long-tail exposure and popularity-proxy analysis | minimum_row_count | pass | expected at least 840, found 840 |
| exposure | critical | Long-tail exposure and popularity-proxy analysis | all_runs_present | pass | expected=40, actual=40, missing=0 sample_missing=[], unexpected=0 sample_unexpected=[] |
| exposure | critical | Long-tail exposure and popularity-proxy analysis | required_metrics_per_run | pass | expected=280, actual=280, missing=0 sample_missing=[], unexpected=0 sample_unexpected=[] |
| exposure | critical | Long-tail exposure and popularity-proxy analysis | no_nulls_value | pass | 0 null value(s) |
| exposure | critical | Long-tail exposure and popularity-proxy analysis | no_nulls_fairness_scope | pass | 0 null value(s) |
| exposure | critical | Long-tail exposure and popularity-proxy analysis | scope_restricted_to_proxies | pass | allowed=['long_tail_exposure_proxy_only', 'popularity_calibration_proxy', 'provider_proxy_if_metadata_available'], invalid=[] |
| pq_diagnostics | critical | Product-quantization diagnostics | file_exists | pass | results\analyses\pq_diagnostics\pq_diagnostics_all.csv |
| pq_diagnostics | critical | Product-quantization diagnostics | required_columns | pass | all present |
| pq_diagnostics | critical | Product-quantization diagnostics | no_duplicate_keys | pass | 0 rows have duplicate key ['dataset', 'weighting', 'dim', 'method', 'metric', 'decile', 'seed']; sample=[] |
| pq_diagnostics | critical | Product-quantization diagnostics | uniform_weighting | pass | weighting='bm25' on every row |
| pq_diagnostics | critical | Product-quantization diagnostics | uniform_dim | pass | dim=128 on every row |
| pq_diagnostics | critical | Product-quantization diagnostics | uniform_seed | pass | seed=42 on every row |
| pq_diagnostics | critical | Product-quantization diagnostics | minimum_row_count | pass | expected at least 24, found 248 |
| pq_diagnostics | critical | Product-quantization diagnostics | required_diagnostics_per_dataset_method | pass | expected=24, actual=24, missing=0 sample_missing=[], unexpected=0 sample_unexpected=[] |
| pq_diagnostics | critical | Product-quantization diagnostics | no_nulls_value | pass | 0 null value(s) |
| decision_framework | critical | ANN selection decision framework | file_exists | pass | results\analyses\decision_framework\ann_decision_framework_scores.csv |
| decision_framework | critical | ANN selection decision framework | required_columns | pass | all present |
| decision_framework | critical | ANN selection decision framework | no_duplicate_keys | pass | 0 rows have duplicate key ['dataset', 'weighting', 'dim', 'modality', 'method', 'seed']; sample=[] |
| decision_framework | critical | ANN selection decision framework | uniform_weighting | pass | weighting='bm25' on every row |
| decision_framework | critical | ANN selection decision framework | uniform_dim | pass | dim=128 on every row |
| decision_framework | critical | ANN selection decision framework | uniform_seed | pass | seed=42 on every row |
| decision_framework | critical | ANN selection decision framework | exact_grid | pass | expected=40, actual=40, missing=0 sample_missing=[], unexpected=0 sample_unexpected=[] |
| decision_framework | critical | ANN selection decision framework | expected_row_count | pass | expected 40, found 40 |
| decision_framework | critical | ANN selection decision framework | no_nulls_deployment_score | pass | 0 null value(s) |
| decision_framework | critical | ANN selection decision framework | no_nulls_deployment_rank | pass | 0 null value(s) |
| decision_framework | critical | ANN selection decision framework | no_nulls_recommended_use_case | pass | 0 null value(s) |
| decision_framework | critical | ANN selection decision framework | no_nulls_recommendation_reason | pass | 0 null value(s) |
| decision_framework | critical | ANN selection decision framework | required_value_w_quality | pass | w_quality=0.45 on every row |
| decision_framework | critical | ANN selection decision framework | required_value_w_latency | pass | w_latency=0.3 on every row |
| decision_framework | critical | ANN selection decision framework | required_value_w_memory | pass | w_memory=0.15 on every row |
| decision_framework | critical | ANN selection decision framework | required_value_w_exposure | pass | w_exposure=0.1 on every row |
| dataset_stats | critical | Dataset statistics | file_exists | pass | results\paper\tables\dataset_stats.csv |
| dataset_stats | critical | Dataset statistics | required_columns | pass | all present |
| dataset_stats | critical | Dataset statistics | no_duplicate_keys | pass | 0 rows have duplicate key ['dataset']; sample=[] |
| dataset_stats | critical | Dataset statistics | exact_datasets | pass | expected=['amazon-books', 'goodbooks', 'ml-1m', 'ml-20m'], found=['amazon-books', 'goodbooks', 'ml-1m', 'ml-20m'] |
| dataset_stats | critical | Dataset statistics | expected_row_count | pass | expected 4, found 4 |
| dataset_stats | critical | Dataset statistics | required_value_min_user_interactions_filter | pass | min_user_interactions_filter=5 on every row |
| dataset_stats | critical | Dataset statistics | no_nulls_users | pass | 0 null value(s) |
| dataset_stats | critical | Dataset statistics | no_nulls_items | pass | 0 null value(s) |
| dataset_stats | critical | Dataset statistics | no_nulls_interactions | pass | 0 null value(s) |
| dataset_stats | critical | Dataset statistics | no_nulls_density | pass | 0 null value(s) |
| dataset_stats | critical | Dataset statistics | no_nulls_inter_per_user_median | pass | 0 null value(s) |
| dataset_stats | critical | Dataset statistics | no_nulls_inter_per_user_p95 | pass | 0 null value(s) |
| dataset_stats | critical | Dataset statistics | no_nulls_train_interactions | pass | 0 null value(s) |
| dataset_stats | critical | Dataset statistics | no_nulls_test_users | pass | 0 null value(s) |
| dataset_stats | critical | Dataset statistics | no_nulls_popularity_gini | pass | 0 null value(s) |
| dataset_stats | critical | Dataset statistics | no_nulls_min_user_interactions_filter | pass | 0 null value(s) |
| hardware | critical | CPU-only scope; Hardware provenance | file_exists | pass | results\_meta\hardware.json |
| hardware | critical | CPU-only scope; Hardware provenance | required_fields | pass | missing fields: [] |
| hardware | critical | CPU-only scope; Hardware provenance | required_value_main_experiments_accelerator_used | pass | expected=False, found=False |
| hardware | critical | CPU-only scope; Hardware provenance | accelerator_present_is_boolean | pass | found=True |
| hardware | critical | CPU-only scope; Hardware provenance | no_gpu_execution_fields | pass | forbidden fields present: [] |
| run_manifest | critical | Full-run provenance; CPU-only scope | file_exists | pass | results\_meta\run_manifest.json |
| run_manifest | critical | Full-run provenance; CPU-only scope | required_fields | pass | missing fields: [] |
| run_manifest | critical | Full-run provenance; CPU-only scope | required_value_project | pass | expected='IndexWise-Recsys', found='IndexWise-Recsys' |
| run_manifest | critical | Full-run provenance; CPU-only scope | required_value_status | pass | expected='completed', found='completed' |
| run_manifest | critical | Full-run provenance; CPU-only scope | exact_datasets | pass | expected=['ml-1m', 'ml-20m', 'goodbooks', 'amazon-books'], found=['ml-1m', 'ml-20m', 'goodbooks', 'amazon-books'] |
| run_manifest | critical | Full-run provenance; CPU-only scope | exact_methods | pass | expected=['flat', 'hnsw', 'ivfflat', 'ivfpq', 'flatpq'], found=['flat', 'hnsw', 'ivfflat', 'ivfpq', 'flatpq'] |
| run_manifest | critical | Full-run provenance; CPU-only scope | exact_modalities | pass | expected=['u2i', 'i2i'], found=['u2i', 'i2i'] |
| run_manifest | critical | Full-run provenance; CPU-only scope | no_failed_combinations | pass | failed_combinations=[] |
| run_manifest | critical | Full-run provenance; CPU-only scope | required_output_recorded | pass | required='results/main/summary_main.csv', outputs=['results/main/run_config.json', 'results/main/summary_main.csv'] |
| run_manifest | critical | Full-run provenance; CPU-only scope | resolved_configuration_mapping | pass | type=dict |
| run_manifest | critical | Full-run provenance; CPU-only scope | resolved_project.name | pass | expected='IndexWise-Recsys', found='IndexWise-Recsys' |
| run_manifest | critical | Full-run provenance; CPU-only scope | resolved_reproducibility.seed | pass | expected=42, found=42 |
| run_manifest | critical | Full-run provenance; CPU-only scope | resolved_embedding.weighting | pass | expected='bm25', found='bm25' |
| run_manifest | critical | Full-run provenance; CPU-only scope | resolved_embedding.dim | pass | expected=128, found=128 |
| run_manifest | critical | Full-run provenance; CPU-only scope | resolved_hardware.cpu_only | pass | expected=True, found=True |
| run_manifest | critical | Full-run provenance; CPU-only scope | resolved_datasets_exact | pass | expected=['ml-1m', 'ml-20m', 'goodbooks', 'amazon-books'], found=['ml-1m', 'ml-20m', 'goodbooks', 'amazon-books'] |
| run_manifest | critical | Full-run provenance; CPU-only scope | resolved_retrieval.methods_exact | pass | expected=['flat', 'hnsw', 'ivfflat', 'ivfpq', 'flatpq'], found=['flat', 'hnsw', 'ivfflat', 'ivfpq', 'flatpq'] |
| run_manifest | critical | Full-run provenance; CPU-only scope | resolved_retrieval.modalities_exact | pass | expected=['u2i', 'i2i'], found=['u2i', 'i2i'] |
| run_manifest | critical | Full-run provenance; CPU-only scope | configuration_hash_matches | pass | expected=12d2e2806429, found=12d2e2806429 |
| run_manifest | critical | Full-run provenance; CPU-only scope | run_id_contains_config_hash | pass | run_id='20260712T173025Z__12d2e2806429' |
| run_manifest | critical | Full-run provenance; CPU-only scope | started_at_utc_populated | pass | found='2026-07-12T17:30:25.517021+00:00' |
| run_manifest | critical | Full-run provenance; CPU-only scope | finished_at_utc_populated | pass | found='2026-07-12T17:50:46.213827+00:00' |
| run_manifest | critical | Full-run provenance; CPU-only scope | git_commit_populated | pass | found='fdcf37d2f4bc1c75ea30916730e4e218c0a9d0d8' |
| run_manifest | critical | Full-run provenance; CPU-only scope | python_version_populated | pass | found='3.11.0 (main, Oct 24 2022, 18:26:48) [MSC v.1933 64 bit (AMD64)]' |
| run_manifest | critical | Full-run provenance; CPU-only scope | manifest_declares_cpu_only | pass | main_experiments_accelerator_used=False |
| run_manifest | critical | Full-run provenance; CPU-only scope | no_gpu_execution_fields | pass | forbidden fields present: [] |
| claim_support_audit | optional | Claim-to-evidence traceability table | file_exists | pass | results\paper\tables\claim_support_audit.csv |
| claim_support_audit | optional | Claim-to-evidence traceability table | required_columns | pass | all present |
| claim_support_audit | optional | Claim-to-evidence traceability table | no_duplicate_keys | pass | 0 rows have duplicate key ['claim_area']; sample=[] |
| claim_support_audit | optional | Claim-to-evidence traceability table | expected_row_count | pass | expected 9, found 9 |
| claim_support_audit | optional | Claim-to-evidence traceability table | boolean_evidence_supported | pass | values=['false', 'true'], nulls=0 |
