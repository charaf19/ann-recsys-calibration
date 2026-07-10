# Result schema

Schemas of every file the pipeline writes under `results/`. All CSVs include
a `seed` column where sampling is involved. Nothing under `results/` is
committed with fabricated values; files appear only after the corresponding
stage is run.

Conventions:
- `modality` columns always use the canonical labels `u2i` / `i2i`
  (`utils.common.normalize_modality_label`; validate_results.py rejects
  anything else).
- Reproducibility fields: `seed` and (where indexes are built) `omp_threads`.
  With `omp_threads=1` builds are bit-reproducible; with more threads exact
  numeric agreement may vary slightly across runs (see
  docs/limitations_code_level.md).

## data/index_*/index_meta.json (written by build_index.py)

- method, N, D, budget_mb, method hyperparameters (M/efConstruction or
  nlist/m/bits/opq + train_sample_size), `omp_threads`, `seed`, index_file.

## results/hardware/

- `hardware.json` — platform, cpu {physical_cores, logical_cores, freq},
  memory.total_gb, python, packages{name: version}, threads (env vars +
  faiss_omp_max_threads), `main_experiments_gpu_used` (bool), label,
  captured_at_utc.
- `hardware.md` — human-readable rendering of the same.
- `env_freeze.txt` — `pip freeze` snapshot.

## results/main/

- `run_config.json` — exact orchestrator configuration of the run.
- `summary_main.csv` — one row per dataset × modality × method:

| column | meaning |
| --- | --- |
| dataset, weighting, modality, method | grid coordinates (modality ∈ {u2i, i2i}) |
| dim, budget_mb, queries | run parameters |
| recall_at_k_mean, ndcg_at_k_mean, hr_at_k_mean, precision_at_k_mean, map_at_k_mean, mrr_at_k_mean | end-to-end quality vs held-out positive (metric_topk) |
| ann_recall_vs_exact_at_k_mean | ANN agreement with exact search (not user relevance) |
| coverage_at_k | fraction of catalog recommended at least once |
| gini_exposure | inequality of the exposure distribution |
| long_tail_exposure | share of top-k exposure going to tail items (bottom 20% by training popularity) |
| long_tail_uplift | long_tail_exposure minus the tail's share of training interactions |
| calibration_target, calibrated_param_name, calibrated_param_value | operating point used (ef / nprobe; empty for flat/flatpq) |
| latency_p50_ms, latency_p95_ms, rss_mb_after | serving cost at that operating point |
| omp_threads | FAISS build threads (1 = bit-reproducible builds) |
| seed | RNG seed |

- `{dataset}__{weighting}__{modality}__{method}.json` — the same aggregate
  metrics plus N, D, ef, nprobe, tail_frac.

### results/main/perquery/ (git-ignored)

- `{dataset}__{weighting}__{modality}__{method}.npz` with arrays
  `recall, precision, hr, ndcg, map, mrr, ann_recall_vs_exact` (one value per
  query, aligned across methods within a dataset/weighting/modality group),
  `exposure_proxy` (per-item normalized exposure), and `meta` (JSON string).

### results/main/calibration/

- `{dataset}__{weighting}__{method}__t{target}.json` — target_recall,
  target_reached, param_name, calibrated_param_value,
  achieved_recall_vs_exact, latency_ms_at_calibrated {mean,p50,p95},
  n_calibration_queries, full `sweep` list.

## results/calibration_sensitivity/

- `calibration_sensitivity.csv` — dataset, weighting, method, target_recall,
  target_reached, param_name, calibrated_param_value,
  achieved_recall_vs_exact, latency_{mean,p50,p95}_ms, topk,
  n_calibration_queries, seed. Plus per-combination JSON sweeps.

## results/bootstrap/

- `bootstrap_cis.csv` — dataset, weighting, modality, method, metric, mean,
  ci_low, ci_high (95% percentile bootstrap), n, n_boot.
- `paired_tests.csv` — dataset, weighting, modality, method, baseline,
  metric, mean_diff, ci_low, ci_high, p_value (two-sided bootstrap),
  significant_at_0.05, n, n_boot.

## results/effect_sizes/

- `effect_sizes.csv/.md/.tex` — dataset, weighting, modality, method,
  baseline, metric, mean_diff, cohens_d (+magnitude), cliffs_delta
  (+magnitude), n. Magnitude bins: Cohen |d| 0.2/0.5/0.8; Cliff |δ|
  0.147/0.33/0.474.

## results/deployment_guidance/

- `deployment_guidance.csv/.md/.tex` — dataset, weighting,
  latency_budget_p95_ms, recall_floor, recommended_method, param_name,
  param_value, calibration_target, achieved_recall_vs_exact, latency_p95_ms,
  rationale, plus u2i/i2i NDCG and long_tail_exposure when
  `summary_main.csv` exists.
- `deployment_notes.md` — generated rule-of-thumb narrative.

## results/paper_tables/

- `dataset_stats.*` — users, items, interactions, density, per-user
  quantiles, temporal-split sizes, popularity_gini.
- `table_main_quality.*`, `table_long_tail_exposure.*`,
  `table_calibration_sensitivity.*`, `table_bootstrap_cis.*`,
  `table_significance_effect_sizes.*` — reshaped views of the CSVs above,
  emitted as `.csv`, `.md`, and `.tex`.

## results/figures_paper/ (git-ignored binaries)

- `fig_latency_vs_ndcg`, `fig_calibration_sensitivity`,
  `fig_long_tail_exposure`, `fig_effect_sizes`, `fig_scaling` — each as
  `.png` and `.pdf`.

## results/scaling/

- `scaling.csv` — n_items, n_users, n_interactions, method, target_recall,
  target_reached, param_name, calibrated_param_value,
  achieved_recall_vs_exact, latency_p50_ms, latency_p95_ms, seed.

## Reviewer-limitation module outputs

- `results/deployment_guidance/ann_decision_framework_scores.csv` (+
  `paper_tables/ann_decision_framework_scores.{csv,md,tex}`) — dataset,
  modality, method, weighting, embedding_backend, ndcg_at_10, recall_at_100,
  ann_recall_vs_flat_at_100, latency_p95_ms, qps, rss_mb, long_tail_uplift,
  delta_ndcg_vs_flat, effect_size_label, deployment_score, deployment_rank,
  recommended_use_case (exact_reference | online_serving_recommended |
  memory_constrained_online_serving | offline_batch_only | not_recommended),
  recommendation_reason.
- `results/pq_diagnostics/pq_diagnostics_all.csv` — long format: dataset,
  weighting, dim, method, metric, decile, value, seed. Summary
  (`paper_tables/pq_diagnostics_summary.*`) adds interpretation_label
  (compression_hurts_quality | compression_preserves_quality |
  compression_may_smooth_noise | insufficient_evidence) and the
  cross-dataset correlations (NaN when <3 points).
- `results/exposure_analysis/exposure_analysis_all.csv` — long format:
  dataset, weighting, modality, method, metric, k, decile, group, value,
  fairness_scope (long_tail_exposure_proxy_only |
  popularity_calibration_proxy | provider_proxy_if_metadata_available |
  not_full_fairness_evaluation), notes.
- `results/embedding_sensitivity/embedding_backbone_sensitivity_all.csv` —
  dataset, backbone, embedding_backend, modality, method, dim, ndcg_at_10,
  recall_at_10, ann_recall_vs_exact_at_k_mean, long_tail_uplift,
  ann_ranking_stability (Spearman of method ranking vs svd_bm25), seed.
- `results/scale_stress/scale_stress_all.csv` — n_items, dim, method,
  build_wall_time_sec, index_size_mb, rss_mb_after, rss_mb_delta,
  calibration fields, latency_p50/p95_ms, **quality_measured=false**, seed.
- `results/optional_backends/optional_ann_backend_comparison.csv` — backend,
  backend_available, backend_error_message, vectors_source, n_items, dim,
  build_time_sec, recall_vs_exact_at_10, latency_p50/p95_ms, seed.
- `results/energy/energy_measurement_all.csv` — dataset, modality, method,
  measurement_backend, direct_energy_available, cpu_energy_joules,
  gpu_energy_joules, wall_time_sec, queries, energy_per_query_joules,
  cpu_utilization_mean, rss_mb, notes (NA energy fields when
  direct_energy_available=false; see docs/energy_measurement_protocol.md).
- `results/paper_tables/claim_support_audit.{csv,md,tex}` — see
  docs/claim_support_schema.md.

## Operational outputs

- `results/status/status_{dataset}_{method}.json` (run_revision_experiments.py)
  — dataset, method, started/finished_at_utc, `overall`
  (ok | partial | failed | skipped | running) and `steps.{build, calibration,
  latency, eval_u2i, eval_i2i}` each with status (ok | failed | skipped |
  pending), error, finished_at_utc. A failed step skips downstream steps for
  that method; the grid continues.
- `results/validation/validation_report.{csv,md}` (validate_results.py) —
  artifact, stage, required, status (ok | missing_required |
  missing_optional | schema_mismatch | empty | unreadable), rows,
  missing_fields, detail.
## Schema addenda (reviewer-hardening pass)

- `embedding_backbone_sensitivity_all.csv` additionally carries
  `backend_available` (bool), `status` (ok | skipped_missing_dependency),
  `error_message` (e.g. `torch_not_installed`) on every row, so skipped
  optional backbones remain visible.
- `scale_stress_all.csv` additionally carries `quality_metric` ("none") and
  `quality_notes` alongside `quality_measured=false`.
- `ann_decision_framework_scores.csv` additionally carries the audit columns
  `quality_retention`, `latency_score`, `memory_score`, `exposure_score`,
  `w_quality`, `w_latency`, `w_memory`, `w_exposure`, so every
  `deployment_score` is recomputable from its own row.
- `hardware.json` distinguishes GPU presence (`cuda_available`,
  `faiss_gpu_available`, `gpu.*`, probed) from usage
  (`main_experiments_gpu_used`, declared).

### Per-query npz extensions (eval_modalities.py)

In addition to the per-query metric arrays, each npz now stores:
`recall_at_100`, `ann_recall_vs_exact_at_100` (per query),
`exposure_counts_at_k`, `exposure_counts_at_100`, `pop_counts` (per item),
`recs_at_k` (queries × metric_topk, -1 padded), `hist_pop_mean` (per query) —
consumed by run_exposure_analysis.py.
