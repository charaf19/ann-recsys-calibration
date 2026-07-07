# Result schema

Schemas of every file the pipeline writes under `results/`. All CSVs include
a `seed` column where sampling is involved. Nothing under `results/` is
committed with fabricated values; files appear only after the corresponding
stage is run.

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
