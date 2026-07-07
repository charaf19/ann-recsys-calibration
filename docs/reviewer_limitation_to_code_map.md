# Reviewer limitation → code map

Maps every reviewer weakness to the concrete artifact that addresses it.
Status legend: **[EXISTING]** = already addressed by the first revision layer
(not duplicated); **[NEW]** = module added in this reviewer-limitation layer.

| # | Reviewer concern | Missing evidence | Existing code response | Missing code response (now added) | Script/file added | Expected output |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Limited novelty beyond benchmarking | No decision procedure a practitioner could apply; results were flat tables | [EXISTING] calibration + effect sizes + `deployment_guidance.py` (constraint filtering) | [NEW] a calibrated, modality-aware, effect-size-aware ANN *selection framework* with deployment scores, ranks, and use-case labels; negligible-ΔNDCG methods compete only on latency unless a clear memory/exposure advantage exists | `src/ann_decision_framework.py`, `configs/decision_framework.yml` | `results/deployment_guidance/ann_decision_framework_scores.csv`, `results/paper_tables/ann_decision_framework_scores.tex` |
| 2 | SVD-only embeddings | No check that conclusions survive an embedding swap | [EXISTING] BM25/TF-IDF/none weighting varies the SVD input (`utils/weighting.py`) | [NEW] non-SVD backbones (NumPy BPR-MF; optional PyTorch two-tower MLP) + `ann_ranking_stability` (Spearman of ANN method ranking vs `svd_bm25`) | `src/train_neural_embeddings.py`, `src/run_embedding_backbone_sensitivity.py`, `configs/embedding_backbones.yml` | `results/embedding_sensitivity/embedding_backbone_sensitivity_all.csv`, summary tables, `figures_paper/embedding_backbone_sensitivity.pdf` |
| 3 | "PQ may act as implicit regularization" is speculative | No mechanism-level measurements behind the sentence | [EXISTING] end-to-end quality deltas per method (`summary_main.csv`) | [NEW] measurable diagnostics: reconstruction error, norm/pairwise-distance distortion, top-10/100 overlap with Flat, score variance, popularity-decile effects, exposure shift, correlations with ΔNDCG; conservative labels only (`compression_may_smooth_noise`, never "is regularization") | `src/pq_diagnostics.py`, `src/run_pq_diagnostics.py` | `results/pq_diagnostics/pq_diagnostics_all.csv`, `paper_tables/pq_diagnostics_summary.{csv,tex}`, 3 `figures_paper/pq_*.pdf` |
| 4 | Fairness narrowly defined as long-tail uplift | Single scalar; no decile/coverage/calibration breakdown; risk of overbroad "fairness" language | [EXISTING] `long_tail_exposure` / `long_tail_uplift` / `exposure_proxy` terminology in `utils/metrics.py` | [NEW] full exposure-proxy analysis (decile exposure, head/mid/tail shares, Gini, Coverage@10/@100, user popularity-calibration error, optional provider proxy) with a mandatory `fairness_scope` column bounding every number | `src/exposure_analysis.py`, `src/run_exposure_analysis.py` | `results/exposure_analysis/exposure_analysis_all.csv`, `paper_tables/exposure_analysis_summary.{csv,tex}`, 3 exposure figures |
| 5 | ML-20M catalog below production/web scale | No cost measurements beyond ~27k items | [EXISTING] `src/synthetic_scaling.py` (moderate scales via full pipeline) | [NEW] stress grid to 1M items × dims {64,128,256} × all 5 methods measuring build time, index size, RSS, calibrated latency — with `quality_measured=false` on every row to block quality overclaims | `src/run_scale_stress.py`, `configs/scale_stress.yml` | `results/scale_stress/scale_stress_all.csv`, `paper_tables/scale_stress_summary.{csv,tex}`, 3 `figures_paper/scale_stress_*.pdf` |
| 6 | 95% calibration threshold is arbitrary | Single-threshold results | [EXISTING] **already addressed**: `src/run_calibration_sensitivity.py` + `configs/calibration_thresholds.yml` sweep 0.90/0.95/0.98 and report parameter/latency movement | — (no new module needed) | — | `results/calibration_sensitivity/calibration_sensitivity.csv` |
| 7 | No ScaNN/NGT comparison | FAISS(+hnswlib) only | [EXISTING] `register_backend()` plug-in point in `utils/ann_io.py` | [NEW] concrete adapters (FAISS-HNSW, hnswlib, ScaNN, NGT) + comparison runner; missing packages degrade gracefully to rows with `backend_available=false`, `backend_error_message=package_not_installed`; ScaNN/NGT stay OUT of requirements.txt | `src/backends/{faiss_backend,hnswlib_backend,optional_scann_backend,optional_ngt_backend}.py`, `src/run_optional_ann_backend_comparison.py` | `results/optional_backends/optional_ann_backend_comparison.csv`, `paper_tables/optional_ann_backend_comparison.tex` |
| 8 | Flat-PQ should be offline/batch-only | No policy encoding; Flat-PQ ranked alongside online methods | [EXISTING] latency measurements expose Flat-PQ's full-scan cost | [NEW] explicit role policy: Flat-PQ is labeled `offline_batch_only` in the decision framework unless `allow_flatpq_online: true` is set deliberately | `configs/decision_framework.yml` (`method_roles`, `allow_flatpq_online`), enforced in `src/ann_decision_framework.py` | `recommended_use_case=offline_batch_only` rows in `ann_decision_framework_scores.csv` |
| 9 | Effect sizes should dominate p-values | Risk of significance-star reasoning | [EXISTING] **already addressed**: `src/effect_size_tables.py` (paired Cohen's d, Cliff's delta) alongside `bootstrap_significance.py` | [NEW, reinforcing] the decision framework consumes effect-size *labels* (not p-values) for quality retention; the claim audit encodes the rule | `src/ann_decision_framework.py` (`effect_size_label` drives scoring), `src/claim_support_audit.py` | `effect_size_label` column in scores CSV; audit row for each claim area |
| 10 | No direct energy/power measurement | Latency/RSS only; no joules | [EXISTING] none (gap) | [NEW] Intel RAPL measurement on Linux, optional NVML as supplementary, and an honest Windows/macOS fallback that reports timing+utilization with `direct_energy_available=false` and NA energy — never fabricated values | `src/energy_measurement.py`, `src/run_energy_measurement.py`, `docs/energy_measurement_protocol.md` | `results/energy/energy_measurement_all.csv`, `paper_tables/energy_measurement_summary.{csv,tex}` |

## Cross-cutting guardrail

[NEW] `src/claim_support_audit.py` + `docs/claim_support_schema.md` link all
eleven claim areas to their required evidence files, the maximum claim
strength allowed, and the unsafe interpretation each row exists to block.
Output: `results/paper_tables/claim_support_audit.{csv,tex}`.

## Optional dependencies (deliberately NOT in requirements.txt)

| Package | Enables | Behavior when absent |
| --- | --- | --- |
| `torch` | `two_tower_mlp` backbone | backbone skipped with a warning; other backbones run |
| `hnswlib` | hnswlib backend row in the comparison | `backend_available=false` row |
| `scann` (Linux only) | ScaNN backend row | `backend_available=false` row |
| `ngt` (`ngtpy`) | NGT backend row | `backend_available=false` row |
| `pynvml` | supplementary GPU energy counter | `gpu_energy_joules=NA` |
