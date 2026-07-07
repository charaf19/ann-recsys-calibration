# Deployment guidance notes (generated)

Inputs: `results/calibration_sensitivity/calibration_sensitivity.csv`, `results/main/summary_main.csv`


These rules are derived from the measured benchmark CSVs referenced above.
They apply to CPU-only serving of item-embedding indexes.

1. **Exact Flat is the default below ~100k items.** If measured Flat p95
   latency already fits the budget, ANN adds parameter-tuning and recall risk
   for no benefit.
2. **HNSW is the general-purpose ANN choice** when latency dominates and the
   full float32 vectors fit in memory (index RAM ~= vectors + graph
   overhead). Calibrate `ef` per the calibration-sensitivity table.
3. **IVF-PQ is the memory-constrained choice**: compressed codes shrink RAM
   at the cost of recall ceiling; verify the 0.98 target is reachable before
   choosing it for quality-sensitive surfaces.
4. **Calibration target selection**: 0.95 agreement recall is the balanced
   default; use 0.98 for U2I home-feed style surfaces where end-to-end NDCG
   effect sizes vs Flat should stay negligible, and 0.90 for I2I
   related-items widgets where latency budgets are tighter.
5. **Recalibrate after re-embedding.** Calibrated ef/nprobe values are only
   valid for the embedding geometry they were tuned on (weighting scheme,
   dim, dataset snapshot).
6. **Watch long_tail_exposure, not just accuracy.** PQ-compressed indexes can
   shift exposure toward head items; compare `long_tail_exposure` /
   `long_tail_uplift` columns against Flat before shipping.
