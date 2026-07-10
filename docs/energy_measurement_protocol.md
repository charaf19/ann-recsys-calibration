# CPU energy measurement protocol

The optional energy analysis measures CPU package energy using Intel RAPL when readable. If direct counters are unavailable, it reports timing, utilization, and RSS while marking direct energy unavailable; it never estimates or fabricates joules.

Outputs are `results/analyses/energy/energy_measurement_all.csv` and `results/paper/tables/energy_measurement_summary.{csv,md,tex}`. No GPU energy or GPU dependency is part of this analysis.
