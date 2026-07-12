# Canonical result schema and write policy

Only four top-level result directories are valid: `results/_meta/`,
`results/main/`, `results/analyses/`, and `results/paper/`. Canonical
scripts read canonical filenames only — there is no legacy-path or
alternate-filename fallback, and missing canonical inputs fail with a clear
error.

## Metadata (`results/_meta/`)

- `hardware.json` / `hardware.md`: CPU/platform/package/thread metadata,
  `accelerator_present` (passive disclosure), and
  `main_experiments_accelerator_used=false`.
- `environment.txt`: `pip freeze` snapshot.
- `run_manifest.json`: run id, resolved configuration + hash, Git commit and
  dirty status, package versions, requested grid, outputs, failures.
- `validation_report.{csv,md,json}`: strict paper-evidence validation.

## Main evidence (`results/main/`)

- `summary_main.csv`: exactly one row per dataset × modality × method, with
  natural key `(dataset, weighting, dim, modality, method, seed)` and
  provenance columns (`run_id`, `config_hash`, `code_commit`,
  `created_at_utc`).
- `run_config.json`: the resolved run configuration.
- `aggregates/{dataset}__{weighting}__d{dim}__{modality}__{method}.json`
- `perquery/{dataset}__{weighting}__d{dim}__{modality}__{method}.npz`
  (aligned per-query metrics; input to paired inference)
- `calibration/{dataset}__{weighting}__d{dim}__{method}__target_{t:.2f}.json`
- `status/{dataset}__{weighting}__d{dim}__{method}.status.json`

Agreement recall (`ann_recall_vs_exact_*`, index fidelity vs exact Flat) and
held-out recommendation relevance (`recall/ndcg/...`) are distinct fields
and must never be conflated.

## Analyses (`results/analyses/`)

One named subdirectory per analysis: `calibration_sensitivity/`,
`bootstrap/`, `effect_sizes/`, `embedding_sensitivity/`, `exposure/`,
`pq_diagnostics/`, `scale_stress/`, `decision_framework/`,
`optional_backends/`, `energy/`.

## Paper artifacts (`results/paper/`)

`tables/` (CSV/Markdown/LaTeX + the claim-support audit), `figures/`
(PNG+PDF, 300 dpi), `supplementary/`. Every generated table/figure has a
`<name>.sources.json` sidecar with source files, hashes, config hash,
timestamp, script, and Git commit.

## Safe writes

All consolidated evidence is written through `src/utils/result_io.py`:
atomic temp-file writes, three modes (`fail_if_exists` — the default for
experiment producers, `replace`, `merge` by natural key), duplicate-key
validation, and deterministic row ordering. A failed write never damages
the previous file; conflicting rows with the same key raise an error.

`src/validate_paper_evidence.py` validates the complete contract from
`configs/paper_evidence_manifest.yml` (40 main rows, 72 calibration rows,
20 embedding-sensitivity rows, 20 scale rows, `n_boot=2000`, CPU-only
scope) and writes its reports to `results/_meta/`.
