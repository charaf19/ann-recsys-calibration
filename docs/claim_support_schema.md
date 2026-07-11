# Claim-support audit schema

`src/claim_support_audit.py` writes
`results/paper/tables/claim_support_audit.{csv,md,tex}` with one row per
claim area. The audit consumes the strict validation report
(`results/_meta/validation_report.json`, produced by
`src/validate_paper_evidence.py`) — a claim is marked supported ONLY when
every validation section it depends on passes, never merely because a file
exists on disk. Before results exist, every claim is honestly unsupported.

## Columns

| column | meaning |
| --- | --- |
| `claim_area` | The topic a paper sentence might make a claim about (fixed list below). |
| `claim_strength_allowed` | The *maximum* strength the artifact supports: `strong` (directly measured under the stated protocol), `moderate` (measured but scoped/diagnostic), or `weak` (sensitivity/proxy only). |
| `required_validation_sections` | Validation-report sections that must pass before the claim may appear (`;`-separated). |
| `evidence_supported` | `True`/`False` — whether those sections pass in the current validation report. |
| `failing_checks` | The specific failed validator checks blocking the claim (empty when supported). |
| `safe_interpretation` | A phrasing of the claim that the evidence supports. |
| `unsafe_interpretation_to_avoid` | The overclaim this row exists to block. |

## Claim areas (fixed)

1. ANN method selection
2. larger-catalog generalization (Amazon Books)
3. U2I vs I2I modality effect
4. statistical significance
5. embedding-sensitivity generalization
6. PQ compression behavior
7. long-tail exposure
8. production-scale catalogs
9. CPU-only scope

## Workflow position

Run order: pipeline stages → `validate_paper_evidence.py` →
`claim_support_audit.py`. Regenerating the audit without re-running the
validator reuses the last validation report and says so via its
`.sources.json` sidecar.
