# Claim-support audit schema

`src/claim_support_audit.py` writes
`results/paper_tables/claim_support_audit.{csv,md,tex}` with one row per
claim area. The audit is regenerated on demand; `evidence_available` always
reflects the filesystem at audit time — it is honest by construction (rows
are `False` until the corresponding pipeline stage has produced its output).

## Columns

| column | meaning |
| --- | --- |
| `claim_area` | The topic a paper sentence might make a claim about (fixed list below). |
| `claim_strength_allowed` | The *maximum* strength the artifact supports: `strong` (directly measured under the stated protocol), `moderate` (measured but scoped/diagnostic), `weak` (sensitivity/proxy only), `conditional` (depends on a runtime flag such as `direct_energy_available`), or `none` for a stated aspect. |
| `required_evidence_file` | Result file(s) that must exist (and be produced by real runs) before the claim may appear. Multiple files are `;`-separated. |
| `evidence_available` | `True`/`False` — whether those files exist on disk right now. |
| `safe_interpretation` | A phrasing of the claim that the evidence supports. |
| `unsafe_interpretation_to_avoid` | The overclaim this row exists to block. |

## Claim areas (fixed)

1. ANN method selection
2. U2I vs I2I modality effect
3. SVD-based embedding scope
4. neural embedding generalization
5. PQ compression behavior
6. long-tail exposure
7. fairness
8. production-scale catalogs
9. energy consumption
10. FAISS-specific scope
11. Flat-PQ deployment role

## Usage rules

- Regenerate the audit after every pipeline stage:
  `python src/claim_support_audit.py`
- A paper statement in a claim area with `evidence_available=false` must be
  removed or rewritten as future work.
- A statement stronger than `claim_strength_allowed` must be weakened to the
  row's `safe_interpretation`.
- The audit table itself is a paper appendix candidate
  (`claim_support_audit.tex`).
