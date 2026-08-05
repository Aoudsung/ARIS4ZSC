# DELTA-ZSC protocol index

This index identifies the only authoritative active documents.

| Question | Authority |
|---|---|
| What scientific problem is being tested? | `SCIENTIFIC_SPEC.md` |
| What is the exact algorithm? | `METHOD_SPEC.md` |
| What is proved, approximated, or explicitly not claimed? | `THEORY.md` |
| Which equation maps to which source file? | `ARCHITECTURE.md` |
| What comparisons and statistics are valid? | `EVALUATION_SPEC.md` |
| Which numerical settings are frozen? | `FORMAL_EXPERIMENT_PROTOCOL.md` |
| What is executed next? | `RESEARCH_PROGRAM.md` |
| What is the bounded development matrix? | `research/DEVELOPMENT_MATRIX.md` |
| What was fixed before confirmatory evaluation? | `research/STATISTICAL_PREREGISTRATION.md` |
| How does the work become a paper? | `PAPER_OUTLINE.md` |

Active implementation identity is defined in `src/delta_zsc/config.py`. Active
execution begins at `experiments/overcooked_v2/delta_zsc.py`. Documents under
`docs/legacy/` and code under `legacy/implementation_v8/` are historical and
cannot override these contracts.
