# DELTA-ZSC v4 protocol index

These are the only authoritative active documents.

| Question | Authority |
|---|---|
| What scientific problem and hypotheses are registered? | `SCIENTIFIC_SPEC.md` |
| What is the exact v4 algorithm? | `METHOD_SPEC.md` |
| What is proved, approximated, or not claimed? | `THEORY.md` |
| Which equation maps to which source file? | `ARCHITECTURE.md` |
| Which comparisons and statistics are valid? | `EVALUATION_SPEC.md` |
| Which numerical settings are frozen? | `FORMAL_EXPERIMENT_PROTOCOL.md` |
| What is the execution order? | `RESEARCH_PROGRAM.md` |
| What is the bounded development matrix? | `research/DEVELOPMENT_MATRIX.md` |
| What is preregistered statistically? | `research/STATISTICAL_PREREGISTRATION.md` |
| How does the work become a paper? | `PAPER_OUTLINE.md` |
| Where is requirement-to-code traceability? | `../IMPLEMENTATION_MATRIX.md` |

Active method identity is defined in `src/delta_zsc/config.py`; active execution
begins at `experiments/overcooked_v2/delta_zsc.py`. `transition.py` is absent by
method design. Documents under `docs/legacy/` and code under
`legacy/implementation_v8/` cannot override these contracts.
