# DELTA-ZSC v6 protocol index

These are the only authoritative active documents.

| Question | Authority |
|---|---|
| What scientific problem and hypotheses are registered? | `SCIENTIFIC_SPEC.md` |
| What is the exact v6 algorithm? | `METHOD_SPEC.md` |
| What is proved, approximated, or not claimed? | `THEORY.md` |
| Which equation maps to which source file? | `ARCHITECTURE.md` |
| Which numerical settings are registered? | `RESEARCH_PLAN.md` §1 |
| Which comparisons and statistics are valid? | `RESEARCH_PLAN.md` §2 |
| What is the bounded development matrix? | `RESEARCH_PLAN.md` §3 |
| What is preregistered statistically? | `RESEARCH_PLAN.md` §4 |
| What is the execution order? | `RESEARCH_PLAN.md` §5 |
| How does the work become a paper? | `RESEARCH_PLAN.md` §6 |
| Where is requirement-to-code traceability? | `../IMPLEMENTATION_MATRIX.md` |

Active method identity is defined in `src/delta_zsc/config.py`; active execution
begins at `experiments/overcooked_v2/delta_zsc.py`. `transition.py` is absent by
method design. All DEPI v8 and earlier code and documents have been removed
from the tree; nothing historical defines or overrides these contracts.
