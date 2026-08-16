# CETR-ZSC protocol index

These are the only authoritative active documents for the current method.

| Question | Authority |
|---|---|
| What scientific problem and hypotheses are registered? | `SCIENTIFIC_SPEC.md` |
| What is the exact CETR-ZSC algorithm and its two identity contracts? | `METHOD_SPEC.md` |
| What is proved, approximated, or not claimed? | `THEORY.md` |
| Which equation maps to which source file? | `ARCHITECTURE.md` |
| Which numerical settings and identity fields are registered? | [`src/cetr_zsc/config.py`](../src/cetr_zsc/config.py) |
| Which partner panels, comparisons, and statistics are valid? | `RESEARCH_PLAN.md` |
| What is the bounded development and confirmatory design? | `RESEARCH_PLAN.md` |
| What is the execution order and failure policy? | `RESEARCH_PLAN.md` |
| Where is requirement-to-code traceability? | `../IMPLEMENTATION_MATRIX.md` |
| What is the current implementation and evidence status? | `status/DASHBOARD.md` |

The active method identity is defined only in [`src/cetr_zsc/config.py`](../src/cetr_zsc/config.py). Its active
execution entry point is `experiments/overcooked_v2/cetr_zsc.py`. The configuration
file is the sole authority for registered budgets, seeds, schema versions, and
other numerical identity fields; the contracts link to it rather than restating
those registrations.

The active method is CETR-ZSC (constrained episodic tail-robust zero-shot
coordination), with `METHOD_VERSION =
"constrained_episodic_tail_robust_zsc_v1"`. V6 DELTA and DEPI, including their
active code paths, contracts, diagnostic machinery, and result claims, are
retired and removed from the active tree. Their prior revisions are preserved
only by git history; they do not define or override any current contract.
