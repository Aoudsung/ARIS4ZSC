# Revision summary

The prior V6 method is retired and is not part of the active implementation.
Its source, contracts, experiments, and results are preserved only in git
history and must not be interpreted as CETR evidence.

The active method is CETR-ZSC, identified by
`constrained_episodic_tail_robust_zsc_v1` in `src/cetr_zsc/config.py`, with
configuration contract `version: 5`, checkpoint schema `7`, and deployment
bundle version `6`.

CETR uses one partner-agnostic recurrent actor, complete undiscounted episodic
returns, parent-level lower-half CVaR, double cross-fitting, and a
reference-derived self-play constraint updated by a primal-dual transaction.
Its deployment input is local observation and action history only.

No CETR training result, CUDA acceptance result, formal evaluation, or
performance claim is established by this source-tree migration.
