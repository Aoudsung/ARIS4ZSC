# Revision summary

The prior V6 method is retired and is not part of the active implementation.
Its source, contracts, experiments, and results are preserved only in git
history and must not be interpreted as CETR evidence.

The active method is CETR-ZSC, identified by
`constrained_episodic_tail_robust_zsc_v1` in `src/cetr_zsc/config.py`, with
configuration contract `version: 6` and checkpoint schema `8`.

The first scientific-audit repair group fixes the training estimand: complete
episode returns, deterministic full external parent coverage, fold×role
cross-fitting, and exactly one shared advantage normalization on the complete
`EpisodeBatch` before minibatch slicing. No observed-subset re-normalization or
random parent sampling remains part of the contract.

The application-layer repair is landed: a version-2 τ artifact with resolved
absolute checkpoint equality, seed-`-1` preflight mapping to seed 0, crossed
ego-run/parent-lineage bootstrap and paired seed-index SP differences, exactly
the `{"fcp"}` baseline method set, and deployment bundle version `7`.
The deployment bundle contains only actor parameters; reference and training
metadata are same-directory provenance records, including `provenance.json`.

The current dashboard and `cetr-ci` workflow are the authorities for
implementation status and CI conclusions. No CETR training result, CUDA
acceptance result, formal evaluation, GO/NO-GO decision, or performance claim is
established; **CETR 尚无训练结果**。
