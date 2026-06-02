# SRVF-MAPPO Vectorization Audit - 2026-06-02

## Scope

- Target file: `raob/srvf_mappo.py`
- Test file: `tests/test_srvf_mappo.py`
- Goal: reduce Python loop and CPU-GPU transfer overhead while preserving the V0-anchored posterior-predictive SRVF-MAPPO method.

## Method Design Correspondence

- `RolloutBatch` and `SourceBatch` schemas are unchanged.
- `SourceBatch` construction still flattens `(state, source_partner)` rows in state-major order.
- Belief updates still use only `(g, action, Delta z)` and `NeuralSRVFHeads`.
- `beta` still enters through SRVF factor terms only; no partner id, target reward, target action label, or target identity is introduced.
- `alpha` calibration still uses response reconstruction error, source beta support distance, and posterior contraction.
- Classic Overcooked environment interaction remains primitive-action step based; environment loops were not replaced with surrogate dynamics.

## Changes Audited

- Vectorized `source_table_to_batch` row construction with `repeat_interleave`, `repeat`, and advanced indexing.
- Removed `.tolist()` from `iter_source_batches`; minibatch rows now use Tensor flat indices.
- Replaced rollout discounted-return and GAE reverse Python recurrence with Tensor discounted cumulative sum under the accepted numerical-equivalence tolerance.
- Wrapped rollout/evaluation policy forward paths in inference mode and belief updates in `no_grad`.
- Kept action `.item()` only at the classic environment API boundary.
- Batched V0 predictions in IRF table collection and aggregated response/residual records with `index_add_`.
- Batched offline target-regret posterior updates per partner.

## Result

No method-design mismatch found. The optimization changes are implementation-level efficiency improvements and do not alter the scientific object, training leakage boundary, or public dataclass contracts.

## Validation Status

Server validation completed on `zsc` under `/apps/users/cxw/Document/CodeSpace/Selfs/RAOB_ZSC`:

- `PYTHONPATH=. .venv/bin/python -m ruff check raob tests`: passed
- `PYTHONPATH=. .venv/bin/python -m compileall -q raob tests`: passed
- `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`: 10 passed
- `python -m raob.srvf_mappo formal-classic --smoke --run-id srvf_mappo_vectorization_smoke_20260602 --workers 2 --learner-epochs 2`: complete
