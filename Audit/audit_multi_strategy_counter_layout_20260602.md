# Audit: Multi-Strategy Counter Classic Layout Support

Date: 2026-06-02

## Scope

- Add classic Overcooked layout-name support for the GAMMA paper's Multi-Strategy Counter setting.
- Keep SRVF-MAPPO method logic unchanged.
- Keep the benchmark family as classic Overcooked-AI, not OGC and not OvercookedV2.

## Source Check

- `Method_design.md` defines SRVF-MAPPO for zero-shot coordination in classic Overcooked-style cooperative Markov games.
- The implementation change is limited to the environment adapter and formal-run layout canonicalization.
- No V0, SRVF belief, alpha calibration, MAPPO loss, source/target split, or target-leakage path was changed.

## Layout Mapping

- GAMMA repository install guidance uses `mapbt` and its `overcooked_berkeley` classic Overcooked package.
- GAMMA scripts and config paths use `diverse_counter_circuit_6x5` as the internal multi-strategy counter family label.
- The installed classic layout file that matches the multi-food, multi-order counter design is `counter_circuit_6x5_2pots_3orders.layout`.
- RAOB now maps:
  - `Multi-Strategy Counter`
  - `multi-strategy-counter`
  - `multi_strategy_counter`
  - `diverse_counter_circuit_6x5`
  to `counter_circuit_6x5_2pots_3orders`.

## Code Correspondence

- `raob/benchmarks/overcooked_classic.py`
  - Adds `canonical_classic_layout_name`.
  - Adds Multi-Strategy Counter aliases.
  - Keeps existing layout names unchanged.
  - Preserves `requested_layout` only when an alias was used, so canonical existing runs keep the same public output keys.
- `raob/srvf_mappo.py`
  - Canonicalizes `args.layout` once at formal-classic entry before partner specs, adapter construction, config payload, worker config, and summary generation.
  - Worker initialization repeats canonicalization defensively for resumed or externally prepared configs.
- `tests/test_srvf_mappo.py`
  - Adds alias-resolution coverage without constructing a server-only Overcooked environment.

## Validation

Server validation completed after synchronization:

- `PYTHONPATH=. .venv/bin/python -m ruff check raob tests`: passed.
- `PYTHONPATH=. .venv/bin/python -m compileall -q raob tests`: passed.
- `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`: 11 passed.
- Server-side classic layout construction smoke for `Multi-Strategy Counter`: passed.
  - Canonical layout: `counter_circuit_6x5_2pots_3orders`.
  - Requested layout preserved in observation metadata: `Multi-Strategy Counter`.
  - Ego observation tensor size: 780.

## Audit Result

Passed. The change is limited to classic Overcooked layout support and does not modify SRVF-MAPPO method logic.
