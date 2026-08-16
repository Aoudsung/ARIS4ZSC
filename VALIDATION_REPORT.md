# CETR-ZSC validation report

Date: 2026-08-16

## Scope

This report records the source-tree migration to
`constrained_episodic_tail_robust_zsc_v1`. It covers repository boundaries,
active package identity, application entry points, workflow wiring, and
text-level import/reference review. It does not establish training returns,
CUDA execution, formal evaluation, or benchmark superiority.

The retired V6 method and its implementation are intentionally absent from the
active tree. Historical source, contracts, and results remain recoverable from
git history only; no historical artifact defines CETR.

## Active artifact identity

- package version: `0.7.0`;
- configuration contract: `version: 5`;
- checkpoint schema: `7`;
- deployment bundle version: `6`;
- partner manifest version: `2`;
- method identity: `src/cetr_zsc/config.py`;
- command entry point: `python -m experiments.overcooked_v2.cetr_zsc`.

## Source-level changes established by inspection

- the retired scientific package, applications, tests, configurations, contract
  script, diagnostic generator, and Python caches were removed;
- the active application set is CETR-only and the active workflow is
  `.github/workflows/cetr-ci.yml`;
- the repository protection test now checks the CETR namespace, exact active
  file sets, package/dependency identity, workflow identity, and authoritative
  CETR documents;
- the CI configuration loads `cetr_*.yaml`, compiles the CETR package, smokes
  the CETR CLI and its 14 subcommands, and passes the required reference-SP and
  initializer artifacts to the fail-closed CUDA preflight;
- `validation/run_all_isolated_tests.py` discovers only `test_cetr_*.py` and
  labels its output as CETR.

## Verification boundary

Per the migration requirement, no Python interpreter, project code, package
manager, compiler, or test runner was executed. Verification was limited to
file operations, Read/Grep/Glob inspection, and `git status --short`.

Consequently, this report does not claim that the test suite, compile gate,
workflow parser, CLI smoke commands, configuration loader, or CUDA preflight
has run successfully after the migration.

## Not established

- CUDA mechanical preflight or peak-memory acceptance;
- development-support artifacts or confirmatory evaluation artifacts;
- formal ten-seed training or held-out external evaluation;
- any CETR performance result, GO/NO-GO decision, or SOTA claim.
