# DELTA-ZSC v4 validation report

Date: 2026-08-06

## Scope

This report validates the source contracts of
`delta_episode_static_centered_residual_delayed_exact_voi_v4`. It does not
claim benchmark performance. The v3 CUDA pilot is not v4 evidence and is not
included in any v4 result.

## Artifact identity

- Upstream source commit:
  `15e90b1be0d50ef99df0fa5837312d70fa913643`.
- Local source-archive baseline:
  `018dd8202abd06e2a685872406c8d4a6e1804e69`.
- Config schema: 3.
- Checkpoint/deployment schema: 4.
- Manifest schema: 2.
- Package: `aris4zsc==0.6.0`.
- Official source commit: `5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e`.

## Local environment

The recorded regression was executed on CPU. Exact Python, JAX, NumPy,
duration and per-test records are machine-readable in
`validation/ISOLATED_TEST_RESULTS.json` and mirrored in
`validation/LOCAL_TEST_RESULTS.txt`.

The installable project remains pinned to Python 3.10, JAX/JAXLIB 0.4.38 and
NumPy below 2 through `pyproject.toml`. The local container used to edit the
source has a newer CPU-only Python/JAX stack; the isolated subprocess runner
prevents cross-test JAX executable retention.

## Executed acceptance paths

### Static and command contracts

`validation/run_contract_validation.sh` verifies:

- compilation of active source, experiment applications and validation tools;
- Python 3.10 grammar compatibility for every active Python source;
- loading all six Simple/Wide mechanical/development/formal configs;
- the root CLI and 16 subcommand help paths;
- workflow YAML parsing;
- one active scientific namespace;
- absence of active `transition.py`;
- no retired DEPI objective symbols in the active namespace;
- no committed credential patterns or local absolute paths;
- exact dependency equality between `pyproject.toml` and `requirements.txt`;
- regenerated exact delayed-VOI synthetic diagnostics.

### Isolated pytest regression

`validation/run_all_isolated_tests.py` discovers each top-level
`test_delta_*.py` function and runs it in a fresh bounded subprocess. The suite
covers:

| Area | Contract coverage |
|---|---|
| CLI and matrix orchestration | argument wiring, seed defaults, fresh-process matrix cells, K-specific initializer roots |
| Repository identity | active namespace, package discovery, workflow, v4 identity, authoritative exact-VOI documents |
| Observation semantics | frame alignment, interact exclusion, recipe/interface masks, delayed two-step target and terminal masking |
| Episode-static Bayes filter | persistence, reset, decision isolation, shared-occurrence isolation |
| Centered emissions | response residuals, current decision residuals, successor decision residuals, shared variance |
| Semantic initializer | simplex centering, SVD artifact round trip, label rejection, layout/protocol/lineage provenance |
| Proper scores and optimization | exact three-channel sum, response-only anchor exclusion, parameter ownership, latent-before-PPO ordering |
| CRN anchors | base-policy rollout, sparse capture equivalence, delayed base bridge, successor reward exclusion, passive/active schemas |
| Exact active value | 66 normalized outcomes, uninformative zero value, decision-irrelevant information, mandatory probe-conditioned utility |
| Deployment/storage | checkpoint round trip, initializer provenance in deployment, one-step active base bridge |
| Final audit | probe-axis reduction before anchor masking, finite action-selectivity and component diagnostics |

The final frozen-worktree regression discovered and executed **50** tests:
**50/50 passed**, with **0 failures** and
**0 timeouts**, in **354.538 seconds**.

Per-file distribution:

| Test file | Count |
|---|---:|
| `test_delta_unified_cli.py` | 3 |
| `test_delta_unified_core.py` | 11 |
| `test_delta_unified_manifest.py` | 4 |
| `test_delta_unified_repository.py` | 7 |
| `test_delta_unified_runner_storage.py` | 4 |
| `test_delta_unified_training.py` | 6 |
| `test_delta_unified_voi.py` | 4 |
| `test_delta_v4_semantics.py` | 11 |
| **Total** | **50** |

The authoritative per-test evidence is in the generated result files rather
than being inferred from this prose.

## Exact-VOI synthetic acceptance

`validation/voi_synthetic_diagnostics.json` records:

- exactly 66 outcomes;
- maximum component outcome-mass error below `1e-7`;
- uninformative delayed response: VOI and information gain at float32 zero;
- decision-revealing response: positive VOI and positive information gain;
- component-identifying but successor-decision-irrelevant response: positive
  information gain and float32-zero VOI.

The implementation contains no Halton sequence, quadrature sample count,
quadrature convergence gate, VOI clamp, or information-gain reward.

## What this validation establishes

The executed checks establish that the implementation can represent and run:

- an episode-static response posterior;
- shared occurrence and component-semantic likelihoods;
- centered response/current-decision/successor-decision residuals;
- an unlabeled spectral-simplex initializer with strict provenance;
- separately normalized shared, semantic and decision proper scores;
- causal delayed probe-response timing;
- probe-conditioned t+2 CRN action values;
- exact 66-outcome active VOI with `gamma^2` temporal placement;
- one committed base bridge after an active probe;
- versioned checkpoint and deployment artifacts.

## Not executed in this package

The following remain empirical work and are not represented as passed:

1. construction of fitted K=2/4/8 semantic initializers from the registered
   real calibration lineages;
2. restoration and execution of real Official SP/OP parent checkpoints under
   v4;
3. one-update CUDA mechanical acceptance on the registered Python 3.10/JAX
   0.4.38 stack;
4. paired five-seed Simple/Wide development matrices;
5. ten-run formal training/evaluation and preregistered H1/H2/H3 inference;
6. any comparison with SOTA.

A successful source regression is necessary but not evidence that latent
components become partner-semantic or that return improves. Those questions
must be answered by the registered development and formal protocols.
