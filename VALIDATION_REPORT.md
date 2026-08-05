# DELTA-ZSC VOI v2 validation report

Date: 2026-08-05

## Scope

This report covers the final source package, completed VOI v2, estimator
boundaries, mocked end-to-end mechanics, configuration/CLI contracts, active
namespace isolation, and artifact integrity. It does not contain or imply
formal OvercookedV2 performance evidence.

## Local validation environment

- Python: 3.13.5
- JAX: 0.9.0.1, CPU backend
- NumPy: 2.3.5
- Formal registered environment: Python 3.10, JAX/JAXLIB 0.4.38 and pinned
  Official OvercookedV2 dependencies in `pyproject.toml`

The local environment differs from the formal environment. Tests were therefore
run as isolated CPU processes. The GitHub workflow installs and checks the
registered Python 3.10 dependency set; the optional self-hosted job performs a
real CUDA mechanical preflight.

## Regression results

| Test file | Result | Tests |
|---|---:|---:|
| `test_delta_unified_voi.py` | passed | 7 |
| `test_delta_unified_core.py` | passed | 6 |
| `test_delta_unified_training.py` | passed | 4 |
| `test_delta_unified_runner_storage.py` | passed | 2 |
| `test_delta_unified_manifest.py` | passed | 1 |
| `test_delta_unified_repository.py` | passed | 6 |
| **Total** | **passed** | **26** |

The tests cover deterministic multidimensional Halton construction,
uninformative/revealing/decision-irrelevant response cases, exact binary
marginalization, probe-conditioned JIT execution, exact filtering, KL
satisfaction, all method variants, belief independence from decision-only
parameters, separate finite base/latent updates, latent-before-PPO transaction
ordering, response-only exclusion of decision anchors, legal rollout and CRN
anchor collection, base-only training behavior, checkpoint/deployment round
trip, manifest lineage, and active/legacy repository boundaries.

## Synthetic VOI acceptance values

`validation/voi_synthetic_diagnostics.json` records:

- uninformative response: VOI `0.0`, information gain `0.0`, quadrature error
  `0.0`;
- decision-revealing response: VOI approximately `0.999329`, information gain
  approximately `0.690129`;
- component-identifying but decision-irrelevant response: information gain
  approximately `0.690129`, VOI `0.0`.

This directly checks that active DELTA values information only through its
consequence for action choice, not through partner identifiability itself.

## Contract validation

- Active source and experiment applications compile successfully.
- All six registered Simple/Wide mechanical/development/formal configurations
  load and pass exact-field validation.
- All 15 CLI subcommands and the top-level CLI parse their help paths.
- `.github/workflows/delta-unified-ci.yml` parses as YAML and contains the CPU
  acceptance and optional self-hosted CUDA jobs.
- `src/path_c/` and the old root analysis track are absent from the active tree.
- Active applications contain no `src.path_c` import.
- Retired comparator/separation/context-dropout/capability/gradient-routing/
  decision-regret/cross-log-likelihood tokens are absent from `src/delta_zsc`.
- No credential-like private-key, GitHub-token, or AWS-key pattern was found.
- The complete old package, applications, configs, tests, workflow, analysis,
  and status ledgers remain available only under explicit legacy paths.

## Configuration identity

The six registered configurations are

`experiments/overcooked_v2/configs/delta_unified_{simple,wide}_{development,formal,mechanical}.yaml`.

Configuration identity is the resolved configuration itself, not a digest of it.
Every run writes its complete resolved config into `run_identity.json`, into the
checkpoint descriptor identity, and into `deployment_bundle.json`; `ensure_run_identity`
refuses to reuse an output directory whose recorded config differs. Changing a
configuration therefore still changes the experiment identity, and the difference is
readable rather than opaque.

## Not executed in this container

The following require resources not present in the local runtime and are not
represented as passed:

1. editable installation under the exact registered Python 3.10 dependency
   environment;
2. restoration and execution of real Official partner checkpoints;
3. the self-hosted single-GPU CUDA mechanical preflight;
4. paired five-seed Simple/Wide development matrices;
5. ten-run formal training/evaluation and H1/H2/H3 inference.

The repository provides executable commands and a fail-closed workflow for
these operations. Their absence prevents a performance or SOTA claim, but does
not invalidate the CPU-side implementation acceptance described above.

## Source identity

The source of record is the git history of this repository: a revision is named
by its commit, not by a checksum manifest. `SOURCE_MANIFEST_SHA256.txt` and the
ZIP checksum sidecar were removed together with every other digest in the tree.
