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
trip, manifest hashing/lineage, and active/legacy repository boundaries.

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

## Configuration fingerprints

| Configuration | SHA-256 fingerprint |
|---|---|
| Simple development | `7df8546e90ede4ea99a3816f88f5912064d02b20aff768d8cada8605d493ef52` |
| Simple formal | `80464cbbc35f3192b579316377673d0bc4c505c76261312fa0417cd5979558da` |
| Simple mechanical | `5f3fb4d108732819c2e7aca57de16f08ed58c1091f9413cac6f1a4a031bc2294` |
| Wide development | `7e3678b2fd00470534e6804139567c3d7daf543fd09f69222a99832023b77ec6` |
| Wide formal | `73a436d6dab1ef91ec1e800f92af98012b8241212c8d9ac3acd0111907ffd2c8` |
| Wide mechanical | `948619d86346f433b10733135dbb7bdccf5d65a94f4f06db8cb738ba8bb5bb6d` |

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

## Integrity

`SOURCE_MANIFEST_SHA256.txt` hashes every included source file except the
manifest itself. The distributable ZIP has a separate SHA-256 sidecar and was
re-extracted before delivery; the extracted files were checked against the
embedded source manifest.
