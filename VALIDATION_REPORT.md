# DELTA-ZSC validation report

Date: 2026-08-06

## Scope

The local regression results below cover the superseded VOI v2 package. The
exact-VOI v3 revision has not been executed locally, but it has compiled and
completed three fresh development-budget runs plus a real Official-environment
evaluation on the registered remote CUDA stack. This is development-support
pilot evidence, not confirmatory OvercookedV2 evidence or a SOTA claim.

## Remote CUDA v3 execution

- Runtime: Python 3.10.19, JAX 0.4.38, one NVIDIA A10 CUDA device.
- Method identity: `delta_active`,
  `delta_joint_geometry_interface_decision_exact_voi_v3`, config schema 2,
  checkpoint schema 3.
- Training: seeds 0, 1 and 2 each completed all 1,228,800 registered
  development ego-policy steps, 150 updates, 12 anchor triggers and final
  deployment export; no base or latent non-finite update was recorded.
- Evaluation: 3,600 real Official-environment episodes against the available
  `development_support` SP/OP pilot panel; mean raw return `6.6777777778`.
- Artifact root:
  `/mnt/workspace/ARIS4ZSC_v5/runs/engineering/SP_OP_results/delta_interface_v3_3seed_20260806`.

The pilot exposes a substantive scientific failure rather than an execution
failure: mean posterior entropy remains within roughly `3e-4` of `log(4)`, final
top-action agreement is only `0.0625`--`0.125`, and exact VOI/information gain
remain near zero. The SP-partner subset averages `17.6778`, while the OP-partner
subset averages `-4.3222`; role 0 averages `13.8778` and role 1 `-0.5222`.
A descriptive one-seed SP comparator on the same panel averages `5.6167`, versus
`6.6778` for the three DELTA seeds. This is not an inferential comparison: the
SP ego is seed 201, shares lineage with the panel's SP parent, and is evaluated
against its own staged checkpoints.

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

The listed v2 tests covered the prior numerical integration and
uninformative/revealing/decision-irrelevant response cases, binary
marginalization, probe-conditioned JIT execution, exact filtering, KL
satisfaction, all method variants, belief independence from decision-only
parameters, separate finite base/latent updates, latent-before-PPO transaction
ordering, response-only exclusion of decision anchors, legal rollout and CRN
anchor collection, base-only training behavior, checkpoint/deployment round
trip, manifest lineage, and active/legacy repository boundaries.

## Synthetic VOI acceptance values

`validation/voi_synthetic_diagnostics.json` records:

- uninformative response: VOI `0.0`, information gain `0.0`;
- decision-revealing response: VOI approximately `0.999329`, information gain
  approximately `0.690129`;
- component-identifying but decision-irrelevant response: information gain
  approximately `0.690129`, VOI `0.0`.

This directly checks that active DELTA values information only through its
consequence for action choice, not through partner identifiability itself.

## Superseded v2 contract validation

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

## Not executed

The following require resources not present in the local runtime and are not
represented as passed:

1. v3 local CPU regression and synthetic diagnostics regeneration;
2. a lineage-disjoint multi-parent calibration/confirmatory panel;
3. paired five-seed Simple/Wide development matrices;
4. ten-run formal training/evaluation and H1/H2/H3 inference.

Their absence prevents a performance or SOTA claim. The remote pilot establishes
that the revised implementation executes end to end; it does not establish that
the current learned posterior or active policy is scientifically effective.

## Source identity

The source of record is the git history of this repository: a revision is named
by its commit, not by a checksum manifest. `SOURCE_MANIFEST_SHA256.txt` and the
ZIP checksum sidecar were removed together with every other digest in the tree.
