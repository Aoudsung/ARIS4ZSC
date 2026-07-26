# VQBC V4.1 Root-Repair Validation Record

Date: 2026-07-26

This record documents local static validation and the completed remote CUDA/JaxMARL development rerun. It creates no scientific readout and does not authorize formal ten-unit training.

## Scope

- Independent latent twin-dueling Q experts.
- Full-episode joint responsibility evidence.
- Persistent slot-level Bayesian filtering.
- Conservative upper-confidence value equivalence.
- Assignment-independent response code targets.
- Per-epoch E-step, exact lane partitions, and per-minibatch Polyak updates.
- Scale-aware response-contrast triggering.
- V4.1 config/checkpoint incompatibility with V4.

## Local commands

```bash
PYTHONPATH=. python -m pytest -q \
  experiments/overcooked_v2/tests/test_path_c_vqbc_math.py \
  experiments/overcooked_v2/tests/test_path_c_vqbc_contracts.py \
  experiments/overcooked_v2/tests/test_path_c_vqbc_evaluation.py \
  experiments/overcooked_v2/tests/test_path_c_vqbc_training.py \
  --junitxml=VQBC_V4_1_LOCAL_TESTS.xml -rs

python -m compileall -q \
  src/path_c/vqbc \
  experiments/overcooked_v2/model_dock \
  experiments/overcooked_v2/tests

git diff --check
```

## Result

- 35 passed.
- 2 skipped: the Flax-dependent training module and one JaxMARL-dependent contract test could not run in the local execution environment.
- Compile check passed.
- Whitespace/diff check passed.
- Local pytest log SHA-256: `5cf06cf29429e557cbcfbee3ce4eda708399487adb3f520f8eb28ea708942e9b`.
- Local JUnit XML SHA-256: `b4d59e75acb9fd9ae795d90486f8a2eb6245bb69b366d61a9d2eb2abd0a08688`.
- Flax, Optax and JaxMARL dependent training/checkpoint tests were not executable locally because those packages are absent and unavailable from the configured package index.

## Remote validation

- Isolated code directory: `/apps/users/cxw/Document/CodeSpace/Selfs/CPR_REPO_VQBC_V4_1_20260726`.
- Isolated result root: `/apps/users/cxw/Document/CodeSpace/Selfs/results/path_c_vqbc_v4_1_20260726`.
- Device: GPU 4; memory use was 77 MiB before and after the run, with zero volatile and aggregate uncorrectable memory errors.
- Uploaded source combined SHA-256 before the remote-only config-path adaptation: `1ecb87e71c3b0cd787b5a22645c09a60bbb145d22ab288794389f0405b11a1f3`.
- Remote adapted config SHA-256: `4f171a4d06342933881b4efbf22c9fce9da4e4ab33bf8aab98555ce2f1b5cf06`.
- The frozen official-source dependency was validated through the supported `PATH_C_OFFICIAL_SOURCE_MIRROR_ROOT` route. The final successful runs did not modify the main remote repository or the historical V4 results.

The four registered VQBC test files completed with 50 passed, 0 failed, 0 errors, and 0 skipped in 156.325 seconds. The test log SHA-256 is `5952b218382e51f578768c0a348723d5418c12143cc6d5f49d527f1d1b978a79`; the JUnit report SHA-256 is `d0747dec97931437110d3d19e225d16aebe53b906c638e5005678843703cbf8c`.

The seed-100 development training completed 1,228,800 effective environment steps, 3,072 complete episodes, and 96 updates. It produced 13 metric rows, 12 periodic checkpoints, and one final checkpoint. Relevant SHA-256 values are:

- training summary: `58f6656be6f17a7f8b9288bb29efd1cba38de1ced7d78996c3c8552b4881361e`;
- metrics: `9dfd4dc9e42e4c5f2ff088fdd3dbc80469bf5de722619e94a67b25bba69d3a8a`;
- pipeline state: `5435572d0fde199dadb1ed81917b664d2fff81b6609237afab60b19b3ba82a20`;
- final checkpoint manifest: `bf528edefe53e53b75a844e1b12abb0a283ffbbd9958ca5c9d5a76ace74609af`;
- training state file: `bb61d3f008007e2f058371a77077b48dc3776dc5530fc11f320338b36a727cc8`;
- deployment file: `2ada67a6b63dca4d84dc1d69e004e8bd187155a3d3dbf44faff19ff909160143`.

The four-mode evaluation completed 500 matched 400-step episodes per mode, for 2,000 episodes and 800,000 effective environment steps. Raw rows SHA-256 is `8431513520d418a39b23ba0cb37d6f018c2eb913e0dfb805f1c7338750e02f04`; summary SHA-256 is `f5347fabb6f4290733f761a3a6212f10e061bfd45851e550fdbf849562ca8e2a`. Mean returns were 168.92 for posterior use, 169.04 for fixed prior, and 168.84 for both reference-only and generic-response-information deployment. Posterior use and fixed prior differed in reference-action deviation count on 452 of 500 matched episodes, so posterior use changed ordinary behavior but did not improve mean return.

The response contrast completed 500 matched rows and 600,000 effective branch environment steps. Raw rows SHA-256 is `d64082e9d1f1f630a1db22b94ccc22ae0a713c7e6ae368ee16fd0f85a0997851`; summary SHA-256 is `a63fa8bb03ba10c61d412e1663e13836cc5b6bb2191a16f2a30a2ddf69851918`. Every row triggered at step 0 with predicted information net value between 0.139714 and 0.278620, yet every response-use minus response-mask return effect was zero, including all ten equal-frequency bins and the highest bin.

Some initial invocations failed on copied macOS metadata, a relative official-config path, the missing official-source mirror binding, and passing YAML where resolved JSON was required. Those failed logs were retained; the copied metadata files were removed only from the new isolated remote directory, and the corrected successful invocations have separate logs and zero exit status.

## Adjudication boundary

The V4 uniform-slot failure was repaired in the narrow implementation sense: responsibilities became non-uniform, all eight value quotients were active in the latest rollout states, and posterior use changed ordinary actions. Responsibility mass nevertheless remained concentrated in a few experts, and the registered causal response-use contrast remained exactly zero. The implementation may therefore be marked `tested`, but the mechanism is not accepted and formal ten-unit training remains prohibited. The follow-up control-path trace is recorded below.

## Remote control-path trace

A follow-up development diagnostic used the same final checkpoint without parameter updates. Two 32-episode step-zero probes consumed 32 shared environment steps each, and one 32-episode, 400-step mask/use branch trace consumed 25,600 branch environment steps. The added effective budget was 25,664 environment steps; all outputs retain `scientific_readout_allowed: false`.

The step-zero trace showed that the response path is wired. Five actual response codes appeared. The mean log-likelihood spread of the observed code across eight slots was 1.522, and the mean L1 distance between response-use and response-mask slot beliefs was 0.367. Nevertheless, the mean total-variation distance between their next execution policies was only 0.000741, with zero sampled-action and zero greedy-action differences at the next step.

The complete branch trace found left-policy action differences in 3 of 32 episodes, first at steps 20, 26, and 309. Three episodes later differed in observation or response code. No right-policy action, per-step reward, or final-return difference occurred. This proves that masking is not a no-op, while also showing that its control effect is rare and never reached reward in the traced episodes.

The source trace identifies an objective/executor mismatch. `bellman_control_values` uses an unconstrained maximum over next actions inside both response-use and response-mask values. Actual deployment uses a reference-anchored, Kullback–Leibler-constrained action distribution. The execution distribution assigned a mean probability of 0.939 to the sampled step-zero action and the earlier evaluation recorded small Kullback–Leibler divergence, so most response-induced value changes could not overcome its existing action preference. Triggering also records the maximum information score across actions while the A2 branches execute an action sampled from the response-use policy; only 6 of 32 traced episodes executed the maximum-score action, although every executed action still had a positive score.

Trace artifact SHA-256 values:

- step-zero control trace: `495d3698bcf1c18f770203c9beda51a7c233ca50a583c355218d8a4d75d83e18`;
- complete branch trace: `fbeba85f20d102b4a17167e6f7b3c66ac62e3d27b834dd5c8d52b97cfaec04d2`;
- trigger-score trace: `c148993f0c7ca67bbfaa6162196f377b2b8c0a35c724133ca385651a20e63612`.

Formal ten-unit training remains prohibited. The next implementation change should make the response-value continuation operator match the action distribution that deployment actually executes.
