# V4.4 delivery notes

This package contains the complete repository tree for:

```text
path_c_v4_4_retrace_calibrated_control_r1
```

## Evidence motivating the repair

The completed V4.3 development run established that response use can change the deployed trajectory, but did not establish positive task value. In 500 matched response-contrast episodes, 230 produced action-trajectory differences, 498 produced identical returns, two were worse under response use, and none improved. The V4.3 checkpoint therefore remains a development diagnostic and the formal ten-unit run remains closed.

## Unified repair

V4.4 changes the complete credit-and-calibration path while retaining TD-only latent semantics:

- one-step Bellman targets are replaced by full-episode off-policy Retrace targets;
- the behavior probability of every recorded action is used in the importance ratio;
- one latent `(estimator, slot)` hypothesis is sampled per episode and supplies a coherent exploratory policy;
- a small uniform floor retains non-zero support for all six primitive actions;
- the outcome model's predicted standard deviations define lower-confidence next-action values;
- runtime control, response-contrast triggering, and gain binning use uncertainty-calibrated values;
- response contrast triggers on the executed action's shift-invariant policy-gain LCB plus the registered next-policy-TV floor;
- a development-only fixed-partner panel evaluates one trained ego against the registered frozen official partners;
- slot responsibility remains based only on full-episode TD evidence;
- every E-step row records responsibility, TD and outcome evidence, bootstrap support, mean importance ratio, and mean trace coefficient.

## Version boundary

V4.4 changes the configuration schema, Bellman target, behavior policy, checkpoint state, decision rows, response-contrast rows, and evaluation surface. V4.3 checkpoints and output directories are intentionally incompatible.

## Validation in this packaging environment

- all active Python files compile;
- 40 locally executable tests pass;
- two dependency suites are skipped because this container lacks Flax and Hydra/JaxMARL;
- no V4.4 training or environment evaluation was run while assembling this package.

The locked remote environment must run the complete suite with zero skips before any V4.4 development training.
