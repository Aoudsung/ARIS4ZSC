# V4.3 delivery notes

This package contains the complete repository tree for:

```text
path_c_v4_3_executable_response_value_r1
```

## Root repair

The previous development run showed posterior changes without executable task value. V4.3 replaces the complete latent-assignment-to-execution path:

- slot responsibility is computed only from full-episode Bellman evidence;
- response, reward, next-Q, and next-reference errors are retained as audit evidence and outcome losses, but never assign latent slots;
- the outcome model predicts response-conditioned next-action Q vectors for use and mask beliefs, plus the next reference logits;
- `J_use` and `J_mask` use the same physical posterior and the same KL-regularized policy operator as runtime;
- the primary actionable score is a shift-invariant policy-mediated gain together with expected next-policy total variation;
- training behavior uses a registered 0.1 uniform support mixture, while deployment remains unchanged;
- KL temperatures are solved by deterministic log-space bisection instead of a nearly static dual update;
- response codes represent current-to-next centered-advantage changes;
- every E-step responsibility and evidence component is written without truncation.

## Version boundary

V4.3 changes the model tree, response signature width, transition batch, checkpoint state, evaluation rows, and configuration schema. V4.2 checkpoints and output directories are intentionally incompatible.

## Local validation in this packaging environment

- every active Python file compiled successfully;
- pure-JAX/config/evaluation/data tests passed;
- model and official-integration suites are collected but require the locked Flax, Optax, Orbax, Hydra, JaxMARL, and official-experiments dependencies;
- no training or environment evaluation was run while assembling this artifact.

Run the complete test suite in the locked remote environment before any development training.
