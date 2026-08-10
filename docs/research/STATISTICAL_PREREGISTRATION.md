# STATISTICAL_PREREGISTRATION — DELTA-ZSC v5

`authoritative: true`

## Registered before confirmatory evaluation

- v5 method/source identity and schemas;
- Simple/Wide configs and semantic initializer artifacts;
- training, initializer-calibration, posterior-calibration, and confirmatory
  partner manifests;
- seed indexes 0..9;
- all registered baseline labels;
- 500 episodes per pairing and both roles;
- raw return as primary endpoint;
- crossed ego-run/partner-run bootstrap with 9,999 replicates;
- one-sided alpha 0.05;
- material effect 20 points;
- closed hierarchy `H1 -> H2 -> H3`;
- active-minus-passive as secondary;
- exact delayed 66-outcome active marginal and probe-successor estimand.

## H1

For every layout and baseline `m`, estimate

\[
\Delta_m=J(\text{delta-active})-J(m).
\]

H1 passes only when every contrast has one-sided 95% lower bound above zero and
point estimate at least 20. This is an intersection-union rule; no baseline is
removed after results.

## H2

Using paired development seed means, estimate

\[
J(\text{delta-passive})-J(\text{response-only}).
\]

Both layouts require a 95% interval lower endpoint above zero. The closed claim
hierarchy interprets H2 after H1, but H2 execution and reporting do not wait on
or change in response to H1.

## H3

At same-world anchors, compare source continuation value of correct-belief and
task-matched shuffled-belief mirror policies. Bootstrap ego and partner nodes.
Both layouts require a one-sided lower bound above zero. The closed claim
hierarchy interprets H3 after H1 and H2, but all intervention jobs execute on
the preassigned schedule regardless of those results.

## Secondary active result

Always report `delta_active - delta_passive`, exact VOI, information gain,
action-wise spreads, active/passive policy TV, and successor decision quality.
No minimum active effect is imposed after results, and a null result is not
redefined as success.

## Diagnostic outputs

Always report initializer singular values and conditional oracle gain,
shared/semantic/decision scores and counts, component JS, partner separation,
filter KL, posterior entropy, phase drift, current/successor regret, gradient
alignment, exact-VOI numerical diagnostics, K sensitivity, negative transfer,
and full resources. None may replace the primary endpoint.

## Missingness and failures

A missing run is not replaced by another seed. Infrastructure continuation
keeps the same recorded identity. Exclusion requires pre-existing mechanical invalidity such as an
unreadable checkpoint, lineage mismatch, non-finite state, malformed initializer,
or incomplete raw artifact. Poor performance, uniform belief, or zero active
increment are never exclusion criteria.
