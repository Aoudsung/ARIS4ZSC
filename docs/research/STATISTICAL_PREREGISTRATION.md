# STATISTICAL_PREREGISTRATION

`authoritative: true`

## Frozen before confirmatory evaluation

- method identity and source revision;
- Simple/Wide configs;
- training and evaluation partner manifests;
- seed indexes 0..9;
- all registered baseline labels;
- 500 episodes per pairing and both roles;
- raw return as primary endpoint;
- crossed ego-run/partner-run bootstrap with 9,999 replicates;
- one-sided alpha 0.05;
- material effect 20 points;
- closed hierarchy H1 -> H2 -> H3;
- active-VOI increment as secondary;
- Halton sample count 16 and nested 8/16 error diagnostic.

## H1

For each layout and every registered baseline `m`, estimate

\[
\Delta_m=J(\text{delta-active})-J(m).
\]

H1 passes only if every contrast has one-sided 95% lower bound above zero and
point estimate at least 20. Requiring all contrasts is an intersection-union
rule; no baseline is dropped after viewing results.

## H2

Using paired development seed means, compute

\[
J(\text{delta-passive})-J(\text{response-only}).
\]

Both layouts require a 95% interval lower endpoint above zero. H2 is
confirmatory only when H1 passes.

## H3

At same-world intervention anchors, compute empirical source continuation value
of correct-belief minus shuffled-belief policies. Bootstrap ego and partner
nodes. Both layouts require a one-sided lower bound above zero. H3 is
confirmatory only when H1 and H2 pass.

## Secondary and diagnostic outputs

Always report active-minus-passive, K sensitivity, response and decision log
scores, posterior entropy, information gain, raw/clamped VOI, quadrature error,
adaptation KL, resource accounting and negative-transfer summaries. None may be
promoted into a new primary claim after results are observed.

## Missingness and failures

A failed or missing run is not replaced by another seed. Root-cause reruns must
use the same immutable identity. Exclusions require a pre-existing mechanical
invalidity such as an unreadable checkpoint, a manifest/lineage mismatch,
non-finite state, or an incomplete raw artifact; poor performance is never an exclusion
criterion.
