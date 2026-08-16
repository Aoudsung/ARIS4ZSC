# SCIENTIFIC_SPEC — held-out external-partner robustness

`authoritative: true`

## 1. Registered substrate

The substrate is the OvercookedV2 Test-Time Protocol Formation benchmark. The
active layouts, observation contract, episode protocol, action space, environment
randomization, and all sample-size registrations are defined in
[`src/cetr_zsc/config.py`](../src/cetr_zsc/config.py). The active resolved
configuration is `version: 6`; this document does not duplicate the registered
budget or seed tables.

## 2. Scientific problem

A single deployed policy must coordinate with external teammates whose concrete
parameters and training lineage were not available during training. The primary
question is whether one partner-agnostic recurrent actor can improve the mean
and lower tail of complete raw episodic return on a lineage-disjoint held-out
external-partner panel while preserving reference-level self-play.

The registered question is the main-report choice A: held-out external-partner
robustness. Training-support performance and a population cross-play matrix are
not substitutes for this estimand.

The method identity under test is
`constrained_episodic_tail_robust_zsc_v1`. Its objective is

\[
\max_\theta\ \rho_{\mathrm{ext}}(\theta)
\quad\text{s.t.}\quad
J_{\mathrm{SP}}(\theta)\ge\tau_{\mathrm{SP}},
\]

where `\rho_ext` is the parent-level lower-half risk objective and `\tau_SP` is
derived from the measured seed-matched Official-SP reference. The complete
objective and estimator ordering are authoritative in `METHOD_SPEC.md`.

## 3. Legal deployment information boundary

The deployed actor may read only:

- the current local observation;
- its own previously executed action history;
- its recurrent carry derived from that local history;
- episode-start and terminal boundaries.

The deployment policy may not read or reconstruct partner run ID, algorithm,
checkpoint stage, family or mechanism label, parent group, co-training lineage,
hidden simulator state, future observation, counterfactual return, or any
manifest field. It has no online partner classifier, posterior, VOI controller,
partner-specific adapter, or deployment critic.

The following are training-only variables and are never part of deployment
state or action selection:

- parent groups and mechanism strata;
- nominal and adversarial parent weights `p0` and `q`;
- cross-fitting fold membership;
- the measured reference target `\tau_SP`;
- the adaptive Lagrange multiplier `\lambda`;
- partner manifests, lineage records, and checkpoint metadata.

Thus the legal deployment path is exactly local observation/history into one
recurrent actor and then an action. The boundary is about information, not about
whether a local observation happens to contain behaviorally informative events.

## 4. Registered hypotheses

### H1 — external mean and lower-tail robustness

On each configured layout, CETR is compared with the registered baselines on the
held-out external panel. The two primary external endpoints are

\[
J_{\mathrm{ext,mean}}(\theta),
\qquad
J_{\mathrm{ext,CVaR50}}(\theta),
\]

where the second is the empirical parent-level lower-half CVaR induced by the
uncertainty set in `METHOD_SPEC.md`. The unit of inference is an independent ego
run crossed with an independent held-out parent lineage; repeated episodes within
a pairing are repeated measurements, not independent ZSC nodes.

### H2 — self-play non-inferiority

CETR must preserve the measured reference-level self-play target. The endpoint is

\[
J_{\mathrm{SP}}(\theta)-\tau_{\mathrm{SP}}.
\]

The target is reference-derived, has no manually chosen tolerance band, and is
reported with its uncertainty interval. The version-2 `cetr_reference_sp`
artifact stores a resolved absolute `source_checkpoint`; training requires
textual equality with the resolved `--sp-initializer` path. Engineering
preflight seed `-1` uses the seed-0 initializer and τ artifact.

### H3 — external robustness under a single actor

The combined claim concerns one partner-agnostic actor satisfying both the
external lower-tail objective and the self-play constraint. A high external
mean alone is insufficient if the lower tail fails; a high lower tail alone is
insufficient if self-play falls below the reference contract.

## 5. Panel and estimand boundary

The development-support panel supplies the training distribution over SP, OP,
SA, and FCP mechanisms. Its parents are grouped by independent parent lineage;
checkpoint stages within one parent remain one group. The formal support uses
four mechanisms and four independent parents per mechanism—sixteen parent
groups with forty-eight stage members—as registered in `config.py`; development
and mechanical support use one parent per mechanism. Nominal mechanism mass and
within-mechanism parent mass are defined by `config.py`.

External lanes are deterministic and fully covered on every update: each parent
receives fold×role cells `A0`, `A1`, `B0`, and `B1`, with checkpoint stage slot
`(update_index + lane_slot) mod 3`. There is no random parent sampling and no
observed-subset re-normalization.

The confirmatory panel is held out from training and development decisions. Every
confirmatory parent and its co-training lineage is parent- and lineage-disjoint
from development_support. A co-training population is not split between the two
sides. The trained actor therefore encounters the confirmatory partner parameters
and their lineage only at evaluation time.

The population `10×10` matrix is a supplementary report object. It can describe
the ordered population behavior of the final policies, but it is not the primary
external-partner estimand and cannot replace the lineage-disjoint confirmatory
panel.

## 6. Decisive comparison

The decisive experiment compares, under matched layout, evaluation keys, ego
training cost, and upstream partner cost:

- Official-SP;
- Official-OP;
- Official-FCP;
- the retired V6 DELTA-active result as a historical comparator only;
- CETR-ZSC.

The V6 comparator is not an active method, is not reimplemented by this contract,
and cannot be used to define CETR identity. All exact seeds, run counts,
episodes per pairing, bootstrap settings, and resource budgets are registered
in `config.py`.

Report for every method and layout:

- `J_ext,mean` on the held-out external panel;
- `J_ext,CVaR50` on independent parent groups;
- `J_SP − τ_SP`;
- raw episode rows, role balance, lineage records, and resource accounting.

The population matrix is supplementary. No checkpoint is selected because it has
the best observed score, no failed seed is replaced, and no formal result is
restarted under a changed method.

## 7. Decision rules

The decision is made against exactly the FCP method set `{"fcp"}` on the
confirmatory held-out external panel. `summarize-evaluations` is descriptive
only; `claim` is the unique GO/NO-GO/INCONCLUSIVE entry point.

External contrasts use crossed bootstrap: within each replicate, the ego-run and
parent-lineage axes are independently resampled, with methods aligned by
run/seed index. The SP contrast uses paired per-seed differences
`d_s = J_SP^(s) − τ_s` and bootstrap resamples seeds. The lower confidence bound
and all replicate/alpha registrations are read from `config.py`.

### GO

CETR receives GO only if all three conditions hold:

\[
\operatorname{LCB}_{95}\left[
J_{\mathrm{ext,mean}}^{\mathrm{CETR}}-
J_{\mathrm{ext,mean}}^{\mathrm{FCP}}
\right]>0,
\]

\[
\operatorname{LCB}_{95}\left[
J_{\mathrm{ext,CVaR50}}^{\mathrm{CETR}}-
J_{\mathrm{ext,CVaR50}}^{\mathrm{FCP}}
\right]>0,
\]

\[
\operatorname{LCB}_{95}\left[
J_{\mathrm{SP}}^{\mathrm{CETR}}-\tau_{\mathrm{SP}}
\right]\ge 0.
\]

### NO-GO

NO-GO is declared if either the self-play constraint is below target with an
interval excluding zero, or CETR fails to improve both held-out external mean
and lower-half CVaR relative to FCP. The scientific conclusion is then that
this single partner-agnostic robust policy did not jointly preserve reference-
level self-play and improve held-out external robustness. The response-latent,
VOI, or other retired V6 mechanisms are not reintroduced as a rescue.

### INCONCLUSIVE

If the self-play constraint is supported but the external mean or lower-tail
contrast interval crosses zero, the result is INCONCLUSIVE. The registered
response is to increase confirmatory episode counts under the existing contract,
not to alter the method, retune the tail, change the panel, or select another
checkpoint.

## 8. Non-claims

This specification does not claim:

- recovery of a teammate's identity, algorithm, or hidden state;
- universal robustness outside the registered held-out panel;
- that parent-level lower-half risk equals population cross-play;
- global optimality or convergence of PPO or the primal-dual iteration;
- that the self-play target is guaranteed before formal evaluation;
- SOTA performance or any performance gain before the decisive experiment;
- that the retired V6 evidence is CETR evidence;
- that training-only `q`, `lambda`, parent groups, or lineage metadata are legal
deployment information.

No training result or performance claim exists until the registered runs and
raw evaluation artifacts are complete.
