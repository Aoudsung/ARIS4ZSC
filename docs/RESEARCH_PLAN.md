# RESEARCH_PLAN — CETR-ZSC registered research plan

`authoritative: true`

This plan defines the panel boundary, decisive comparison, inference units, and
execution order for CETR-ZSC. The sole authority for registered budgets, seeds,
versions, schema fields, and sample sizes is
[`src/cetr_zsc/config.py`](../src/cetr_zsc/config.py), whose active contract is
`version: 5`. This document describes what is compared and how it is decided;
it does not duplicate the registration table.

## 1. Scientific contract

The primary estimand is held-out external-partner robustness:

\[
\max_\theta\ \rho_{\mathrm{ext}}(\theta)
\quad\text{s.t.}\quad
J_{\mathrm{SP}}(\theta)\ge\tau_{\mathrm{SP}}.
\]

`rho_ext` is parent-level lower-half CVaR over the nominal mechanism-balanced
parent distribution. The policy is one partner-agnostic recurrent actor. The
training signal is the complete undiscounted raw return-to-go, with one shared
advantage normalization. External weights are computed from completed returns
with double cross-fitting. The SP constraint uses bilateral self-play gradients
and an adaptive dual variable.

The exact method identity is
`constrained_episodic_tail_robust_zsc_v1`; it is implemented and registered in
`src/cetr_zsc/config.py` and specified in `METHOD_SPEC.md`.

## 2. Panel structure

### 2.1 Development support

`development_support` is the only partner panel used to construct the training
support distribution. It covers the registered SP, OP, SA, and FCP mechanisms.
Independent parent runs are the sampling units. All checkpoint stages from one
parent remain in one parent group and are sampled within that group.

The nominal distribution is mechanism-uniform followed by parent-uniform within
mechanism. The actor never receives the mechanism, parent, stage, or lineage
label. The panel is used for training-support collection, mechanical checks, and
development diagnostics; development returns do not redefine the method or
select the tail constant.

### 2.2 Confirmatory external partners

`confirmatory` is a lineage-disjoint held-out external-partner panel. No
confirmatory parent, parent family, or co-training lineage may overlap with
`development_support`. A co-training population is not split between the two
panels. The confirmatory panel is frozen before formal evaluation and is not
read during development decisions.

The primary inference unit is independent ego run crossed with independent held-
out parent lineage. Episodes within a pairing are repeated observations, not
independent scientific nodes. Both ego roles are retained when the protocol
requires them. Exact parent counts, run counts, episode counts, and bootstrap
settings are read from `config.py`.

### 2.3 Supplementary population object

The population `10×10` matrix is supplementary. It can report the ordered
population behavior of final policies, but it is not the causal validation of
training-support robustness and cannot replace the lineage-disjoint confirmatory
panel. It is never allowed to mask a failure on the primary external estimand.

## 3. Training transaction and data boundary

Each update follows one end-to-end transaction:

1. collect complete episodes from the configured self-play and external lanes;
2. record raw complete-episode returns and scalar-baseline outputs;
3. assign external parents to cross-fitting folds;
4. use one fold's completed returns to compute the other fold's lower-half `q`
   weights;
5. compute the complete raw return-to-go and freeze advantages;
6. normalize all policy samples once with one shared normalization;
7. apply one clipped PPO primal update with the external risk term and SP dual
   contribution;
8. apply one projected dual update using the self-play return shortfall.

The self-play contribution is bilateral: both agents execute the current policy
with independent recurrent states. The scalar value baseline only reduces
policy-gradient variance. Parent groups, `q`, folds, `lambda`, and reference
metadata are training-only and never enter deployment.

No pretraining/freeze stage, best-checkpoint selection, restart after a low
score, post-hoc tail tuning, or alternative loss is permitted.

## 4. Decisive experiment

The decisive comparison uses matched layout, evaluation keys, total ego
interaction cost, upstream partner-training cost, and registered resource
acceptance. It compares:

1. Official-SP;
2. Official-OP;
3. Official-FCP;
4. retired V6 DELTA-active as a historical comparator only;
5. CETR-ZSC.

The V6 row is not an active implementation and does not supply any CETR
training input. Its role is to contextualize the final method revision; V6 and
DEPI are otherwise removed from the active tree.

For each method and layout, preserve raw episode rows and report:

- `J_ext,mean` on the held-out external panel;
- `J_ext,CVaR50` over independent parent groups;
- `J_SP − τ_SP`;
- ego-run and parent-lineage inference units;
- both roles where required;
- resource and missingness ledgers.

All exact formal settings are linked to `config.py`. A formal failure is reported
as observed. No failed seed is replaced, no poor result triggers a rerun under a
changed identity, and no checkpoint is selected because it maximizes a measured
score.

## 5. Statistical decision rules

The confirmatory comparison is CETR against FCP. Confidence bounds use the
registered one-sided run/parent bootstrap and alpha from `config.py`.

### 5.1 GO

Declare GO only when all conditions hold:

\[
\operatorname{LCB}_{95}[J_{\mathrm{ext,mean}}^{CETR}
-J_{\mathrm{ext,mean}}^{FCP}]>0,
\]

\[
\operatorname{LCB}_{95}[J_{\mathrm{ext,CVaR50}}^{CETR}
-J_{\mathrm{ext,CVaR50}}^{FCP}]>0,
\]

\[
\operatorname{LCB}_{95}[J_{\mathrm{SP}}^{CETR}-\tau_{\mathrm{SP}}]\ge0.
\]

### 5.2 NO-GO

Declare NO-GO if the self-play constraint is below `tau_SP` with an interval
excluding zero, or if CETR fails to improve both held-out external mean and
lower-half CVaR relative to FCP. This stops the core hypothesis; it does not
license adding a posterior, VOI, latent component, or other retired V6
machinery.

### 5.3 INCONCLUSIVE

If the self-play constraint is supported but either external contrast interval
crosses zero, declare INCONCLUSIVE. Increase confirmatory episode counts under
the existing registration if the pre-registered rule calls for more precision;
do not change the method, tail definition, panel lineage, or checkpoint choice.

## 6. Development and execution order

The implementation sequence is:

1. bind method identity and resolved config from `src/cetr_zsc/config.py`;
2. complete the CETR package modules and the Overcooked application entry
   points;
3. verify whole-episode collection, parent grouping, cross-fitting, the shared
   advantage normalization, bilateral self-play gradient, and dual transaction;
4. verify deployment contains only local observation/history and the recurrent
   actor;
5. freeze development-support and confirmatory manifests with lineage records;
6. run mechanical and development-support checks without reading confirmatory
   returns;
7. run the decisive baseline/CETR comparison;
8. apply the claim builder without changing the registered method.

The current dashboard records which of these steps have actually occurred.
No claim is inferred from compilation, a mechanical check, or an unrun formal
job.

## 7. Failure and missingness policy

A missing or invalid artifact is excluded only for a pre-existing mechanical
reason such as unreadable data, identity mismatch, lineage mismatch, non-finite
state, or incomplete raw artifact. Poor return, low self-play, low tail value,
or a null comparison is scientific evidence, not an exclusion reason.

The project reports numerical failures as-is. It does not change seeds, delete a
failed run, choose a best checkpoint, or restart under a different method. The
formal decision remains GO, NO-GO, or INCONCLUSIVE under the rules above.

## 8. Non-claims and retired methods

Before the decisive experiment, there is no performance claim, SOTA claim, or
training-result claim. V6 DELTA and DEPI are retired historical revisions;
this plan does not treat their active mechanisms as CETR components. In
particular, a prior posterior, anchor, mirror, VOI, or snapshot self-play result
cannot be used as evidence for CETR.
