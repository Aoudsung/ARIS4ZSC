# RESEARCH_PROGRAM — Direct route to a publishable DELTA-ZSC result

The objective is benchmark superiority supported by a small, falsifiable
mechanism story. Engineering checks exist only to certify that the experiment
is valid; end-to-end runs proceed once the §1 checks are archived with the
exact commit.

## 1. Final implementation acceptance

Run the active compilation, six isolated test files, all six configuration
loads, CLI smoke tests, lineage-bound manifest validation, and one real CUDA
mechanical update. Archive the validation report with the exact commit.

## 2. Paired development execution

Execute the registered five-seed matrix on Simple and Wide without adding new
losses or variants. Read the results in this order:

1. base/history task competence;
2. response-only versus passive DELTA;
3. passive versus active DELTA;
4. total-interaction controls;
5. bounded K sensitivity;
6. posterior, KL and exact-VOI numerical diagnostics.

A weak result triggers diagnosis of the corresponding estimator or data
distribution; the next revision repairs that defect and preserves the single
latent factorization.

## 3. Formal benchmark execution

Freeze code, manifests, configs and seeds. Train ten DELTA-active runs per
layout and all registered baselines under the same Official protocol. Evaluate
all ego/partner/role pairings and generate raw matrices, resource tables,
posterior diagnostics and same-world belief interventions.

## 4. Paper decision

The paper is strongest when all three claims close:

- H1: material superiority to every baseline on both layouts;
- H2: decision observation adds value beyond response prediction;
- H3: legal-history belief has same-world causal decision value.

The active-VOI increment is a pre-registered secondary contribution. A null
increment is reported as such; it does not invalidate passive decision-relevant
adaptation, and post-hoc additions to the registered method remain excluded:
any revision must pass the three-link test in §5.

## 5. Revision discipline

Any future change must answer one of three questions:

1. Does legal response modeling improve the posterior?
2. Does the posterior predict counterfactual action ordering?
3. Does the analytic policy convert that ordering into higher held-out return?

Changes that cannot be mapped to one of these links do not enter the active
method.
