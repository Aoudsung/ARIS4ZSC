# RESEARCH_PROGRAM — Direct execution path for DELTA-ZSC v4

The objective is benchmark superiority supported by a falsifiable mechanism
chain. Engineering acceptance certifies experiment validity; it is not a
reason to postpone end-to-end runs.

## 1. Source and semantic-initializer acceptance

For the exact v4 commit:

1. run compilation, configuration, CLI, repository-boundary, core, VOI,
   semantic, training, runner/storage, and manifest tests;
2. generate the v4 synthetic exact-VOI artifact;
3. build one initializer per layout from the frozen calibration panel;
4. verify initializer centering, singular values, unlabeled construction, and
   parent-disjoint conditional oracle artifact;
5. execute one real CUDA mechanical update with current and successor anchors,
   save/restore, deployment export, and final audit.

## 2. Paired development matrix

Run the registered five-seed Simple/Wide matrix without adding new losses or
post-hoc variants. Read results in this causal order:

1. base/history task competence and reproducibility;
2. semantic response specialization and partner separation;
3. response-only versus passive DELTA;
4. current and successor decision quality;
5. passive versus active DELTA;
6. total-interaction controls and bounded K sensitivity;
7. resource and numerical diagnostics.

The v4 mechanism is considered operational only when:

- component event distributions are non-identical;
- beliefs differ by partner more than by episode phase alone;
- component decision residuals alter action ordering;
- same-world belief intervention has positive empirical value;
- active VOI has action-wise spread and changes the passive policy when its
  secondary contribution is claimed.

These are interpretation requirements, not training gates.

## 3. Formal execution

Freeze source, configs, initializer artifacts, manifests, and seeds. Train ten
`delta_active` runs per layout and all registered baselines under the Official
protocol. Produce full ego x partner x role matrices, resource ledgers,
posterior diagnostics, final current/successor audits, and belief interventions.

## 4. Paper decision

The strongest paper closes:

- H1: material superiority to every baseline on both layouts;
- H2: decision supervision adds value beyond response inference;
- H3: the legal belief has same-world causal decision value.

The active increment is secondary. A null active increment is reported without
inventing a replacement mechanism. A negative passive result sends diagnosis
back to the corresponding response, decision, or mirror-policy link.

## 5. Revision discipline

A future modification enters the active method only when it repairs one of:

1. legal semantic response -> partner-relevant posterior;
2. posterior -> component-conditioned action ordering;
3. action ordering -> higher held-out return;
4. probe -> action-selective delayed information value.

It must retain one shared actor-critic, avoid partner-label supervision, and add
no arbitrary anti-collapse gate or loss without a separately identified
estimand.
