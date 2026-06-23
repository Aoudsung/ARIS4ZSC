# Toy Factor Game Experiments

This directory contains the current ARIS-Bellman toy implementation for
zero-shot coordination via interaction-factor beliefs.

## ARIS-Bellman Neural Toy Path

Train a full-graph model:

```bash
python experiments/toy_factor_game/train.py \
  --seed 0 \
  --n_episodes 8000 \
  --eval_every 200 \
  --method aris_bellman \
  --graph_variant full_support \
  --max_steps 50 \
  --output_dir results/toy
```

Run Exp 1 policy/baseline evaluation:

```bash
python experiments/toy_factor_game/evaluate.py \
  --seed 0 \
  --experiments 1 \
  --methods base_only,aris_bellman,flat_latent,global_gru,oracle_belief_factorq,oracle_belief_flatq,random_policy \
  --exp1_graph_variants full_support,overcomplete \
  --n_per_conv 5 \
  --max_steps 50 \
  --output_dir results/toy
```

Train graph variants for Exp 4 by changing `--graph_variant` to each of:

- `full_support`
- `overcomplete`
- `overcomplete_minus_low_ce`
- `minus_critical`
- `random_same_size`
- `complete_option_graph`
- `shuffled_routes`
- `shuffled_relevance`

Then run:

```bash
python experiments/toy_factor_game/evaluate.py \
  --seed 0 \
  --experiments 4 \
  --methods aris_bellman \
  --graph_variants full_support,overcomplete,overcomplete_minus_low_ce,minus_critical,random_same_size,complete_option_graph,shuffled_routes,shuffled_relevance \
  --n_per_conv 5 \
  --max_steps 50 \
  --output_dir results/toy
```

The neural implementation no longer trains or evaluates deployment-time
`gtvoi`, `mi`, `passive`, or `oracle` selectors. G-TVOI and MI are post-hoc
trajectory diagnostics computed from real belief updates under the learned
Bellman policy.

V4.1 support graphs are induced from the all-convention CE expectation by
default. One-convention CE estimates are available only as explicit diagnostics
and are not used for graph induction.

V4.1 checkpoints use schema `aris_bellman_v4.1`. Older `aris_bellman_v2`
checkpoints were induced from the pre-fix CE path and should be treated as
pre-v4.1 legacy outputs, not mixed into V4.1 tables.

Exp 3 has two blocks: method comparison on `full_support`, and routing/relevance
controls for `aris_bellman` on `full_support`, `shuffled_routes`,
`shuffled_relevance`, and `random_same_size`. The routing/relevance block
requires those checkpoints to exist and fails visibly if they are missing.

The main baseline names are:

- `base_only`: task-only `Q_base(s, option)`.
- `aris_bellman`: factor-local belief state and factor-local Q residuals.
- `flat_latent`: unrestricted latent-belief Q baseline.
- `global_gru`: global-history shortcut baseline.
- `oracle_belief_factorq`: true factor labels with factor-local Q.
- `oracle_belief_flatq`: true factor labels with unrestricted flat Q.
- `random_policy`: uniformly sampled valid options.

## Symbolic Pilot

`run_symbolic_pilot.py` is a retained historical smoke/debug tool for the first
planned symbolic comparison:

- `gtvoi`
- `mi`
- `passive`
- `random`
- `oracle`

It is not the formal neural Exp 1, Exp 3, or Exp 4 implementation and should not
be used for ARIS-Bellman claims. It evaluates against `ToyFactorGameEnv`
ground-truth `ConventionAssignment` labels, not another model's predictions.

Example command:

```bash
python experiments/toy_factor_game/run_symbolic_pilot.py \
  --seeds 0,1,2 \
  --episodes_per_convention 1 \
  --max_steps 50 \
  --output_dir results/toy_symbolic/sanity
```

Tiny remote smoke command:

```bash
python experiments/toy_factor_game/run_symbolic_pilot.py \
  --seeds 0 \
  --episodes_per_convention 1 \
  --max_conventions 1 \
  --max_steps 5 \
  --progress_every 1 \
  --output_dir results/toy_symbolic/smoke
```

Expected outputs:

- `summary.json`
- `episodes.csv`

## Current Execution Boundary

Project guidance in `AGENTS.md` and `CLAUDE.md` forbids local experiment
execution and remote SSH execution unless explicitly authorized by the user. The
command above is therefore documented but was not run during static bridge work.
