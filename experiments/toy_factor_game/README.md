# Toy Factor Game Experiments

This directory contains the toy-game implementation for the first
experiment-bridge milestone in `idea-stage/refine-logs/EXPERIMENT_PLAN.md`.

## ARIS-Bellman Neural Toy Path

Train a full-graph model:

```bash
python experiments/toy_factor_game/train.py \
  --seed 0 \
  --n_episodes 8000 \
  --eval_every 200 \
  --method aris_bellman \
  --graph_variant full_graph \
  --max_steps 50 \
  --output_dir results/toy
```

Run Exp 1 policy/baseline evaluation:

```bash
python experiments/toy_factor_game/evaluate.py \
  --seed 0 \
  --experiments 1 \
  --methods aris_bellman,flat_latent,global_gru,oracle_belief,random_policy \
  --exp1_graph_variants full_graph,plus_irrelevant \
  --n_per_conv 5 \
  --max_steps 50 \
  --output_dir results/toy
```

Train graph variants for Exp 4 by changing `--graph_variant` to each of:

- `full_graph`
- `plus_irrelevant`
- `minus_noncritical`
- `minus_critical`
- `random_graph`
- `complete_graph`

Then run:

```bash
python experiments/toy_factor_game/evaluate.py \
  --seed 0 \
  --experiments 4 \
  --methods aris_bellman \
  --graph_variants full_graph,plus_irrelevant,minus_noncritical,minus_critical,random_graph,complete_graph \
  --n_per_conv 5 \
  --max_steps 50 \
  --output_dir results/toy
```

The neural implementation no longer trains or evaluates deployment-time
`gtvoi`, `mi`, `passive`, or `oracle` selectors. G-TVOI and MI are post-hoc
trajectory diagnostics computed from real belief updates under the learned
Bellman policy.

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
