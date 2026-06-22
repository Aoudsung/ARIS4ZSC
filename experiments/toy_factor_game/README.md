# Toy Factor Game Experiments

This directory contains the toy-game implementation for the first
experiment-bridge milestone in `idea-stage/refine-logs/EXPERIMENT_PLAN.md`.

## Neural Toy Path

Train a full-graph model:

```bash
python experiments/toy_factor_game/train.py \
  --seed 0 \
  --n_episodes 2000 \
  --eval_every 200 \
  --graph_variant full_graph \
  --loss_variant full \
  --mode gtvoi \
  --max_steps 50 \
  --output_dir results/toy
```

Run Exp 1 on the same trained full-graph model:

```bash
python experiments/toy_factor_game/evaluate.py \
  --seed 0 \
  --experiments 1 \
  --modes gtvoi,mi,passive,random,oracle \
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
  --graph_variants full_graph,plus_irrelevant,minus_noncritical,minus_critical,random_graph,complete_graph \
  --n_per_conv 5 \
  --max_steps 50 \
  --output_dir results/toy
```

## Symbolic Pilot

`run_symbolic_pilot.py` is the current sanity-stage runner for the first planned
toy comparison:

- `gtvoi`
- `mi`
- `passive`
- `random`
- `oracle`

It is retained as a smoke/debug tool. It is not the formal neural Exp 1 or Exp 4
implementation. It evaluates against `ToyFactorGameEnv` ground-truth
`ConventionAssignment` labels, not another model's predictions.

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
