"""Top-level standard-matrix and response-contrast evaluation application."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.overcooked_v2.response_contrast_app import (
    evaluate_response_contrast,
)
from experiments.overcooked_v2.standard_evaluation_app import evaluate_standard
from src.path_c.experiment import load_config, load_population
from src.path_c.storage import (
    ensure_run_identity,
    evaluation_identity,
    write_json,
    write_run_metadata,
)


def run_evaluation(args: argparse.Namespace) -> None:
    config = load_config(args.config, run_kind=args.run_kind)
    population = load_population(args.manifest)
    if population.layout != config.environment.layout:
        raise ValueError("Population and configuration layouts differ.")

    output = Path(args.output).resolve()
    ensure_run_identity(
        output,
        evaluation_identity(
            config=config,
            seed=int(args.seed),
            population=population.to_mapping(),
        ),
    )
    write_json(output / "resolved_config.json", config.to_mapping())
    write_json(output / "resolved_population.json", population.to_mapping())

    if population.evaluation_kind == "standard_matrix":
        completed_episodes, effective_environment_steps = evaluate_standard(
            config=config,
            population=population,
            output=output,
            evaluation_seed=int(args.seed),
            resume=bool(args.resume),
        )
    elif population.evaluation_kind == "response_contrast":
        matched_episode_count, effective_environment_steps = (
            evaluate_response_contrast(
                config=config,
                population=population,
                output=output,
                evaluation_seed=int(args.seed),
                resume=bool(args.resume),
            )
        )
        completed_episodes = 3 * matched_episode_count
    else:  # The population parser should make this unreachable.
        raise ValueError(
            f"Unknown evaluation kind: {population.evaluation_kind}"
        )

    write_run_metadata(
        output / "run_metadata.json",
        config=config,
        seed=int(args.seed),
        effective_environment_steps=int(effective_environment_steps),
        update_count=0,
        completed_episodes=int(completed_episodes),
    )
    print(f"Complete evaluation rows: {output}")


__all__ = ["run_evaluation"]
