"""Official SP/OP upstream training application for Path C."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.overcooked_v2.official_adapter import (
    compose_official_config,
    train_upstream,
)
from src.path_c.experiment import load_config
from src.path_c.storage import (
    ensure_run_identity,
    upstream_identity,
    write_array_chunks,
    write_json,
    write_run_metadata,
)


def run_upstream(args: argparse.Namespace) -> None:
    """Run one locked official upstream training job and preserve all outputs."""

    config = load_config(args.config, run_kind=args.run_kind)
    output = Path(args.output).resolve()
    ensure_run_identity(
        output,
        upstream_identity(
            config=config,
            seed=int(args.seed),
            algorithm=str(args.algorithm),
        ),
    )
    official = compose_official_config(
        config,
        algorithm=args.algorithm,
        seed=int(args.seed),
        output_directory=output,
    )
    result = train_upstream(
        official,
        seed=int(args.seed),
        checkpoint_progress=config.upstream.checkpoint_progress,
    )
    write_json(output / "resolved_config.json", config.to_mapping())
    write_json(output / "official_resolved_config.json", official)

    metric_paths: dict[str, list[str]] = {}
    for name, values in result["metrics"].items():
        metric_paths[str(name)] = [
            str(path)
            for path in write_array_chunks(
                output / "metrics",
                name=str(name),
                values=values,
                rows_per_chunk=1024,
            )
        ]
    write_json(
        output / "upstream_summary.json",
        {
            "algorithm": str(args.algorithm),
            "seed": int(args.seed),
            "effective_environment_steps": int(
                result["effective_environment_steps"]
            ),
            "completed_episodes": int(result["completed_episodes"]),
            "update_count": int(result["update_count"]),
            "checkpoint_paths": [
                str(path) for path in result["checkpoint_paths"]
            ],
            "metric_files": metric_paths,
        },
    )
    write_run_metadata(
        output / "run_metadata.json",
        config=config,
        seed=int(args.seed),
        effective_environment_steps=int(result["effective_environment_steps"]),
        update_count=int(result["update_count"]),
        completed_episodes=int(result["completed_episodes"]),
    )
    print(f"Complete upstream metrics: {output / 'metrics'}")


__all__ = ["run_upstream"]
