"""Build the two manifests consumed by Path C from completed run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.path_c.experiment import (
    MANIFEST_VERSION,
    METHOD_VERSION,
    Population,
    PopulationEntry,
    load_training_unit_manifest,
    write_population,
)
from src.path_c.storage import read_run_identity, write_json


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON mapping in {path}.")
    return payload


def _upstream_checkpoints(
    run_directory: Path, *, layout: str, run_kind: str
) -> tuple[Path, ...]:
    identity = read_run_identity(run_directory)
    if (
        identity.get("stage") != "upstream"
        or identity.get("layout") != layout
        or identity.get("run_kind") != run_kind
    ):
        raise ValueError(
            f"{run_directory} is not a {run_kind} upstream run for {layout}."
        )
    summary = _load_json(run_directory / "upstream_summary.json")
    checkpoints = tuple(Path(value).resolve() for value in summary.get("checkpoint_paths", ()))
    if len(checkpoints) != 3:
        raise ValueError(f"{run_directory} must expose start, midpoint, and final checkpoints.")
    return checkpoints


def build_unit_manifest(
    *, plan_path: str | Path, output_path: str | Path, run_kind: str
) -> Path:
    """Expand upstream run directories into the literal training-unit manifest."""

    plan_file = Path(plan_path).resolve()
    plan = _load_json(plan_file)
    if set(plan) != {"version", "layout", "units"} or int(plan["version"]) != 1:
        raise ValueError("Unit plan fields must be version, layout, and units.")
    layout = str(plan["layout"])
    if not isinstance(plan["units"], Sequence):
        raise ValueError("Unit plan units must be a sequence.")
    units = []
    for index, raw in enumerate(plan["units"]):
        if not isinstance(raw, Mapping) or set(raw) != {
            "outer_unit_id",
            "reference_run",
            "partner_runs",
        }:
            raise ValueError(f"unit[{index}] has the wrong fields.")
        reference_run = (plan_file.parent / str(raw["reference_run"])).resolve()
        reference_checkpoints = _upstream_checkpoints(
            reference_run, layout=layout, run_kind=run_kind
        )
        partner_runs = raw["partner_runs"]
        if not isinstance(partner_runs, Sequence) or isinstance(partner_runs, (str, bytes)):
            raise ValueError("partner_runs must be a sequence.")
        partner_checkpoints = tuple(
            checkpoint
            for value in partner_runs
            for checkpoint in _upstream_checkpoints(
                (plan_file.parent / str(value)).resolve(),
                layout=layout,
                run_kind=run_kind,
            )
        )
        units.append(
            {
                "outer_unit_id": int(raw["outer_unit_id"]),
                "reference_checkpoint": str(reference_checkpoints[-1]),
                "partner_checkpoints": [str(path) for path in partner_checkpoints],
            }
        )
    output = write_json(
        output_path,
        {"version": MANIFEST_VERSION, "layout": layout, "units": units},
    )
    # The same parser used by training is the final check; no parallel validation exists.
    load_training_unit_manifest(output, expected_layout=layout, run_kind=run_kind)
    return output


def build_population(
    *,
    run_directories: Sequence[str | Path],
    name: str,
    evaluation_kind: str,
    output_path: str | Path,
) -> Path:
    runs = tuple(Path(value).resolve() for value in run_directories)
    if len(runs) != 10:
        raise ValueError("A standard population requires ten training run directories.")
    identities = [read_run_identity(path) for path in runs]
    if any(
        identity.get("stage") != "train" or identity.get("method") != METHOD_VERSION
        for identity in identities
    ):
        raise ValueError("Every population member must be a completed run of this method.")
    layouts = {str(identity.get("layout")) for identity in identities}
    configs = {json.dumps(identity.get("config"), sort_keys=True) for identity in identities}
    outer_units = [int(identity.get("outer_unit_id", -1)) for identity in identities]
    if len(layouts) != 1 or len(configs) != 1 or outer_units != list(range(10)):
        raise ValueError("Population runs must share one config and be ordered outer units 0 through 9.")
    population = Population(
        name=str(name),
        layout=next(iter(layouts)),
        evaluation_kind=str(evaluation_kind),
        entries=tuple(
            PopulationEntry(outer_unit_id=index, run_directory=path)
            for index, path in enumerate(runs)
        ),
    )
    return write_population(output_path, population)


def add_manifest_commands(commands: argparse._SubParsersAction) -> None:
    units = commands.add_parser("build-units")
    units.add_argument("--plan", required=True)
    units.add_argument("--run-kind", choices=("development", "formal"), required=True)
    units.add_argument("--output", required=True)
    units.set_defaults(
        function=lambda args: print(
            build_unit_manifest(
                plan_path=args.plan,
                output_path=args.output,
                run_kind=args.run_kind,
            )
        ),
        manages_output=False,
    )

    population = commands.add_parser("build-population")
    population.add_argument("--name", required=True)
    population.add_argument(
        "--evaluation-kind",
        choices=("standard_matrix", "response_contrast"),
        required=True,
    )
    population.add_argument("--runs", nargs=10, required=True)
    population.add_argument("--output", required=True)
    population.set_defaults(
        function=lambda args: print(
            build_population(
                run_directories=args.runs,
                name=args.name,
                evaluation_kind=args.evaluation_kind,
                output_path=args.output,
            )
        ),
        manages_output=False,
    )


__all__ = ["add_manifest_commands", "build_population", "build_unit_manifest"]
