"""Single command entry for the active DELTA-ZSC v5 implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.overcooked_v2.calibration_app import run_calibration
from experiments.overcooked_v2.evaluation_app import run_evaluation
from experiments.overcooked_v2.manifest_app import add_manifest_command
from experiments.overcooked_v2.training_app import run_training
from experiments.overcooked_v2.upstream_app import run_upstream
from src.path_c.experiment import RUN_KINDS, load_config, load_partner_manifest
from src.path_c.storage import CompleteConsoleLog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.overcooked_v2.path_c"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    add_manifest_command(commands)

    upstream = commands.add_parser("upstream")
    upstream.add_argument("--config", required=True)
    upstream.add_argument("--algorithm", choices=("rnn-sp", "rnn-op"), required=True)
    upstream.add_argument("--seed", type=int, required=True)
    upstream.add_argument("--run-kind", choices=("development", "formal"), required=True)
    upstream.add_argument("--output", required=True)
    upstream.set_defaults(function=run_upstream, manages_output=True)

    validate = commands.add_parser("validate-manifest")
    validate.add_argument("--config", required=True)
    validate.add_argument("--partner-manifest", required=True)
    validate.add_argument("--run-kind", choices=RUN_KINDS, required=True)
    validate.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )

    def validate_manifest(args: argparse.Namespace) -> None:
        config = load_config(args.config, run_kind=args.run_kind)
        manifest = load_partner_manifest(
            args.partner_manifest,
            expected_layout=config.environment.layout,
            verify_files=not bool(args.skip_manifest_hash_check),
        )
        print(f"Valid DELTA-ZSC manifest: {len(manifest.runs)} runs")

    validate.set_defaults(function=validate_manifest, manages_output=False)

    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--partner-manifest", required=True)
    train.add_argument("--ego-run-id", required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--run-kind", choices=RUN_KINDS, required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )
    train.set_defaults(function=run_training, manages_output=True)

    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--partner-manifest", required=True)
    calibrate.add_argument("--training-run", required=True)
    calibrate.add_argument("--seed", type=int, required=True)
    calibrate.add_argument("--run-kind", choices=RUN_KINDS, required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )
    calibrate.set_defaults(function=run_calibration, manages_output=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--partner-manifest", required=True)
    evaluate.add_argument("--deployments", nargs="+", required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--run-kind", choices=RUN_KINDS, required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--allow-scientific-readout", action="store_true")
    evaluate.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )
    evaluate.set_defaults(function=run_evaluation, manages_output=True)

    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.manages_output:
        args.function(args)
        return
    output = Path(args.output).resolve()
    with CompleteConsoleLog(output / "logs") as log:
        print(f"Complete stdout: {log.stdout_path}")
        print(f"Complete stderr: {log.stderr_path}")
        args.function(args)


if __name__ == "__main__":
    main()
