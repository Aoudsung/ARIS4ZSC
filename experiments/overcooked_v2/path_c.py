"""Single command entry for Path C experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.overcooked_v2.evaluation_app import run_evaluation
from experiments.overcooked_v2.manifest_app import add_manifest_commands
from experiments.overcooked_v2.training_app import run_training
from experiments.overcooked_v2.upstream_app import run_upstream
from src.path_c.storage import CompleteConsoleLog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.overcooked_v2.path_c")
    commands = parser.add_subparsers(dest="command", required=True)

    upstream = commands.add_parser("upstream")
    upstream.add_argument("--config", required=True)
    upstream.add_argument("--algorithm", choices=("rnn-sp", "rnn-op"), required=True)
    upstream.add_argument("--seed", type=int, required=True)
    upstream.add_argument("--run-kind", choices=("development", "formal"), required=True)
    upstream.add_argument("--output", required=True)
    upstream.set_defaults(function=run_upstream, manages_output=True)

    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--unit-manifest", required=True)
    train.add_argument("--outer-unit", type=int, choices=range(10), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--run-kind", choices=("development", "formal"), required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true")
    train.set_defaults(function=run_training, manages_output=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--run-kind", choices=("development", "formal"), required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.set_defaults(function=run_evaluation, manages_output=True)
    add_manifest_commands(commands)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.manages_output:
        args.function(args)
        return
    output = Path(args.output).resolve()
    with CompleteConsoleLog(output / "logs") as console_log:
        print(f"Complete standard output: {console_log.stdout_path}")
        print(f"Complete standard error: {console_log.stderr_path}")
        args.function(args)


if __name__ == "__main__":
    main()
