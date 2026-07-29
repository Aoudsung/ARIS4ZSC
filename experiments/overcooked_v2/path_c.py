"""Single command entry for Path C experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.overcooked_v2.counterfactual_audit_app import (
    run_counterfactual_value_audit,
)
from experiments.overcooked_v2.evaluation_app import run_evaluation
from experiments.overcooked_v2.manifest_app import add_manifest_commands
from experiments.overcooked_v2.partner_panel_app import run_partner_panel
from experiments.overcooked_v2.training_app import run_training
from experiments.overcooked_v2.upstream_app import run_upstream
from src.path_c.experiment import RUN_KINDS
from src.path_c.storage import CompleteConsoleLog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.overcooked_v2.path_c")
    commands = parser.add_subparsers(dest="command", required=True)

    upstream = commands.add_parser("upstream")
    upstream.add_argument("--config", required=True)
    upstream.add_argument("--algorithm", choices=("rnn-sp", "rnn-op"), required=True)
    upstream.add_argument("--seed", type=int, required=True)
    upstream.add_argument(
        "--run-kind", choices=("development", "formal"), required=True
    )
    upstream.add_argument("--output", required=True)
    upstream.set_defaults(function=run_upstream, manages_output=True)

    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--unit-manifest", required=True)
    train.add_argument("--outer-unit", type=int, choices=range(10), required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--run-kind", choices=RUN_KINDS, required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true")
    train.set_defaults(function=run_training, manages_output=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument(
        "--run-kind", choices=("development", "formal"), required=True
    )
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.set_defaults(function=run_evaluation, manages_output=True)

    panel = commands.add_parser("evaluate-panel")
    panel.add_argument("--config", required=True)
    panel.add_argument("--unit-manifest", required=True)
    panel.add_argument("--run-directory", required=True)
    panel.add_argument("--outer-unit", type=int, choices=range(10), required=True)
    panel.add_argument("--seed", type=int, required=True)
    panel.add_argument(
        "--run-kind", choices=("mechanical", "development"), required=True
    )
    panel.add_argument("--output", required=True)
    panel.add_argument("--resume", action="store_true")
    panel.add_argument(
        "--final-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the final checkpoint from each start/midpoint/final partner run.",
    )
    panel.add_argument(
        "--episodes-per-pairing",
        type=int,
        help=(
            "Override the panel episode count only for a mechanical wiring check."
        ),
    )
    panel.set_defaults(function=run_partner_panel, manages_output=True)

    audit = commands.add_parser("audit-counterfactual-value")
    audit.add_argument("--config", required=True)
    audit.add_argument("--run-directory", required=True)
    audit.add_argument("--checkpoint-step", type=int, required=True)
    audit.add_argument("--response-contrast", required=True)
    audit.add_argument("--panel-manifest", required=True)
    audit.add_argument("--replicas", type=int, default=128)
    audit.add_argument("--output", required=True)
    audit.add_argument("--resume", action="store_true")
    audit.set_defaults(
        function=run_counterfactual_value_audit, manages_output=True
    )

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
