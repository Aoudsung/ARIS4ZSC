"""Command entry for the unified DELTA-ZSC method."""

from __future__ import annotations

import argparse

from experiments.overcooked_v2.unified_calibration_app import run_calibration
from experiments.overcooked_v2.unified_causal_app import run_causal_evaluation
from experiments.overcooked_v2.unified_evaluation_app import run_evaluation
from experiments.overcooked_v2.unified_summary_app import run_summary
from experiments.overcooked_v2.unified_training_app import (
    TRAINING_VARIANTS,
    run_training,
)
from src.path_c.experiment import ENGINEERING_SEED_INDEX


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.overcooked_v2.delta_zsc",
        description=(
            "Unified response-decision Bayesian coordination with analytic "
            "KL-constrained policy improvement."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--partner-manifest", required=True)
    train.add_argument("--variant", choices=TRAINING_VARIANTS, required=True)
    train.add_argument(
        "--seed-index",
        type=int,
        choices=(ENGINEERING_SEED_INDEX, *range(10)),
        required=True,
    )
    train.add_argument(
        "--run-kind",
        choices=("mechanical", "development", "formal"),
        required=True,
    )
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true", default=False)
    train.add_argument("--require-cuda", action="store_true", default=False)
    train.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )
    train.set_defaults(function=run_training)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--deployment", required=True)
    evaluate.add_argument("--partner-manifest", required=True)
    evaluate.add_argument(
        "--partner-role", default="confirmatory", choices=("confirmatory", "calibration")
    )
    evaluate.add_argument("--variant-override", choices=("joint", "full"))
    evaluate.add_argument("--episodes", type=int)
    evaluate.add_argument("--evaluation-seed", type=int, default=0)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )
    evaluate.set_defaults(function=run_evaluation)

    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--deployment", required=True)
    calibrate.add_argument("--partner-manifest", required=True)
    calibrate.add_argument("--episodes-per-run", type=int, default=32)
    calibrate.add_argument("--seed", type=int, default=0)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )
    calibrate.set_defaults(function=run_calibration)

    causal = commands.add_parser("evaluate-causal-belief")
    causal.add_argument("--deployment", required=True)
    causal.add_argument("--partner-manifest", required=True)
    causal.add_argument(
        "--partner-role", default="confirmatory", choices=("confirmatory", "calibration")
    )
    causal.add_argument("--num-envs", type=int, default=64)
    causal.add_argument("--anchor-count", type=int, default=64)
    causal.add_argument("--fit-replicas", type=int, default=4)
    causal.add_argument("--evaluation-replicas", type=int, default=8)
    causal.add_argument("--seed", type=int, default=0)
    causal.add_argument("--output", required=True)
    causal.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )
    causal.set_defaults(function=run_causal_evaluation)

    summary = commands.add_parser("summarize")
    summary.add_argument(
        "--evaluation",
        action="append",
        required=True,
        help=(
            "LABEL=/evaluation/directory. Supply all base/response_only/joint/full "
            "seed runs and any external baselines."
        ),
    )
    summary.add_argument("--causal-evaluation", required=True)
    summary.add_argument("--bootstrap-replicates", type=int, default=9999)
    summary.add_argument("--bootstrap-seed", type=int, default=0)
    summary.add_argument("--minimum-effect", type=float, default=20.0)
    summary.add_argument("--output", required=True)
    summary.set_defaults(function=run_summary)

    return parser


def main() -> None:
    args = _parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
