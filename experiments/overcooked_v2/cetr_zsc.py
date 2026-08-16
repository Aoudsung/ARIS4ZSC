"""Command-line entry point for the CETR-ZSC application layer."""

from __future__ import annotations

import argparse

from .baseline_app import BASELINE_METHODS, run_official_baseline
from .claim_app import build_claim
from .evaluation_app import (
    build_policy_manifest,
    run_evaluation,
    summarize_evaluations,
    summarize_population_matrices,
)
from .partner_manifest_app import build_partner_manifest, validate_partner_manifest_command
from .reference_sp_app import measure_reference_sp
from .resource_report_app import run_resource_report
from .training_app import run_cuda_preflight, run_training
from .upstream_app import train_official_parent
from .upstream_pipeline_app import run_upstream


_RUN_KINDS = ("mechanical", "development", "formal")


def _common_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-kind", choices=_RUN_KINDS, required=True)
    parser.add_argument("--partner-manifest", required=True)
    parser.add_argument("--skip-manifest-file-check", action="store_true")


def _reference_sp(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-kind", choices=_RUN_KINDS, required=True)
    parser.add_argument("--sp-checkpoint", required=True)
    parser.add_argument("--seed-index", type=int, choices=tuple(range(-1, 10)), required=True)
    parser.add_argument("--output", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.overcooked_v2.cetr_zsc",
        description="Constrained episodic tail-robust zero-shot coordination.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("build-partner-manifest")
    manifest.add_argument("--plan", required=True)
    manifest.add_argument("--expected-layout")
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(function=build_partner_manifest)

    validate_manifest = commands.add_parser("validate-partner-manifest")
    validate_manifest.add_argument("--partner-manifest", required=True)
    validate_manifest.add_argument("--expected-layout")
    validate_manifest.add_argument("--skip-manifest-file-check", action="store_true")
    validate_manifest.set_defaults(function=validate_partner_manifest_command)

    reference = commands.add_parser("measure-reference-sp")
    _reference_sp(reference)
    reference.set_defaults(function=measure_reference_sp)

    train = commands.add_parser("train")
    _common_run(train)
    train.add_argument("--ego-run-id", required=True)
    train.add_argument("--seed-index", type=int, choices=tuple(range(-1, 10)), required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--require-cuda", action="store_true")
    train.add_argument("--sp-initializer", required=True)
    train.add_argument("--reference-sp-artifact", required=True)
    train.set_defaults(function=run_training)

    preflight = commands.add_parser("cuda-preflight")
    _common_run(preflight)
    preflight.add_argument("--ego-run-id", default="cetr-cuda-preflight")
    preflight.add_argument("--seed-index", type=int, default=-1)
    preflight.add_argument("--output", required=True)
    preflight.add_argument("--sp-initializer", required=True)
    preflight.add_argument("--reference-sp-artifact", required=True)
    preflight.set_defaults(function=run_cuda_preflight)

    policy_manifest = commands.add_parser("build-policy-manifest")
    policy_manifest.add_argument("--method", required=True)
    policy_manifest.add_argument("--layout", required=True)
    policy_manifest.add_argument(
        "--policy-kind", choices=("cetr_deployment", "official_checkpoint"), required=True
    )
    policy_manifest.add_argument("--policy", action="append", required=True)
    policy_manifest.add_argument("--training-lineage-manifest")
    policy_manifest.add_argument("--run-count", type=int, required=True)
    policy_manifest.add_argument("--output", required=True)
    policy_manifest.set_defaults(function=build_policy_manifest)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--run-kind", choices=_RUN_KINDS, required=True)
    evaluate.add_argument(
        "--evaluation-mode",
        choices=("common_partner", "population_matrix"),
        default="common_partner",
    )
    evaluate.add_argument("--partner-manifest")
    evaluate.add_argument("--skip-manifest-file-check", action="store_true")
    evaluate.add_argument("--policy-manifest")
    evaluate.add_argument("--left-policy-manifest")
    evaluate.add_argument("--right-policy-manifest")
    evaluate.add_argument("--partner-role", default="confirmatory")
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(function=run_evaluation)

    summarize = commands.add_parser("summarize-evaluations")
    summarize.add_argument("--evaluation", action="append", required=True)
    summarize.add_argument("--bootstrap-replicates", type=int, default=9_999)
    summarize.add_argument("--seed", type=int, default=0)
    summarize.add_argument("--output", required=True)
    summarize.set_defaults(function=summarize_evaluations)

    population_summary = commands.add_parser("summarize-population-matrices")
    population_summary.add_argument("--evaluation", action="append", required=True)
    population_summary.add_argument("--output", required=True)
    population_summary.set_defaults(function=summarize_population_matrices)

    claim = commands.add_parser("claim")
    claim.add_argument("--cetr-evaluation", action="append", required=True)
    claim.add_argument("--baseline-evaluation", action="append", required=True)
    claim.add_argument("--cetr-population", action="append", required=True)
    claim.add_argument("--reference-sp", action="append", required=True)
    claim.add_argument("--output", required=True)
    claim.set_defaults(function=build_claim)

    baseline = commands.add_parser("train-baseline")
    baseline.add_argument("--method", choices=BASELINE_METHODS, required=True)
    baseline.add_argument("--layout", required=True)
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--fcp-population")
    baseline.add_argument("--fcp-population-ledger")
    baseline.add_argument("--cetr-deployment")
    baseline.add_argument("--shared-training-ledger")
    baseline.add_argument("--training-lineage-manifest")
    baseline.set_defaults(function=run_official_baseline)

    parent = commands.add_parser("train-official-parent")
    parent.add_argument("--config", required=True)
    parent.add_argument("--run-kind", choices=_RUN_KINDS, required=True)
    parent.add_argument("--method", choices=("sp", "op"), required=True)
    parent.add_argument("--root-seed", type=int, required=True)
    parent.add_argument("--population-size", type=int, required=True)
    parent.add_argument("--seed-index", type=int, required=True)
    parent.add_argument("--parent-training-run-id", required=True)
    parent.add_argument("--output", required=True)
    parent.set_defaults(function=train_official_parent)

    upstream = commands.add_parser("run-upstream")
    upstream.add_argument("--config", required=True)
    upstream.add_argument("--output", required=True)
    upstream.set_defaults(function=run_upstream)

    resources = commands.add_parser("resource-report")
    resources.add_argument("--ledger", action="append", required=True)
    resources.add_argument("--output", required=True)
    resources.set_defaults(function=run_resource_report)

    return parser


def main() -> None:
    args = _parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
