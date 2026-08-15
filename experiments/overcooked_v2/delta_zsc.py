"""Command-line entry for the unified DELTA-ZSC implementation."""

from __future__ import annotations

import argparse

from .baseline_app import BASELINE_METHODS, run_official_baseline
from .calibration_app import (
    build_semantic_initializer,
    run_posterior_predictive_diagnostics,
)
from .development_matrix_app import (
    evaluate_development_matrix,
    run_development_matrix,
    summarize_development_matrix,
)
from .evaluation_app import (
    build_policy_manifest,
    run_evaluation,
    summarize_evaluations,
    summarize_population_matrices,
)
from .formal_claim_app import build_formal_claim_report
from .intervention_app import (
    run_belief_value_intervention,
    run_decision_chain_audit,
    summarize_decision_chain_audit,
)
from .delta_manifest_app import build_partner_manifest, validate_partner_manifest_command
from .resource_report_app import run_resource_report
from .training_app import run_cuda_preflight, run_training
from .upstream_app import train_official_parent
from .upstream_pipeline_app import run_upstream


def _common_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run-kind", choices=("mechanical", "development", "formal"), required=True
    )
    parser.add_argument("--partner-manifest", required=True)
    parser.add_argument("--skip-manifest-file-check", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.overcooked_v2.delta_zsc",
        description=(
            "Unified response-decision latent coordination with exact filtering, "
            "KL mirror improvement, and deterministic Bayesian VOI."
        ),
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

    train = commands.add_parser("train")
    _common_run(train)
    train.add_argument("--ego-run-id", required=True)
    train.add_argument("--seed-index", type=int, choices=tuple(range(-1, 10)), required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--require-cuda", action="store_true")
    train.add_argument("--semantic-initializer")
    train.add_argument(
        "--sp-initializer",
        required=True,
        help=(
            "Seed-matched Official SP checkpoint directory embedded as the "
            "immutable reference actor. The trainable residual starts at zero."
        ),
    )
    train.set_defaults(function=run_training)

    preflight = commands.add_parser("cuda-preflight")
    _common_run(preflight)
    preflight.add_argument("--ego-run-id", default="delta-cuda-preflight")
    preflight.add_argument("--seed-index", type=int, default=-1)
    preflight.add_argument("--output", required=True)
    preflight.add_argument("--semantic-initializer")
    preflight.add_argument("--sp-initializer", required=True)
    preflight.set_defaults(function=run_cuda_preflight)

    policy_manifest = commands.add_parser("build-policy-manifest")
    policy_manifest.add_argument("--method", required=True)
    policy_manifest.add_argument("--layout", required=True)
    policy_manifest.add_argument(
        "--policy-kind", choices=("delta_deployment", "official_checkpoint"), required=True
    )
    policy_manifest.add_argument("--policy", action="append", required=True)
    policy_manifest.add_argument("--training-lineage-manifest")
    policy_manifest.add_argument("--run-count", type=int, required=True)
    policy_manifest.add_argument("--output", required=True)
    policy_manifest.set_defaults(function=build_policy_manifest)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument(
        "--run-kind", choices=("mechanical", "development", "formal"), required=True
    )
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
    evaluate.add_argument(
        "--execution-mode",
        choices=("reference_only", "residual", "passive", "active"),
        default="active",
    )
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

    initializer = commands.add_parser("build-semantic-initializer")
    _common_run(initializer)
    initializer.add_argument("--deployment", required=True)
    initializer.add_argument(
        "--partner-role", choices=("calibration",), default="calibration"
    )
    initializer.add_argument(
        "--component-count",
        action="append",
        type=int,
        choices=(2, 4, 8),
        help=(
            "Build one or more K-specific simplex artifacts from the same "
            "unlabeled residual panel. Defaults to K from --config."
        ),
    )
    initializer.add_argument("--output", required=True)
    initializer.set_defaults(function=build_semantic_initializer)

    diagnostics = commands.add_parser("posterior-diagnostics")
    _common_run(diagnostics)
    diagnostics.add_argument("--deployment", required=True)
    diagnostics.add_argument(
        "--partner-role",
        choices=("calibration", "development_support"),
        default="calibration",
        help=("Confirmatory is deliberately not offered: reading it during "
              "development is what panel disjointness forbids."),
    )
    diagnostics.add_argument("--output", required=True)
    diagnostics.set_defaults(function=run_posterior_predictive_diagnostics)

    intervention = commands.add_parser("belief-intervention")
    _common_run(intervention)
    intervention.add_argument("--deployment", action="append", required=True)
    intervention.add_argument("--output", required=True)
    intervention.set_defaults(function=run_belief_value_intervention)

    decision_audit = commands.add_parser("decision-chain-audit")
    _common_run(decision_audit)
    decision_audit.add_argument("--deployment", action="append", required=True)
    decision_audit.add_argument(
        "--partner-role",
        choices=("development_support",),
        default="development_support",
    )
    decision_audit.add_argument("--seed", type=int, default=0)
    decision_audit.add_argument("--trajectory-steps", type=int, default=256)
    decision_audit.add_argument("--anchor-states", type=int, default=240)
    decision_audit.add_argument("--output", required=True)
    decision_audit.set_defaults(function=run_decision_chain_audit)

    decision_report = commands.add_parser("summarize-decision-chain-audit")
    decision_report.add_argument("--audit", required=True)
    decision_report.add_argument("--evaluation", action="append", required=True)
    decision_report.add_argument("--bootstrap-replicates", type=int, default=9_999)
    decision_report.add_argument("--seed", type=int, default=0)
    decision_report.add_argument("--output", required=True)
    decision_report.set_defaults(function=summarize_decision_chain_audit)

    matrix = commands.add_parser("run-development-matrix")
    _common_run(matrix)
    matrix.add_argument("--seed-index", action="append", type=int, required=True)
    matrix.add_argument("--output", required=True)
    matrix.add_argument("--resume", action="store_true")
    matrix.add_argument("--require-cuda", action="store_true")
    matrix.add_argument("--semantic-initializer", required=True)
    matrix.add_argument("--sp-initializer-root", required=True)
    matrix.set_defaults(function=run_development_matrix)

    matrix_eval = commands.add_parser("evaluate-development-matrix")
    matrix_eval.add_argument("--matrix", required=True)
    matrix_eval.add_argument("--config", required=True)
    matrix_eval.add_argument("--partner-manifest", required=True)
    matrix_eval.add_argument("--layout", required=True)
    matrix_eval.add_argument("--output", required=True)
    matrix_eval.add_argument("--skip-manifest-file-check", action="store_true")
    matrix_eval.set_defaults(function=evaluate_development_matrix)

    matrix_summary = commands.add_parser("summarize-development-matrix")
    matrix_summary.add_argument("--evaluations", required=True)
    matrix_summary.add_argument("--layout", required=True)
    matrix_summary.add_argument("--output", required=True)
    matrix_summary.set_defaults(function=summarize_development_matrix)

    baseline = commands.add_parser("train-baseline")
    baseline.add_argument("--method", choices=BASELINE_METHODS, required=True)
    baseline.add_argument("--layout", required=True)
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--fcp-population")
    baseline.add_argument("--fcp-population-ledger")
    baseline.add_argument("--delta-deployment")
    baseline.add_argument("--shared-training-ledger")
    baseline.add_argument("--training-lineage-manifest")
    baseline.set_defaults(function=run_official_baseline)

    parent = commands.add_parser("train-official-parent")
    parent.add_argument("--config", required=True)
    parent.add_argument(
        "--run-kind", choices=("mechanical", "development", "formal"), required=True
    )
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

    formal = commands.add_parser("formal-claim")
    formal.add_argument("--official-summary", action="append", required=True)
    formal.add_argument("--development-summary", action="append", required=True)
    formal.add_argument("--belief-intervention", action="append", required=True)
    formal.add_argument("--posterior-diagnostics", action="append", required=True)
    formal.add_argument("--resource-report", required=True)
    formal.add_argument("--output", required=True)
    formal.set_defaults(function=build_formal_claim_report)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
