"""Single command entry for DELTA-ZSC V6 E2E."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.overcooked_v2.calibration_app import run_safety_calibration
from experiments.overcooked_v2.common_partner_app import run_common_partner_evaluation
from experiments.overcooked_v2.manifest_app import add_manifest_command
from experiments.overcooked_v2.mechanical_e2e_app import run_mechanical_e2e
from experiments.overcooked_v2.official_baseline_app import run_official_baseline
from experiments.overcooked_v2.official_br_prox_app import run_common_br_prox
from experiments.overcooked_v2.official_evaluation_app import (
    run_build_delta_policy_manifest,
    run_capacity_summary,
    run_official_evaluation,
    run_official_summary,
)
from experiments.overcooked_v2.training_app import run_cuda_preflight, run_training
from experiments.overcooked_v2.signal_audit_app import run_signal_audit
from experiments.overcooked_v2.upstream_app import run_upstream
from experiments.overcooked_v2.resource_report_app import run_resource_report
from experiments.overcooked_v2.formal_claim_app import run_formal_claim_report
from src.path_c.experiment import (
    ENGINEERING_SEED_INDEX,
    RUN_KINDS,
    load_config,
    load_partner_manifest,
    validate_seed_training_manifest,
)
from src.path_c.storage import CompleteConsoleLog


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.overcooked_v2.path_c"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    add_manifest_command(commands)

    mechanical = commands.add_parser("mechanical-e2e")
    mechanical.add_argument(
        "--config",
        default=(
            "experiments/overcooked_v2/configs/"
            "delta_zsc_simple_mechanical_e2e.yaml"
        ),
    )
    mechanical.add_argument("--partner-manifest", required=True)
    mechanical.add_argument("--ego-run-id", default="delta-zsc-v6-engineering")
    mechanical.add_argument("--require-cuda", action="store_true", default=False)
    mechanical.add_argument("--resume", action="store_true", default=False)
    mechanical.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )
    mechanical.add_argument("--output", required=True)
    mechanical.set_defaults(function=run_mechanical_e2e, manages_output=True)

    upstream = commands.add_parser("upstream")
    upstream.add_argument("--config", required=True)
    upstream.add_argument("--algorithm", choices=("rnn-sp", "rnn-op"), required=True)
    upstream.add_argument("--seed-index", type=int, choices=range(10), required=True)
    upstream.add_argument("--run-kind", choices=RUN_KINDS, required=True)
    upstream.add_argument(
        "--jax-prng-key",
        type=int,
        nargs=2,
        metavar=("WORD0", "WORD1"),
        help=(
            "Explicit two-word key for mechanical lineage fixtures only; "
            "formal and development runs always use split(PRNGKey(42), 10)."
        ),
    )
    upstream.add_argument("--output", required=True)
    upstream.set_defaults(function=run_upstream, manages_output=True)

    baseline = commands.add_parser("train-official-baseline")
    baseline.add_argument(
        "--method",
        choices=("sp", "state-augmented", "op", "fcp", "ippo-large"),
        required=True,
    )
    baseline.add_argument(
        "--layout", choices=("test_time_simple", "test_time_wide"), required=True
    )
    baseline.add_argument("--fcp-population")
    baseline.add_argument("--fcp-population-ledger")
    baseline.add_argument("--training-lineage-manifest")
    baseline.add_argument("--delta-deployment")
    baseline.add_argument("--output", required=True)
    baseline.set_defaults(function=run_official_baseline, manages_output=True)

    validate = commands.add_parser("validate-manifest")
    validate.add_argument("--config", required=True)
    validate.add_argument("--partner-manifest", required=True)
    validate.add_argument("--run-kind", choices=RUN_KINDS, required=True)
    validate.add_argument(
        "--seed-index",
        type=int,
        choices=(ENGINEERING_SEED_INDEX, *range(10)),
        help="Required for per-seed V6 initialization and training manifests.",
    )
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
        if args.run_kind == "formal" and args.seed_index is None:
            raise ValueError("Formal V6 manifest validation requires --seed-index.")
        if args.seed_index is not None:
            validate_seed_training_manifest(
                manifest,
                owner_seed_index=int(args.seed_index),
                formal=(args.run_kind == "formal"),
            )
        print(f"Valid DELTA-ZSC manifest: {len(manifest.runs)} runs")

    validate.set_defaults(function=validate_manifest, manages_output=False)

    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--partner-manifest", required=True)
    train.add_argument("--ego-run-id", required=True)
    train.add_argument(
        "--seed-index",
        type=int,
        choices=(ENGINEERING_SEED_INDEX, *range(10)),
        required=True,
    )
    train.add_argument("--run-kind", choices=RUN_KINDS, required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )
    train.set_defaults(function=run_training, manages_output=True)

    cuda_preflight = commands.add_parser("cuda-preflight")
    cuda_preflight.add_argument("--config", required=True)
    cuda_preflight.add_argument("--partner-manifest", required=True)
    cuda_preflight.add_argument(
        "--seed-index",
        type=int,
        choices=(ENGINEERING_SEED_INDEX,),
        default=ENGINEERING_SEED_INDEX,
    )
    cuda_preflight.add_argument(
        "--ego-run-id", default="delta-zsc-formal-cuda-preflight"
    )
    cuda_preflight.add_argument("--output", required=True)
    cuda_preflight.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )
    cuda_preflight.set_defaults(
        function=run_cuda_preflight,
        manages_output=True,
        run_kind="formal",
        resume=False,
    )

    calibrate = commands.add_parser("calibrate-safety")
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--partner-manifest", required=True)
    calibrate.add_argument("--training-run", required=True)
    calibrate.add_argument(
        "--seed-index",
        type=int,
        choices=(ENGINEERING_SEED_INDEX, *range(10)),
        required=True,
    )
    calibrate.add_argument("--run-kind", choices=RUN_KINDS, required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )
    calibrate.set_defaults(function=run_safety_calibration, manages_output=True)

    audit = commands.add_parser("audit-signals")
    audit.add_argument("--training-run", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(function=run_signal_audit, manages_output=True)

    delta_manifest = commands.add_parser("build-delta-policy-manifest")
    delta_manifest.add_argument("--deployments", nargs=10, required=True)
    delta_manifest.add_argument("--output", required=True)
    delta_manifest.set_defaults(
        function=run_build_delta_policy_manifest, manages_output=False
    )

    official = commands.add_parser("evaluate-official")
    official.add_argument("--config", required=True)
    official.add_argument("--policy-manifest", required=True)
    official.add_argument("--output", required=True)
    official.set_defaults(function=run_official_evaluation, manages_output=True)

    official_summary = commands.add_parser("summarize-official")
    official_summary.add_argument(
        "--result",
        action="append",
        required=True,
        help="LAYOUT:METHOD=/official/evaluation/directory; provide all ten",
    )
    official_summary.add_argument("--output", required=True)
    official_summary.set_defaults(function=run_official_summary, manages_output=True)

    capacity_summary = commands.add_parser("summarize-capacity-control")
    capacity_summary.add_argument(
        "--result",
        action="append",
        required=True,
        help="LAYOUT:METHOD=/official/evaluation/directory; DELTA and IPPO-Large on both layouts",
    )
    capacity_summary.add_argument("--output", required=True)
    capacity_summary.set_defaults(function=run_capacity_summary, manages_output=True)

    common = commands.add_parser("evaluate-common")
    common.add_argument("--config", required=True)
    common.add_argument("--partner-manifest", required=True)
    common.add_argument(
        "--policy-manifest",
        action="append",
        required=True,
        help="METHOD=/policy_manifest.json; provide all five methods",
    )
    common.add_argument("--br-prox-result", required=True)
    common.add_argument("--output", required=True)
    common.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )
    common.set_defaults(function=run_common_partner_evaluation, manages_output=True)

    common_br = commands.add_parser("evaluate-common-br-prox")
    common_br.add_argument("--config", required=True)
    common_br.add_argument("--partner-manifest", required=True)
    common_br.add_argument(
        "--policy-manifest",
        action="append",
        required=True,
        help="METHOD=/policy_manifest.json; provide all five methods",
    )
    common_br.add_argument("--output", required=True)
    common_br.add_argument(
        "--skip-manifest-hash-check", action="store_true", default=False
    )
    common_br.set_defaults(function=run_common_br_prox, manages_output=True)

    resources = commands.add_parser("summarize-resources")
    resources.add_argument(
        "--ledger",
        action="append",
        required=True,
        help="METHOD=/path/resource_ledger.json; repeat for every independent artifact",
    )
    resources.add_argument("--output", required=True)
    resources.set_defaults(function=run_resource_report, manages_output=True)

    claims = commands.add_parser("build-formal-claim-report")
    claims.add_argument("--official-summary", required=True)
    claims.add_argument("--common-simple", required=True)
    claims.add_argument("--common-wide", required=True)
    claims.add_argument("--capacity-summary", required=True)
    claims.add_argument("--resource-report", required=True)
    claims.add_argument("--ablation-gate")
    claims.add_argument("--output", required=True)
    claims.set_defaults(function=run_formal_claim_report, manages_output=True)

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
