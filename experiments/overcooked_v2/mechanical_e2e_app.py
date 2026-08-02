"""Non-scientific, small-budget end-to-end acceptance run for DELTA-ZSC.

The command intentionally executes the same public applications used by real
runs.  Only execution budgets and replica counts are reduced.  Its artifacts
must never be interpreted as benchmark or mechanism evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from src.path_c.experiment import (
    METHOD_VERSION,
    OFFICIAL_SOURCE_COMMIT,
    load_config,
    official_training_key,
)
from src.path_c.curriculum import phase_from_qualification
from src.path_c.fallback import deployment_tier
from src.path_c.qualification import GateDecision, SignalQualification
from src.path_c.signal_audit import fixture_contract_outcome
from src.path_c.storage import write_json


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return value


def generator_update_executed(metrics: Mapping[str, Any]) -> bool:
    """Return whether telemetry proves that the real generator update ran."""

    if float(metrics.get("generator_update_skipped", 1.0)) == 0.0:
        return True
    # Compatibility with run telemetry written before the explicit zero marker:
    # the loss bundle exists only after evaluating the score-function objective.
    return (
        "generator_total_loss" in metrics
        and float(metrics.get("generator_active_lanes", 0.0)) > 0.0
    )


def mechanical_fixture_key(label: str) -> tuple[int, tuple[int, int]]:
    """Derive one stable fixture key from a named, non-scientific domain."""

    import jax
    import numpy as np

    payload = f"delta-zsc-v5/mechanical-e2e/{label}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
    key = np.asarray(jax.random.PRNGKey(seed), dtype=np.uint32)
    return seed, (int(key[0]), int(key[1]))


def mechanical_upstream_seed_slot(sequence_index: int) -> int:
    """Map non-scientific fixtures onto the ten legal Official run slots."""

    index = int(sequence_index)
    if index < 0:
        raise ValueError("Mechanical upstream sequence indexes are non-negative.")
    return index % 10


def _run(
    command: Sequence[str],
    *,
    repository: Path,
    status_path: Path,
    stage: str,
    completed: list[str],
) -> None:
    write_json(
        status_path,
        {
            "status": "running",
            "stage": stage,
            "completed_stages": completed,
            "scientific_readout_allowed": False,
        },
    )
    try:
        subprocess.run(tuple(command), cwd=repository, check=True)
    except subprocess.CalledProcessError as error:
        write_json(
            status_path,
            {
                "status": "failed",
                "stage": stage,
                "completed_stages": completed,
                "exit_code": int(error.returncode),
                "scientific_readout_allowed": False,
                "note": (
                    "The mechanical acceptance stopped at the first failing "
                    "real application stage. No later stage was started."
                ),
            },
        )
        raise
    completed.append(stage)


def _upstream_command(
    *,
    config: Path,
    algorithm: str,
    seed_index: int,
    output: Path,
    key: tuple[int, int] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.overcooked_v2.path_c",
        "upstream",
        "--config",
        str(config),
        "--algorithm",
        algorithm,
        "--seed-index",
        str(seed_index),
        "--run-kind",
        "mechanical",
        "--output",
        str(output),
    ]
    if key is not None:
        command.extend(("--jax-prng-key", str(key[0]), str(key[1])))
    return command


def _plan_run(
    *,
    run_id: str,
    role: str,
    checkpoint: str,
    parent: str,
    mechanism: str,
    seed: int,
    seed_index: int | None,
    key: tuple[int, int],
    owner: int | None,
) -> Mapping[str, Any]:
    return {
        "run_id": run_id,
        "role": role,
        "checkpoint": checkpoint,
        "parent_training_run_id": parent,
        "generation_mechanism": mechanism,
        "seed": int(seed),
        "seed_index": seed_index,
        "jax_prng_key": list(key),
        "owner_seed_index": owner,
        "co_training_group_id": None,
        "partner_type_id": None,
    }


def _verify(output: Path) -> Mapping[str, Any]:
    upstream_summaries = sorted(output.glob("upstream/*/upstream_summary.json"))
    if len(upstream_summaries) != 11:
        raise RuntimeError("Mechanical E2E did not complete all eleven upstream fixtures.")
    for path in upstream_summaries:
        summary = _read_json(path)
        if (
            int(summary["effective_environment_steps"]) != 1_024
            or int(summary["update_count"]) != 16
            or len(summary["checkpoint_paths"]) != 3
        ):
            raise RuntimeError(f"Unexpected mechanical upstream budget: {path}")

    training = _read_json(output / "delta_train" / "run_metadata.json")
    if (
        int(training["effective_environment_steps"]) != 1_024
        or int(training["update_count"]) != 16
    ):
        raise RuntimeError("DELTA mechanical trajectory budget is incomplete.")
    fixture = _read_json(output / "signal_contract_fixture.json")
    if fixture["qualified_path"]["final_phase"] != "FULL_DELTA":
        raise RuntimeError("Mechanical C0--C5 state-machine fixture did not complete.")
    if fixture["fallback_path"]["deployment_tier"] != "owner_sp_source_fallback":
        raise RuntimeError("Mechanical qualification-failure fallback was not exercised.")
    checkpoint_identities = sorted(
        (output / "delta_train" / "checkpoint_identity").glob("*.json")
    )
    if not checkpoint_identities:
        raise RuntimeError("r3 checkpoint/resume identity sidecar is missing.")
    metrics = []
    for path in sorted(
        (output / "delta_train" / "records" / "metrics").glob("*.jsonl")
    ):
        metrics.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not any(generator_update_executed(row["generator"]) for row in metrics):
        raise RuntimeError("The continuous partner generator never updated.")
    ledger = _read_json(output / "delta_train" / "resource_ledger.json")
    if int(ledger["generator_training_steps"]) != 16 * 4 * 400:
        raise RuntimeError("Mechanical complete-episode generator budget is incomplete.")

    calibration = _read_json(output / "calibration" / "run_metadata.json")
    if int(calibration["calibration_partner_runs"]) != 2:
        raise RuntimeError("Run-block calibration did not use both fixture runs.")
    deployment = output / "calibration" / "deployment"
    if not deployment.is_dir():
        raise RuntimeError("Pruned deployment bundle is missing.")

    evaluation = _read_json(output / "evaluation" / "summary.json")
    if bool(evaluation["scientific_readout_allowed"]):
        raise RuntimeError("Mechanical evaluation must never enable scientific readout.")
    if int(evaluation["episode_rows_per_condition"]) != 8:
        raise RuntimeError("Mechanical confirmatory evaluation is incomplete.")
    if evaluation["local_br_prox"] is None:
        raise RuntimeError("Mechanical empirical BR-Prox path did not execute.")

    return {
        "status": "complete",
        "method": METHOD_VERSION,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "scope": "mechanical_end_to_end_acceptance_only",
        "scientific_readout_allowed": False,
        "upstream_runs": len(upstream_summaries),
        "upstream_steps_per_run": 1_024,
        "delta_updates": int(training["update_count"]),
        "delta_trajectory_steps": int(training["effective_environment_steps"]),
        "signal_contract_fixture": str(output / "signal_contract_fixture.json"),
        "checkpoint_identity_count": len(checkpoint_identities),
        "calibration_partner_runs": int(
            calibration["calibration_partner_runs"]
        ),
        "evaluation_rows_per_condition": int(
            evaluation["episode_rows_per_condition"]
        ),
        "br_prox_anchor_rows": int(evaluation["local_br_prox"]["anchor_rows"]),
        "deployment_bundle": str(deployment),
        "note": (
            "Passing this run establishes mechanical reachability only. It is "
            "not evidence of ZSC performance, calibration coverage, or a "
            "scientific hypothesis."
        ),
    }


def _write_signal_contract_fixture(output: Path) -> None:
    """Exercise every state-machine edge without manufacturing science data."""

    def decision(name: str, passed: bool) -> GateDecision:
        return GateDecision(
            passed=passed,
            statistic=1.0 if passed else -1.0,
            lower_bound=0.5 if passed else -1.5,
            upper_bound=1.5 if passed else -0.5,
            sample_count=4,
            random_domain=f"mechanical_fixture/{name}",
            artifact_fingerprint=hashlib.sha256(name.encode("utf-8")).hexdigest(),
            reason="synthetic path-coverage fixture; never a scientific decision",
        )

    qualification = SignalQualification()
    phases = []
    for name in ("C0", "C1", "C2", "C3", "C4"):
        qualification = qualification.with_decision(name, decision(name, True))
        phases.append(
            phase_from_qualification(
                qualification,
                generator_admitted=(name == "C4"),
            ).name
        )
    qualification = qualification.with_decision("C5", decision("C5", True))
    phases.append(
        phase_from_qualification(qualification, generator_admitted=True).name
    )
    failed = SignalQualification({"C0": decision("C0-failure", False)})
    fixtures = {
        "decision_irrelevant": fixture_contract_outcome(
            decision_relevant=False,
            history_identifiable=False,
            regret_calibrated=False,
        ),
        "decision_relevant_unidentifiable": fixture_contract_outcome(
            decision_relevant=True,
            history_identifiable=False,
            regret_calibrated=False,
        ),
        "decision_relevant_identifiable": fixture_contract_outcome(
            decision_relevant=True,
            history_identifiable=True,
            regret_calibrated=True,
        ),
    }
    write_json(output / "signal_contract_fixture.json", {
        "qualified_path": {
            "phases": phases,
            "final_phase": phase_from_qualification(
                qualification, generator_admitted=True
            ).name,
            "qualification": qualification.to_mapping(),
        },
        "fallback_path": {
            "deployment_tier": deployment_tier(failed).value,
            "qualification": failed.to_mapping(),
        },
        "three_registered_fixtures": fixtures,
        "scientific_readout_allowed": False,
        "note": "Path coverage only; no qualification statistic is inferred.",
    })


def run_mechanical_e2e(args: argparse.Namespace) -> None:
    repository = Path(__file__).resolve().parents[2]
    config_path = Path(args.config).resolve()
    config = load_config(config_path, run_kind="mechanical")
    if int(config.upstream.total_timesteps) != 1_024:
        raise ValueError("Use the registered small mechanical E2E config.")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "mechanical_e2e_status.json"
    if any(
        child.name != "logs"
        for child in output.iterdir()
    ):
        raise RuntimeError(
            "Mechanical E2E output must be new; preserve failed artifacts and "
            "choose a fresh output directory."
        )
    completed: list[str] = []
    _write_signal_contract_fixture(output)

    upstream_root = output / "upstream"
    specifications = [
        ("train_owner_sp", "rnn-sp", 0, None),
        ("train_init_sp", "rnn-sp", 1, None),
        ("train_init_op", "rnn-op", 2, None),
        ("train_init_sa_surrogate", "rnn-sp", 3, None),
        ("train_init_fcp_surrogate", "rnn-sp", 4, None),
        ("train_development_sp", "rnn-sp", 5, None),
        ("train_development_op", "rnn-op", 6, None),
    ]
    fixture_metadata: dict[str, tuple[int, tuple[int, int]]] = {}
    for index, label in enumerate(
        ("calibration_a", "calibration_b", "confirmatory_a", "confirmatory_b"),
        start=7,
    ):
        fixture_metadata[label] = mechanical_fixture_key(label)
        specifications.append(
            (
                label,
                "rnn-sp",
                mechanical_upstream_seed_slot(index),
                fixture_metadata[label][1],
            )
        )
    for label, algorithm, seed_index, key in specifications:
        _run(
            _upstream_command(
                config=config_path,
                algorithm=algorithm,
                seed_index=seed_index,
                output=upstream_root / label,
                key=key,
            ),
            repository=repository,
            status_path=status_path,
            stage=f"upstream:{label}",
            completed=completed,
        )

    train_key = tuple(official_training_key(0))
    train_runs: list[Mapping[str, Any]] = []
    owner_summary = _read_json(
        upstream_root / "train_owner_sp" / "upstream_summary.json"
    )
    train_runs.append(
        _plan_run(
            run_id="mechanical_owner_sp",
            role="owner_source",
            checkpoint=str(owner_summary["checkpoint_paths"][-1]),
            parent="mechanical_train_owner_sp_parent",
            mechanism="rnn-sp",
            seed=42,
            seed_index=0,
            key=train_key,
            owner=None,
        )
    )
    for label, mechanism, role, seed_index in (
        ("train_init_sp", "rnn-sp", "generator_init_source", 1),
        ("train_init_op", "rnn-op", "generator_init_source", 2),
        ("train_init_sa_surrogate", "mechanical-sa-surrogate", "generator_init_source", 3),
        ("train_init_fcp_surrogate", "mechanical-fcp-surrogate", "generator_init_source", 4),
        ("train_development_sp", "rnn-sp", "development_support", 5),
        ("train_development_op", "rnn-op", "development_support", 6),
    ):
        summary = _read_json(upstream_root / label / "upstream_summary.json")
        train_runs.append(
            _plan_run(
                run_id=f"mechanical_{label}",
                role=role,
                checkpoint=str(summary["checkpoint_paths"][-1]),
                parent=f"mechanical_{label}_parent",
                mechanism=mechanism,
                seed=42,
                seed_index=seed_index,
                key=tuple(official_training_key(seed_index)),
                owner=None,
            )
        )
    evaluation_runs: list[Mapping[str, Any]] = []
    for role, labels in (
        ("calibration", ("calibration_a", "calibration_b")),
        ("confirmatory", ("confirmatory_a", "confirmatory_b")),
    ):
        for label in labels:
            seed, key = fixture_metadata[label]
            summary = _read_json(upstream_root / label / "upstream_summary.json")
            evaluation_runs.append(
                _plan_run(
                    run_id=f"mechanical_{label}",
                    role=role,
                    checkpoint=str(summary["checkpoint_paths"][-1]),
                    parent=f"mechanical_{label}_parent",
                    mechanism=f"mechanical-e2e-{label}",
                    seed=seed,
                    seed_index=None,
                    key=key,
                    owner=None,
                )
            )
    plan = output / "partner_manifest_plan.json"
    write_json(
        plan,
        {
            "version": 2,
            "layout": config.environment.layout,
            "runs": [*train_runs, *evaluation_runs],
        },
    )
    manifest = output / "partner_manifest.json"
    _run(
        (
            sys.executable,
            "-m",
            "experiments.overcooked_v2.path_c",
            "build-partner-manifest",
            "--plan",
            str(plan),
            "--output",
            str(manifest),
        ),
        repository=repository,
        status_path=status_path,
        stage="build-partner-manifest",
        completed=completed,
    )
    _run(
        (
            sys.executable,
            "-m",
            "experiments.overcooked_v2.path_c",
            "validate-manifest",
            "--config",
            str(config_path),
            "--partner-manifest",
            str(manifest),
            "--run-kind",
            "mechanical",
        ),
        repository=repository,
        status_path=status_path,
        stage="validate-manifest",
        completed=completed,
    )
    training = output / "delta_train"
    _run(
        (
            sys.executable,
            "-m",
            "experiments.overcooked_v2.path_c",
            "train",
            "--config",
            str(config_path),
            "--partner-manifest",
            str(manifest),
            "--ego-run-id",
            "mechanical-e2e-engineering-seed",
            "--seed-index",
            "-1",
            "--run-kind",
            "mechanical",
            "--output",
            str(training),
        ),
        repository=repository,
        status_path=status_path,
        stage="delta-train",
        completed=completed,
    )
    _run(
        (
            sys.executable,
            "-m",
            "experiments.overcooked_v2.path_c",
            "train",
            "--config",
            str(config_path),
            "--partner-manifest",
            str(manifest),
            "--ego-run-id",
            "mechanical-e2e-engineering-seed",
            "--seed-index",
            "-1",
            "--run-kind",
            "mechanical",
            "--output",
            str(training),
            "--resume",
        ),
        repository=repository,
        status_path=status_path,
        stage="delta-train-resume-noop",
        completed=completed,
    )
    calibration = output / "calibration"
    _run(
        (
            sys.executable,
            "-m",
            "experiments.overcooked_v2.path_c",
            "calibrate",
            "--config",
            str(config_path),
            "--partner-manifest",
            str(manifest),
            "--training-run",
            str(training),
            "--seed-index",
            "-1",
            "--run-kind",
            "mechanical",
            "--output",
            str(calibration),
        ),
        repository=repository,
        status_path=status_path,
        stage="calibrate-and-export",
        completed=completed,
    )
    _run(
        (
            sys.executable,
            "-m",
            "experiments.overcooked_v2.path_c",
            "evaluate",
            "--config",
            str(config_path),
            "--partner-manifest",
            str(manifest),
            "--deployments",
            str(calibration / "deployment"),
            "--seed",
            "0",
            "--run-kind",
            "mechanical",
            "--output",
            str(output / "evaluation"),
        ),
        repository=repository,
        status_path=status_path,
        stage="confirmatory-evaluate",
        completed=completed,
    )
    report = _verify(output)
    write_json(output / "mechanical_e2e_report.json", report)
    write_json(
        status_path,
        {
            **report,
            "stage": "complete",
            "completed_stages": completed,
        },
    )
    print(f"Complete non-scientific mechanical E2E acceptance: {output}")


__all__ = [
    "mechanical_fixture_key",
    "mechanical_upstream_seed_slot",
    "run_mechanical_e2e",
]
