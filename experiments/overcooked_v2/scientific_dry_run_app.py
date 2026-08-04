"""Two-ego real-simulator artifact-chain dry run.

This execution path intentionally uses the mechanical budget and cannot be
consumed by the formal claim builder.  It exists to prove that training,
fresh-partner calibration, raw development episodes, mechanism continuations,
and the terminal claim-style report can be generated from one current-method
lineage before expensive confirmatory runs begin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.overcooked_v2.calibration_app import run_posterior_calibration
from experiments.overcooked_v2.common_partner_app import _official_environment
from experiments.overcooked_v2.deployment import load_deployment
from experiments.overcooked_v2.identifiability_app import (
    collect_mechanism_raw_artifacts,
    run_identifiability_evaluation,
    run_recoverable_value_evaluation,
)
from experiments.overcooked_v2.official_adapter import official_pairing_rollouts
from experiments.overcooked_v2.official_policy import OfficialDEPIPolicy
from experiments.overcooked_v2.training_app import run_training
from src.path_c.experiment import METHOD_VERSION, load_config
from src.path_c.storage import sha256_path, write_json, write_parquet


DRY_RUN_EGO_SEEDS = (0, 1)
DRY_RUN_EPISODES = 50
DRY_RUN_FRESH_PARTNERS = 2
DRY_RUN_STAGE_ORDER = (
    "two_ego_training",
    "fresh_partner_posterior_calibration",
    "raw_development_evaluation",
    "real_crn_mechanism_collection",
    "identifiability",
    "recoverable_value",
    "dry_run_claim_report",
)


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Dry-run artifact is not a JSON object: {path}")
    return value


def _development_dry_run(
    *, config: Any, training_runs: tuple[Path, Path], output: Path
) -> Path:
    import jax

    policies = tuple(
        OfficialDEPIPolicy(
            load_deployment(training_run / "final_deployment", config)
        )
        for training_run in training_runs
    )
    environment = _official_environment(config)
    rows = []
    for left, right in ((0, 1), (1, 0)):
        root = jax.random.fold_in(jax.random.PRNGKey(0), 10_000 + 10 * left + right)
        rollout, keys = official_pairing_rollouts(
            left_policy=policies[left],
            right_policy=policies[right],
            environment=environment,
            root_key=root,
            episodes=DRY_RUN_EPISODES,
        )
        returns = np.asarray(rollout.total_reward, dtype=np.float64)
        if returns.shape != (DRY_RUN_EPISODES,) or not np.all(np.isfinite(returns)):
            raise RuntimeError("Scientific dry-run evaluator returned invalid episodes.")
        rows.extend(
            {
                "left_seed_index": left,
                "right_seed_index": right,
                "episode_index": episode,
                "environment_key": [int(value) for value in np.asarray(keys[episode])],
                "raw_return": float(raw_return),
            }
            for episode, raw_return in enumerate(returns)
        )
    raw_path = output / "development_episode_returns.parquet"
    write_parquet(raw_path, rows)
    artifact_path = output / "development_score.json"
    write_json(
        artifact_path,
        {
            "version": 1,
            "artifact_type": "depi_scientific_dry_run_development_score",
            "method": METHOD_VERSION,
            "ego_seed_indexes": list(DRY_RUN_EGO_SEEDS),
            "episodes_per_ordered_cross_pair": DRY_RUN_EPISODES,
            "episode_count": len(rows),
            "xp_mean_recomputed": float(
                np.mean([float(row["raw_return"]) for row in rows])
            ),
            "raw_episodes": {
                "path": str(raw_path),
                "sha256": sha256_path(raw_path),
            },
            "real_official_simulator": True,
            "scientific_readout_allowed": False,
        },
    )
    return artifact_path


def run_scientific_dry_run(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    manifest_path = Path(args.partner_manifest).resolve()
    comparator_path = Path(args.pair_comparator).resolve()
    output = Path(args.output).resolve()
    config = load_config(config_path, run_kind="mechanical")
    if config.method_variant != "b2":
        raise ValueError("Scientific dry run is registered for B2 only.")
    training_runs = tuple(output / "training" / f"seed-{seed}" for seed in DRY_RUN_EGO_SEEDS)
    for seed, training_run in zip(DRY_RUN_EGO_SEEDS, training_runs, strict=True):
        run_training(
            argparse.Namespace(
                config=str(config_path),
                partner_manifest=str(manifest_path),
                ego_run_id=f"depi-scientific-dry-run-seed-{seed}",
                seed_index=seed,
                run_kind="mechanical",
                output=str(training_run),
                pair_comparator=str(comparator_path),
                resume=bool(args.resume),
                skip_manifest_hash_check=bool(args.skip_manifest_hash_check),
                _execution_scope="scientific-dry-run",
            )
        )

    calibration_artifacts = []
    for seed, training_run in zip(DRY_RUN_EGO_SEEDS, training_runs, strict=True):
        calibration_output = output / "calibration" / f"seed-{seed}"
        run_posterior_calibration(
            argparse.Namespace(
                config=str(config_path),
                partner_manifest=str(manifest_path),
                training_run=str(training_run),
                seed_index=seed,
                run_kind="mechanical",
                output=str(calibration_output),
                skip_manifest_hash_check=bool(args.skip_manifest_hash_check),
                maximum_runs=DRY_RUN_FRESH_PARTNERS,
            )
        )
        calibration_artifacts.append(
            calibration_output / "posterior_calibration.json"
        )

    development_path = _development_dry_run(
        config=config,
        training_runs=(training_runs[0], training_runs[1]),
        output=output / "development",
    )
    policy_manifest_path = output / "mechanism" / "dry_run_policy_manifest.json"
    raw_paths = collect_mechanism_raw_artifacts(
        config_path=config_path,
        policy_manifest_path=policy_manifest_path,
        partner_manifest_path=manifest_path,
        output=output / "mechanism" / "raw",
        skip_manifest_hash_check=bool(args.skip_manifest_hash_check),
        dry_run_training_runs=training_runs,
        dry_run_partner_count=DRY_RUN_FRESH_PARTNERS,
        dry_run_episodes=DRY_RUN_EPISODES,
    )
    identifiability_output = output / "mechanism" / "identifiability"
    recoverable_output = output / "mechanism" / "recoverable_value"
    run_identifiability_evaluation(
        argparse.Namespace(
            raw_input=str(raw_paths["identifiability"]),
            config=None,
            policy_manifest=None,
            partner_manifest=None,
            output=str(identifiability_output),
            skip_manifest_hash_check=False,
        )
    )
    run_recoverable_value_evaluation(
        argparse.Namespace(
            raw_input=str(raw_paths["recoverable_value"]),
            config=None,
            policy_manifest=None,
            partner_manifest=None,
            output=str(recoverable_output),
            skip_manifest_hash_check=False,
        )
    )

    identifiability_path = (
        identifiability_output / "identifiability_evaluation.json"
    )
    recoverable_path = (
        recoverable_output / "recoverable_value_evaluation.json"
    )
    report_path = output / "scientific_dry_run_claim_report.json"
    identifiability_payload = _json(identifiability_path)
    recoverable_payload = _json(recoverable_path)
    calibration_payloads = [_json(path) for path in calibration_artifacts]
    if (
        identifiability_payload.get("scientific_readout_allowed") is not False
        or recoverable_payload.get("scientific_readout_allowed") is not False
        or any(
            payload.get("artifact_type") != "depi_posterior_calibration"
            or payload.get("scientific_readout_allowed") is not False
            for payload in calibration_payloads
        )
    ):
        raise RuntimeError("Dry-run claim rehearsal received an evidentiary artifact.")
    claim_vector = {
        name: False
        for name in (
            "performance_claim",
            "filter_architecture_claim",
            "decision_supervision_claim",
            "decision_supervision_cost_efficiency_claim",
            "predictive_calibration_claim",
            "history_dependence_claim",
            "context_causal_value_claim",
            "recoverable_value_claim",
            "capacity_explanation_rejected",
            "common_partner_generalization_claim",
        )
    }
    write_json(
        report_path,
        {
            "version": 1,
            "artifact_type": "depi_scientific_dry_run_claim_report",
            "method": METHOD_VERSION,
            "stage_order": list(DRY_RUN_STAGE_ORDER),
            "ego_seed_indexes": list(DRY_RUN_EGO_SEEDS),
            "episodes_per_pairing": DRY_RUN_EPISODES,
            "fresh_partner_count_per_stage": DRY_RUN_FRESH_PARTNERS,
            "small_anchor_replicas": {
                "fit": int(config.evaluation.mechanism_fit_replicas),
                "evaluation": int(config.evaluation.mechanism_evaluation_replicas),
            },
            "sources": {
                "development_score": {
                    "path": str(development_path),
                    "sha256": sha256_path(development_path),
                },
                "posterior_calibration": [
                    {"path": str(path), "sha256": sha256_path(path)}
                    for path in calibration_artifacts
                ],
                "identifiability": {
                    "path": str(identifiability_path),
                    "sha256": sha256_path(identifiability_path),
                },
                "recoverable_value": {
                    "path": str(recoverable_path),
                    "sha256": sha256_path(recoverable_path),
                },
            },
            "artifact_chain_complete": all(
                path.is_file()
                for path in (
                    development_path,
                    *calibration_artifacts,
                    identifiability_path,
                    recoverable_path,
                )
            ),
            "formal_claim_report_rehearsal": True,
            "claim_vector": claim_vector,
            "dry_run_descriptive_passes": {
                "posterior_calibration": [
                    bool(payload.get("pass", {}).get("overall", False))
                    for payload in calibration_payloads
                ],
                "identifiability": bool(
                    identifiability_payload.get("pass", {}).get("overall", False)
                ),
                "recoverable_value": bool(
                    recoverable_payload.get("pass", {}).get("overall", False)
                ),
            },
            "real_official_simulator": True,
            "scientific_readout_allowed": False,
            "note": (
                "Execution/schema rehearsal only; this report is rejected by the "
                "formal claim builder and is not paper evidence."
            ),
        },
    )
    if _json(report_path).get("artifact_chain_complete") is not True:
        raise RuntimeError("Scientific dry-run artifact chain is incomplete.")


__all__ = [
    "DRY_RUN_EGO_SEEDS",
    "DRY_RUN_EPISODES",
    "DRY_RUN_FRESH_PARTNERS",
    "DRY_RUN_STAGE_ORDER",
    "run_scientific_dry_run",
]
