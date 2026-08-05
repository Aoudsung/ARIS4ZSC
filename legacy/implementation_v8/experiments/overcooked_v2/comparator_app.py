"""K-independent, method-independent frozen matched-history comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from experiments.overcooked_v2.common_partner_app import _official_environment
from experiments.overcooked_v2.official_adapter import (
    official_policy,
    restore_official_checkpoint,
    validate_official_runtime,
)
from experiments.overcooked_v2.official_br_prox_app import (
    empirical_all_action_continuations_from_snapshots,
    record_official_history_anchor_snapshots,
)
from src.path_c.anchor_sampling import (
    COMPARATOR_RUN_ID_CAPACITY,
    FrozenPairComparator,
    fit_pair_comparator,
    pair_legal_history_features,
    task_matched_decision_distinction_pair_dataset,
)
from src.path_c.comparator_contract import (
    ComparatorContract,
    comparator_contract_for_run,
    comparator_reference_policy_set_hash,
)
from src.path_c.continuation import ContinuationContract, validate_continuation_contract
from src.path_c.decision_geometry import centered_action_values
from src.path_c.experiment import (
    OFFICIAL_SOURCE_COMMIT,
    load_config,
    load_partner_manifest,
    validate_partner_manifest,
)
from src.path_c.resources import ResourceLedger, gpu_hours_for_wall_seconds
from src.path_c.task_encoder import (
    instantaneous_partner_observation,
    task_only_observation,
)
from src.path_c.storage import (
    ensure_run_identity,
    runtime_provenance,
    sha256_path,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
)


COMPARATOR_ARTIFACT_VERSION = 3
COMPARATOR_SOURCE_VERSION = 4
MINIMUM_HISTORIES_PER_SPLIT = 256
MINIMUM_RUNS_PER_SPLIT = 8
COMPARATOR_HISTORIES_PER_ROLE_RUN = 16
COMPARATOR_REFERENCE_ROOT_SEED = 42
COMPARATOR_TASK_MATCH_EPSILON = 4.0
COMPARATOR_EPISODE_TIME_TOLERANCE = 0


def _training_cost_source(checkpoint: Path, *, label: str) -> Mapping[str, Any]:
    """Resolve an explicit upstream ledger for a shared frozen policy."""

    candidates: list[Path] = []
    for directory in (checkpoint, *checkpoint.parents[:6]):
        candidates.extend(
            (directory / "resource_ledger.json", directory / "upstream_summary.json")
        )
    ledger_path = next((path for path in candidates if path.is_file()), None)
    if ledger_path is None:
        raise RuntimeError(f"{label} has no explicit upstream resource ledger.")
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    try:
        ledger = ResourceLedger.from_mapping(payload)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"{label} upstream resource ledger is not the registered schema."
        ) from error
    steps = ledger.total_training_simulator_steps
    if int(steps) <= 0:
        raise RuntimeError(f"{label} upstream resource ledger has no positive step total.")
    return {
        "path": str(ledger_path.resolve()),
        "sha256": sha256_path(ledger_path),
        "training_simulator_steps": int(steps),
        "gpu_hours": float(ledger.gpu_hours),
        "wall_clock_hours": float(ledger.wall_clock_hours),
    }


def collect_pair_comparator_source(args: argparse.Namespace) -> None:
    """Collect a lineage-bound comparator source in the pinned simulator."""

    import jax

    started = time.perf_counter()

    validate_formal_repository_state()
    validate_registered_python_runtime()
    official_runtime = validate_official_runtime()
    config_path = Path(args.config).resolve()
    manifest_path = Path(args.partner_manifest).resolve()
    reference_checkpoint = Path(args.reference_ego_checkpoint).resolve()
    config = load_config(config_path, run_kind="formal")
    manifest = load_partner_manifest(
        manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    validate_partner_manifest(manifest)
    split_runs = {
        "fit_histories": manifest.by_role("comparator_fit"),
        "validation_histories": manifest.by_role("comparator_validation"),
    }
    parents_by_split: dict[str, set[str]] = {}
    for label, runs in split_runs.items():
        parents = {str(run.parent_training_run_id) for run in runs}
        parents_by_split[label] = parents
        if len(parents) < MINIMUM_RUNS_PER_SPLIT:
            raise ValueError(
                f"Comparator {label} needs at least {MINIMUM_RUNS_PER_SPLIT} "
                "independent partner parents."
            )
        if any(run.owner_seed_index is not None for run in runs):
            raise ValueError("Comparator source partners must be shared and owner-free.")
    if parents_by_split["fit_histories"] & parents_by_split["validation_histories"]:
        raise ValueError("Comparator source fit/validation parents overlap.")
    reference_resource = _training_cost_source(
        reference_checkpoint, label="Comparator reference ego"
    )
    comparator_parent_resources: dict[str, Mapping[str, Any]] = {}
    for runs in split_runs.values():
        for run in runs:
            parent = str(run.parent_training_run_id)
            if parent not in comparator_parent_resources:
                comparator_parent_resources[parent] = _training_cost_source(
                    Path(run.checkpoint).resolve(),
                    label=f"Comparator partner parent {parent}",
                )
    reference_checkpoint_sha = sha256_path(reference_checkpoint)
    comparator_checkpoint_hashes = {
        str(run.checkpoint_sha256)
        for runs in split_runs.values()
        for run in runs
    }
    if reference_checkpoint_sha in comparator_checkpoint_hashes:
        raise ValueError("Comparator reference ego is also a comparator partner.")
    reference_policy_set_hash = comparator_reference_policy_set_hash(
        reference_checkpoint_sha256=reference_checkpoint_sha,
        manifest=manifest,
    )
    comparator_contract = comparator_contract_for_run(
        config=config,
        manifest=manifest,
        reference_policy_set_hash=reference_policy_set_hash,
    )
    if comparator_contract.task_match_epsilon != COMPARATOR_TASK_MATCH_EPSILON:
        raise ValueError("Config comparator task-match epsilon differs from registration.")

    reference_config, reference_params = restore_official_checkpoint(
        reference_checkpoint
    )
    reference_policy = official_policy(reference_params, reference_config)
    environment = _official_environment(config)
    probe_steps = 16
    anchors_per_role_run = COMPARATOR_HISTORIES_PER_ROLE_RUN
    fit_replicas = int(config.anchors.fit_replicas)
    evaluation_replicas = int(config.anchors.evaluation_replicas)
    horizon = int(config.anchors.continuation_horizon)
    episode_count = int(config.evaluation.episodes_per_pairing)
    rows_by_split: dict[str, list[Mapping[str, Any]]] = {
        "fit_histories": [],
        "validation_histories": [],
    }
    trajectory_steps = 0
    continuation_steps = 0
    root = jax.random.PRNGKey(COMPARATOR_REFERENCE_ROOT_SEED)
    for split_index, (label, runs) in enumerate(split_runs.items()):
        split_root = jax.random.fold_in(root, 60_001 + split_index)
        for run_index, run in enumerate(runs):
            partner_config, partner_params = restore_official_checkpoint(run.checkpoint)
            partner_policy = official_policy(partner_params, partner_config)
            for ego_role in (0, 1):
                pairing_root = jax.random.fold_in(
                    jax.random.fold_in(split_root, run_index), ego_role
                )
                left, right = (
                    (reference_policy, partner_policy)
                    if ego_role == 0
                    else (partner_policy, reference_policy)
                )
                recorded = record_official_history_anchor_snapshots(
                    left=left,
                    right=right,
                    ego_role=ego_role,
                    environment=environment,
                    root_key=pairing_root,
                    anchors=anchors_per_role_run,
                    history_steps=probe_steps,
                    continuation_horizon=horizon,
                    episodes=episode_count,
                )
                continuations = empirical_all_action_continuations_from_snapshots(
                    left=left,
                    right=right,
                    ego_role=ego_role,
                    environment=environment,
                    selected=recorded["selected"],
                    anchor_roots=recorded["anchor_roots"],
                    fit_replicas=fit_replicas,
                    evaluation_replicas=evaluation_replicas,
                    continuation_horizon=horizon,
                    gamma=float(config.ppo.gamma),
                    continuation_policy_fingerprint=reference_policy_set_hash,
                )
                features = np.asarray(
                    pair_legal_history_features(recorded["history"]),
                    dtype=np.float64,
                )
                returns = np.asarray(
                    continuations["fit_returns_by_action"], dtype=np.float64
                )
                evaluation_returns = np.asarray(
                    continuations["evaluation_returns_by_action"], dtype=np.float64
                )
                ego_name = "agent_0" if ego_role == 0 else "agent_1"
                current_observation = np.asarray(
                    recorded["selected"]["observations"][ego_name],
                    dtype=np.float32,
                )
                task_observation = np.asarray(
                    task_only_observation(current_observation), dtype=np.float32
                )
                task_features = task_observation.reshape(
                    (anchors_per_role_run, -1)
                )
                instant_features = np.asarray(
                    instantaneous_partner_observation(current_observation),
                    dtype=np.float32,
                ).reshape((anchors_per_role_run, -1))
                task_channel_counts = np.sum(task_observation, axis=(1, 2))
                history_observations = np.asarray(
                    recorded["history"].ego_observations, dtype=np.float32
                )
                history_actions = np.asarray(
                    recorded["history"].ego_actions, dtype=np.int32
                )
                episode_times = np.asarray(recorded["time_indexes"], dtype=np.int64)
                if (
                    features.shape[0] != anchors_per_role_run
                    or returns.shape != (anchors_per_role_run, 6)
                    or evaluation_returns.shape != (anchors_per_role_run, 6)
                ):
                    raise RuntimeError("Comparator source collection shape differs.")
                rows_by_split[label].extend(
                    {
                        "history_features": features[index].tolist(),
                        "fit_returns_by_action": returns[index].tolist(),
                        "evaluation_returns_by_action": evaluation_returns[
                            index
                        ].tolist(),
                        "partner_run_id": str(run.run_id),
                        "partner_mechanism": str(run.generation_mechanism),
                        "task_features": task_features[index].tolist(),
                        "episode_time": int(episode_times[index]),
                        "recipe_order_state": hashlib.sha256(
                            np.round(task_channel_counts[index], 4).tobytes()
                        ).hexdigest(),
                        "ego_role": int(ego_role),
                        "instant_partner_features": instant_features[index].tolist(),
                        "task_state_hash": hashlib.sha256(
                            np.round(task_features[index], 4).tobytes()
                        ).hexdigest(),
                        "history_observations": history_observations[index].tolist(),
                        "history_actions": history_actions[index].tolist(),
                        "current_observation": current_observation[index].tolist(),
                    }
                    for index in range(anchors_per_role_run)
                )
                trajectory_steps += anchors_per_role_run * int(environment.max_steps)
                continuation_steps += (
                    anchors_per_role_run
                    * 6
                    * (fit_replicas + evaluation_replicas)
                    * horizon
                )

    for label, rows in rows_by_split.items():
        _rows(rows, label=label)
    output = Path(args.output).resolve()
    reference = {
        "path": str(reference_checkpoint),
        "sha256": reference_checkpoint_sha,
    }
    partner_source = {
        "path": str(manifest_path),
        "sha256": sha256_path(manifest_path),
    }
    elapsed_wall_seconds = time.perf_counter() - started
    comparator_gpu_hours = gpu_hours_for_wall_seconds(elapsed_wall_seconds)
    resource_ledger = ResourceLedger(
        comparator_continuation_steps=continuation_steps,
        comparator_history_collection_steps=trajectory_steps,
        reference_ego_upstream_steps=int(
            reference_resource["training_simulator_steps"]
        ),
        shared_pretraining_cost=sum(
            int(value["training_simulator_steps"])
            for value in comparator_parent_resources.values()
        ),
        gpu_hours=(
            float(reference_resource["gpu_hours"])
            + sum(
                float(value["gpu_hours"])
                for value in comparator_parent_resources.values()
            )
            + comparator_gpu_hours
        ),
        comparator_gpu_hours=comparator_gpu_hours,
        wall_clock_hours=(
            elapsed_wall_seconds / 3_600.0
            + float(reference_resource["wall_clock_hours"])
            + sum(
                float(value["wall_clock_hours"])
                for value in comparator_parent_resources.values()
            )
        ),
    ).to_mapping()
    payload = {
        "version": COMPARATOR_SOURCE_VERSION,
        "artifact_type": "depi_pair_comparator_source",
        "layout": config.environment.layout,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "config_fingerprint": config.fingerprint,
        "reference_ego_checkpoint": reference,
        "reference_ego_resource": reference_resource,
        "comparator_partner_resources": comparator_parent_resources,
        "partner_manifest": partner_source,
        "probe_steps": probe_steps,
        "signature_distance_threshold": 1.0,
        "comparator_contract": comparator_contract.to_mapping(),
        "comparator_contract_fingerprint": comparator_contract.fingerprint,
        "continuation_contract": ContinuationContract(
            gamma=float(config.ppo.gamma),
            horizon=horizon,
            continuation_policy_fingerprint=reference_policy_set_hash,
        ).to_mapping(),
        "collection": {
            "reference_policy": "fixed_official_checkpoint",
            "root_key": [0, COMPARATOR_REFERENCE_ROOT_SEED],
            "histories_per_partner_role": anchors_per_role_run,
            "fit_replicas": fit_replicas,
            "evaluation_replicas": evaluation_replicas,
            "continuation_horizon": horizon,
        },
        "resource_ledger": resource_ledger,
        **rows_by_split,
    }
    ensure_run_identity(
        output,
        {
            "stage": "collect-pair-comparator-source",
            "layout": config.environment.layout,
            "official_runtime": official_runtime,
            "repository_runtime": runtime_provenance(),
            "config": {"path": str(config_path), "fingerprint": config.fingerprint},
            "reference_ego_checkpoint": reference,
            "reference_ego_resource": reference_resource,
            "comparator_partner_resources": comparator_parent_resources,
            "partner_manifest": partner_source,
            "collection": payload["collection"],
            "comparator_contract": payload["comparator_contract"],
            "comparator_contract_fingerprint": payload[
                "comparator_contract_fingerprint"
            ],
        },
    )
    write_json(output / "pair_comparator_source.json", payload)
    write_json(output / "budget_ledger.json", resource_ledger)


def _rows(
    payload: Any, *, label: str
) -> Mapping[str, np.ndarray]:
    if not isinstance(payload, list) or len(payload) < MINIMUM_HISTORIES_PER_SPLIT:
        raise ValueError(
            f"Comparator {label} needs at least {MINIMUM_HISTORIES_PER_SPLIT} histories."
        )
    expected = {
        "history_features",
        "fit_returns_by_action",
        "evaluation_returns_by_action",
        "partner_run_id",
        "partner_mechanism",
        "task_features",
        "episode_time",
        "recipe_order_state",
        "ego_role",
        "instant_partner_features",
        "task_state_hash",
        "history_observations",
        "history_actions",
        "current_observation",
    }
    if any(not isinstance(row, Mapping) or set(row) != expected for row in payload):
        raise ValueError(f"Comparator {label} history row schema differs.")
    features = np.asarray([row["history_features"] for row in payload], dtype=np.float64)
    returns = np.asarray([row["fit_returns_by_action"] for row in payload], dtype=np.float64)
    evaluation_returns = np.asarray(
        [row["evaluation_returns_by_action"] for row in payload], dtype=np.float64
    )
    runs = np.asarray([str(row["partner_run_id"]) for row in payload])
    mechanisms = np.asarray([str(row["partner_mechanism"]) for row in payload])
    tasks = np.asarray([row["task_features"] for row in payload], dtype=np.float64)
    times = np.asarray([row["episode_time"] for row in payload], dtype=np.int64)
    regimes = np.asarray([str(row["recipe_order_state"]) for row in payload])
    roles = np.asarray([row["ego_role"] for row in payload], dtype=np.int64)
    instant = np.asarray(
        [row["instant_partner_features"] for row in payload], dtype=np.float64
    )
    hashes = np.asarray([str(row["task_state_hash"]) for row in payload])
    history_observations = np.asarray(
        [row["history_observations"] for row in payload], dtype=np.float32
    )
    history_actions = np.asarray(
        [row["history_actions"] for row in payload], dtype=np.int32
    )
    current_observations = np.asarray(
        [row["current_observation"] for row in payload], dtype=np.float32
    )
    if (
        features.ndim != 2
        or returns.shape != (features.shape[0], 6)
        or evaluation_returns.shape != returns.shape
        or tasks.ndim != 2
        or tasks.shape[0] != features.shape[0]
        or instant.ndim != 2
        or instant.shape[0] != features.shape[0]
        or history_observations.shape[0] != features.shape[0]
        or history_actions.shape[:2] != history_observations.shape[:2]
        or current_observations.shape[0] != features.shape[0]
        or times.shape != runs.shape
        or roles.shape != runs.shape
        or hashes.shape != runs.shape
        or mechanisms.shape != runs.shape
    ):
        raise ValueError(f"Comparator {label} feature/return shapes differ.")
    if (
        not np.all(np.isfinite(features))
        or not np.all(np.isfinite(returns))
        or not np.all(np.isfinite(evaluation_returns))
        or not np.all(np.isfinite(tasks))
        or not np.all(np.isfinite(instant))
        or not np.all(np.isfinite(history_observations))
        or not np.all(np.isfinite(current_observations))
    ):
        raise ValueError(f"Comparator {label} contains non-finite values.")
    if np.unique(runs).size < MINIMUM_RUNS_PER_SPLIT:
        raise ValueError(
            f"Comparator {label} needs at least {MINIMUM_RUNS_PER_SPLIT} partner runs."
        )
    if (
        np.any((roles != 0) & (roles != 1))
        or any(len(value) != 64 for value in hashes)
        or any(len(value) != 64 for value in regimes)
        or any(not value for value in mechanisms)
    ):
        raise ValueError(f"Comparator {label} role/hash values differ.")
    return {
        "history_features": features,
        "fit_signatures": centered_action_values(returns),
        "evaluation_signatures": centered_action_values(evaluation_returns),
        "partner_run_ids": runs,
        "partner_mechanisms": mechanisms,
        "task_features": tasks,
        "episode_times": times,
        "recipe_order_states": regimes,
        "ego_roles": roles,
        "instant_partner_features": instant,
        "task_state_hashes": hashes,
        "history_observations": history_observations,
        "history_actions": history_actions,
        "current_observations": current_observations,
    }


def fit_frozen_pair_comparator(args: argparse.Namespace) -> None:
    source = Path(args.source).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {
        "version",
        "artifact_type",
        "layout",
        "official_source_commit",
        "config_fingerprint",
        "reference_ego_checkpoint",
        "reference_ego_resource",
        "comparator_partner_resources",
        "partner_manifest",
        "probe_steps",
        "signature_distance_threshold",
        "comparator_contract",
        "comparator_contract_fingerprint",
        "continuation_contract",
        "collection",
        "resource_ledger",
        "fit_histories",
        "validation_histories",
    }:
        raise ValueError("Comparator source schema differs.")
    if (
        int(payload["version"]) != COMPARATOR_SOURCE_VERSION
        or payload["artifact_type"] != "depi_pair_comparator_source"
        or payload["official_source_commit"] != OFFICIAL_SOURCE_COMMIT
        or not isinstance(payload["layout"], str)
        or not payload["layout"]
        or not isinstance(payload["config_fingerprint"], str)
        or len(payload["config_fingerprint"]) != 64
        or int(payload["probe_steps"]) != 16
        or float(payload["signature_distance_threshold"]) != 1.0
    ):
        raise ValueError("Comparator source registration differs.")
    comparator_contract = ComparatorContract.from_mapping(
        payload["comparator_contract"]
    )
    if comparator_contract.fingerprint != payload["comparator_contract_fingerprint"]:
        raise ValueError("Comparator contract fingerprint differs.")
    validate_continuation_contract(
        payload["continuation_contract"],
        expected=ContinuationContract(
            gamma=float(comparator_contract.gamma),
            horizon=int(comparator_contract.continuation_horizon),
            continuation_policy_fingerprint=(
                comparator_contract.reference_policy_set_hash
            ),
        ),
    )
    for field in ("reference_ego_checkpoint", "partner_manifest"):
        descriptor = payload[field]
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256"}:
            raise ValueError(f"Comparator source {field} identity differs.")
        if sha256_path(Path(str(descriptor["path"])).resolve()) != descriptor["sha256"]:
            raise ValueError(f"Comparator source {field} hash differs.")
    reference_resource = payload["reference_ego_resource"]
    if (
        not isinstance(reference_resource, Mapping)
        or set(reference_resource) != {
            "path", "sha256", "training_simulator_steps", "gpu_hours", "wall_clock_hours"
        }
        or sha256_path(Path(str(reference_resource["path"])).resolve())
        != reference_resource["sha256"]
        or int(reference_resource["training_simulator_steps"]) <= 0
    ):
        raise ValueError("Comparator reference-ego resource provenance differs.")
    parent_resources = payload["comparator_partner_resources"]
    if not isinstance(parent_resources, Mapping) or not parent_resources:
        raise ValueError("Comparator partner resource provenance differs.")
    for descriptor in parent_resources.values():
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {
                "path", "sha256", "training_simulator_steps", "gpu_hours", "wall_clock_hours"
            }
            or sha256_path(Path(str(descriptor["path"])).resolve())
            != descriptor["sha256"]
            or int(descriptor["training_simulator_steps"]) <= 0
        ):
            raise ValueError("Comparator partner resource provenance differs.")
    collection = payload["collection"]
    if (
        not isinstance(collection, Mapping)
        or collection.get("reference_policy") != "fixed_official_checkpoint"
        or collection.get("root_key") != [0, COMPARATOR_REFERENCE_ROOT_SEED]
        or int(collection.get("histories_per_partner_role", -1))
        != COMPARATOR_HISTORIES_PER_ROLE_RUN
        or min(
            int(collection.get("fit_replicas", 0)),
            int(collection.get("evaluation_replicas", 0)),
            int(collection.get("continuation_horizon", 0)),
        )
        <= 0
    ):
        raise ValueError("Comparator source collection registration differs.")
    ledger_payload = payload["resource_ledger"]
    try:
        ledger = ResourceLedger.from_mapping(ledger_payload)
    except (TypeError, ValueError) as error:
        raise ValueError("Comparator source resource ledger differs.") from error
    if (
        ledger.comparator_continuation_steps <= 0
        or ledger.comparator_history_collection_steps <= 0
        or ledger.reference_ego_upstream_steps
        != int(reference_resource["training_simulator_steps"])
        or ledger.shared_pretraining_cost
        != sum(
            int(value["training_simulator_steps"])
            for value in parent_resources.values()
        )
        or not np.isclose(
            ledger.gpu_hours,
            float(reference_resource["gpu_hours"])
            + sum(float(value["gpu_hours"]) for value in parent_resources.values())
            + float(ledger.comparator_gpu_hours),
        )
        or ledger.wall_clock_hours
        < float(reference_resource["wall_clock_hours"])
        + sum(
            float(value["wall_clock_hours"])
            for value in parent_resources.values()
        )
    ):
        raise ValueError("Comparator source resource ledger differs.")
    collector_identity_path = source.parent / "run_identity.json"
    collector_budget_path = source.parent / "budget_ledger.json"
    collector_identity = json.loads(
        collector_identity_path.read_text(encoding="utf-8")
    )
    collector_budget = json.loads(collector_budget_path.read_text(encoding="utf-8"))
    if (
        collector_identity.get("stage") != "collect-pair-comparator-source"
        or collector_identity.get("layout") != payload["layout"]
        or collector_identity.get("config", {}).get("fingerprint")
        != payload["config_fingerprint"]
        or collector_identity.get("reference_ego_checkpoint")
        != payload["reference_ego_checkpoint"]
        or collector_identity.get("reference_ego_resource")
        != payload["reference_ego_resource"]
        or collector_identity.get("comparator_partner_resources")
        != payload["comparator_partner_resources"]
        or collector_identity.get("partner_manifest") != payload["partner_manifest"]
        or collector_identity.get("collection") != payload["collection"]
        or collector_identity.get("comparator_contract")
        != payload["comparator_contract"]
        or collector_identity.get("comparator_contract_fingerprint")
        != payload["comparator_contract_fingerprint"]
        or collector_budget != ledger_payload
    ):
        raise ValueError("Comparator source collector provenance differs.")
    fit_source = _rows(payload["fit_histories"], label="fit")
    validation_source = _rows(payload["validation_histories"], label="validation")
    fit_features = fit_source["history_features"]
    fit_signatures = fit_source["fit_signatures"]
    fit_runs = fit_source["partner_run_ids"]
    validation_features = validation_source["history_features"]
    validation_signatures = validation_source["evaluation_signatures"]
    validation_runs = validation_source["partner_run_ids"]
    if fit_features.shape[1] != validation_features.shape[1]:
        raise ValueError("Comparator fit/validation feature dimensions differ.")
    if set(fit_runs.tolist()) & set(validation_runs.tolist()):
        raise ValueError("Comparator fit and validation partner runs overlap.")
    fit_rows, fit_labels, fit_blocks, fit_match_report = (
        task_matched_decision_distinction_pair_dataset(
            fit_features,
            fit_signatures,
            fit_source["task_features"],
            fit_runs,
            signature_distance_threshold=1.0,
            task_match_epsilon=float(comparator_contract.task_match_epsilon),
            episode_times=fit_source["episode_times"],
            episode_time_tolerance=COMPARATOR_EPISODE_TIME_TOLERANCE,
            recipe_order_states=fit_source["recipe_order_states"],
            ego_roles=fit_source["ego_roles"],
        )
    )
    validation_rows, validation_labels, validation_blocks, validation_match_report = (
        task_matched_decision_distinction_pair_dataset(
            validation_features,
            validation_signatures,
            validation_source["task_features"],
            validation_runs,
            signature_distance_threshold=1.0,
            task_match_epsilon=float(comparator_contract.task_match_epsilon),
            episode_times=validation_source["episode_times"],
            episode_time_tolerance=COMPARATOR_EPISODE_TIME_TOLERANCE,
            recipe_order_states=validation_source["recipe_order_states"],
            ego_roles=validation_source["ego_roles"],
            require_both_classes=True,
        )
    )
    comparator = fit_pair_comparator(
        fit_rows,
        fit_labels,
        validation_pair_features=validation_rows,
        validation_pair_labels=validation_labels,
        history_feature_dim=int(fit_features.shape[1]),
        training_block_ids=fit_blocks,
        validation_block_ids=validation_blocks,
        bootstrap_replicates=9_999,
        bootstrap_seed=0,
    )
    output = Path(args.output).resolve()
    result = {
        "version": COMPARATOR_ARTIFACT_VERSION,
        "artifact_type": "depi_frozen_pair_comparator",
        "estimand": (
            "P(discounted decision-signature distance > 1.0 | reciprocal "
            "task-state-matched legal history pair)"
        ),
        "method_independent": True,
        "protocol_component_count_independent": True,
        "state_matching_space": "fixed_simulator_task_planes",
        "state_matching_algorithm": "run_disjoint_mutual_nearest_neighbour",
        "fit_match_report": fit_match_report,
        "validation_match_report": validation_match_report,
        "comparator_contract": payload["comparator_contract"],
        "comparator_contract_fingerprint": payload[
            "comparator_contract_fingerprint"
        ],
        "continuation_contract": payload["continuation_contract"],
        "history_feature_dim": int(comparator.history_feature_dim),
        "pair_feature_dim": int(comparator.pair_feature_dim),
        "weights": np.asarray(comparator.weights).tolist(),
        "bias": float(comparator.bias),
        "train_accuracy": float(comparator.train_accuracy),
        "validation_accuracy": float(comparator.validation_accuracy),
        "validation_accuracy_interval": list(comparator.validation_accuracy_interval),
        "fit_pair_count": int(comparator.development_row_count),
        "validation_pair_count": int(comparator.validation_row_count),
        "fit_history_count": int(fit_features.shape[0]),
        "validation_history_count": int(validation_features.shape[0]),
        "fit_partner_run_count": int(np.unique(fit_runs).size),
        "validation_partner_run_count": int(np.unique(validation_runs).size),
        "source_registration": {
            "layout": str(payload["layout"]),
            "official_source_commit": str(payload["official_source_commit"]),
            "config_fingerprint": str(payload["config_fingerprint"]),
            "reference_ego_checkpoint": payload["reference_ego_checkpoint"],
            "reference_ego_resource": payload["reference_ego_resource"],
            "comparator_partner_resources": payload[
                "comparator_partner_resources"
            ],
            "partner_manifest": payload["partner_manifest"],
            "collection": payload["collection"],
            "resource_ledger": payload["resource_ledger"],
            "comparator_contract": payload["comparator_contract"],
            "comparator_contract_fingerprint": payload[
                "comparator_contract_fingerprint"
            ],
            "continuation_contract": payload["continuation_contract"],
        },
        "source_provenance": {
            "run_identity": {
                "path": str(collector_identity_path),
                "sha256": sha256_path(collector_identity_path),
            },
            "resource_ledger": {
                "path": str(collector_budget_path),
                "sha256": sha256_path(collector_budget_path),
            },
        },
        "source": {"path": str(source), "sha256": sha256_path(source)},
    }
    ensure_run_identity(
        output,
        {
            "stage": "fit-pair-comparator",
            "source": result["source"],
            "registration": {
                "signature_distance_threshold": 1.0,
                "minimum_histories_per_split": MINIMUM_HISTORIES_PER_SPLIT,
                "minimum_runs_per_split": MINIMUM_RUNS_PER_SPLIT,
            },
            "repository_runtime": runtime_provenance(),
        },
    )
    write_json(output / "frozen_pair_comparator.json", result)


def load_frozen_pair_comparator(
    path: str | Path,
    *,
    expected_contract: ComparatorContract,
    expected_history_feature_dim: int,
) -> FrozenPairComparator:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "version", "artifact_type", "estimand", "method_independent",
        "protocol_component_count_independent", "state_matching_space",
        "state_matching_algorithm", "fit_match_report", "validation_match_report",
        "comparator_contract", "comparator_contract_fingerprint",
        "continuation_contract",
        "history_feature_dim", "pair_feature_dim", "weights", "bias",
        "train_accuracy", "validation_accuracy", "validation_accuracy_interval",
        "fit_pair_count", "validation_pair_count", "fit_history_count",
        "validation_history_count", "fit_partner_run_count",
        "validation_partner_run_count", "source",
        "source_registration",
        "source_provenance",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("Frozen comparator artifact schema differs.")
    if (
        int(payload["version"]) != COMPARATOR_ARTIFACT_VERSION
        or payload["artifact_type"] != "depi_frozen_pair_comparator"
        or payload["method_independent"] is not True
        or payload["protocol_component_count_independent"] is not True
        or payload["state_matching_space"] != "fixed_simulator_task_planes"
        or payload["state_matching_algorithm"]
        != "run_disjoint_mutual_nearest_neighbour"
        or payload["comparator_contract"] != expected_contract.to_mapping()
        or payload["comparator_contract_fingerprint"]
        != expected_contract.fingerprint
        or int(payload["history_feature_dim"]) != int(expected_history_feature_dim)
        or int(payload["pair_feature_dim"]) != 2 * int(expected_history_feature_dim)
        or int(payload["fit_history_count"]) < MINIMUM_HISTORIES_PER_SPLIT
        or int(payload["validation_history_count"]) < MINIMUM_HISTORIES_PER_SPLIT
        or int(payload["fit_partner_run_count"]) < MINIMUM_RUNS_PER_SPLIT
        or int(payload["validation_partner_run_count"]) < MINIMUM_RUNS_PER_SPLIT
    ):
        raise ValueError("Frozen comparator registration differs.")
    validate_continuation_contract(
        payload["continuation_contract"],
        expected=ContinuationContract(
            gamma=float(expected_contract.gamma),
            horizon=int(expected_contract.continuation_horizon),
            continuation_policy_fingerprint=(
                expected_contract.reference_policy_set_hash
            ),
        ),
    )
    for name in ("fit_match_report", "validation_match_report"):
        report = payload[name]
        if (
            not isinstance(report, Mapping)
            or int(report.get("matched_pair_count", 0)) <= 0
            or not 0.0 < float(report.get("match_coverage", 0.0)) <= 1.0
            or float(report.get("task_distance_max", float("inf")))
            > float(expected_contract.task_match_epsilon)
        ):
            raise ValueError("Frozen comparator task-matching report differs.")
    if not isinstance(payload["source"], Mapping) or set(payload["source"]) != {
        "path",
        "sha256",
    }:
        raise ValueError("Frozen comparator raw-source identity differs.")
    source_provenance = payload["source_provenance"]
    if (
        not isinstance(source_provenance, Mapping)
        or set(source_provenance) != {"run_identity", "resource_ledger"}
    ):
        raise ValueError("Frozen comparator source provenance differs.")
    provenance_paths = {}
    for name in ("run_identity", "resource_ledger"):
        ref = source_provenance[name]
        if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}:
            raise ValueError("Frozen comparator source provenance differs.")
        provenance_path = Path(str(ref["path"])).resolve()
        if sha256_path(provenance_path) != ref["sha256"]:
            raise ValueError("Frozen comparator source provenance hash differs.")
        provenance_paths[name] = provenance_path
    source_registration = payload["source_registration"]
    registration_fields = {
        "layout",
        "official_source_commit",
        "config_fingerprint",
        "reference_ego_checkpoint",
        "reference_ego_resource",
        "comparator_partner_resources",
        "partner_manifest",
        "collection",
        "resource_ledger",
        "comparator_contract",
        "comparator_contract_fingerprint",
        "continuation_contract",
    }
    if (
        not isinstance(source_registration, Mapping)
        or set(source_registration) != registration_fields
        or source_registration["official_source_commit"] != OFFICIAL_SOURCE_COMMIT
        or source_registration["comparator_contract"]
        != expected_contract.to_mapping()
        or source_registration["comparator_contract_fingerprint"]
        != expected_contract.fingerprint
        or not isinstance(source_registration["layout"], str)
        or not source_registration["layout"]
        or not isinstance(source_registration["config_fingerprint"], str)
        or len(source_registration["config_fingerprint"]) != 64
    ):
        raise ValueError("Frozen comparator source registration differs.")
    for field in ("reference_ego_checkpoint", "partner_manifest"):
        descriptor = source_registration[field]
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"path", "sha256"}
            or sha256_path(Path(str(descriptor["path"])).resolve())
            != descriptor["sha256"]
        ):
            raise ValueError("Frozen comparator source lineage differs.")
    reference_resource = source_registration["reference_ego_resource"]
    parent_resources = source_registration["comparator_partner_resources"]
    if (
        not isinstance(reference_resource, Mapping)
        or set(reference_resource) != {
            "path", "sha256", "training_simulator_steps", "gpu_hours", "wall_clock_hours"
        }
        or sha256_path(Path(str(reference_resource["path"])).resolve())
        != reference_resource["sha256"]
        or int(reference_resource["training_simulator_steps"]) <= 0
        or not isinstance(parent_resources, Mapping)
        or not parent_resources
    ):
        raise ValueError("Frozen comparator shared-resource lineage differs.")
    for descriptor in parent_resources.values():
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {
                "path", "sha256", "training_simulator_steps", "gpu_hours", "wall_clock_hours"
            }
            or sha256_path(Path(str(descriptor["path"])).resolve())
            != descriptor["sha256"]
            or int(descriptor["training_simulator_steps"]) <= 0
        ):
            raise ValueError("Frozen comparator shared-resource lineage differs.")
    collection = source_registration["collection"]
    if (
        not isinstance(collection, Mapping)
        or collection.get("reference_policy") != "fixed_official_checkpoint"
        or collection.get("root_key") != [0, COMPARATOR_REFERENCE_ROOT_SEED]
        or int(collection.get("histories_per_partner_role", -1))
        != COMPARATOR_HISTORIES_PER_ROLE_RUN
        or min(
            int(collection.get("fit_replicas", 0)),
            int(collection.get("evaluation_replicas", 0)),
            int(collection.get("continuation_horizon", 0)),
        )
        <= 0
    ):
        raise ValueError("Frozen comparator collection registration differs.")
    try:
        source_ledger = ResourceLedger.from_mapping(
            source_registration["resource_ledger"]
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Frozen comparator source resource ledger differs.") from error
    if (
        source_ledger.comparator_continuation_steps <= 0
        or source_ledger.comparator_history_collection_steps <= 0
        or source_ledger.reference_ego_upstream_steps
        != int(reference_resource["training_simulator_steps"])
        or source_ledger.shared_pretraining_cost
        != sum(
            int(value["training_simulator_steps"])
            for value in parent_resources.values()
        )
        or not np.isclose(
            source_ledger.gpu_hours,
            float(reference_resource["gpu_hours"])
            + sum(float(value["gpu_hours"]) for value in parent_resources.values())
            + float(source_ledger.comparator_gpu_hours),
        )
        or source_ledger.wall_clock_hours
        < float(reference_resource["wall_clock_hours"])
        + sum(
            float(value["wall_clock_hours"])
            for value in parent_resources.values()
        )
    ):
        raise ValueError("Frozen comparator source resource ledger differs.")
    raw_source = Path(str(payload["source"]["path"])).resolve()
    if sha256_path(raw_source) != payload["source"]["sha256"]:
        raise ValueError("Frozen comparator raw-source hash differs.")
    raw_payload = json.loads(raw_source.read_text(encoding="utf-8"))
    raw_fields = {
        "version",
        "artifact_type",
        "layout",
        "official_source_commit",
        "config_fingerprint",
        "reference_ego_checkpoint",
        "reference_ego_resource",
        "comparator_partner_resources",
        "partner_manifest",
        "probe_steps",
        "signature_distance_threshold",
        "comparator_contract",
        "comparator_contract_fingerprint",
        "continuation_contract",
        "collection",
        "resource_ledger",
        "fit_histories",
        "validation_histories",
    }
    if (
        not isinstance(raw_payload, Mapping)
        or set(raw_payload) != raw_fields
        or int(raw_payload.get("version", -1)) != COMPARATOR_SOURCE_VERSION
        or raw_payload.get("artifact_type") != "depi_pair_comparator_source"
        or any(
            raw_payload.get(field) != source_registration[field]
            for field in registration_fields
        )
    ):
        raise ValueError("Frozen comparator raw source and registration differ.")
    collector_identity = json.loads(
        provenance_paths["run_identity"].read_text(encoding="utf-8")
    )
    collector_budget = json.loads(
        provenance_paths["resource_ledger"].read_text(encoding="utf-8")
    )
    if (
        provenance_paths["run_identity"] != raw_source.parent / "run_identity.json"
        or provenance_paths["resource_ledger"]
        != raw_source.parent / "budget_ledger.json"
        or collector_identity.get("stage") != "collect-pair-comparator-source"
        or collector_identity.get("layout") != source_registration["layout"]
        or collector_identity.get("config", {}).get("fingerprint")
        != source_registration["config_fingerprint"]
        or collector_identity.get("reference_ego_checkpoint")
        != source_registration["reference_ego_checkpoint"]
        or collector_identity.get("reference_ego_resource")
        != source_registration["reference_ego_resource"]
        or collector_identity.get("comparator_partner_resources")
        != source_registration["comparator_partner_resources"]
        or collector_identity.get("partner_manifest")
        != source_registration["partner_manifest"]
        or collector_identity.get("collection") != source_registration["collection"]
        or collector_identity.get("comparator_contract")
        != source_registration["comparator_contract"]
        or collector_identity.get("comparator_contract_fingerprint")
        != source_registration["comparator_contract_fingerprint"]
        or collector_budget != source_registration["resource_ledger"]
    ):
        raise ValueError("Frozen comparator collector provenance differs.")
    raw_fit = _rows(raw_payload["fit_histories"], label="fit")
    raw_validation = _rows(raw_payload["validation_histories"], label="validation")
    raw_fit_features = raw_fit["history_features"]
    raw_validation_features = raw_validation["history_features"]
    raw_fit_runs = raw_fit["partner_run_ids"]
    raw_validation_runs = raw_validation["partner_run_ids"]
    if (
        raw_fit_features.shape[1] != int(expected_history_feature_dim)
        or raw_validation_features.shape[1] != int(expected_history_feature_dim)
        or raw_fit_features.shape[0] != int(payload["fit_history_count"])
        or raw_validation_features.shape[0]
        != int(payload["validation_history_count"])
        or np.unique(raw_fit_runs).size != int(payload["fit_partner_run_count"])
        or np.unique(raw_validation_runs).size
        != int(payload["validation_partner_run_count"])
        or set(raw_fit_runs.tolist()) & set(raw_validation_runs.tolist())
    ):
        raise ValueError("Frozen comparator raw-source counts differ.")
    weights = np.asarray(payload["weights"], dtype=np.float32)
    if weights.shape != (2 * int(expected_history_feature_dim),):
        raise ValueError("Frozen comparator weight shape differs.")
    return FrozenPairComparator(
        weights=weights,
        bias=np.asarray(payload["bias"], dtype=np.float32),
        train_accuracy=np.asarray(payload["train_accuracy"], dtype=np.float32),
        validation_accuracy=np.asarray(payload["validation_accuracy"], dtype=np.float32),
        validation_accuracy_interval=tuple(payload["validation_accuracy_interval"]),
        history_feature_dim=int(payload["history_feature_dim"]),
        pair_feature_dim=int(payload["pair_feature_dim"]),
        development_row_count=int(payload["fit_pair_count"]),
        validation_row_count=int(payload["validation_pair_count"]),
        validation_partner_run_count=int(payload["validation_partner_run_count"]),
        development_partner_run_ids=np.full(
            (COMPARATOR_RUN_ID_CAPACITY,), -1, dtype=np.int32
        ),
        development_partner_run_count=0,
    )


def load_component_diagnostic_panel(
    path: str | Path, *, maximum_histories: int = 32
) -> Mapping[str, Any]:
    """Load the immutable validation-history panel shared by every ego seed."""

    artifact_path = Path(path).resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if (
        artifact.get("version") != COMPARATOR_ARTIFACT_VERSION
        or artifact.get("artifact_type") != "depi_frozen_pair_comparator"
        or not isinstance(artifact.get("source"), Mapping)
    ):
        raise ValueError("Component panel requires the active comparator artifact.")
    source_path = Path(str(artifact["source"]["path"])).resolve()
    if sha256_path(source_path) != artifact["source"]["sha256"]:
        raise ValueError("Component panel comparator source hash differs.")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("version") != COMPARATOR_SOURCE_VERSION:
        raise ValueError("Component panel source schema is stale.")
    rows = source.get("validation_histories")
    parsed = _rows(rows, label="validation")
    count = min(int(maximum_histories), int(parsed["history_features"].shape[0]))
    if count < 2:
        raise ValueError("Component diagnostic panel needs at least two histories.")
    selected = np.arange(count, dtype=np.int64)
    panel_ids = [
        hashlib.sha256(
            (
                str(rows[index]["partner_run_id"])
                + ":"
                + str(rows[index]["ego_role"])
                + ":"
                + str(rows[index]["episode_time"])
                + ":"
                + str(rows[index]["task_state_hash"])
            ).encode("utf-8")
        ).hexdigest()
        for index in selected
    ]
    return {
        "panel_ids": panel_ids,
        "state_hashes": [str(rows[index]["task_state_hash"]) for index in selected],
        "history_observations": parsed["history_observations"][selected],
        "history_actions": parsed["history_actions"][selected],
        "current_observations": parsed["current_observations"][selected],
        "source": {"path": str(source_path), "sha256": sha256_path(source_path)},
    }


__all__ = [
    "fit_frozen_pair_comparator",
    "collect_pair_comparator_source",
    "load_frozen_pair_comparator",
    "load_component_diagnostic_panel",
]
