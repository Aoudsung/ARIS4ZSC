"""K-independent, method-independent frozen matched-history comparator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    decision_distinction_pair_dataset,
    fit_pair_comparator,
    pair_legal_history_features,
)
from src.path_c.decision_geometry import centered_action_values
from src.path_c.experiment import (
    OFFICIAL_SOURCE_COMMIT,
    load_config,
    load_partner_manifest,
    validate_partner_manifest,
)
from src.path_c.resources import ResourceLedger
from src.path_c.storage import (
    ensure_run_identity,
    runtime_provenance,
    sha256_path,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
)


COMPARATOR_ARTIFACT_VERSION = 1
COMPARATOR_SOURCE_VERSION = 2
MINIMUM_HISTORIES_PER_SPLIT = 256
MINIMUM_RUNS_PER_SPLIT = 8
COMPARATOR_HISTORIES_PER_ROLE_RUN = 16
COMPARATOR_REFERENCE_ROOT_SEED = 42


def collect_pair_comparator_source(args: argparse.Namespace) -> None:
    """Collect a lineage-bound comparator source in the pinned simulator."""

    import jax

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
    reference_checkpoint_sha = sha256_path(reference_checkpoint)
    comparator_checkpoint_hashes = {
        str(run.checkpoint_sha256)
        for runs in split_runs.values()
        for run in runs
    }
    if reference_checkpoint_sha in comparator_checkpoint_hashes:
        raise ValueError("Comparator reference ego is also a comparator partner.")

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
    resource_ledger = ResourceLedger(
        counterfactual_continuation_steps=continuation_steps,
        evaluation_steps=trajectory_steps,
    ).to_mapping()
    payload = {
        "version": COMPARATOR_SOURCE_VERSION,
        "artifact_type": "depi_pair_comparator_source",
        "layout": config.environment.layout,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "config_fingerprint": config.fingerprint,
        "reference_ego_checkpoint": reference,
        "partner_manifest": partner_source,
        "probe_steps": probe_steps,
        "signature_distance_threshold": 1.0,
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
            "partner_manifest": partner_source,
            "collection": payload["collection"],
        },
    )
    write_json(output / "pair_comparator_source.json", payload)
    write_json(output / "budget_ledger.json", resource_ledger)


def _rows(
    payload: Any, *, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(payload, list) or len(payload) < MINIMUM_HISTORIES_PER_SPLIT:
        raise ValueError(
            f"Comparator {label} needs at least {MINIMUM_HISTORIES_PER_SPLIT} histories."
        )
    expected = {
        "history_features",
        "fit_returns_by_action",
        "evaluation_returns_by_action",
        "partner_run_id",
    }
    if any(not isinstance(row, Mapping) or set(row) != expected for row in payload):
        raise ValueError(f"Comparator {label} history row schema differs.")
    features = np.asarray([row["history_features"] for row in payload], dtype=np.float64)
    returns = np.asarray([row["fit_returns_by_action"] for row in payload], dtype=np.float64)
    evaluation_returns = np.asarray(
        [row["evaluation_returns_by_action"] for row in payload], dtype=np.float64
    )
    runs = np.asarray([str(row["partner_run_id"]) for row in payload])
    if (
        features.ndim != 2
        or returns.shape != (features.shape[0], 6)
        or evaluation_returns.shape != returns.shape
    ):
        raise ValueError(f"Comparator {label} feature/return shapes differ.")
    if (
        not np.all(np.isfinite(features))
        or not np.all(np.isfinite(returns))
        or not np.all(np.isfinite(evaluation_returns))
    ):
        raise ValueError(f"Comparator {label} contains non-finite values.")
    if np.unique(runs).size < MINIMUM_RUNS_PER_SPLIT:
        raise ValueError(
            f"Comparator {label} needs at least {MINIMUM_RUNS_PER_SPLIT} partner runs."
        )
    return (
        features,
        centered_action_values(returns),
        centered_action_values(evaluation_returns),
        runs,
    )


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
        "partner_manifest",
        "probe_steps",
        "signature_distance_threshold",
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
    for field in ("reference_ego_checkpoint", "partner_manifest"):
        descriptor = payload[field]
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256"}:
            raise ValueError(f"Comparator source {field} identity differs.")
        if sha256_path(Path(str(descriptor["path"])).resolve()) != descriptor["sha256"]:
            raise ValueError(f"Comparator source {field} hash differs.")
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
        ledger.counterfactual_continuation_steps <= 0
        or ledger.evaluation_steps <= 0
        or ledger.total_training_simulator_steps
        != ledger.counterfactual_continuation_steps
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
        or collector_identity.get("partner_manifest") != payload["partner_manifest"]
        or collector_identity.get("collection") != payload["collection"]
        or collector_budget != ledger_payload
    ):
        raise ValueError("Comparator source collector provenance differs.")
    (
        fit_features,
        fit_signatures,
        unused_fit_evaluation_signatures,
        fit_runs,
    ) = _rows(payload["fit_histories"], label="fit")
    (
        validation_features,
        unused_validation_fit_signatures,
        validation_signatures,
        validation_runs,
    ) = _rows(payload["validation_histories"], label="validation")
    del unused_fit_evaluation_signatures, unused_validation_fit_signatures
    if fit_features.shape[1] != validation_features.shape[1]:
        raise ValueError("Comparator fit/validation feature dimensions differ.")
    if set(fit_runs.tolist()) & set(validation_runs.tolist()):
        raise ValueError("Comparator fit and validation partner runs overlap.")
    fit_rows, fit_labels, fit_blocks = decision_distinction_pair_dataset(
        fit_features,
        fit_signatures,
        signature_distance_threshold=1.0,
        block_ids=fit_runs,
    )
    validation_rows, validation_labels, validation_blocks = (
        decision_distinction_pair_dataset(
            validation_features,
            validation_signatures,
            signature_distance_threshold=1.0,
            block_ids=validation_runs,
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
        "estimand": "P(decision-signature distance > 1.0 | legal history pair)",
        "method_independent": True,
        "protocol_component_count_independent": True,
        "state_matching_space": "fixed_simulator_task_planes",
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
            "partner_manifest": payload["partner_manifest"],
            "collection": payload["collection"],
            "resource_ledger": payload["resource_ledger"],
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
    path: str | Path, *, expected_history_feature_dim: int
) -> FrozenPairComparator:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "version", "artifact_type", "estimand", "method_independent",
        "protocol_component_count_independent", "state_matching_space",
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
        or int(payload["history_feature_dim"]) != int(expected_history_feature_dim)
        or int(payload["pair_feature_dim"]) != 2 * int(expected_history_feature_dim)
        or int(payload["fit_history_count"]) < MINIMUM_HISTORIES_PER_SPLIT
        or int(payload["validation_history_count"]) < MINIMUM_HISTORIES_PER_SPLIT
        or int(payload["fit_partner_run_count"]) < MINIMUM_RUNS_PER_SPLIT
        or int(payload["validation_partner_run_count"]) < MINIMUM_RUNS_PER_SPLIT
    ):
        raise ValueError("Frozen comparator registration differs.")
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
        "partner_manifest",
        "collection",
        "resource_ledger",
    }
    if (
        not isinstance(source_registration, Mapping)
        or set(source_registration) != registration_fields
        or source_registration["official_source_commit"] != OFFICIAL_SOURCE_COMMIT
        or not isinstance(source_registration["layout"], str)
        or not source_registration["layout"]
        or not isinstance(source_registration["config_fingerprint"], str)
        or len(source_registration["config_fingerprint"]) != 64
    ):
        raise ValueError("Frozen comparator source registration differs.")
    for field in ("reference_ego_checkpoint", "partner_manifest"):
        descriptor = source_registration[field]
        if not isinstance(descriptor, Mapping) or set(descriptor) != {"path", "sha256"}:
            raise ValueError("Frozen comparator source lineage differs.")
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
        source_ledger.counterfactual_continuation_steps <= 0
        or source_ledger.evaluation_steps <= 0
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
        "partner_manifest",
        "probe_steps",
        "signature_distance_threshold",
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
        or collector_identity.get("partner_manifest")
        != source_registration["partner_manifest"]
        or collector_identity.get("collection") != source_registration["collection"]
        or collector_budget != source_registration["resource_ledger"]
    ):
        raise ValueError("Frozen comparator collector provenance differs.")
    raw_fit_features, unused_a, unused_b, raw_fit_runs = _rows(
        raw_payload["fit_histories"], label="fit"
    )
    raw_validation_features, unused_c, unused_d, raw_validation_runs = _rows(
        raw_payload["validation_histories"], label="validation"
    )
    del unused_a, unused_b, unused_c, unused_d
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


__all__ = [
    "fit_frozen_pair_comparator",
    "collect_pair_comparator_source",
    "load_frozen_pair_comparator",
]
