"""Real-simulator DEPI identifiability and recoverable-value controls.

The live collector replays legal histories in the pinned Official simulator,
clones simulator and recurrent-policy state, and applies only registered
recurrent-state interventions.  Every comparison keeps the ego checkpoint,
current environment, partner policy and carry, role, observation, episode
time, action RNG, and continuation RNG fixed.  All-action returns use fit and
evaluation replica blocks that are disjoint by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.overcooked_v2.common_partner_app import (
    _load_common_panel_policies,
    _official_environment,
    _validate_common_panel,
)
from experiments.overcooked_v2.deployment import load_deployment
from experiments.overcooked_v2.official_adapter import (
    official_policy,
    restore_official_checkpoint,
    validate_official_runtime,
)
from experiments.overcooked_v2.official_br_prox_app import (
    _stack_initial_hstate,
    empirical_all_action_continuations_from_snapshots,
    record_official_anchor_snapshots,
)
from experiments.overcooked_v2.official_evaluation_app import (
    _load_policies,
    _load_policy_manifest,
)
from experiments.overcooked_v2.official_policy import OfficialDEPIPolicy
from src.path_c.experiment import (
    METHOD_VERSION,
    OFFICIAL_SOURCE_COMMIT,
    load_config,
    load_partner_manifest,
    official_training_domain_keys,
)
from src.path_c.identifiability_controls import (
    LEAKAGE_PROBE_EXCESS_THRESHOLD,
    SHUFFLE_EPSILON_QUANTILE,
    balanced_linear_probe_accuracy,
    continuation_swap_causal_consistency,
    match_different_partner_task_states,
    paired_bootstrap_drop,
    task_representation_drift_under_shuffle,
)
from src.path_c.storage import (
    ensure_run_identity,
    runtime_provenance,
    sha256_path,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
)


IDENTIFIABILITY_SCHEMA_VERSION = 3
RECOVERABLE_VALUE_SCHEMA_VERSION = 3
MECHANISM_RAW_SCHEMA_VERSION = 3


_RAW_BASE_FIELDS = {
    "version",
    "artifact_type",
    "method",
    "method_variant",
    "layout",
    "official_source_commit",
    "collector",
    "config_fingerprint",
    "policy_manifest_sha256",
    "partner_manifest_sha256",
    "registration",
    "resource_ledger",
    "rows",
}


_COMMON_ROW_FIELDS = {
    "unit_id",
    "ego_run_id",
    "ego_checkpoint_sha256",
    "partner_run_id",
    "partner_mechanism",
    "donor_partner_run_id",
    "donor_unit_id",
    "episode_id",
    "episode_time",
    "environment_key",
    "ego_role",
    "current_observation_sha256",
    "task_state_sha256",
    "continuation_horizon",
    "fit_replica_count",
    "evaluation_replica_count",
}


def _load_raw(path: str | Path, artifact_type: str) -> tuple[Path, Mapping[str, Any]]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = set(_RAW_BASE_FIELDS)
    if artifact_type == "depi_identifiability_raw":
        required.add("leakage_rows")
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError(
            f"Mechanism raw artifact fields differ from schema {MECHANISM_RAW_SCHEMA_VERSION}."
        )
    if (
        int(payload["version"]) != MECHANISM_RAW_SCHEMA_VERSION
        or payload["artifact_type"] != artifact_type
    ):
        raise ValueError("Mechanism raw artifact type/version mismatch.")
    if payload["method"] != METHOD_VERSION or payload["method_variant"] != "b2":
        raise ValueError("Mechanism raw artifact is not from the registered DEPI B2 method.")
    if payload["official_source_commit"] != OFFICIAL_SOURCE_COMMIT:
        raise ValueError("Mechanism raw artifact is not from the pinned simulator.")
    if payload["collector"] != "real_environment_crn_continuations":
        raise ValueError("Mechanism controls require real simulator continuations.")
    if not isinstance(payload["rows"], Sequence) or not payload["rows"]:
        raise ValueError("Mechanism raw artifact contains no matched rows.")
    return source, payload


def _validate_common_row(row: Mapping[str, Any]) -> None:
    key = row["environment_key"]
    if not isinstance(key, Sequence) or len(key) != 2:
        raise ValueError("Every mechanism row needs one two-word CRN environment key.")
    if int(row["ego_role"]) not in {0, 1}:
        raise ValueError("ego_role must be zero or one.")
    if min(
        int(row["continuation_horizon"]),
        int(row["fit_replica_count"]),
        int(row["evaluation_replica_count"]),
    ) <= 0:
        raise ValueError("Real continuation horizon and replica counts must be positive.")
    if not row["current_observation_sha256"] or not row["task_state_sha256"]:
        raise ValueError("Current observation and task-state identities are required.")
    if not row["ego_checkpoint_sha256"]:
        raise ValueError("Every mechanism row must identify the frozen ego checkpoint.")
    if str(row["partner_run_id"]) == str(row["donor_partner_run_id"]):
        raise ValueError("History/context donor must come from a different partner run.")


def _run_means(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["ego_run_id"]), []).append(float(row[field]))
    if len(grouped) < 2:
        raise ValueError("Run-level mechanism inference needs at least two ego runs.")
    return np.asarray(
        [np.mean(grouped[name]) for name in sorted(grouped)], dtype=np.float64
    )


def _tree_take(tree: Any, indexes: Any) -> Any:
    import jax
    import jax.numpy as jnp

    index = jnp.asarray(indexes, dtype=jnp.int32)
    return jax.tree_util.tree_map(
        lambda value: None if value is None else jnp.asarray(value)[index],
        tree,
        is_leaf=lambda value: value is None,
    )


def _tree_concat(trees: Sequence[Any]) -> Any:
    import jax
    import jax.numpy as jnp

    if not trees:
        raise ValueError("Cannot concatenate an empty state collection.")
    return jax.tree_util.tree_map(
        lambda *values: (
            None
            if values[0] is None
            else jnp.concatenate(
                tuple(jnp.asarray(value) for value in values), axis=0
            )
        ),
        *trees,
        is_leaf=lambda value: value is None,
    )


def _squeeze_official_hstate(state: Any) -> Any:
    """Remove the adapter's per-policy singleton batch axis."""

    import jax
    import jax.numpy as jnp

    def squeeze(value: Any) -> Any:
        array = jnp.asarray(value)
        if array.ndim < 2 or int(array.shape[1]) != 1:
            raise ValueError("DEPI Official state lacks its singleton policy axis.")
        return array[:, 0]

    return jax.tree_util.tree_map(squeeze, state)


def _task_features(policy: Any, hstate: Any, observation: Any) -> np.ndarray:
    import jax.numpy as jnp

    if not isinstance(policy, OfficialDEPIPolicy):
        raise TypeError("Mechanism controls require a DEPI deployment policy.")
    state = _squeeze_official_hstate(hstate)
    count = int(np.asarray(observation).shape[0])
    _, output = policy.deployment.model.apply(
        {"params": policy.deployment.params},
        state,
        jnp.asarray(observation),
        jnp.zeros((count,), dtype=jnp.bool_),
        method=policy.deployment.model.step,
    )
    return np.asarray(output.task_features, dtype=np.float64)


def _policy_probabilities(policy: Any, hstate: Any, observation: Any) -> np.ndarray:
    """Read the deployed action distribution without sampling an action."""

    import jax
    import jax.numpy as jnp

    if not isinstance(policy, OfficialDEPIPolicy):
        raise TypeError("Mechanism controls require a DEPI deployment policy.")
    state = _squeeze_official_hstate(hstate)
    count = int(np.asarray(observation).shape[0])
    _, output = policy.deployment.model.apply(
        {"params": policy.deployment.params},
        state,
        jnp.asarray(observation),
        jnp.zeros((count,), dtype=jnp.bool_),
        method=policy.deployment.model.step,
    )
    return np.asarray(jax.nn.softmax(output.policy_logits, axis=-1), dtype=np.float64)


def _history_intervention(
    source: Any,
    donor: Any,
    *,
    mode: str,
    capability_dim: int,
) -> Any:
    """Replace only registered capability/protocol state fields."""

    import jax.numpy as jnp

    if mode == "history":
        return source._replace(
            capability_carry=donor.capability_carry,
            protocol_carry=donor.protocol_carry,
            context_summary=donor.context_summary,
        )
    source_summary = jnp.asarray(source.context_summary)
    donor_summary = jnp.asarray(donor.context_summary)
    cut = int(capability_dim)
    if mode == "swap_u":
        return source._replace(
            capability_carry=donor.capability_carry,
            context_summary=jnp.concatenate(
                (donor_summary[..., :cut], source_summary[..., cut:]), axis=-1
            ),
        )
    if mode == "swap_c":
        return source._replace(
            protocol_carry=donor.protocol_carry,
            context_summary=jnp.concatenate(
                (source_summary[..., :cut], donor_summary[..., cut:]), axis=-1
            ),
        )
    raise ValueError(f"Unknown recurrent-state intervention: {mode}")


def _pytree_sha256(tree: Any) -> str:
    import jax

    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _center_signature(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array - np.mean(array, axis=-1, keepdims=True)


def _collector_registration(config: Any) -> Mapping[str, Any]:
    return {
        "anchors_per_pairing": int(config.evaluation.mechanism_anchors_per_pairing),
        "fit_replicas": int(config.evaluation.mechanism_fit_replicas),
        "evaluation_replicas": int(
            config.evaluation.mechanism_evaluation_replicas
        ),
        "continuation_horizon": int(
            config.evaluation.mechanism_continuation_horizon
        ),
        "task_match_epsilon_quantile": SHUFFLE_EPSILON_QUANTILE,
        "recoverable_signal_threshold": float(
            config.evaluation.recoverable_signal_threshold
        ),
        "fit_and_evaluation_replicas_disjoint": True,
        "history_shuffle_fields": [
            "capability_carry",
            "protocol_carry",
            "context_summary",
        ],
        "fixed_fields": [
            "ego_checkpoint",
            "environment_state",
            "partner_policy_and_carry",
            "ego_role",
            "current_observation",
            "task_carry",
            "episode_time",
            "continuation_keys",
        ],
    }


def collect_mechanism_raw_artifacts(
    *,
    config_path: str | Path,
    policy_manifest_path: str | Path,
    partner_manifest_path: str | Path,
    output: str | Path,
    skip_manifest_hash_check: bool = False,
    dry_run_training_runs: Sequence[str | Path] | None = None,
    dry_run_partner_count: int = 2,
    dry_run_episodes: int = 50,
) -> Mapping[str, Path]:
    """Collect both mechanism artifacts from real Official continuations."""

    import jax
    import jax.numpy as jnp

    dry_run = dry_run_training_runs is not None
    if not dry_run:
        validate_formal_repository_state()
        validate_registered_python_runtime()
    validate_official_runtime()
    config = load_config(config_path, run_kind="mechanical" if dry_run else "formal")
    if config.method_variant != "b2":
        raise ValueError("Mechanism attribution is registered only for the B2 deployment.")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    policy_path = Path(policy_manifest_path).resolve()
    partner_path = Path(partner_manifest_path).resolve()
    partner_manifest = load_partner_manifest(
        partner_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(skip_manifest_hash_check),
    )
    if dry_run:
        training_runs = tuple(Path(value).resolve() for value in dry_run_training_runs or ())
        if len(training_runs) != 2:
            raise ValueError("Scientific dry run requires exactly two ego training runs.")
        runs = []
        ego_policies = []
        for index, training_run in enumerate(training_runs):
            deployment_path = training_run / "final_deployment"
            deployment = load_deployment(deployment_path, config)
            ego_policies.append(OfficialDEPIPolicy(deployment))
            runs.append(
                {
                    "run_index": index,
                    "run_id": deployment.ego_run_id,
                    "checkpoint": str(deployment_path),
                    "checkpoint_sha256": sha256_path(deployment_path),
                }
            )
        manifest = {
            "method": "depi",
            "policy_kind": "depi_deployment",
            "runs": runs,
        }
        write_json(
            policy_path,
            {
                "version": 1,
                "artifact_type": "depi_scientific_dry_run_policy_manifest",
                "method": METHOD_VERSION,
                "layout": config.environment.layout,
                "runs": runs,
                "scientific_readout_allowed": False,
            },
        )
        partners = tuple(partner_manifest.by_role("confirmatory"))[
            : int(dry_run_partner_count)
        ]
        if len(partners) < 2 or len(
            {run.parent_training_run_id for run in partners}
        ) != len(partners):
            raise ValueError(
                "Scientific dry run needs at least two independent fresh confirmatory partners."
            )
        partner_policies = []
        for run in partners:
            partner_config, partner_params = restore_official_checkpoint(run.checkpoint)
            partner_policies.append(official_policy(partner_params, partner_config))
        partners = tuple(partners)
        partner_policies = tuple(partner_policies)
        ego_policies = tuple(ego_policies)
    else:
        manifest = _load_policy_manifest(
            policy_path, expected_layout=config.environment.layout
        )
        if manifest["method"] != "depi" or manifest["policy_kind"] != "depi_deployment":
            raise ValueError("Mechanism controls require the ten-run DEPI policy manifest.")
        trained_partners = _validate_common_panel(
            partner_manifest, {"depi": manifest}
        )
        partners, partner_policies = _load_common_panel_policies(trained_partners)
        ego_policies = _load_policies(manifest, config)
    environment = _official_environment(config)
    anchors = int(config.evaluation.mechanism_anchors_per_pairing)
    fit_replicas = int(config.evaluation.mechanism_fit_replicas)
    evaluation_replicas = int(config.evaluation.mechanism_evaluation_replicas)
    horizon = int(config.evaluation.mechanism_continuation_horizon)

    identifiability_rows: list[Mapping[str, Any]] = []
    recoverable_rows: list[Mapping[str, Any]] = []
    leakage_rows: list[Mapping[str, Any]] = []
    trajectory_replay_steps = 0
    continuation_steps = 0

    for ego_index, ego_policy in enumerate(ego_policies):
        ego_run = manifest["runs"][ego_index]
        ego_run_id = str(ego_run["run_id"])
        ego_checkpoint = str(ego_run["checkpoint_sha256"])
        domain_root = jnp.asarray(
            official_training_domain_keys(ego_index)["identifiability"],
            dtype=jnp.uint32,
        )
        for ego_role in (0, 1):
            root = jax.random.fold_in(domain_root, ego_role)
            snapshots = []
            offset = 0
            for partner_index, (partner_descriptor, partner_policy) in enumerate(
                zip(partners, partner_policies, strict=True)
            ):
                left, right = (
                    (ego_policy, partner_policy)
                    if ego_role == 0
                    else (partner_policy, ego_policy)
                )
                recorded = record_official_anchor_snapshots(
                    left=left,
                    right=right,
                    environment=environment,
                    root_key=root,
                    anchors=anchors,
                    continuation_horizon=horizon,
                    episodes=(
                        int(dry_run_episodes)
                        if dry_run
                        else int(config.evaluation.episodes_per_pairing)
                    ),
                )
                selected = recorded["selected"]
                ego_hstate = (
                    selected["left_hstate"]
                    if ego_role == 0
                    else selected["right_hstate"]
                )
                observation = selected["observations"][
                    "agent_0" if ego_role == 0 else "agent_1"
                ]
                features = _task_features(ego_policy, ego_hstate, observation)
                legal = empirical_all_action_continuations_from_snapshots(
                    left=left,
                    right=right,
                    ego_role=ego_role,
                    environment=environment,
                    selected=selected,
                    anchor_roots=recorded["anchor_roots"],
                    fit_replicas=fit_replicas,
                    evaluation_replicas=evaluation_replicas,
                    continuation_horizon=horizon,
                    ego_hstate=ego_hstate,
                )
                continuation_steps += (
                    anchors * 6 * (fit_replicas + evaluation_replicas) * horizon
                )
                trajectory_replay_steps += anchors * int(environment.max_steps)
                snapshots.append(
                    {
                        **recorded,
                        "ego_hstate": ego_hstate,
                        "task_features": features,
                        "legal": legal,
                        "partner_index": partner_index,
                        "partner_run_id": str(partner_descriptor.run_id),
                        "partner_mechanism": str(
                            partner_descriptor.generation_mechanism
                        ),
                        "offset": offset,
                    }
                )
                offset += anchors

            all_features = np.concatenate(
                [snapshot["task_features"] for snapshot in snapshots], axis=0
            )
            all_partner_ids = np.concatenate(
                [
                    np.repeat(snapshot["partner_run_id"], anchors)
                    for snapshot in snapshots
                ]
            )
            all_states = _tree_concat(
                [snapshot["ego_hstate"] for snapshot in snapshots]
            )
            all_legal_fit = np.concatenate(
                [snapshot["legal"]["fit_returns_by_action"] for snapshot in snapshots],
                axis=0,
            )
            all_sources, all_donors, unused_all_epsilon, unused_all_distances = (
                match_different_partner_task_states(
                    all_features, all_partner_ids, epsilon_quantile=1.0
                )
            )
            del unused_all_epsilon, unused_all_distances
            if not np.array_equal(all_sources, np.arange(all_features.shape[0])):
                raise AssertionError("Every snapshot needs a different-partner donor.")
            matched_sources, matched_donors, epsilon, matched_distances = (
                match_different_partner_task_states(
                    all_features,
                    all_partner_ids,
                    epsilon_quantile=SHUFFLE_EPSILON_QUANTILE,
                )
            )
            donor_by_source = np.asarray(all_donors, dtype=np.int64)
            matched_donor_by_source = {
                int(source): (int(donor), float(distance))
                for source, donor, distance in zip(
                    matched_sources, matched_donors, matched_distances, strict=True
                )
            }

            for snapshot in snapshots:
                start = int(snapshot["offset"])
                stop = start + anchors
                source_global = np.arange(start, stop, dtype=np.int64)
                source_state = snapshot["ego_hstate"]
                all_donor_state = _tree_take(
                    all_states, donor_by_source[source_global]
                )
                all_history_state = _history_intervention(
                    source_state,
                    all_donor_state,
                    mode="history",
                    capability_dim=config.model.capability_dim,
                )
                all_shuffled_features = _task_features(
                    ego_policy,
                    all_history_state,
                    snapshot["selected"]["observations"][
                        "agent_0" if ego_role == 0 else "agent_1"
                    ],
                )
                from src.path_c.task_encoder import task_only_observation

                snapshot_observations = np.asarray(
                    snapshot["selected"]["observations"][
                        "agent_0" if ego_role == 0 else "agent_1"
                    ]
                )
                fixed_task_state_features = np.asarray(
                    task_only_observation(snapshot_observations)
                ).reshape((anchors, -1))
                for local_index, global_index in enumerate(source_global):
                    donor_global = int(donor_by_source[global_index])
                    leakage_rows.append(
                        {
                            "unit_id": (
                                f"{ego_run_id}:r{ego_role}:p{snapshot['partner_index']}:"
                                f"a{local_index}"
                            ),
                            "ego_run_id": ego_run_id,
                            "partner_run_label": snapshot["partner_run_id"],
                            "donor_partner_run_id": str(all_partner_ids[donor_global]),
                            "episode_group_id": (
                                f"{ego_run_id}:r{ego_role}:p{snapshot['partner_index']}:"
                                f"e{int(np.asarray(snapshot['episode_indexes'])[local_index])}"
                            ),
                            "task_state_features": fixed_task_state_features[
                                local_index
                            ].tolist(),
                            "task_features": all_features[global_index].tolist(),
                            "task_features_history_shuffled": (
                                all_shuffled_features[local_index].tolist()
                            ),
                        }
                    )

                matched_local = np.asarray(
                    [
                        global_index - start
                        for global_index in source_global
                        if int(global_index) in matched_donor_by_source
                    ],
                    dtype=np.int64,
                )
                if matched_local.size == 0:
                    continue
                matched_global = start + matched_local
                donor_global = np.asarray(
                    [
                        matched_donor_by_source[int(index)][0]
                        for index in matched_global
                    ],
                    dtype=np.int64,
                )
                distances = np.asarray(
                    [
                        matched_donor_by_source[int(index)][1]
                        for index in matched_global
                    ],
                    dtype=np.float64,
                )
                selected = _tree_take(snapshot["selected"], matched_local)
                roots = np.asarray(snapshot["anchor_roots"])[matched_local]
                source_state = _tree_take(snapshot["ego_hstate"], matched_local)
                donor_state = _tree_take(all_states, donor_global)
                history_state = _history_intervention(
                    source_state,
                    donor_state,
                    mode="history",
                    capability_dim=config.model.capability_dim,
                )
                swap_u_state = _history_intervention(
                    source_state,
                    donor_state,
                    mode="swap_u",
                    capability_dim=config.model.capability_dim,
                )
                swap_c_state = _history_intervention(
                    source_state,
                    donor_state,
                    mode="swap_c",
                    capability_dim=config.model.capability_dim,
                )
                state_only = _stack_initial_hstate(ego_policy, matched_local.size)
                left, right = (
                    (ego_policy, partner_policies[snapshot["partner_index"]])
                    if ego_role == 0
                    else (partner_policies[snapshot["partner_index"]], ego_policy)
                )

                def evaluate(candidate_state: Any) -> Mapping[str, Any]:
                    nonlocal continuation_steps
                    continuation_steps += (
                        matched_local.size
                        * 6
                        * (fit_replicas + evaluation_replicas)
                        * horizon
                    )
                    return empirical_all_action_continuations_from_snapshots(
                        left=left,
                        right=right,
                        ego_role=ego_role,
                        environment=environment,
                        selected=selected,
                        anchor_roots=roots,
                        fit_replicas=fit_replicas,
                        evaluation_replicas=evaluation_replicas,
                        continuation_horizon=horizon,
                        ego_hstate=candidate_state,
                    )

                shuffled = evaluate(history_state)
                state_only_values = evaluate(state_only)
                swapped_u = evaluate(swap_u_state)
                swapped_c = evaluate(swap_c_state)
                source_observations = selected["observations"][
                    "agent_0" if ego_role == 0 else "agent_1"
                ]
                original_policy_probabilities = _policy_probabilities(
                    ego_policy, source_state, source_observations
                )
                swapped_c_policy_probabilities = _policy_probabilities(
                    ego_policy, swap_c_state, source_observations
                )
                shuffled_features = _task_features(
                    ego_policy, history_state, source_observations
                )
                legal = snapshot["legal"]
                source_signature = _center_signature(
                    legal["fit_returns_by_action"][matched_local]
                )
                target_signature = _center_signature(all_legal_fit[donor_global])

                for index, local_index in enumerate(matched_local):
                    global_index = int(matched_global[index])
                    donor_index = int(donor_global[index])
                    partner_run_id = snapshot["partner_run_id"]
                    donor_partner_id = str(all_partner_ids[donor_index])
                    unit_id = (
                        f"{ego_run_id}:r{ego_role}:p{snapshot['partner_index']}:"
                        f"a{int(local_index)}"
                    )
                    donor_unit_id = f"{ego_run_id}:r{ego_role}:g{donor_index}"
                    current_observation = np.asarray(source_observations[index])
                    environment_state = _tree_take(
                        selected["environment_state"], np.asarray([index])
                    )
                    source_task_carry = _tree_take(
                        source_state.task_carry, np.asarray([index])
                    )
                    common = {
                        "unit_id": unit_id,
                        "ego_run_id": ego_run_id,
                        "ego_checkpoint_sha256": ego_checkpoint,
                        "partner_run_id": partner_run_id,
                        "partner_mechanism": snapshot["partner_mechanism"],
                        "donor_partner_run_id": donor_partner_id,
                        "donor_unit_id": donor_unit_id,
                        "episode_id": int(
                            np.asarray(snapshot["episode_indexes"])[int(local_index)]
                        ),
                        "episode_time": int(
                            np.asarray(snapshot["time_indexes"])[int(local_index)]
                        ),
                        "environment_key": [int(value) for value in roots[index]],
                        "ego_role": ego_role,
                        "current_observation_sha256": _pytree_sha256(
                            current_observation
                        ),
                        "task_state_sha256": _pytree_sha256(
                            (environment_state, source_task_carry)
                        ),
                        "continuation_horizon": horizon,
                        "fit_replica_count": fit_replicas,
                        "evaluation_replica_count": evaluation_replicas,
                    }
                    original_returns = np.asarray(
                        legal["evaluation_returns_by_action"][int(local_index)],
                        dtype=np.float64,
                    )
                    identifiability_rows.append(
                        {
                            **common,
                            "task_features": all_features[global_index].tolist(),
                            "task_features_history_shuffled": shuffled_features[
                                index
                            ].tolist(),
                            "partner_run_label": partner_run_id,
                            "legal_history_return": float(
                                legal["selected_evaluation_return"][int(local_index)]
                            ),
                            "history_shuffled_return": float(
                                shuffled["selected_evaluation_return"][index]
                            ),
                            "task_match_distance": float(distances[index]),
                            "task_match_epsilon": float(epsilon),
                            "swap_u_original_returns_by_action": original_returns.tolist(),
                            "swap_u_returns_by_action": np.asarray(
                                swapped_u["evaluation_returns_by_action"][index]
                            ).tolist(),
                            "swap_u_source_signature": source_signature[index].tolist(),
                            "swap_u_target_signature": target_signature[index].tolist(),
                            "swap_c_original_returns_by_action": original_returns.tolist(),
                            "swap_c_returns_by_action": np.asarray(
                                swapped_c["evaluation_returns_by_action"][index]
                            ).tolist(),
                            "swap_c_source_signature": source_signature[index].tolist(),
                            "swap_c_target_signature": target_signature[index].tolist(),
                            "original_policy_probabilities": (
                                original_policy_probabilities[index].tolist()
                            ),
                            "swap_c_policy_probabilities": (
                                swapped_c_policy_probabilities[index].tolist()
                            ),
                        }
                    )
                    recoverable_rows.append(
                        {
                            **common,
                            "g1": float(
                                legal["selected_evaluation_return"][int(local_index)]
                            ),
                            "g2": float(shuffled["selected_evaluation_return"][index]),
                            "g3": float(
                                state_only_values["selected_evaluation_return"][index]
                            ),
                            "g4": float(
                                legal["oracle_evaluation_return"][int(local_index)]
                            ),
                            "g4_selection_stability": float(
                                np.mean(
                                    np.argmax(
                                        legal["fit_replica_returns_by_action"][
                                            int(local_index)
                                        ],
                                        axis=0,
                                    )
                                    == int(legal["fit_oracle_action"][int(local_index)])
                                )
                            ),
                        }
                    )

    expected_ego_count = 2 if dry_run else 10
    if len({row["ego_run_id"] for row in identifiability_rows}) != expected_ego_count:
        raise RuntimeError(
            "Mechanism collection did not retain matched rows for every ego."
        )
    registration = _collector_registration(config)
    if dry_run:
        registration = {
            **registration,
            "execution_scope": "two-ego-real-simulator-scientific-dry-run",
            "episodes_per_pairing": int(dry_run_episodes),
            "scientific_readout_allowed": False,
        }
    resource_ledger = {
        "trajectory_replay_steps": int(trajectory_replay_steps),
        "continuation_steps": int(continuation_steps),
        "total_simulator_steps": int(trajectory_replay_steps + continuation_steps),
    }
    common_artifact = {
        "version": MECHANISM_RAW_SCHEMA_VERSION,
        "method": METHOD_VERSION,
        "method_variant": config.method_variant,
        "layout": config.environment.layout,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "collector": "real_environment_crn_continuations",
        "config_fingerprint": config.fingerprint,
        "policy_manifest_sha256": sha256_path(policy_path),
        "partner_manifest_sha256": sha256_path(partner_path),
        "registration": registration,
        "resource_ledger": resource_ledger,
    }
    ident_path = root / "depi_identifiability_raw.json"
    recoverable_path = root / "depi_recoverable_value_raw.json"
    write_json(
        ident_path,
        {
            **common_artifact,
            "artifact_type": "depi_identifiability_raw",
            "leakage_rows": leakage_rows,
            "rows": identifiability_rows,
        },
    )
    write_json(
        recoverable_path,
        {
            **common_artifact,
            "artifact_type": "depi_recoverable_value_raw",
            "rows": recoverable_rows,
        },
    )
    return {"identifiability": ident_path, "recoverable_value": recoverable_path}


def _resolve_raw_input(
    args: argparse.Namespace, *, artifact: str, output: Path
) -> tuple[Path, bool]:
    raw_input = getattr(args, "raw_input", None)
    if raw_input:
        return Path(raw_input).resolve(), False
    required = {
        "config": getattr(args, "config", None),
        "policy_manifest": getattr(args, "policy_manifest", None),
        "partner_manifest": getattr(args, "partner_manifest", None),
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise ValueError(
            "Live mechanism collection requires " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    identity = {
        "stage": f"evaluate-{artifact.replace('_', '-')}",
        "method": METHOD_VERSION,
        "collector": "real_environment_crn_continuations",
        "repository_runtime": runtime_provenance(),
        "config": str(Path(required["config"]).resolve()),
        "policy_manifest_sha256": sha256_path(required["policy_manifest"]),
        "partner_manifest_sha256": sha256_path(required["partner_manifest"]),
    }
    ensure_run_identity(output, identity)
    paths = collect_mechanism_raw_artifacts(
        config_path=required["config"],
        policy_manifest_path=required["policy_manifest"],
        partner_manifest_path=required["partner_manifest"],
        output=output / "raw",
        skip_manifest_hash_check=bool(
            getattr(args, "skip_manifest_hash_check", False)
        ),
    )
    return paths[artifact], True


def run_identifiability_evaluation(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    raw_path, live = _resolve_raw_input(
        args, artifact="identifiability", output=output
    )
    source, payload = _load_raw(raw_path, "depi_identifiability_raw")
    required = _COMMON_ROW_FIELDS | {
        "task_features",
        "task_features_history_shuffled",
        "partner_run_label",
        "legal_history_return",
        "history_shuffled_return",
        "task_match_distance",
        "task_match_epsilon",
        "swap_u_original_returns_by_action",
        "swap_u_returns_by_action",
        "swap_u_source_signature",
        "swap_u_target_signature",
        "swap_c_original_returns_by_action",
        "swap_c_returns_by_action",
        "swap_c_source_signature",
        "swap_c_target_signature",
        "original_policy_probabilities",
        "swap_c_policy_probabilities",
    }
    rows = list(payload["rows"])
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError(
                f"Identifiability raw row fields differ from schema {MECHANISM_RAW_SCHEMA_VERSION}."
            )
        _validate_common_row(row)
        if float(row["task_match_distance"]) > float(row["task_match_epsilon"]):
            raise ValueError("Context swap pair is not task-state matched.")
        for name in (
            "swap_u_original_returns_by_action",
            "swap_u_returns_by_action",
            "swap_u_source_signature",
            "swap_u_target_signature",
            "swap_c_original_returns_by_action",
            "swap_c_returns_by_action",
            "swap_c_source_signature",
            "swap_c_target_signature",
            "original_policy_probabilities",
            "swap_c_policy_probabilities",
        ):
            values = np.asarray(row[name], dtype=np.float64)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain six finite action continuations.")

    leakage_rows = list(payload["leakage_rows"])
    leakage_required = {
        "unit_id",
        "ego_run_id",
        "partner_run_label",
        "donor_partner_run_id",
        "episode_group_id",
        "task_state_features",
        "task_features",
        "task_features_history_shuffled",
    }
    if not leakage_rows:
        raise ValueError("Identifiability artifact contains no task-leakage rows.")
    if any(
        not isinstance(row, Mapping) or set(row) != leakage_required
        for row in leakage_rows
    ):
        raise ValueError(
            f"Task-leakage raw rows differ from schema {MECHANISM_RAW_SCHEMA_VERSION}."
        )
    task_features = np.asarray(
        [row["task_features"] for row in leakage_rows], dtype=np.float64
    )
    task_state_features = np.asarray(
        [row["task_state_features"] for row in leakage_rows], dtype=np.float64
    )
    shuffled_features = np.asarray(
        [row["task_features_history_shuffled"] for row in leakage_rows],
        dtype=np.float64,
    )
    labels = np.asarray([row["partner_run_label"] for row in leakage_rows])
    probe_groups = np.asarray([row["episode_group_id"] for row in leakage_rows])
    class_count = int(np.unique(labels).size)
    if class_count < 2:
        raise ValueError("Task leakage control needs at least two held-out partner runs.")
    scientific_readout_allowed = bool(
        payload["registration"].get("scientific_readout_allowed", True)
    )
    probe_fold_count = 5
    if not scientific_readout_allowed:
        # The two-ego rehearsal has one small anchor per partner/role.  It
        # still performs a genuinely group-held-out probe, but it may not
        # impersonate the registered five-fold scientific estimator.
        minimum_groups = min(
            np.unique(probe_groups[labels == label]).size
            for label in np.unique(labels)
        )
        probe_fold_count = min(probe_fold_count, int(minimum_groups))
        if probe_fold_count < 2:
            raise ValueError(
                "Scientific dry-run leakage probe needs two independent groups per class."
            )
    leakage_accuracy = float(
        balanced_linear_probe_accuracy(
            task_features,
            labels,
            group_ids=probe_groups,
            fold_count=probe_fold_count,
        )
    )
    task_state_baseline_accuracy = float(
        balanced_linear_probe_accuracy(
            task_state_features,
            labels,
            group_ids=probe_groups,
            fold_count=probe_fold_count,
        )
    )
    excess_leakage_accuracy = leakage_accuracy - task_state_baseline_accuracy
    leakage_chance = 1.0 / class_count
    task_drift = float(
        task_representation_drift_under_shuffle(task_features, shuffled_features)
    )

    normal = _run_means(rows, "legal_history_return")
    shuffled = _run_means(rows, "history_shuffled_return")
    shuffle_drop, shuffle_low, shuffle_high = paired_bootstrap_drop(
        normal, shuffled, replications=9_999, seed=0
    )

    def stack(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=np.float64)

    swap_u = continuation_swap_causal_consistency(
        swapped_returns_by_action=stack("swap_u_returns_by_action"),
        original_returns_by_action=stack("swap_u_original_returns_by_action"),
        source_signature=stack("swap_u_source_signature"),
        target_signature=stack("swap_u_target_signature"),
    )
    swap_c_directional_descriptive = continuation_swap_causal_consistency(
        swapped_returns_by_action=stack("swap_c_returns_by_action"),
        original_returns_by_action=stack("swap_c_original_returns_by_action"),
        source_signature=stack("swap_c_source_signature"),
        target_signature=stack("swap_c_target_signature"),
    )
    original_policy = stack("original_policy_probabilities")
    swapped_c_policy = stack("swap_c_policy_probabilities")
    source_values = stack("swap_c_original_returns_by_action")
    context_sensitivity = float(
        np.mean(0.5 * np.sum(np.abs(original_policy - swapped_c_policy), axis=-1))
    )
    alignment_rows = np.sum(
        (original_policy - swapped_c_policy) * source_values, axis=-1
    )
    alignment_by_run = np.asarray(
        [
            np.mean(
                alignment_rows[
                    np.asarray([str(item["ego_run_id"]) == run for item in rows])
                ]
            )
            for run in sorted({str(item["ego_run_id"]) for item in rows})
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(77)
    alignment_draws = rng.integers(
        0,
        alignment_by_run.size,
        size=(9_999, alignment_by_run.size),
    )
    alignment_bootstrap = np.mean(alignment_by_run[alignment_draws], axis=-1)
    alignment_ci = [
        float(np.quantile(alignment_bootstrap, 0.005)),
        float(np.quantile(alignment_bootstrap, 0.995)),
    ]
    pass_flags = {
        "task_excess_leakage": bool(
            excess_leakage_accuracy <= LEAKAGE_PROBE_EXCESS_THRESHOLD
        ),
        "protocol_state_transplant": bool(shuffle_drop > 0.0 and shuffle_low > 0.0),
        "source_world_context_value": bool(alignment_ci[0] > 0.0),
    }
    result = {
        "version": IDENTIFIABILITY_SCHEMA_VERSION,
        "artifact_type": "depi_identifiability_evaluation",
        "method": METHOD_VERSION,
        "method_variant": payload["method_variant"],
        "layout": payload["layout"],
        "policy_manifest_sha256": payload["policy_manifest_sha256"],
        "partner_manifest_sha256": payload["partner_manifest_sha256"],
        "paired_crn": True,
        "scientific_readout_allowed": scientific_readout_allowed,
        "run_count": len({str(row["ego_run_id"]) for row in rows}),
        "row_count": len(rows),
        "task_leakage": {
            "row_count": len(leakage_rows),
            "group_held_out_fold_count": probe_fold_count,
            "representation_shuffle_drift": task_drift,
            "learned_representation_balanced_accuracy": leakage_accuracy,
            "task_state_baseline_balanced_accuracy": task_state_baseline_accuracy,
            "excess_balanced_accuracy": excess_leakage_accuracy,
            "chance_accuracy": leakage_chance,
            "maximum_excess_over_task_state": LEAKAGE_PROBE_EXCESS_THRESHOLD,
        },
        "protocol_state_transplant": {
            "mean_return_drop": shuffle_drop,
            "bootstrap_99_percent_ci": [shuffle_low, shuffle_high],
            "primary_unit": "ego_run",
        },
        "context_swap": {
            "swap_u_continuation_consistency": swap_u,
            "swap_c_donor_direction_consistency_descriptive": (
                swap_c_directional_descriptive
            ),
            "action_distribution_total_variation": context_sensitivity,
            "source_world_value_alignment": float(np.mean(alignment_by_run)),
            "source_world_value_alignment_bootstrap_99_percent_ci": alignment_ci,
            "measurement": "real_crn_all_action_continuation",
            "task_match_epsilon_quantile": SHUFFLE_EPSILON_QUANTILE,
        },
        "pass": {**pass_flags, "overall": bool(all(pass_flags.values()))},
        "source": {"path": str(source), "sha256": sha256_path(source)},
        "resource_ledger": payload["resource_ledger"],
    }
    if not live:
        ensure_run_identity(
            output,
            {
                "stage": "evaluate-identifiability",
                "method": METHOD_VERSION,
                "repository_runtime": runtime_provenance(),
                "source": result["source"],
            },
        )
    write_json(output / "identifiability_evaluation.json", result)


def _paired_run_bootstrap(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, *, seed: int
) -> Mapping[str, Any]:
    left_values = _run_means(rows, left)
    right_values = _run_means(rows, right)
    point, low, high = paired_bootstrap_drop(
        left_values, right_values, replications=9_999, seed=seed
    )
    return {"point_difference": point, "bootstrap_99_percent_ci": [low, high]}


def _recoverable_fraction(
    rows: Sequence[Mapping[str, Any]], *, threshold: float, seed: int = 4
) -> Mapping[str, Any]:
    run_ids = sorted({str(row["ego_run_id"]) for row in rows})
    numerator = np.asarray(
        [
            np.mean(
                [
                    float(row["g1"]) - float(row["g2"])
                    for row in rows
                    if str(row["ego_run_id"]) == run
                ]
            )
            for run in run_ids
        ],
        dtype=np.float64,
    )
    denominator = np.asarray(
        [
            np.mean(
                [
                    float(row["g4"]) - float(row["g2"])
                    for row in rows
                    if str(row["ego_run_id"]) == run
                ]
            )
            for run in run_ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(run_ids), size=(9_999, len(run_ids)))
    denominator_draws = np.mean(denominator[draws], axis=1)
    denominator_interval = [
        float(np.quantile(denominator_draws, 0.005)),
        float(np.quantile(denominator_draws, 0.995)),
    ]
    estimable = bool(
        len(run_ids) >= 2 and denominator_interval[0] >= float(threshold)
    )
    result: dict[str, Any] = {
        "definition": "run-level (G1-G2)/(G4-G2) when LCB(G4-G2) >= threshold",
        "signal_threshold": float(threshold),
        "denominator_point": float(np.mean(denominator)),
        "denominator_bootstrap_99_percent_ci": denominator_interval,
        "total_row_count": int(len(rows)),
        "signal_run_count": int(len(run_ids)),
        "primary_unit": "ego_run",
        "estimable": estimable,
        "point": None,
        "bootstrap_99_percent_ci": None,
    }
    if not estimable:
        return result
    ratio = np.mean(numerator) / np.mean(denominator)
    boot = np.mean(numerator[draws], axis=1) / np.maximum(
        np.mean(denominator[draws], axis=1), 1.0e-8
    )
    result["point"] = float(ratio)
    result["bootstrap_99_percent_ci"] = [
        float(np.quantile(boot, 0.005)),
        float(np.quantile(boot, 0.995)),
    ]
    return result


def run_recoverable_value_evaluation(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    raw_path, live = _resolve_raw_input(
        args, artifact="recoverable_value", output=output
    )
    source, payload = _load_raw(raw_path, "depi_recoverable_value_raw")
    required = _COMMON_ROW_FIELDS | {
        "g1", "g2", "g3", "g4", "g4_selection_stability"
    }
    rows = list(payload["rows"])
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != required:
            raise ValueError(
                f"Recoverable-value raw row fields differ from schema {MECHANISM_RAW_SCHEMA_VERSION}."
            )
        _validate_common_row(row)
        if not all(
            np.isfinite(float(row[name]))
            for name in ("g1", "g2", "g3", "g4", "g4_selection_stability")
        ):
            raise ValueError("G1--G4 returns must be finite.")
    run_means = {
        name: float(np.mean(_run_means(rows, name))) for name in ("g1", "g2", "g3", "g4")
    }
    comparisons = {
        "g1_minus_g2": _paired_run_bootstrap(rows, "g1", "g2", seed=1),
        "g1_minus_g3": _paired_run_bootstrap(rows, "g1", "g3", seed=2),
        "g4_minus_g1": _paired_run_bootstrap(rows, "g4", "g1", seed=3),
    }
    signal_threshold = float(payload["registration"]["recoverable_signal_threshold"])
    recoverable = _recoverable_fraction(rows, threshold=signal_threshold)
    pass_flags = {
        "legal_history_beats_shuffled": comparisons["g1_minus_g2"][
            "bootstrap_99_percent_ci"
        ][0]
        > 0.0,
        "legal_history_beats_state_only": comparisons["g1_minus_g3"][
            "bootstrap_99_percent_ci"
        ][0]
        > 0.0,
        "recoverable_fraction_estimable": bool(recoverable["estimable"]),
    }
    result = {
        "version": RECOVERABLE_VALUE_SCHEMA_VERSION,
        "artifact_type": "depi_recoverable_value_evaluation",
        "method": METHOD_VERSION,
        "method_variant": payload["method_variant"],
        "layout": payload["layout"],
        "policy_manifest_sha256": payload["policy_manifest_sha256"],
        "partner_manifest_sha256": payload["partner_manifest_sha256"],
        "conditions": {
            "g1": "legal_history",
            "g2": "shuffled_capability_and_protocol_history",
            "g3": "state_only_uniform_context",
            "g4": "fit_selected_cross_fitted_proxy_on_independent_evaluation_replicas",
        },
        "paired_crn": True,
        "scientific_readout_allowed": bool(
            payload["registration"].get("scientific_readout_allowed", True)
        ),
        "run_level_means": run_means,
        "comparisons": comparisons,
        "cross_fitted_proxy": {
            "mean_top_action_selection_stability": float(
                np.mean([row["g4_selection_stability"] for row in rows])
            ),
            "not_a_mathematical_upper_bound": True,
        },
        "recoverable_fraction": recoverable,
        "pass": {**pass_flags, "overall": bool(all(pass_flags.values()))},
        "source": {"path": str(source), "sha256": sha256_path(source)},
        "resource_ledger": payload["resource_ledger"],
    }
    if not live:
        ensure_run_identity(
            output,
            {
                "stage": "evaluate-recoverable-value",
                "method": METHOD_VERSION,
                "repository_runtime": runtime_provenance(),
                "source": result["source"],
            },
        )
    write_json(output / "recoverable_value_evaluation.json", result)


__all__ = [
    "IDENTIFIABILITY_SCHEMA_VERSION",
    "MECHANISM_RAW_SCHEMA_VERSION",
    "RECOVERABLE_VALUE_SCHEMA_VERSION",
    "collect_mechanism_raw_artifacts",
    "run_identifiability_evaluation",
    "run_recoverable_value_evaluation",
]
