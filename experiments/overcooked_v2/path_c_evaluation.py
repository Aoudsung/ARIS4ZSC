from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from experiments.overcooked_v2.path_c_audit_battery import FrozenAuditBatteryV1
from experiments.overcooked_v2.path_c_belief_audit import (
    recompute_instrument_measurement_from_evidence_v2,
)
from experiments.overcooked_v2.path_c_protocol import (
    baseline_selection_payload,
    estimate_audit_cost,
    estimate_locked_primary_endpoint,
    matched_policy_metrics,
    select_strongest_baseline_on_design,
)
from experiments.overcooked_v2.path_c_probe_contract import (
    normalized_probe_numeric_fields,
)
from experiments.overcooked_v2.path_c_return_artifacts import ReturnPointLedgerV1
from experiments.overcooked_v2.path_c_response_summary import ResponseSummarySpecV1
from experiments.overcooked_v2.path_c_seed import (
    canonical_uint64_seed,
    derive_ocv2_execution_seed,
)
from experiments.overcooked_v2.path_c_sequence import EgoEvidenceSpecV1
from experiments.overcooked_v2.path_c_split import SplitManifestV1


PATH_C_ARTIFACT_SCHEMA = "path_c_artifacts_v3"
PATH_C_MEASUREMENT_SCHEMA = "path_c_measurement_v3"
PATH_C_DECISION_SCHEMA = "path_c_decision_v3"
PATH_C_PRIMARY_VARIANT = "probing_ego"
PATH_C_PROBE_PROVENANCE_IDS = {
    "none": 0,
    "exploration": 1,
    "scripted": 2,
    "disabled": 3,
    "threshold": 4,
    "return_floor": 5,
    "selected": 6,
}
PATH_C_PROBE_RULE_IDS = {
    "none": 0,
    "max_normalized_advantage_disagreement": 1,
    "random_probe_valid": 2,
    "direct_information": 3,
}
PATH_C_SOFTWARE_CHECKS = (
    "snapshot_replay_determinism",
    "fork_rng_stream_isolation",
    "option_distribution_generation_inference_shared",
    "exact_mode_no_positive_mass_pruning",
    "response_vocabulary_frozen",
    "sequence_target_alignment",
    "episode_bootstrap_scope",
    "acting_policy_protocol_complete",
    "four_role_split_feasible",
    "artifact_semantic_hashes_bound",
    "static_cost_estimate_within_budget",
)
_BOUND_MEASUREMENT_ENVELOPE_KEYS = {
    "measurement_schema_version",
    "preregistration_sha256",
    "resolved_path_c_sha256",
    "semantic_bindings",
    "artifact_sha256",
}
_POLICY_METRIC_FIELDS = (
    "training_environment_steps",
    "gradient_updates",
    "evaluation_environment_step_limit",
    "evaluation_schedule_id",
    "probe_budget_grid_sha256",
    "probe_cost_per_use",
    "action_support_sha256",
    "trainable_parameters",
    "training_flops",
    "wall_clock_seconds",
    "inference_latency_ms",
)
PATH_C_LEAKAGE_CHANNELS = (
    "fingerprint",
    "identity",
    "seed",
    "layout_style",
    "surface_action_frequency",
    "trajectory_source",
)

_FORBIDDEN_RV_COLUMNS = {
    "identity",
    "seed",
    "style",
    "layout_style",
    "trajectory_source",
    "resp_rv_value_event_target_x",
    "resp_rv_value_event_target_y",
}
_ALLOWED_COLLECTION_ROLES = {
    "train",
    "design",
    "calibration",
    "locked_audit",
    "data_collection_only",
    "synthetic_power",
    "terminal_axis_null",
    "recovery",
}
_EXPECTED_VARIANT_INPUT_CONTRACTS = {
    "probing_ego": "ego_evidence_spec_v1",
    "base_only": "public_context_only",
    "full_history_rnn": "ego_evidence_spec_v1",
    "global_gru": "ego_evidence_spec_v1",
    "partner_id": "public_context_plus_partner_id_oracle",
    "exact_belief_filter": "ego_evidence_spec_v1",
    "learned_hmm_filter": "ego_evidence_spec_v1",
    "particle_belief_filter": "ego_evidence_spec_v1",
    "rnn_residualized": "ego_evidence_spec_v1",
    "random_probe": "ego_evidence_spec_v1",
    "no_probe": "ego_evidence_spec_v1",
    "direct_information": "synthetic_oracle_diagnostic_only",
    "no_admission": "public_context_only",
    "unpruned_ensemble": "public_context_plus_all_candidate_coordinates",
}

_REQUIRED_PREREGISTRATION_SECTIONS = (
    "method_contract_v2",
    "semantic_bindings",
    "semantic_sources",
    "evidence_spec",
    "ensemble",
    "probe",
    "response_summary_spec",
    "fingerprint",
    "synthetic_factorial",
    "split",
    "budget",
    "baselines",
    "primary_endpoint",
    "secondary_endpoints",
    "power_analysis",
    "belief_kernel_audit",
    "audit_battery",
    "artifact_contract",
    "decision",
)
_PREREGISTRATION_SECTION_KEYS = {
    "method_contract_v2": {
        "backbone",
        "adaptation",
        "partner_pool",
        "response_probe",
        "forbidden_inputs",
        "primary_evaluation_metric",
        "raw_reward_decomposition",
    },
    "semantic_bindings": {
        "code_commit",
        "resolved_config_sha256",
        "module_registry_sha256",
        "partner_registry_sha256",
        "option_policy_sha256",
        "evidence_spec_sha256",
        "response_vocabulary_sha256",
        "audit_battery_sha256",
        "split_manifest_sha256",
        "rng_key_schedule_version",
    },
    "semantic_sources": {
        "module_registry",
        "partner_registry",
        "option_policy",
        "split_manifest",
        "audit_battery",
    },
    "evidence_spec": {
        "schema_version",
        "enable",
        "observation_dim",
        "num_primitive_actions",
        "num_options",
        "max_primitive_actions_per_decision",
        "max_episode_decisions",
        "progress_event_dim",
        "observation_schema",
        "primitive_action_names",
        "option_names",
        "progress_event_names",
        "public_observation",
        "previous_decision_window_fields",
        "forbidden_fields",
        "shared_by",
        "sequence_training",
    },
    "ensemble": {
        "architecture",
        "n_heads",
        "bootstrap_p",
        "bootstrap_scope",
        "prior_scale",
        "prior_network",
        "recurrent_state",
        "dueling_advantage_center",
        "disagreement_stat",
        "disagreement_target",
        "tie_atol",
    },
    "probe": {
        "enable",
        "collection_enable",
        "locked_audit_adaptive_probe_enable",
        "rule",
        "disagreement_threshold",
        "return_floor",
        "record_candidate_mask",
        "record_all_scores",
        "record_propensity",
        "record_budget_and_cost",
        "random_probe_support_matched",
        "direct_information_baseline",
        "min_selected_probes",
        "min_probe_opportunities",
        "min_context_coverage",
        "min_action_coverage",
    },
    "response_summary_spec": {
        "schema_version",
        "enable",
        "selection_role",
        "primary_kind",
        "window_decisions",
        "latency_bin_upper_bounds",
        "response_classes",
        "structured_multilabel_role",
        "q_source",
        "structured_multi_label_readout",
        "forbidden_fields",
    },
    "fingerprint": {
        "enable",
        "kind",
        "vocab",
        "positive_control_kind",
        "negative_control_kind",
        "admission_required",
        "value_source",
        "mechanism_crossed",
    },
    "synthetic_factorial": {
        "true_value_class_source",
        "value_mechanisms",
        "independent_identity_style_realizations_per_mechanism",
        "fingerprints_per_cell",
        "near_complete_mechanism_fingerprint_cross",
        "overlap_posterior_registry",
        "admission_value_source",
    },
    "split": {
        "schema_version",
        "enable",
        "manifest_sha256",
        "roles",
        "deterministic",
        "mechanism_balanced",
        "group_disjoint_fields",
        "minimum_groups_per_mechanism",
        "preferred_groups_per_mechanism",
        "primary_shift",
        "secondary_shift",
        "role_internal_value_cross_fitting",
        "fail_closed_on_infeasible_role",
    },
    "budget": {
        "effective_episode_floor",
        "effective_transition_floor",
        "training_environment_steps",
        "training_gradient_updates",
        "evaluation_environment_step_limit",
        "evaluation_schedule_id",
        "probe_budget_grid_sha256",
        "action_support_sha256",
        "maximum_audit_primitive_steps",
        "split_unit",
        "matched_environment_steps",
        "matched_gradient_updates",
        "matched_evaluation_schedule",
        "matched_action_support",
        "matched_probe_cost",
        "report_training_flops",
        "report_wall_clock",
        "report_inference_latency",
        "required_policy_keys",
    },
    "baselines": {
        "acting_protocol_version",
        "deployable_required",
        "design_only_diagnostic",
        "oracle_only",
        "strongest_baseline_candidates",
        "selection_role",
        "selection_rule",
        "locked_selection_artifact_sha256",
    },
    "primary_endpoint": {
        "schema_version",
        "enable",
        "name",
        "probe_budget_grid",
        "normalization_rule",
        "normalization_lower",
        "normalization_upper",
        "probe_cost_per_use",
        "preregistered_margin",
        "confidence_level",
        "bootstrap_iterations",
        "cluster_unit",
        "shift",
        "layout_shift_endpoint",
        "selected_baseline_source",
    },
    "secondary_endpoints": {
        "representation_scope",
        "public_base_residual",
        "response_prediction",
        "ecological_value_class_source",
        "mechanism_label_role",
        "cluster_permutation",
        "leakage_channels",
        "thresholds",
    },
    "power_analysis": {
        "positive_and_null_cluster_simulation_required",
        "joint_operating_characteristics_required",
        "equivalence_test",
        "primary_alpha",
        "secondary_multiplicity",
        "selection_role",
        "cluster_unit",
        "clusters_per_trial",
        "simulation_repetitions",
        "bootstrap_iterations",
        "seed",
        "equivalence_margin",
        "minimum_positive_power",
        "minimum_null_equivalence_power",
        "minimum_joint_power",
    },
    "belief_kernel_audit": {
        "schema_version",
        "enable",
        "exact_mode",
        "tier1_hypothesis_prune",
        "posterior_bias_bound",
        "reset_bias_bound",
        "rng_key_schedule_version",
        "estimand",
        "naive_single_checkpoint_fork_admissible",
        "audit_unit",
        "inference_cluster_unit",
        "tv_convention",
        "recovery_condition",
        "outer_replicas_M",
        "simultaneous_cell_count",
        "confidence_delta",
        "inner_forks_L_inner",
        "probe_horizon_T_probe",
        "audit_information_states",
        "instrument_cell_registry_sha256",
        "response_vocabulary_source",
        "primary_kernel_table_granularity",
        "posterior_full_state_sampler",
        "tier1_exact_enumeration_replay",
        "tier1_approximate_mode",
        "tier2_hash_and_match",
        "rng",
        "validity_measurement_schema",
        "validity_checks",
    },
    "audit_battery": {
        "schema_version",
        "enable",
        "method_agnostic_core_required",
        "design_only_extension_frozen_before_locked_audit",
        "invalid_option_fallback_is_part_of_intervention",
        "cost_unit",
        "core_probe_count",
        "battery_sha256",
    },
    "artifact_contract": {
        "schema_version",
        "enable",
        "storage",
        "training_snapshots_default",
        "audit_harvest_separate",
        "required_hashes",
    },
    "decision": {
        "schema_version",
        "order",
        "instrument_valid",
        "primary_effective",
        "missing_primary_result",
        "validity_failure_claim",
        "type_b_review_required",
    },
}
_BELIEF_NESTED_MAPPING_KEYS = {
    "posterior_full_state_sampler": {
        "draw",
        "paired_frozen_probes",
        "rao_blackwell_mixture",
    },
    "tier1_exact_enumeration_replay": {
        "enable",
        "partner_registry_frozen_required",
        "prior_over_theta",
        "likelihood",
        "generation_inference_policy_function",
        "execution_state_resample",
        "tier1_hypothesis_prune",
        "rho_prune",
        "posterior_bias_bound",
        "reset_bias_bound",
        "brute_force_differential_fixture_required",
    },
    "tier1_approximate_mode": {
        "enable",
        "report_discarded_posterior_mass_as_rho_prune",
        "bound_required",
    },
    "tier2_hash_and_match": {
        "enable",
        "group_key",
        "hash",
        "maximum_outer_draws_per_episode_and_key",
        "recurrence_and_positivity_required",
        "support_bias_registration_required",
        "claim_scope",
    },
    "rng": {
        "schedule_version",
        "replay_uses_original_keys",
        "fork_uses_fresh_derived_keys",
        "named_streams",
    },
    "validity_measurement_schema": {
        "required_fields",
        "instrument_valid_rule",
    },
}
_SEMANTIC_SOURCE_NAMES = (
    "module_registry",
    "partner_registry",
    "option_policy",
    "split_manifest",
    "audit_battery",
)
_REQUIRED_THRESHOLDS = (
    "response_advantage",
    "value_advantage",
    "transfer_advantage",
    "conditional_leakage_max",
    "null_equivalence_gain",
    "synthetic_power_min_advantage",
    "value_ablation_min_degradation",
    "nuisance_value_gain_equivalence",
    "kernel_confidence",
    "kernel_min_outer_replicas",
)
_RUNTIME_CONTRACT_FIELDS = {
    "evidence_spec": (
        "enable",
        "schema_version",
        "max_primitive_actions_per_decision",
        "max_episode_decisions",
    ),
    "ensemble": (
        "architecture",
        "n_heads",
        "bootstrap_p",
        "bootstrap_scope",
        "prior_scale",
        "prior_network",
        "recurrent_state",
        "dueling_advantage_center",
        "disagreement_stat",
        "disagreement_target",
        "tie_atol",
    ),
    "probe": (
        "enable",
        "collection_enable",
        "locked_audit_adaptive_probe_enable",
        "rule",
        "disagreement_threshold",
        "return_floor",
        "record_candidate_mask",
        "record_all_scores",
        "record_propensity",
        "record_budget_and_cost",
        "random_probe_support_matched",
        "direct_information_baseline",
        "min_selected_probes",
        "min_probe_opportunities",
        "min_context_coverage",
        "min_action_coverage",
    ),
    "response_summary_spec": ("enable", "schema_version"),
    "fingerprint": (
        "enable",
        "kind",
        "vocab",
        "positive_control_kind",
        "negative_control_kind",
    ),
    "split": (
        "enable",
        "schema_version",
        "roles",
        "minimum_groups_per_mechanism",
        "preferred_groups_per_mechanism",
        "primary_shift",
        "secondary_shift",
    ),
    "budget": (
        "training_environment_steps",
        "training_gradient_updates",
        "evaluation_environment_step_limit",
        "evaluation_schedule_id",
        "probe_budget_grid_sha256",
        "action_support_sha256",
    ),
    "primary_endpoint": (
        "enable",
        "schema_version",
    ),
    "belief_kernel_audit": (
        "enable",
        "schema_version",
        "outer_replicas_M",
        "simultaneous_cell_count",
        "confidence_delta",
        "inner_forks_L_inner",
        "probe_horizon_T_probe",
        "exact_mode",
        "tier1_hypothesis_prune",
        "posterior_bias_bound",
        "reset_bias_bound",
        "rng_key_schedule_version",
    ),
    "audit_battery": ("enable", "schema_version"),
    "artifact_contract": ("enable", "schema_version"),
}


@dataclass(frozen=True)
class FrozenPathCPreregistration:
    path: Path
    sha256: str
    payload: dict[str, Any]
    thresholds: dict[str, Any]
    required_baselines: tuple[str, ...]
    strong_baselines: tuple[str, ...]
    runtime_contract: dict[str, Any]
    runtime_contract_sha256: str
    semantic_bindings: dict[str, Any]
    primary_endpoint: dict[str, Any]
    evidence_spec: EgoEvidenceSpecV1
    response_summary_spec: ResponseSummarySpecV1
    response_vocabulary_sha256: str
    evidence_spec_sha256: str
    split_manifest: SplitManifestV1
    audit_battery: FrozenAuditBatteryV1
    module_registry_test_ids: tuple[str, ...]


@dataclass(frozen=True)
class PathCArtifactManifest:
    path: Path
    sha256: str
    payload: dict[str, Any]
    preregistration_sha256: str
    resolved_path_c_sha256: str
    active_stage: str
    datasets: dict[str, dict[str, Any]]
    checkpoints: dict[str, dict[str, Any]]
    measurement_artifacts: dict[str, dict[str, Any]]
    value_control_artifact: dict[str, Any]
    semantic_bindings: dict[str, Any] | None = None
    dataset_shards: dict[str, dict[str, Any]] | None = None
    policy_artifacts: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class PathCDecision:
    schema_version: str
    software_conformant: bool
    instrument_valid: bool
    primary_effective: bool | None
    primary_estimate: float | None
    primary_ci: tuple[float, float] | None
    secondary: Mapping[str, Any]
    allowed_claim: str
    status: str
    requires_type_b_review: bool = True


@dataclass(frozen=True)
class ValidatedPathCInputs:
    preregistration: FrozenPathCPreregistration
    manifest: PathCArtifactManifest
    measurements: dict[str, dict[str, Any]]


def assemble_path_c_measurements(inputs: ValidatedPathCInputs) -> dict[str, Any]:
    """Return only hash-validated version-3 measurement objects."""

    required = {
        "software_conformance",
        "software_test_report",
        "policy_metrics",
        "audit_cost_estimate",
    }
    missing = sorted(required.difference(inputs.measurements))
    if missing:
        raise ValueError(
            "Path C version-3 inputs are missing measurement artifact(s): "
            + ", ".join(missing)
        )
    return {
        **{name: dict(value) for name, value in inputs.measurements.items()},
        "artifact_contract": artifact_contract_measurement(inputs),
    }


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, trimmed string.")
    return value


def _validate_preregistration_shape(payload: Mapping[str, Any]) -> None:
    """Reject every missing or unregistered version-3 schema key."""

    sections: dict[str, dict[str, Any]] = {}
    for section_name, expected_keys in _PREREGISTRATION_SECTION_KEYS.items():
        section = _mapping(payload[section_name], section_name)
        _require_exact_mapping_keys(section, expected_keys, section_name)
        sections[section_name] = section

    method_contract = sections["method_contract_v2"]
    method_nested_keys = {
        "backbone": {
            "algorithm",
            "policy_action_selection",
            "environment_steps",
            "shaping_horizon_environment_steps",
            "pool_snapshot_environment_steps",
            "budget_reporting",
            "partner_pool_training_seeds",
            "active_snapshots_per_training_seed",
        },
        "adaptation": {
            "environment_steps",
            "policy_action_selection",
            "trunk_gradient_source",
            "actor_input_gradient_boundary",
            "response_model_input_gradient_boundary",
            "shaping_horizon_environment_steps",
            "response_prefit_environment_steps",
            "response_prefit_updates",
            "response_prefit_scientific_readout_allowed",
        },
        "partner_pool": {
            "minimum_independent_training_seeds",
            "active_snapshots_per_training_seed",
            "partner_head_assignment",
        },
        "response_probe": {
            "controller",
            "partner_belief_initialization",
            "partner_belief_update",
            "information_statistic",
            "calibration_episodes",
            "calibration_probe_execution",
            "threshold_quantile",
            "threshold_candidate_rule",
            "no_op_probe_rule",
        },
        "raw_reward_decomposition": {
            "constants",
            "pure_event_fast_path",
            "correct_delivery_count_interpretation",
            "repeated_indicator_example",
        },
    }
    for mapping_name, expected_keys in method_nested_keys.items():
        value = _mapping(
            method_contract[mapping_name],
            f"method_contract_v2.{mapping_name}",
        )
        _require_exact_mapping_keys(
            value,
            expected_keys,
            f"method_contract_v2.{mapping_name}",
        )
    reward_constants = _mapping(
        method_contract["raw_reward_decomposition"]["constants"],
        "method_contract_v2.raw_reward_decomposition.constants",
    )
    _require_exact_mapping_keys(
        reward_constants,
        {"correct_delivery", "wrong_delivery", "indicator_activation"},
        "method_contract_v2.raw_reward_decomposition.constants",
    )
    backbone_contract = _mapping(
        method_contract["backbone"], "method_contract_v2.backbone"
    )
    if int(backbone_contract["environment_steps"]) != 30_000_000:
        raise ValueError("The backbone budget must remain 30,000,000 environment steps.")
    if int(backbone_contract["shaping_horizon_environment_steps"]) != 15_000_000:
        raise ValueError("The backbone shaping horizon must remain 15,000,000 steps.")
    if tuple(map(int, backbone_contract["pool_snapshot_environment_steps"])) != (
        7_500_000,
        15_000_000,
        22_500_000,
        30_000_000,
    ):
        raise ValueError("The backbone partner-snapshot schedule changed.")
    expected_backbone_labels = {
        "algorithm": "recurrent_parameter_shared_ippo",
        "policy_action_selection": "stochastic_temperature_1",
        "budget_reporting": "separate_pool_formation_cost",
        "partner_pool_training_seeds": "at_least_two_independent_runs",
    }
    if any(
        backbone_contract[key] != value
        for key, value in expected_backbone_labels.items()
    ) or int(backbone_contract["active_snapshots_per_training_seed"]) != 1:
        raise ValueError("The backbone formation contract changed.")
    adaptation_contract = _mapping(
        method_contract["adaptation"], "method_contract_v2.adaptation"
    )
    if int(adaptation_contract["environment_steps"]) != 10_000_000:
        raise ValueError("The adaptation budget must remain 10,000,000 environment steps.")
    if int(adaptation_contract["response_prefit_environment_steps"]) != 1_000_000:
        raise ValueError("Response-only prefit must remain 1,000,000 environment steps.")
    if adaptation_contract["response_prefit_scientific_readout_allowed"] is not False:
        raise ValueError("Response-only prefit cannot produce a scientific readout.")
    expected_adaptation_labels = {
        "policy_action_selection": "stochastic_temperature_1",
        "trunk_gradient_source": "critic_td_only",
        "actor_input_gradient_boundary": "detached_trunk_features",
        "response_model_input_gradient_boundary": "detached_trunk_features",
        "response_prefit_updates": "response_heads_only",
    }
    if any(
        adaptation_contract[key] != value
        for key, value in expected_adaptation_labels.items()
    ) or int(adaptation_contract["shaping_horizon_environment_steps"]) != 5_000_000:
        raise ValueError("The adaptation gradient or reward contract changed.")
    partner_pool_contract = _mapping(
        method_contract["partner_pool"], "method_contract_v2.partner_pool"
    )
    if int(partner_pool_contract["minimum_independent_training_seeds"]) < 2:
        raise ValueError("The active partner pool requires at least two independent seeds.")
    if int(partner_pool_contract["active_snapshots_per_training_seed"]) != 1:
        raise ValueError("The active partner pool must select one snapshot per seed.")
    if partner_pool_contract["partner_head_assignment"] != "one_head_per_active_partner":
        raise ValueError("Each active partner must retain its own response head.")
    response_probe_contract = _mapping(
        method_contract["response_probe"], "method_contract_v2.response_probe"
    )
    if int(response_probe_contract["calibration_episodes"]) != 500:
        raise ValueError("Response-probe calibration must contain 500 episodes.")
    if response_probe_contract["calibration_probe_execution"] is not False:
        raise ValueError("Response-probe calibration must not execute probes.")
    threshold_quantile = _finite_number(
        response_probe_contract["threshold_quantile"],
        "method_contract_v2.response_probe.threshold_quantile",
    )
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("The response-probe threshold quantile must lie in (0, 1).")
    expected_probe_labels = {
        "controller": "response_voi",
        "partner_belief_initialization": "uniform",
        "partner_belief_update": "observable_response_likelihood",
        "information_statistic": "weighted_generalized_jensen_shannon_divergence",
        "threshold_candidate_rule": "safe_non_greedy_action_changing",
        "no_op_probe_rule": "candidate_must_differ_from_sampled_actor_action",
    }
    if any(
        response_probe_contract[key] != value
        for key, value in expected_probe_labels.items()
    ):
        raise ValueError("The partner-response probe contract changed.")
    if method_contract["primary_evaluation_metric"] != "raw_episode_return":
        raise ValueError("The primary evaluation metric must remain raw episode return.")
    forbidden_inputs = set(
        map(
            str,
            _sequence(
                method_contract["forbidden_inputs"],
                "method_contract_v2.forbidden_inputs",
            ),
        )
    )
    if forbidden_inputs != {
        "partner_identity",
        "mechanism_label",
        "style_label",
        "privileged_global_state",
    }:
        raise ValueError("The method contract changed its forbidden-input set.")
    if {
        "correct_delivery": int(reward_constants["correct_delivery"]),
        "wrong_delivery": int(reward_constants["wrong_delivery"]),
        "indicator_activation": int(reward_constants["indicator_activation"]),
    } != {
        "correct_delivery": 20,
        "wrong_delivery": -20,
        "indicator_activation": -5,
    }:
        raise ValueError("The raw-reward decomposition constants changed.")
    reward_contract = method_contract["raw_reward_decomposition"]
    if reward_contract["pure_event_fast_path"] is not True or (
        reward_contract["correct_delivery_count_interpretation"]
        != "observable_lower_bound_when_mixed_totals_are_ambiguous"
    ):
        raise ValueError("The raw-reward event interpretation changed.")

    semantic_sources = sections["semantic_sources"]
    for source_name in _SEMANTIC_SOURCE_NAMES:
        reference = _mapping(
            semantic_sources[source_name],
            f"semantic_sources.{source_name}",
        )
        _require_exact_mapping_keys(
            reference,
            {"path", "sha256"},
            f"semantic_sources.{source_name}",
        )

    thresholds = _mapping(
        sections["secondary_endpoints"]["thresholds"],
        "secondary_endpoints.thresholds",
    )
    _require_exact_mapping_keys(
        thresholds,
        set(_REQUIRED_THRESHOLDS),
        "secondary_endpoints.thresholds",
    )

    belief = sections["belief_kernel_audit"]
    nested: dict[str, dict[str, Any]] = {}
    for mapping_name, expected_keys in _BELIEF_NESTED_MAPPING_KEYS.items():
        value = _mapping(
            belief[mapping_name],
            f"belief_kernel_audit.{mapping_name}",
        )
        _require_exact_mapping_keys(
            value,
            expected_keys,
            f"belief_kernel_audit.{mapping_name}",
        )
        nested[mapping_name] = value

    prior = _mapping(
        nested["tier1_exact_enumeration_replay"]["prior_over_theta"],
        "belief_kernel_audit.tier1_exact_enumeration_replay.prior_over_theta",
    )
    _require_exact_mapping_keys(
        prior,
        {"schema_version", "partner_registry_sha256", "support"},
        "belief_kernel_audit.tier1_exact_enumeration_replay.prior_over_theta",
    )
    support = _sequence(
        prior["support"],
        "belief_kernel_audit.tier1_exact_enumeration_replay."
        "prior_over_theta.support",
    )
    for index, raw_item in enumerate(support):
        item = _mapping(
            raw_item,
            "belief_kernel_audit.tier1_exact_enumeration_replay."
            f"prior_over_theta.support[{index}]",
        )
        _require_exact_mapping_keys(
            item,
            {"theta_id", "probability"},
            "belief_kernel_audit.tier1_exact_enumeration_replay."
            f"prior_over_theta.support[{index}]",
        )


def load_frozen_preregistration(path: str | Path) -> FrozenPathCPreregistration:
    """Load only the strict version-3 schema; version-2 artifacts need migration."""

    source = Path(path).resolve()
    payload = _load_mapping(source, "Path C preregistration")
    expected_top_level = {
        "schema_version",
        "version",
        "measurement_schema_version",
        "status",
        "freeze_timestamp",
        *_REQUIRED_PREREGISTRATION_SECTIONS,
    }
    _require_exact_mapping_keys(payload, expected_top_level, "preregistration")
    if payload["schema_version"] != "path_c_preregistration_schema_v3":
        raise ValueError(
            "Legacy Path C preregistration requires an explicit version-3 migration."
        )
    if payload["measurement_schema_version"] != PATH_C_MEASUREMENT_SCHEMA:
        raise ValueError(
            f"measurement_schema_version must be {PATH_C_MEASUREMENT_SCHEMA!r}."
        )
    if str(payload.get("status", "")).strip().lower() != "frozen":
        raise ValueError("Path C preregistration status must be exactly 'frozen'.")
    if payload.get("freeze_timestamp") in {None, ""}:
        raise ValueError("A frozen Path C preregistration needs freeze_timestamp.")
    _reject_legacy_margin_fields(payload)
    _validate_preregistration_shape(payload)

    semantic_bindings = _mapping(payload["semantic_bindings"], "semantic_bindings")
    required_bindings = _PREREGISTRATION_SECTION_KEYS["semantic_bindings"]
    for key in sorted(required_bindings.difference({"code_commit", "rng_key_schedule_version"})):
        if not _is_sha256(semantic_bindings[key]):
            raise ValueError(f"semantic_bindings.{key} must be a SHA-256 digest.")
    code_commit = semantic_bindings["code_commit"]
    if not _is_git_commit_sha(code_commit):
        raise ValueError(
            "semantic_bindings.code_commit must be a forty-character lower-case "
            "Git commit SHA."
        )
    if semantic_bindings["rng_key_schedule_version"] != "path_c_rng_key_schedule_v1":
        raise ValueError("Unknown Path C RNG key-schedule version.")

    semantic_sources = _mapping(payload["semantic_sources"], "semantic_sources")
    semantic_source_paths: dict[str, Path] = {}
    for source_name in _SEMANTIC_SOURCE_NAMES:
        reference = _mapping(
            semantic_sources[source_name],
            f"semantic_sources.{source_name}",
        )
        source_path = _validated_file_reference(
            reference,
            source.parent,
            f"semantic_sources.{source_name}",
        )
        binding_key = f"{source_name}_sha256"
        if (
            source_name not in {"split_manifest", "audit_battery"}
            and reference["sha256"] != semantic_bindings[binding_key]
        ):
            raise ValueError(
                f"semantic_sources.{source_name}.sha256 differs from "
                f"semantic_bindings.{binding_key}."
            )
        semantic_source_paths[source_name] = source_path

    module_registry = _load_mapping(
        semantic_source_paths["module_registry"],
        "Path C module registry semantic source",
    )
    if module_registry.get("source_git_sha") != code_commit:
        raise ValueError(
            "The module registry source_git_sha must equal "
            "semantic_bindings.code_commit."
        )
    if module_registry.get("schema_version") != "path_c_module_registry_v1":
        raise ValueError("The module registry has an unsupported schema version.")
    if module_registry.get("test_execution_status") != "passed":
        raise ValueError(
            "A frozen preregistration requires an archived passing module-registry test run."
        )
    registry_modules = _sequence(
        module_registry.get("modules"), "module_registry.modules"
    )
    if not registry_modules:
        raise ValueError("The module registry must contain registered modules.")
    module_registry_test_ids: list[str] = []
    for index, raw_module in enumerate(registry_modules):
        module = _mapping(raw_module, f"module_registry.modules[{index}]")
        if module.get("status") not in {"tested", "frozen"}:
            raise ValueError(
                "Every module must be tested before a preregistration can be frozen."
            )
        test_ids = _sequence(
            module.get("test_ids"), f"module_registry.modules[{index}].test_ids"
        )
        if not test_ids or any(not str(value).strip() for value in test_ids):
            raise ValueError("Every tested module must bind non-empty test identifiers.")
        module_registry_test_ids.extend(str(value) for value in test_ids)

    split_manifest_payload = _load_mapping(
        semantic_source_paths["split_manifest"],
        "Path C split manifest semantic source",
    )
    frozen_split_manifest = SplitManifestV1.from_mapping(split_manifest_payload)
    if frozen_split_manifest.sha256 != semantic_bindings["split_manifest_sha256"]:
        raise ValueError("The parsed split manifest does not match its semantic binding.")

    audit_battery_payload = _load_mapping(
        semantic_source_paths["audit_battery"],
        "Path C audit battery semantic source",
    )
    frozen_audit_battery = FrozenAuditBatteryV1.from_manifest(audit_battery_payload)
    if frozen_audit_battery.sha256 != semantic_bindings["audit_battery_sha256"]:
        raise ValueError("The parsed audit battery does not match its semantic binding.")

    evidence_spec = _mapping(payload["evidence_spec"], "evidence_spec")
    if evidence_spec.get("enable") is not True:
        raise ValueError("Frozen Path C evidence_spec.enable must be true.")
    forbidden = set(map(str, evidence_spec.get("forbidden_fields", ())))
    if not {"identity", "mechanism", "style", "seed"}.issubset(forbidden):
        raise ValueError("EgoEvidenceSpecV1 does not reject every nuisance label.")
    expected_window_fields = (
        "ego_primitive_actions",
        "partner_primitive_actions",
        "ego_option",
        "option_duration",
        "reward",
        "progress_events",
        "valid_action_mask",
        "terminated",
        "truncated",
    )
    if evidence_spec.get("public_observation") != (
        "current_raw_or_public_observation"
    ):
        raise ValueError("EgoEvidenceSpecV1 must use the current observable state.")
    if tuple(map(str, evidence_spec.get("previous_decision_window_fields", ()))) != (
        expected_window_fields
    ):
        raise ValueError("EgoEvidenceSpecV1 decision-window fields are not version 1.")
    if evidence_spec.get("sequence_training") != "full_episode":
        raise ValueError("EgoEvidenceSpecV1 must be trained as a full episode sequence.")
    if int(evidence_spec.get("max_episode_decisions", 0)) <= 0:
        raise ValueError("EgoEvidenceSpecV1 max_episode_decisions must be positive.")
    frozen_evidence = EgoEvidenceSpecV1.from_dict({
        "schema_version": evidence_spec.get("schema_version"),
        "observation_dim": evidence_spec.get("observation_dim"),
        "num_primitive_actions": evidence_spec.get("num_primitive_actions"),
        "num_options": evidence_spec.get("num_options"),
        "max_primitive_steps_per_decision": evidence_spec.get(
            "max_primitive_actions_per_decision"
        ),
        "progress_event_dim": evidence_spec.get("progress_event_dim"),
        "observation_schema": evidence_spec.get("observation_schema"),
        "primitive_action_names": evidence_spec.get("primitive_action_names"),
        "option_names": evidence_spec.get("option_names"),
        "progress_event_names": evidence_spec.get("progress_event_names"),
    })
    evidence_spec_sha256 = frozen_evidence.sha256()
    if semantic_bindings["evidence_spec_sha256"] != evidence_spec_sha256:
        raise ValueError("Frozen evidence_spec SHA-256 binding does not match its payload.")

    response_payload = _mapping(
        payload["response_summary_spec"], "response_summary_spec"
    )
    response_spec = ResponseSummarySpecV1.from_mapping({
        "schema_version": response_payload.get("schema_version"),
        "response_classes": response_payload.get("response_classes"),
        "latency_bin_upper_bounds": response_payload.get("latency_bin_upper_bounds"),
        "structured_multilabel_role": response_payload.get(
            "structured_multilabel_role"
        ),
    })
    if response_payload.get("enable") is not True:
        raise ValueError("Frozen response_summary_spec.enable must be true.")
    if response_payload.get("selection_role") != "design":
        raise ValueError("The response summary must be selected on design.")
    if response_payload.get("primary_kind") != "canonical_finite_token":
        raise ValueError("The primary response summary must be a canonical finite token.")
    if int(response_payload.get("window_decisions", 0)) <= 0:
        raise ValueError("The response summary needs a positive frozen decision window.")
    if response_payload.get("structured_multi_label_readout") != "secondary_only":
        raise ValueError("Structured multi-label responses must remain secondary-only.")
    response_forbidden = set(map(str, response_payload.get("forbidden_fields", ())))
    if not {
        "identity", "seed", "style", "layout_style", "trajectory_source"
    }.issubset(response_forbidden):
        raise ValueError("The response vocabulary does not exclude nuisance labels.")
    if semantic_bindings["response_vocabulary_sha256"] != response_spec.sha256:
        raise ValueError("Frozen response vocabulary SHA-256 binding is inconsistent.")

    ensemble = _mapping(payload["ensemble"], "ensemble")
    if ensemble.get("architecture") != "recurrent_sequence_v1":
        raise ValueError("Frozen Path C must use recurrent_sequence_v1.")
    if int(ensemble.get("n_heads", 0)) <= 1:
        raise ValueError("Frozen Path C ensemble.n_heads must exceed one.")
    if not 0.0 < float(ensemble.get("bootstrap_p", 0.0)) < 1.0:
        raise ValueError("Frozen ensemble.bootstrap_p must be in (0, 1).")
    if ensemble.get("bootstrap_scope") != "episode":
        raise ValueError("Frozen ensemble bootstrap_scope must be episode.")
    if float(ensemble.get("prior_scale", 0.0)) <= 0.0:
        raise ValueError("Frozen ensemble.prior_scale must be positive.")
    if ensemble.get("dueling_advantage_center") != "valid_actions_only":
        raise ValueError("Frozen dueling heads must center over valid actions only.")
    if ensemble.get("disagreement_target") != "normalized_advantage":
        raise ValueError("Frozen probe disagreement must target normalized advantage.")

    probe = _mapping(payload["probe"], "probe")
    if probe.get("enable") is not True or probe.get("collection_enable") is not True:
        raise ValueError("Frozen Path C adaptive probes must be enabled for collection.")
    if probe.get("locked_audit_adaptive_probe_enable") is not False:
        raise ValueError("Adaptive probes must remain disabled in locked audit.")
    if probe.get("rule") != "max_normalized_advantage_disagreement":
        raise ValueError("Frozen probe rule has the wrong primary target.")
    normalized_probe_numeric_fields(probe, prefix="probe")
    for key in (
        "record_candidate_mask",
        "record_all_scores",
        "record_propensity",
        "record_budget_and_cost",
        "random_probe_support_matched",
    ):
        if probe.get(key) is not True:
            raise ValueError(f"Frozen probe.{key} must be true.")

    split = _mapping(payload["split"], "split")
    if tuple(map(str, split.get("roles", ()))) != (
        "train", "design", "calibration", "locked_audit"
    ):
        raise ValueError("Frozen split must contain the four roles in canonical order.")
    if int(split.get("minimum_groups_per_mechanism", 0)) < 4:
        raise ValueError("Each mechanism needs at least four independent groups.")
    if split.get("primary_shift") != "identity" or split.get("secondary_shift") != "layout":
        raise ValueError("Primary identity shift and secondary layout shift must remain separate.")
    if split.get("manifest_sha256") != semantic_bindings["split_manifest_sha256"]:
        raise ValueError("Frozen split manifest SHA-256 binding is inconsistent.")
    if frozen_split_manifest.sha256 != split.get("manifest_sha256"):
        raise ValueError("Frozen split section does not bind the parsed split manifest.")
    if any(
        count < int(split["minimum_groups_per_mechanism"])
        for count in frozen_split_manifest.independent_group_count_by_mechanism.values()
    ):
        raise ValueError("Parsed split manifest is below the frozen group-count floor.")

    synthetic = _mapping(payload["synthetic_factorial"], "synthetic_factorial")
    if int(synthetic.get("value_mechanisms", 0)) < 3:
        raise ValueError("Synthetic factorial needs at least three value mechanisms.")
    if int(synthetic.get("independent_identity_style_realizations_per_mechanism", 0)) < 4:
        raise ValueError("Synthetic factorial has too few identity/style realizations.")

    budget = _mapping(payload["budget"], "budget")
    if int(budget.get("effective_episode_floor", 0)) < 2000:
        raise ValueError("Path C claim budget must contain at least 2000 episodes.")
    for key in (
        "effective_transition_floor",
        "training_environment_steps",
        "training_gradient_updates",
        "evaluation_environment_step_limit",
        "maximum_audit_primitive_steps",
    ):
        value = _optional_int(budget.get(key))
        if value is None or value <= 0:
            raise ValueError(f"Frozen budget.{key} must be a positive integer.")
    if not str(budget.get("evaluation_schedule_id", "")).strip():
        raise ValueError("Frozen budget.evaluation_schedule_id must be non-empty.")
    for key in ("probe_budget_grid_sha256", "action_support_sha256"):
        if not _is_sha256(budget.get(key)):
            raise ValueError(f"Frozen budget.{key} must be a SHA-256 digest.")
    for key in (
        "matched_environment_steps",
        "matched_gradient_updates",
        "matched_evaluation_schedule",
        "matched_action_support",
        "matched_probe_cost",
    ):
        if budget.get(key) is not True:
            raise ValueError(f"Frozen budget.{key} must be true.")
    required_policy_keys = _string_tuple(
        budget.get("required_policy_keys"),
        "budget.required_policy_keys",
    )

    baselines = _mapping(payload["baselines"], "baselines")
    required_baselines = _string_tuple(
        baselines.get("deployable_required"), "baselines.deployable_required"
    )
    strong_baselines = _string_tuple(
        baselines.get("strongest_baseline_candidates"),
        "baselines.strongest_baseline_candidates",
    )
    if not required_baselines or not strong_baselines:
        raise ValueError("Frozen deployable and strongest baseline lists must be non-empty.")
    if not set(strong_baselines).issubset(required_baselines):
        raise ValueError("Strong baseline candidates must be deployable baselines.")
    expected_required_policy_keys = {
        PATH_C_PRIMARY_VARIANT,
        *required_baselines,
        "direct_information",
    }
    if set(required_policy_keys) != expected_required_policy_keys:
        missing = sorted(expected_required_policy_keys.difference(required_policy_keys))
        extra = sorted(set(required_policy_keys).difference(expected_required_policy_keys))
        raise ValueError(
            "budget.required_policy_keys must exactly equal probing_ego plus every "
            "deployable baseline plus direct_information; "
            f"missing={missing}, extra={extra}."
        )
    if baselines.get("selection_role") != "design":
        raise ValueError("Strongest baseline must be selected on design.")
    selection_hash = baselines.get("locked_selection_artifact_sha256")
    if not _is_sha256(selection_hash):
        raise ValueError("Frozen baseline selection artifact SHA-256 is required.")

    primary = _mapping(payload["primary_endpoint"], "primary_endpoint")
    if primary.get("schema_version") != "path_c_primary_endpoint_v1":
        raise ValueError("Frozen primary endpoint has the wrong schema version.")
    if primary.get("enable") is not True:
        raise ValueError("Frozen primary endpoint must be enabled.")
    if primary.get("name") != (
        "cross_identity_normalized_net_return_probe_budget_auc_difference"
    ):
        raise ValueError("Frozen primary endpoint has the wrong name.")
    if primary.get("normalization_rule") != "affine_without_clipping":
        raise ValueError("Frozen primary endpoint has an unknown normalization rule.")
    for key in (
        "normalization_lower",
        "normalization_upper",
        "probe_cost_per_use",
        "preregistered_margin",
    ):
        _finite_number(primary.get(key), f"primary_endpoint.{key}")
    if float(primary["normalization_upper"]) <= float(primary["normalization_lower"]):
        raise ValueError("Primary return normalization upper bound must exceed lower.")
    if float(primary["probe_cost_per_use"]) < 0.0:
        raise ValueError("Primary probe_cost_per_use must be non-negative.")
    if float(primary["preregistered_margin"]) < 0.0:
        raise ValueError("Primary preregistered_margin must be non-negative.")
    budgets = tuple(int(value) for value in primary.get("probe_budget_grid", ()))
    if not budgets or budgets[0] != 0 or any(
        right <= left for left, right in zip(budgets, budgets[1:], strict=False)
    ):
        raise ValueError("Primary probe-budget grid must start at zero and increase.")
    if budget["probe_budget_grid_sha256"] != canonical_sha256(list(budgets)):
        raise ValueError("Frozen probe-budget grid SHA-256 does not match the grid.")
    if primary.get("shift") != "identity_only":
        raise ValueError("The primary endpoint may vary identity only.")
    confidence_level = _finite_number(
        primary.get("confidence_level"), "primary_endpoint.confidence_level"
    )
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Primary confidence_level must lie in (0, 1).")
    if _optional_int(primary.get("bootstrap_iterations")) is None or int(
        primary["bootstrap_iterations"]
    ) <= 0:
        raise ValueError("Primary bootstrap_iterations must be positive.")
    if primary.get("cluster_unit") != "identity_group":
        raise ValueError("Primary inference must cluster by identity_group.")
    if primary.get("layout_shift_endpoint") != "secondary":
        raise ValueError("Layout shift cannot replace the primary identity shift.")
    if primary.get("selected_baseline_source") != (
        "frozen_design_selection_artifact"
    ):
        raise ValueError("Primary baseline must come from the frozen design artifact.")

    secondary = _mapping(payload["secondary_endpoints"], "secondary_endpoints")
    thresholds = _mapping(secondary.get("thresholds"), "secondary_endpoints.thresholds")
    missing_thresholds = [key for key in _REQUIRED_THRESHOLDS if key not in thresholds]
    if missing_thresholds:
        raise ValueError("Missing secondary threshold(s): " + ", ".join(missing_thresholds))
    for key in _REQUIRED_THRESHOLDS:
        _finite_number(thresholds[key], f"secondary_endpoints.thresholds.{key}")

    power = _mapping(payload["power_analysis"], "power_analysis")
    if power.get("positive_and_null_cluster_simulation_required") is not True:
        raise ValueError("Power analysis must include positive and null cluster simulation.")
    if power.get("joint_operating_characteristics_required") is not True:
        raise ValueError("Power analysis must freeze joint operating characteristics.")
    if power.get("equivalence_test") != "two_one_sided_tests":
        raise ValueError("Power analysis must use the frozen equivalence test.")
    if power.get("secondary_multiplicity") != "hierarchical_gatekeeping":
        raise ValueError("Power analysis changed the frozen secondary multiplicity rule.")
    if power.get("selection_role") != "design":
        raise ValueError("Power analysis must be fixed on the design split.")
    if power.get("cluster_unit") != "identity_group":
        raise ValueError("Primary power simulation must resample identity groups.")
    power_alpha = _finite_number(power.get("primary_alpha"), "power_analysis.primary_alpha")
    if not 0.0 < power_alpha < 0.5:
        raise ValueError(
            "power_analysis.primary_alpha must lie in (0, 0.5) so the frozen "
            "two-one-sided-tests interval has positive confidence."
        )
    minimum_power_counts = {
        "clusters_per_trial": 2,
        "simulation_repetitions": 1,
        "bootstrap_iterations": 2,
    }
    for key, minimum in minimum_power_counts.items():
        value = _optional_int(power.get(key))
        if value is None or value < minimum:
            raise ValueError(
                f"power_analysis.{key} must be an integer greater than or "
                f"equal to {minimum}."
            )
    power_seed = _optional_int(power.get("seed"))
    if power_seed is None or power_seed < 0:
        raise ValueError("power_analysis.seed must be a non-negative integer.")
    equivalence_margin = _finite_number(
        power.get("equivalence_margin"), "power_analysis.equivalence_margin"
    )
    if equivalence_margin <= 0.0:
        raise ValueError("power_analysis.equivalence_margin must be positive.")
    frozen_power_thresholds = {}
    for key in (
        "minimum_positive_power",
        "minimum_null_equivalence_power",
        "minimum_joint_power",
    ):
        value = _finite_number(power.get(key), f"power_analysis.{key}")
        if not 0.0 < value <= 1.0:
            raise ValueError(f"power_analysis.{key} must lie in (0, 1].")
        frozen_power_thresholds[key] = value
    if frozen_power_thresholds["minimum_joint_power"] > min(
        frozen_power_thresholds["minimum_positive_power"],
        frozen_power_thresholds["minimum_null_equivalence_power"],
    ):
        raise ValueError("Minimum joint power cannot exceed either component power target.")

    belief = _mapping(payload["belief_kernel_audit"], "belief_kernel_audit")
    for key in ("outer_replicas_M", "inner_forks_L_inner", "probe_horizon_T_probe"):
        if _optional_int(belief.get(key)) is None or int(belief[key]) <= 0:
            raise ValueError(f"Frozen belief_kernel_audit.{key} must be positive.")
    audit_information_states = tuple(
        map(
            str,
            _sequence(
                belief.get("audit_information_states"),
                "belief_kernel_audit.audit_information_states",
            ),
        )
    )
    if (
        not audit_information_states
        or len(set(audit_information_states)) != len(audit_information_states)
        or any(not value.strip() for value in audit_information_states)
    ):
        raise ValueError(
            "Frozen audit_information_states must be non-empty, unique names."
        )
    instrument_cell_registry_sha256 = belief.get(
        "instrument_cell_registry_sha256"
    )
    if not _is_sha256(instrument_cell_registry_sha256):
        raise ValueError(
            "belief_kernel_audit.instrument_cell_registry_sha256 must freeze the "
            "exact pre-collection cell registry."
        )
    if int(belief["outer_replicas_M"]) < int(
        thresholds["kernel_min_outer_replicas"]
    ):
        raise ValueError("Frozen outer_replicas_M is below the registered minimum.")
    if isinstance(belief.get("simultaneous_cell_count"), bool) or int(
        belief.get("simultaneous_cell_count", 0)
    ) < 3:
        raise ValueError("Frozen simultaneous_cell_count must be at least three.")
    confidence_delta = _finite_number(
        belief.get("confidence_delta"),
        "belief_kernel_audit.confidence_delta",
    )
    if not 0.0 < confidence_delta < 1.0:
        raise ValueError("Frozen belief-kernel confidence_delta must lie in (0, 1).")
    if not isinstance(belief.get("exact_mode"), bool):
        raise ValueError("Frozen belief_kernel_audit.exact_mode must be boolean.")
    exact_mode = bool(belief["exact_mode"])
    prune_threshold = _finite_number(
        belief.get("tier1_hypothesis_prune"),
        "belief_kernel_audit.tier1_hypothesis_prune",
    )
    if not 0.0 <= prune_threshold < 1.0:
        raise ValueError("Tier-1 hypothesis pruning threshold must be in [0, 1).")
    registered_posterior_bias = _finite_number(
        belief.get("posterior_bias_bound"),
        "belief_kernel_audit.posterior_bias_bound",
    )
    registered_reset_bias = _finite_number(
        belief.get("reset_bias_bound"),
        "belief_kernel_audit.reset_bias_bound",
    )
    if any(
        not 0.0 <= value <= 1.0
        for value in (registered_posterior_bias, registered_reset_bias)
    ):
        raise ValueError("Frozen posterior and reset bias bounds must lie in [0, 1].")
    if exact_mode and prune_threshold != 0.0:
        raise ValueError("Exact Tier 1 forbids positive-mass hypothesis pruning.")
    if not exact_mode and prune_threshold > 0.0 and registered_posterior_bias <= 0.0:
        raise ValueError(
            "Approximate Tier 1 with positive pruning must freeze a positive "
            "posterior_bias_bound."
        )
    if exact_mode and (
        registered_posterior_bias != 0.0 or registered_reset_bias != 0.0
    ):
        raise ValueError("Exact Tier 1 must freeze zero posterior and reset bias.")
    if belief.get("rng_key_schedule_version") != semantic_bindings[
        "rng_key_schedule_version"
    ]:
        raise ValueError("Belief audit and semantic binding use different RNG schedules.")
    if belief.get("naive_single_checkpoint_fork_admissible") is not False:
        raise ValueError("A single hidden-state checkpoint fork is not a valid kernel estimator.")
    if belief.get("inference_cluster_unit") != "outer_replica":
        raise ValueError("Kernel inference must cluster by independent outer replica.")
    if belief.get("primary_kernel_table_granularity") != (
        "one_row_per_outer_replica_with_paired_probes"
    ):
        raise ValueError("Kernel tables must retain one row per outer replica with paired probes.")
    posterior_sampler = _mapping(
        belief.get("posterior_full_state_sampler"),
        "belief_kernel_audit.posterior_full_state_sampler",
    )
    if posterior_sampler.get("paired_frozen_probes") is not True:
        raise ValueError("Every outer hidden-state draw must receive the same frozen probes.")
    if posterior_sampler.get("rao_blackwell_mixture") != "secondary_only":
        raise ValueError("A Rao-Blackwell mixture cannot replace primary outer draws.")
    exact_replay = _mapping(
        belief.get("tier1_exact_enumeration_replay"),
        "belief_kernel_audit.tier1_exact_enumeration_replay",
    )
    if exact_replay.get("enable") is not True:
        raise ValueError("The exact Tier-1 reference recursion must remain enabled.")
    if exact_replay.get("partner_registry_frozen_required") is not True:
        raise ValueError("The exact Tier-1 prior must bind the frozen partner registry.")
    prior = _mapping(
        exact_replay.get("prior_over_theta"),
        "belief_kernel_audit.tier1_exact_enumeration_replay.prior_over_theta",
    )
    if prior.get("schema_version") != "path_c_theta_prior_v1":
        raise ValueError("The frozen theta prior has the wrong schema version.")
    if prior.get("partner_registry_sha256") != semantic_bindings[
        "partner_registry_sha256"
    ]:
        raise ValueError("The frozen theta prior does not bind the partner registry.")
    prior_support = _sequence(
        prior.get("support"),
        "belief_kernel_audit.tier1_exact_enumeration_replay."
        "prior_over_theta.support",
    )
    if not prior_support:
        raise ValueError("The frozen theta prior support must be non-empty.")
    theta_ids: list[str] = []
    prior_probabilities: list[float] = []
    for index, raw_item in enumerate(prior_support):
        item = _mapping(
            raw_item,
            "belief_kernel_audit.tier1_exact_enumeration_replay."
            f"prior_over_theta.support[{index}]",
        )
        theta_id = item.get("theta_id")
        if not isinstance(theta_id, str) or not theta_id.strip():
            raise ValueError("Every frozen theta prior support id must be non-empty.")
        probability = _finite_number(
            item.get("probability"),
            "belief_kernel_audit.tier1_exact_enumeration_replay."
            f"prior_over_theta.support[{index}].probability",
        )
        if probability <= 0.0:
            raise ValueError("Every frozen theta prior probability must be positive.")
        theta_ids.append(theta_id)
        prior_probabilities.append(probability)
    if len(theta_ids) != len(set(theta_ids)):
        raise ValueError("The frozen theta prior support ids must be unique.")
    if not np.isclose(
        sum(prior_probabilities),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("The frozen theta prior probabilities must sum to one.")
    if exact_replay.get("generation_inference_policy_function") != "option_distribution":
        raise ValueError("Tier-1 generation and inference must share option_distribution.")
    exact_replay_values = {
        "tier1_hypothesis_prune": _finite_number(
            exact_replay.get("tier1_hypothesis_prune"),
            "belief_kernel_audit.tier1_exact_enumeration_replay."
            "tier1_hypothesis_prune",
        ),
        "rho_prune": _finite_number(
            exact_replay.get("rho_prune"),
            "belief_kernel_audit.tier1_exact_enumeration_replay.rho_prune",
        ),
        "posterior_bias_bound": _finite_number(
            exact_replay.get("posterior_bias_bound"),
            "belief_kernel_audit.tier1_exact_enumeration_replay."
            "posterior_bias_bound",
        ),
        "reset_bias_bound": _finite_number(
            exact_replay.get("reset_bias_bound"),
            "belief_kernel_audit.tier1_exact_enumeration_replay.reset_bias_bound",
        ),
    }
    if any(value != 0.0 for value in exact_replay_values.values()):
        raise ValueError(
            "The exact Tier-1 reference recursion must have zero pruning and bias."
        )
    approximate_mode = _mapping(
        belief.get("tier1_approximate_mode"),
        "belief_kernel_audit.tier1_approximate_mode",
    )
    expected_approximate_enable = not exact_mode
    if approximate_mode.get("enable") is not expected_approximate_enable:
        raise ValueError(
            "The approximate Tier-1 mode enable flag must be the inverse of exact_mode."
        )
    if approximate_mode.get("report_discarded_posterior_mass_as_rho_prune") is not True:
        raise ValueError("Approximate Tier 1 must report discarded mass as rho_prune.")
    if approximate_mode.get("bound_required") is not True:
        raise ValueError("Approximate Tier 1 requires a registered pruning bound.")
    tier2 = _mapping(
        belief.get("tier2_hash_and_match"),
        "belief_kernel_audit.tier2_hash_and_match",
    )
    if tier2.get("group_key") != "exact_ego_observable_history_bytes":
        raise ValueError("Tier 2 must match exact observable histories byte-for-byte.")
    if int(tier2.get("maximum_outer_draws_per_episode_and_key", 0)) != 1:
        raise ValueError("Tier 2 may retain at most one draw per episode and history key.")
    rng = _mapping(belief.get("rng"), "belief_kernel_audit.rng")
    if rng.get("schedule_version") != semantic_bindings["rng_key_schedule_version"]:
        raise ValueError("Nested RNG schedule does not match the semantic binding.")
    if rng.get("replay_uses_original_keys") is not True or rng.get(
        "fork_uses_fresh_derived_keys"
    ) is not True:
        raise ValueError("Snapshot replay and forks must use separate RNG key schedules.")
    if tuple(map(str, rng.get("named_streams", ()))) != (
        "jax", "numpy", "python", "torch"
    ):
        raise ValueError("The frozen RNG schedule must name all supported streams.")
    validity_schema = _mapping(
        belief.get("validity_measurement_schema"),
        "belief_kernel_audit.validity_measurement_schema",
    )
    expected_validity_fields = (
        "alpha_upper",
        "beta_lower",
        "sampling_radius",
        "rho_prune",
        "posterior_bias_bound",
        "reset_bias_bound",
        "exact_or_approximate",
    )
    if tuple(map(str, validity_schema.get("required_fields", ()))) != (
        expected_validity_fields
    ):
        raise ValueError("Instrument validity measurement fields are not version 3.")
    if validity_schema.get("instrument_valid_rule") != "beta_lower > alpha_upper":
        raise ValueError("Instrument validity rule must compare beta_lower and alpha_upper.")

    decision = _mapping(payload["decision"], "decision")
    expected_order = (
        "software_conformance",
        "instrument_validity",
        "design_and_calibration_freeze",
        "locked_primary_efficacy",
        "secondary_mechanisms",
    )
    if tuple(map(str, decision.get("order", ()))) != expected_order:
        raise ValueError("Path C decision order is not the frozen dependency order.")
    battery = _mapping(payload["audit_battery"], "audit_battery")
    core_probe_count = _optional_int(battery.get("core_probe_count"))
    if core_probe_count is None or core_probe_count <= 0:
        raise ValueError("Frozen audit_battery.core_probe_count must be positive.")
    if battery.get("battery_sha256") != semantic_bindings["audit_battery_sha256"]:
        raise ValueError("Frozen audit battery SHA-256 binding is inconsistent.")
    if frozen_audit_battery.sha256 != battery.get("battery_sha256"):
        raise ValueError("Frozen audit-battery section does not bind the parsed battery.")
    if core_probe_count != len(frozen_audit_battery.core_scripts):
        raise ValueError("audit_battery.core_probe_count differs from the parsed battery.")
    if int(frozen_audit_battery.T_probe) != int(belief["probe_horizon_T_probe"]):
        raise ValueError("Audit battery T_probe differs from the belief-audit horizon.")
    if int(frozen_audit_battery.L_inner) != int(belief["inner_forks_L_inner"]):
        raise ValueError("Audit battery L_inner differs from the belief-audit inner forks.")

    runtime_contract = _runtime_contract(payload)
    runtime_contract_sha256 = canonical_sha256(runtime_contract)
    if semantic_bindings["resolved_config_sha256"] != runtime_contract_sha256:
        raise ValueError(
            "semantic_bindings.resolved_config_sha256 does not bind the runtime contract."
        )
    return FrozenPathCPreregistration(
        path=source,
        sha256=file_sha256(source),
        payload=payload,
        thresholds=dict(thresholds),
        required_baselines=required_baselines,
        strong_baselines=strong_baselines,
        runtime_contract=runtime_contract,
        runtime_contract_sha256=runtime_contract_sha256,
        semantic_bindings=dict(semantic_bindings),
        primary_endpoint=dict(primary),
        evidence_spec=frozen_evidence,
        response_summary_spec=response_spec,
        response_vocabulary_sha256=response_spec.sha256,
        evidence_spec_sha256=evidence_spec_sha256,
        split_manifest=frozen_split_manifest,
        audit_battery=frozen_audit_battery,
        module_registry_test_ids=tuple(sorted(set(module_registry_test_ids))),
    )


def validate_runtime_path_c_config(
    runtime_section: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
) -> str:
    observed = _runtime_contract(runtime_section)
    if observed != preregistration.runtime_contract:
        mismatches = []
        for section, expected_values in preregistration.runtime_contract.items():
            observed_values = observed.get(section, {})
            for key, expected in expected_values.items():
                actual = observed_values.get(key)
                if actual != expected:
                    mismatches.append(f"{section}.{key}: runtime={actual!r}, frozen={expected!r}")
        raise ValueError(
            "Runtime Path C config does not match the frozen preregistration: "
            + "; ".join(mismatches)
        )
    return canonical_sha256(observed)


def _load_and_validate_path_c_inputs_v1_removed(*_args: Any, **_kwargs: Any) -> None:
    raise ValueError(
        "Path C version-1 manifests are unsupported; migrate explicitly to path_c_artifacts_v3."
    )
def load_and_validate_path_c_inputs(
    preregistration_path: str | Path,
    manifest_path: str | Path,
    *,
    resolved_path_c: Mapping[str, Any] | None = None,
) -> ValidatedPathCInputs:
    """Load the version-3 append-only manifest and every bound raw measurement."""

    preregistration = load_frozen_preregistration(preregistration_path)
    resolved_sha = (
        validate_runtime_path_c_config(resolved_path_c, preregistration)
        if resolved_path_c is not None
        else preregistration.runtime_contract_sha256
    )
    source = Path(manifest_path).resolve()
    payload = _load_mapping(source, "Path C artifact manifest")
    expected_keys = {
        "schema_version",
        "active_stage",
        "preregistration",
        "resolved_path_c_sha256",
        "semantic_bindings",
        "dataset_shards",
        "policy_artifacts",
        "measurement_artifacts",
    }
    _require_exact_mapping_keys(payload, expected_keys, "manifest")
    if payload["schema_version"] != PATH_C_ARTIFACT_SCHEMA:
        raise ValueError(
            "Legacy Path C artifact manifests require explicit version-3 migration."
        )
    active_stage = str(payload["active_stage"])
    stage_order = (
        "software_conformance",
        "instrument_validity",
        "design_and_calibration_freeze",
        "locked_primary_efficacy",
        "secondary_mechanisms",
    )
    if active_stage not in stage_order:
        raise ValueError("Manifest active_stage is not a registered Path C stage.")
    prereg_ref = _mapping(payload["preregistration"], "manifest.preregistration")
    _require_exact_mapping_keys(
        prereg_ref,
        {"sha256", "version", "schema_version"},
        "manifest.preregistration",
    )
    if prereg_ref != {
        "sha256": preregistration.sha256,
        "version": preregistration.payload["version"],
        "schema_version": preregistration.payload["schema_version"],
    }:
        raise ValueError("Manifest preregistration binding is inconsistent.")
    if payload["resolved_path_c_sha256"] != resolved_sha:
        raise ValueError("Manifest resolved Path C runtime hash is inconsistent.")
    semantic_bindings = _mapping(
        payload["semantic_bindings"], "manifest.semantic_bindings"
    )
    if semantic_bindings != preregistration.semantic_bindings:
        raise ValueError("Manifest semantic bindings differ from the frozen preregistration.")

    policies = _mapping(payload["policy_artifacts"], "manifest.policy_artifacts")
    _validate_v3_policy_artifacts(policies, preregistration, source.parent)

    refs = _mapping(
        payload["measurement_artifacts"], "manifest.measurement_artifacts"
    )
    required_measurements = {
        "software_conformance": "path_c_software_conformance_v1",
        "software_test_report": "path_c_software_test_report_v1",
        "policy_metrics": "path_c_policy_metrics_v1",
        "audit_cost_estimate": "path_c_audit_cost_estimate_v1",
    }
    optional_measurements = {
        "secondary_profile": "path_c_secondary_profile_v1",
        "instrument_evidence": "path_c_instrument_evidence_v2",
        "instrument_measurement": "path_c_instrument_measurement_v1",
        "baseline_selection": "path_c_baseline_selection_v1",
        "primary_endpoint": "path_c_primary_endpoint_v1",
        "design_return_point_ledger": "path_c_return_point_ledger_v2",
        "locked_return_point_ledger": "path_c_return_point_ledger_v2",
    }
    all_measurements = {**required_measurements, **optional_measurements}
    stage_measurements = {
        "software_conformance": set(required_measurements),
        "instrument_validity": {
            *required_measurements,
            "instrument_evidence",
            "instrument_measurement",
        },
        "design_and_calibration_freeze": {
            *required_measurements,
            "instrument_evidence",
            "instrument_measurement",
            "design_return_point_ledger",
            "baseline_selection",
        },
        "locked_primary_efficacy": {
            *required_measurements,
            "instrument_evidence",
            "instrument_measurement",
            "design_return_point_ledger",
            "baseline_selection",
            "primary_endpoint",
        },
        "secondary_mechanisms": set(all_measurements),
    }
    expected_measurements = stage_measurements[active_stage]
    allowed_measurements = set(expected_measurements)
    if active_stage == "locked_primary_efficacy":
        allowed_measurements.add("locked_return_point_ledger")
    missing = sorted(expected_measurements.difference(refs))
    unknown = sorted(set(refs).difference(allowed_measurements))
    if missing or unknown:
        raise ValueError(
            f"Manifest measurement artifacts mismatch; missing={missing}, unknown={unknown}."
        )
    measurements: dict[str, dict[str, Any]] = {}
    for name, schema_version in all_measurements.items():
        if name not in refs:
            continue
        measurement = _load_v3_bound_artifact(
            refs[name],
            base_dir=source.parent,
            artifact_name=name,
            schema_version=schema_version,
            preregistration=preregistration,
            resolved_path_c_sha256=resolved_sha,
        )
        measurements[name] = measurement

    if active_stage == "locked_primary_efficacy":
        primary_available = measurements["primary_endpoint"].get("available")
        locked_ledger_present = "locked_return_point_ledger" in measurements
        if primary_available is True and not locked_ledger_present:
            raise ValueError(
                "A completed locked-primary result requires its return-point ledger."
            )
        if primary_available is False and locked_ledger_present:
            raise ValueError(
                "PRIMARY_NOT_RUN must not reference a locked-audit return ledger."
            )
        if not isinstance(primary_available, bool):
            raise ValueError(
                "locked-primary primary_endpoint.available must be boolean."
            )

    stage_dataset_roles = {
        "software_conformance": {"train"},
        "instrument_validity": {"calibration"},
        "design_and_calibration_freeze": {"design"},
        "locked_primary_efficacy": {"locked_audit"},
        "secondary_mechanisms": {"locked_audit"},
    }
    datasets = _mapping(payload["dataset_shards"], "manifest.dataset_shards")
    _validate_v3_dataset_shards(
        datasets,
        preregistration,
        source.parent,
        required_roles=stage_dataset_roles[active_stage],
        allowed_roles=stage_dataset_roles[active_stage],
    )

    manifest = PathCArtifactManifest(
        path=source,
        sha256=file_sha256(source),
        payload=payload,
        preregistration_sha256=preregistration.sha256,
        resolved_path_c_sha256=resolved_sha,
        active_stage=active_stage,
        datasets=dict(datasets),
        checkpoints=dict(policies),
        measurement_artifacts=dict(refs),
        value_control_artifact={},
        semantic_bindings=dict(semantic_bindings),
        dataset_shards=dict(datasets),
        policy_artifacts=dict(policies),
    )
    return ValidatedPathCInputs(preregistration, manifest, measurements)


def artifact_contract_measurement(inputs: ValidatedPathCInputs) -> dict[str, Any]:
    prereg = inputs.preregistration
    datasets = inputs.manifest.dataset_shards or {}
    policies = inputs.manifest.policy_artifacts or {}
    bindings_valid = inputs.manifest.semantic_bindings == prereg.semantic_bindings
    append_only = all(
        item.get("storage") == "content_addressed_append_only_shards"
        for item in datasets.values()
    )
    required_policy_names = {
        PATH_C_PRIMARY_VARIANT,
        *prereg.required_baselines,
        "random_probe",
        "no_probe",
        "direct_information",
    }
    missing_policies = sorted(required_policy_names.difference(policies))
    roles = {str(item.get("collection_role")) for item in datasets.values()}
    stage_roles = {
        "software_conformance": {"train"},
        "instrument_validity": {"calibration"},
        "design_and_calibration_freeze": {"design"},
        "locked_primary_efficacy": {"locked_audit"},
        "secondary_mechanisms": {"locked_audit"},
    }
    missing_roles = sorted(stage_roles[inputs.manifest.active_stage].difference(roles))
    return {
        "schema_version": PATH_C_ARTIFACT_SCHEMA,
        "preregistration_sha256": prereg.sha256,
        "resolved_path_c_sha256": inputs.manifest.resolved_path_c_sha256,
        "semantic_bindings": dict(inputs.manifest.semantic_bindings or {}),
        "semantic_bindings_valid": bindings_valid,
        "active_stage": inputs.manifest.active_stage,
        "append_only_content_addressed_shards": append_only,
        "missing_roles": missing_roles,
        "missing_policies": missing_policies,
        "effective_budgets": {
            name: {
                "effective_episodes": int(item.get("effective_episodes", 0)),
                "effective_transitions": int(item.get("effective_transitions", 0)),
            }
            for name, item in datasets.items()
        },
        "valid": bool(
            not missing_roles
            and not missing_policies
            and bindings_valid
            and append_only
        ),
    }


def phase_b_go_no_go_rule(
    measurements: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, Any]:
    """Compatibility entry point with version-3 semantics.

    The historical function name remains for callers, but it no longer evaluates
    the retired response, transfer, leakage, or null gates and never emits the old
    binary decision. Its result is the explicit staged decision object.
    """

    decision = evaluate_path_c_decision(measurements, preregistration)
    return {
        **asdict(decision),
        "preregistration_sha256": preregistration.sha256,
    }


def pass_af_claim_rule(
    measurements: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, Any]:
    del measurements, preregistration
    raise ValueError(
        "pass_af_claim_rule was removed in Path C version 3. Migrate to "
        "evaluate_path_c_decision; the old G/epsilon gate has no version-3 alias."
    )


def _audit_cost_estimate_conforms(
    measurement: Any,
    preregistration: FrozenPathCPreregistration,
) -> bool:
    """Recompute every audit-cost dimension from the frozen preregistration."""

    if not isinstance(measurement, Mapping):
        return False
    belief = preregistration.payload["belief_kernel_audit"]
    expected = estimate_audit_cost(
        audit_units=len(belief["audit_information_states"]),
        probes=int(preregistration.payload["audit_battery"]["core_probe_count"]),
        outer_replicas_M=int(belief["outer_replicas_M"]),
        inner_forks_L_inner=int(belief["inner_forks_L_inner"]),
        horizon_T_probe=int(belief["probe_horizon_T_probe"]),
        max_primitive_steps=int(
            preregistration.payload["evidence_spec"][
                "max_primitive_actions_per_decision"
            ]
        ),
        maximum_primitive_step_budget=int(
            preregistration.payload["budget"]["maximum_audit_primitive_steps"]
        ),
        design_audit_units=(
            len(belief["audit_information_states"])
            if preregistration.audit_battery.design_only_scripts
            else 0
        ),
        design_only_probes=len(preregistration.audit_battery.design_only_scripts),
    )
    core_keys = set(expected)
    bound_envelope_keys = {
        "measurement_schema_version",
        "preregistration_sha256",
        "resolved_path_c_sha256",
        "semantic_bindings",
        "artifact_sha256",
    }
    observed_keys = set(measurement)
    if observed_keys not in (core_keys, core_keys | bound_envelope_keys):
        return False
    integer_fields = (
        "audit_units",
        "probes",
        "outer_replicas_M",
        "inner_forks_L_inner",
        "horizon_T_probe",
        "max_primitive_steps",
        "maximum_primitive_step_budget",
        "design_audit_units",
        "design_only_probes",
        "core_continuations",
        "design_extension_continuations",
        "continuations",
        "option_steps",
        "primitive_step_upper_bound",
    )
    if any(
        _optional_int(measurement.get(key)) != int(expected[key])
        for key in integer_fields
    ):
        return False
    if measurement.get("schema_version") != expected["schema_version"]:
        return False
    if measurement.get("within_budget") is not expected["within_budget"]:
        return False
    if observed_keys == core_keys | bound_envelope_keys:
        if (
            measurement.get("measurement_schema_version")
            != PATH_C_MEASUREMENT_SCHEMA
            or measurement.get("preregistration_sha256")
            != preregistration.sha256
            or measurement.get("resolved_path_c_sha256")
            != preregistration.runtime_contract_sha256
            or measurement.get("semantic_bindings")
            != preregistration.semantic_bindings
            or not _is_sha256(measurement.get("artifact_sha256"))
        ):
            return False
    return True


def _measurement_core(
    measurement: Mapping[str, Any],
    expected_keys: set[str],
    name: str,
) -> dict[str, Any]:
    """Extract a strict core while permitting only the standard hash envelope."""

    observed = set(measurement)
    missing = sorted(expected_keys.difference(observed))
    unknown = sorted(
        observed.difference(expected_keys | _BOUND_MEASUREMENT_ENVELOPE_KEYS)
    )
    if missing or unknown:
        raise ValueError(
            f"{name} key mismatch; missing={missing}, unknown={unknown}."
        )
    return {key: measurement[key] for key in expected_keys}


def _validated_software_test_report(
    measurement: Any,
    preregistration: FrozenPathCPreregistration,
) -> bool:
    """Validate archived test outcomes and their exact registry coverage."""

    observed = _mapping(measurement, "software_test_report")
    core = _measurement_core(
        observed,
        {
            "schema_version",
            "module_registry_sha256",
            "source_git_sha",
            "test_execution_status",
            "tests",
        },
        "software_test_report",
    )
    if core["schema_version"] != "path_c_software_test_report_v1":
        raise ValueError("software_test_report has the wrong schema version.")
    if core["module_registry_sha256"] != preregistration.semantic_bindings[
        "module_registry_sha256"
    ]:
        raise ValueError("software_test_report does not bind the frozen module registry.")
    if core["source_git_sha"] != preregistration.semantic_bindings["code_commit"]:
        raise ValueError("software_test_report does not bind the frozen source commit.")
    execution_status = str(core["test_execution_status"])
    if execution_status not in {"passed", "failed"}:
        raise ValueError("software_test_report execution status must be passed or failed.")
    rows = _sequence(core["tests"], "software_test_report.tests")
    if not rows:
        raise ValueError("software_test_report must contain archived test rows.")
    observed_test_ids: list[str] = []
    covered_checks: set[str] = set()
    all_rows_passed = True
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"software_test_report.tests[{index}]")
        _require_exact_mapping_keys(
            row,
            {"test_id", "outcome", "report_sha256", "covers"},
            f"software_test_report.tests[{index}]",
        )
        test_id = str(row["test_id"])
        if not test_id.strip():
            raise ValueError("software_test_report test_id must be non-empty.")
        outcome = str(row["outcome"])
        if outcome not in {"passed", "failed"}:
            raise ValueError("software_test_report row outcome must be passed or failed.")
        if not _is_sha256(row["report_sha256"]):
            raise ValueError("software_test_report rows must bind archived report SHA-256.")
        covers = tuple(
            map(
                str,
                _sequence(
                    row["covers"],
                    f"software_test_report.tests[{index}].covers",
                ),
            )
        )
        if not covers or any(not value.strip() for value in covers):
            raise ValueError("Every software test row must cover named conformance checks.")
        observed_test_ids.append(test_id)
        covered_checks.update(covers)
        all_rows_passed = all_rows_passed and outcome == "passed"
    if len(observed_test_ids) != len(set(observed_test_ids)):
        raise ValueError("software_test_report repeats a test identifier.")
    if set(observed_test_ids) != set(preregistration.module_registry_test_ids):
        raise ValueError("software_test_report does not cover the exact module-registry tests.")
    required_coverage = set(PATH_C_SOFTWARE_CHECKS) | set(
        map(
            str,
            preregistration.payload["belief_kernel_audit"]["validity_checks"],
        )
    )
    if covered_checks != required_coverage:
        raise ValueError(
            "software_test_report does not cover the exact software and instrument checks."
        )
    if (execution_status == "passed") != all_rows_passed:
        raise ValueError("software_test_report aggregate status disagrees with its rows.")
    return all_rows_passed


def _validated_policy_metrics(
    measurement: Any,
    preregistration: FrozenPathCPreregistration,
) -> dict[str, dict[str, Any]] | None:
    """Recompute the policy fairness report and bind it to frozen resources."""

    try:
        observed = _mapping(measurement, "policy_metrics")
        required_policies = tuple(map(
            str,
            preregistration.payload["budget"]["required_policy_keys"],
        ))
        expected_core_keys = {
            "schema_version",
            "required_policies",
            "policies",
            "complete",
            "missing_policies",
            "matched_fields",
            "mismatches",
            "resource_pareto_metrics",
            "fairness_conformant",
        }
        core = _measurement_core(observed, expected_core_keys, "policy_metrics")
        if core["schema_version"] != "path_c_policy_metrics_v1":
            raise ValueError("policy_metrics has the wrong schema version.")
        reported_required = tuple(map(
            str,
            _sequence(core["required_policies"], "policy_metrics.required_policies"),
        ))
        if reported_required != required_policies:
            raise ValueError(
                "policy_metrics changed the frozen required-policy order."
            )
        policies = _mapping(core["policies"], "policy_metrics.policies")
        if set(policies) != set(required_policies):
            raise ValueError(
                "policy_metrics.policies must exactly match the frozen required policies."
            )

        frozen_budget = preregistration.payload["budget"]
        frozen_values = {
            "training_environment_steps": int(
                frozen_budget["training_environment_steps"]
            ),
            "gradient_updates": int(frozen_budget["training_gradient_updates"]),
            "evaluation_environment_step_limit": int(
                frozen_budget["evaluation_environment_step_limit"]
            ),
            "evaluation_schedule_id": str(frozen_budget["evaluation_schedule_id"]),
            "probe_budget_grid_sha256": str(
                frozen_budget["probe_budget_grid_sha256"]
            ),
            "probe_cost_per_use": float(
                preregistration.primary_endpoint["probe_cost_per_use"]
            ),
            "action_support_sha256": str(frozen_budget["action_support_sha256"]),
        }
        validated_policies: dict[str, dict[str, Any]] = {}
        for policy_name in required_policies:
            item = _mapping(
                policies[policy_name],
                f"policy_metrics.policies.{policy_name}",
            )
            _require_exact_mapping_keys(
                item,
                set(_POLICY_METRIC_FIELDS),
                f"policy_metrics.policies.{policy_name}",
            )
            if (
                _optional_int(item["training_environment_steps"])
                != frozen_values["training_environment_steps"]
                or _optional_int(item["gradient_updates"])
                != frozen_values["gradient_updates"]
                or _optional_int(item["evaluation_environment_step_limit"])
                != frozen_values["evaluation_environment_step_limit"]
            ):
                raise ValueError(
                    f"policy_metrics.policies.{policy_name} changed frozen compute."
                )
            for field_name in (
                "evaluation_schedule_id",
                "probe_budget_grid_sha256",
                "action_support_sha256",
            ):
                if item[field_name] != frozen_values[field_name]:
                    raise ValueError(
                        f"policy_metrics.policies.{policy_name} changed frozen "
                        f"field {field_name}."
                    )
            probe_cost = _finite_number(
                item["probe_cost_per_use"],
                f"policy_metrics.policies.{policy_name}.probe_cost_per_use",
            )
            if probe_cost != frozen_values["probe_cost_per_use"]:
                raise ValueError(
                    f"policy_metrics.policies.{policy_name} changed frozen probe cost."
                )
            trainable_parameters = _optional_int(item["trainable_parameters"])
            if trainable_parameters is None or trainable_parameters < 0:
                raise ValueError(
                    f"policy_metrics.policies.{policy_name}.trainable_parameters "
                    "must be non-negative."
                )
            for field_name in (
                "training_flops",
                "wall_clock_seconds",
                "inference_latency_ms",
            ):
                if _finite_number(
                    item[field_name],
                    f"policy_metrics.policies.{policy_name}.{field_name}",
                ) < 0.0:
                    raise ValueError(
                        f"policy_metrics.policies.{policy_name}.{field_name} "
                        "must be non-negative."
                    )
            validated_policies[policy_name] = item

        recomputed = matched_policy_metrics(
            validated_policies,
            required_policies=required_policies,
        )
        reported_recomputed_fields = {
            key: core[key]
            for key in (
                "complete",
                "missing_policies",
                "matched_fields",
                "mismatches",
                "resource_pareto_metrics",
                "fairness_conformant",
            )
        }
        if canonical_sha256(reported_recomputed_fields) != canonical_sha256(
            recomputed
        ):
            raise ValueError(
                "policy_metrics fairness report does not match the raw policy metrics."
            )
        if recomputed.get("fairness_conformant") is not True:
            raise ValueError("policy_metrics do not satisfy the frozen fairness contract.")
        return validated_policies
    except (KeyError, TypeError, ValueError):
        return None


def _validated_return_point_ledger(
    measurement: Any,
    preregistration: FrozenPathCPreregistration,
    policy_metrics: Mapping[str, Mapping[str, Any]],
    *,
    split_role: str,
) -> ReturnPointLedgerV1:
    """Load the raw episode ledger and bind every row to frozen policy resources."""

    artifact_name = (
        "design_return_point_ledger"
        if split_role == "design"
        else "locked_return_point_ledger"
    )
    observed = _mapping(measurement, artifact_name)
    ledger = ReturnPointLedgerV1.from_mapping(observed)
    if ledger.split_role != split_role:
        raise ValueError(
            f"{artifact_name} has split role {ledger.split_role!r}, expected "
            f"{split_role!r}."
        )
    if ledger.split_manifest_sha256 != preregistration.semantic_bindings[
        "split_manifest_sha256"
    ]:
        raise ValueError(
            f"{artifact_name} changed the frozen split-manifest SHA-256."
        )
    ledger.validate_split_manifest(preregistration.split_manifest)
    primary = preregistration.primary_endpoint
    frozen_grid = tuple(map(int, primary["probe_budget_grid"]))
    if ledger.probe_budget_grid != frozen_grid:
        raise ValueError("Return-point ledger changed the frozen probe-budget grid.")
    if ledger.normalization_rule != primary["normalization_rule"]:
        raise ValueError("Return-point ledger changed the frozen normalization rule.")
    for field_name in (
        "normalization_lower",
        "normalization_upper",
        "probe_cost_per_use",
    ):
        if float(getattr(ledger, field_name)) != float(primary[field_name]):
            raise ValueError(
                f"Return-point ledger changed frozen field {field_name}."
            )
    if "preregistration_sha256" in observed and observed[
        "preregistration_sha256"
    ] != preregistration.sha256:
        raise ValueError("Return-point ledger has the wrong preregistration hash.")
    if "resolved_path_c_sha256" in observed and observed[
        "resolved_path_c_sha256"
    ] != preregistration.runtime_contract_sha256:
        raise ValueError("Return-point ledger has the wrong runtime hash.")
    if "semantic_bindings" in observed and observed[
        "semantic_bindings"
    ] != preregistration.semantic_bindings:
        raise ValueError("Return-point ledger changed frozen semantic bindings.")

    if split_role == "design":
        expected_policy_names = {
            PATH_C_PRIMARY_VARIANT,
            *preregistration.strong_baselines,
            "random_probe",
            "no_probe",
            "direct_information",
        }
    else:
        expected_policy_names = {
            PATH_C_PRIMARY_VARIANT,
            *preregistration.required_baselines,
        }
    observed_policy_names: set[str] = set()
    for record in ledger.points:
        point = record.point
        policy_name = str(point.policy_name)
        if policy_name not in expected_policy_names:
            raise ValueError(
                "Return-point ledger contains a policy outside the primary comparison."
            )
        observed_policy_names.add(policy_name)
        policy = policy_metrics.get(policy_name)
        if policy is None:
            raise ValueError(
                f"Return-point ledger policy {policy_name!r} lacks policy metrics."
            )
        expected_fields = {
            "training_environment_steps": int(
                policy["training_environment_steps"]
            ),
            "gradient_updates": int(policy["gradient_updates"]),
            "evaluation_environment_step_limit": int(
                policy["evaluation_environment_step_limit"]
            ),
            "evaluation_schedule_id": str(policy["evaluation_schedule_id"]),
            "probe_cost_per_use": float(policy["probe_cost_per_use"]),
        }
        observed_fields = {
            "training_environment_steps": int(point.training_environment_steps),
            "gradient_updates": int(point.gradient_updates),
            "evaluation_environment_step_limit": int(
                point.evaluation_environment_step_limit
            ),
            "evaluation_schedule_id": str(point.evaluation_schedule_id),
            "probe_cost_per_use": float(point.probe_cost_per_use),
        }
        if observed_fields != expected_fields:
            raise ValueError(
                "Return-point ledger row does not match its frozen policy resources."
            )
        if int(point.episode_environment_steps) > int(
            point.evaluation_environment_step_limit
        ):
            raise ValueError(
                "Return-point ledger episode exceeds the frozen evaluation-step limit."
            )
    if observed_policy_names != expected_policy_names:
        raise ValueError(
            f"Return-point ledger {split_role} rows do not contain the complete "
            "frozen acting-policy comparison set."
        )
    return ledger


def build_secondary_profile_v1(
    source_measurements: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a secondary-only profile from content-addressed source measurements."""

    if not isinstance(source_measurements, Mapping) or not source_measurements:
        raise ValueError("Secondary profile requires source measurements.")
    normalized: dict[str, dict[str, Any]] = {}
    measurements: dict[str, dict[str, Any]] = {}
    expected_keys = {
        "estimand",
        "role",
        "cluster_unit",
        "estimate",
        "confidence_interval",
        "multiplicity_adjustment",
        "source_artifact_sha256",
    }
    for name, raw_item in sorted(source_measurements.items()):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Secondary measurement names must be non-empty.")
        item = _mapping(raw_item, f"secondary source {name}")
        _require_exact_mapping_keys(item, expected_keys, f"secondary source {name}")
        source_sha256 = str(item["source_artifact_sha256"])
        if not _is_sha256(source_sha256):
            raise ValueError("Secondary source artifact hash must be SHA-256.")
        interval = tuple(
            _finite_number(value, f"secondary source {name} confidence interval")
            for value in _sequence(
                item["confidence_interval"],
                f"secondary source {name} confidence interval",
            )
        )
        if len(interval) != 2 or interval[0] > interval[1]:
            raise ValueError("Secondary confidence interval must contain ordered bounds.")
        normalized_item = {
            "estimand": _strict_text(item["estimand"], f"secondary {name} estimand"),
            "role": _strict_text(item["role"], f"secondary {name} role"),
            "cluster_unit": _strict_text(
                item["cluster_unit"],
                f"secondary {name} cluster unit",
            ),
            "estimate": _finite_number(item["estimate"], f"secondary {name} estimate"),
            "confidence_interval": list(map(float, interval)),
            "multiplicity_adjustment": _strict_text(
                item["multiplicity_adjustment"],
                f"secondary {name} multiplicity adjustment",
            ),
            "source_artifact_sha256": source_sha256,
        }
        normalized[name] = normalized_item
        measurements[name] = {
            key: normalized_item[key]
            for key in (
                "estimand",
                "role",
                "cluster_unit",
                "estimate",
                "confidence_interval",
                "multiplicity_adjustment",
            )
        }
    return {
        "schema_version": "path_c_secondary_profile_v1",
        "available": True,
        "decision_eligible": False,
        "source_measurements": normalized,
        "source_measurements_sha256": canonical_sha256(normalized),
        "measurements": measurements,
    }


def evaluate_path_c_decision(
    measurements: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
) -> PathCDecision:
    """Apply software -> instrument -> design freeze -> primary -> secondary."""

    artifact_contract = _mapping(
        measurements.get("artifact_contract"), "artifact_contract"
    )
    active_stage = str(artifact_contract.get("active_stage", ""))
    stage_order = (
        "software_conformance",
        "instrument_validity",
        "design_and_calibration_freeze",
        "locked_primary_efficacy",
        "secondary_mechanisms",
    )
    if active_stage not in stage_order:
        raise ValueError("artifact_contract active_stage is not registered.")

    software = _mapping(
        measurements.get("software_conformance"), "software_conformance"
    )
    software = _measurement_core(
        software,
        {"schema_version", "checks"},
        "software_conformance",
    )
    if software["schema_version"] != "path_c_software_conformance_v1":
        raise ValueError("software_conformance has the wrong schema version.")
    checks = _mapping(software.get("checks"), "software_conformance.checks")
    required_software_checks = set(PATH_C_SOFTWARE_CHECKS)
    missing_software = sorted(required_software_checks.difference(checks))
    unknown_software = sorted(set(checks).difference(required_software_checks))
    cost_estimate = measurements.get("audit_cost_estimate")
    cost_conformant = _audit_cost_estimate_conforms(
        cost_estimate,
        preregistration,
    )
    validated_policy_metrics = _validated_policy_metrics(
        measurements.get("policy_metrics"),
        preregistration,
    )
    software_test_report_conformant = _validated_software_test_report(
        measurements.get("software_test_report"),
        preregistration,
    )
    software_conformant = bool(
        not missing_software
        and not unknown_software
        and all(checks[name] is True for name in required_software_checks)
        and cost_conformant
        and software_test_report_conformant
        and validated_policy_metrics is not None
        and artifact_contract.get("valid") is True
        and artifact_contract.get("semantic_bindings_valid") is True
    )
    if not software_conformant:
        return PathCDecision(
            schema_version=PATH_C_DECISION_SCHEMA,
            software_conformant=False,
            instrument_valid=False,
            primary_effective=None,
            primary_estimate=None,
            primary_ci=None,
            secondary={},
            allowed_claim="diagnostic_only",
            status="SOFTWARE_NONCONFORMANT",
            requires_type_b_review=True,
        )
    if validated_policy_metrics is None:
        raise RuntimeError("Policy-metric validation result was lost after conformance.")
    if active_stage == "software_conformance":
        return PathCDecision(
            schema_version=PATH_C_DECISION_SCHEMA,
            software_conformant=True,
            instrument_valid=False,
            primary_effective=None,
            primary_estimate=None,
            primary_ci=None,
            secondary={},
            allowed_claim="software_conformance_only",
            status="SOFTWARE_CONFORMANT_PENDING_INSTRUMENT",
            requires_type_b_review=True,
        )

    instrument_evidence = _mapping(
        measurements.get("instrument_evidence"), "instrument_evidence"
    )
    instrument_evidence_keys = {
        "schema_version",
        "response_vocabulary_sha256",
        "vocabulary_size_q",
        "simultaneous_cell_count",
        "confidence_delta",
        "exact_or_approximate",
        "rho_prune",
        "posterior_bias_bound",
        "reset_bias_bound",
        "sampling_design",
        "sampling_source",
        "battery_sha256",
        "frozen_probes",
        "summary_id",
        "inner_forks_L_inner",
        "support_violation_token_id",
        "cell_registry_sha256",
        "kernel_records_sha256",
        "cells",
    }
    instrument_evidence = _measurement_core(
        instrument_evidence,
        instrument_evidence_keys,
        "instrument_evidence",
    )
    frozen_belief = preregistration.payload["belief_kernel_audit"]
    if instrument_evidence["schema_version"] != "path_c_instrument_evidence_v2":
        raise ValueError(
            "Instrument evidence must contain complete version-2 outer-replica records."
        )
    if instrument_evidence["response_vocabulary_sha256"] != (
        preregistration.response_vocabulary_sha256
    ):
        raise ValueError("Instrument evidence changes the frozen response vocabulary.")
    if int(instrument_evidence["vocabulary_size_q"]) != (
        preregistration.response_summary_spec.q
    ):
        raise ValueError("Instrument evidence uses the wrong frozen vocabulary size.")
    if int(instrument_evidence["simultaneous_cell_count"]) != int(
        frozen_belief["simultaneous_cell_count"]
    ):
        raise ValueError("Instrument evidence changes the frozen simultaneous cell count.")
    if float(instrument_evidence["confidence_delta"]) != float(
        frozen_belief["confidence_delta"]
    ):
        raise ValueError("Instrument evidence changes the frozen confidence delta.")
    if instrument_evidence["battery_sha256"] != preregistration.semantic_bindings[
        "audit_battery_sha256"
    ]:
        raise ValueError("Instrument evidence changes the frozen audit battery.")
    expected_probes = [
        {
            "probe_id": script.script_id,
            "sampling_probability": float(script.sampling_probability),
        }
        for script in preregistration.audit_battery.scripts_for_role("locked_audit")
    ]
    if instrument_evidence["frozen_probes"] != expected_probes:
        raise ValueError("Instrument evidence changes the frozen locked-audit probes.")
    if int(instrument_evidence["inner_forks_L_inner"]) != int(
        frozen_belief["inner_forks_L_inner"]
    ) or int(instrument_evidence["inner_forks_L_inner"]) != int(
        preregistration.audit_battery.L_inner
    ):
        raise ValueError("Instrument evidence changes frozen L_inner.")
    expected_support_token = preregistration.response_summary_spec.encode(
        support_violation=True
    )
    if int(instrument_evidence["support_violation_token_id"]) != int(
        expected_support_token
    ):
        raise ValueError(
            "Instrument evidence changes the frozen support-violation token."
        )
    if instrument_evidence["cell_registry_sha256"] != frozen_belief[
        "instrument_cell_registry_sha256"
    ]:
        raise ValueError(
            "Instrument evidence does not match the cell registry frozen before collection."
        )
    raw_instrument_cells = _sequence(
        instrument_evidence["cells"], "instrument_evidence.cells"
    )
    registered_audit_units = tuple(map(str, frozen_belief["audit_information_states"]))
    observed_audit_units = []
    for raw_cell in raw_instrument_cells:
        cell = _mapping(raw_cell, "instrument_evidence.cell")
        observed_audit_units.append(str(cell.get("audit_unit_id", "")))
        records = _sequence(
            cell.get("kernel_records"),
            "instrument_evidence.cell.kernel_records",
        )
        if len(records) != int(frozen_belief["outer_replicas_M"]):
            raise ValueError(
                "Every instrument cell must contain the frozen number of complete "
                "outer-replica records."
            )
    if (
        len(observed_audit_units) != len(set(observed_audit_units))
        or set(observed_audit_units) != set(registered_audit_units)
    ):
        raise ValueError(
            "Instrument cells must exactly cover the registered audit information states."
        )
    recomputed_instrument = recompute_instrument_measurement_from_evidence_v2(
        instrument_evidence
    )
    instrument = _mapping(
        measurements.get("instrument_measurement"), "instrument_measurement"
    )
    instrument = _measurement_core(
        instrument,
        set(recomputed_instrument),
        "instrument_measurement",
    )
    if canonical_sha256(instrument) != canonical_sha256(recomputed_instrument):
        raise ValueError("instrument_measurement was not recomputed from instrument_evidence.")
    if instrument["schema_version"] != "path_c_instrument_measurement_v1":
        raise ValueError("instrument_measurement has the wrong schema version.")
    alpha_upper = _finite_number(
        instrument.get("alpha_upper"), "instrument_measurement.alpha_upper"
    )
    beta_lower = _finite_number(
        instrument.get("beta_lower"), "instrument_measurement.beta_lower"
    )
    sampling_radius = _finite_number(
        instrument.get("sampling_radius"),
        "instrument_measurement.sampling_radius",
    )
    rho_prune = _finite_number(
        instrument.get("rho_prune"),
        "instrument_measurement.rho_prune",
    )
    posterior_bias = _finite_number(
        instrument.get("posterior_bias_bound"),
        "instrument_measurement.posterior_bias_bound",
    )
    reset_bias = _finite_number(
        instrument.get("reset_bias_bound"),
        "instrument_measurement.reset_bias_bound",
    )
    if any(
        not 0.0 <= value <= 1.0
        for value in (sampling_radius, rho_prune, posterior_bias, reset_bias)
    ):
        raise ValueError("Instrument radii and bias bounds must lie in [0, 1].")
    if not 0.0 <= alpha_upper <= 1.0 or not 0.0 <= beta_lower <= 1.0:
        raise ValueError("Instrument alpha/beta bounds must lie in [0, 1].")
    if instrument.get("response_vocabulary_sha256") != preregistration.response_vocabulary_sha256:
        raise ValueError("Instrument uses a different frozen response vocabulary.")
    if instrument.get("distance_convention") != "total_variation_half_l1":
        raise ValueError("Instrument uses the wrong total-variation convention.")
    if instrument.get("outer_sampling_unit") != (
        "independent_full_hidden_state_posterior_draw"
    ):
        raise ValueError("Instrument outer replicas must be full hidden-state draws.")
    if instrument.get("inner_forks_are_outer_samples") is not False:
        raise ValueError("Instrument inner forks cannot be counted as outer samples.")
    exact_or_approximate = str(instrument.get("exact_or_approximate", ""))
    if exact_or_approximate not in {"exact", "approximate_bounded"}:
        raise ValueError("Instrument must declare exact or approximate_bounded status.")
    if exact_or_approximate == "exact" and (
        rho_prune != 0.0 or posterior_bias != 0.0 or reset_bias != 0.0
    ):
        raise ValueError("Exact instrument measurements cannot report pruning bias.")
    if exact_or_approximate == "approximate_bounded" and posterior_bias < rho_prune:
        raise ValueError(
            "Approximate posterior_bias_bound must cover the reported rho_prune."
        )
    validity_checks = _mapping(
        instrument.get("validity_checks"),
        "instrument_measurement.validity_checks",
    )
    expected_mechanical_checks = {
        "embedded_kernel_records_schema_valid",
        "battery_snapshot_audit_unit_probe_binding_valid",
        "iid_full_posterior_with_replacement_valid",
        "sampler_seed_schedule_valid",
        "posterior_support_draws_valid",
        "outer_cluster_and_fork_coordinates_unique",
        "no_primary_support_violations",
    }
    if set(validity_checks) != expected_mechanical_checks:
        raise ValueError(
            "Instrument measurement mechanical-check set differs from the "
            "registered version-2 recomputation path."
        )
    effective_outer = _finite_number(
        instrument.get("minimum_effective_outer_sample_size"),
        "instrument_measurement.minimum_effective_outer_sample_size",
    )
    minimum_support_fraction = _finite_number(
        instrument.get("minimum_observed_support_fraction"),
        "instrument_measurement.minimum_observed_support_fraction",
    )
    minimum_posterior_mass = _finite_number(
        instrument.get("minimum_observed_posterior_mass"),
        "instrument_measurement.minimum_observed_posterior_mass",
    )
    if not 0.0 < minimum_support_fraction <= 1.0:
        raise ValueError("Instrument observed-support fraction must lie in (0, 1].")
    if not 0.0 < minimum_posterior_mass <= 1.0:
        raise ValueError("Instrument observed posterior mass must lie in (0, 1].")
    minimum_outer = max(
        int(preregistration.thresholds["kernel_min_outer_replicas"]),
        int(preregistration.payload["belief_kernel_audit"]["outer_replicas_M"]),
    )
    observed_outer = _optional_int(instrument.get("minimum_outer_replicas"))
    expected_instrument_mode = (
        "exact" if frozen_belief["exact_mode"] is True else "approximate_bounded"
    )
    within_frozen_bias_bounds = bool(
        posterior_bias <= float(frozen_belief["posterior_bias_bound"])
        and reset_bias <= float(frozen_belief["reset_bias_bound"])
    )
    instrument_valid = bool(
        all(validity_checks[name] is True for name in expected_mechanical_checks)
        and observed_outer is not None
        and observed_outer >= minimum_outer
        and effective_outer >= float(minimum_outer)
        and exact_or_approximate == expected_instrument_mode
        and within_frozen_bias_bounds
        and beta_lower > alpha_upper
    )
    if not instrument_valid:
        return PathCDecision(
            schema_version=PATH_C_DECISION_SCHEMA,
            software_conformant=True,
            instrument_valid=False,
            primary_effective=None,
            primary_estimate=None,
            primary_ci=None,
            secondary={},
            allowed_claim="diagnostic_only",
            status="INSTRUMENT_INVALID",
            requires_type_b_review=True,
        )
    if active_stage == "instrument_validity":
        return PathCDecision(
            schema_version=PATH_C_DECISION_SCHEMA,
            software_conformant=True,
            instrument_valid=True,
            primary_effective=None,
            primary_estimate=None,
            primary_ci=None,
            secondary={},
            allowed_claim="instrument_valid_only",
            status="INSTRUMENT_VALID",
            requires_type_b_review=True,
        )

    design_return_point_ledger = _validated_return_point_ledger(
        measurements.get("design_return_point_ledger"),
        preregistration,
        validated_policy_metrics,
        split_role="design",
    )
    primary_layout_by_mechanism = dict(
        preregistration.split_manifest.primary_layout_strata
    )
    primary_design_group_ids = tuple(sorted(
        group.group_id
        for group in preregistration.split_manifest.groups
        if preregistration.split_manifest.role_for(group.group_id) == "design"
        and group.layout_stratum == primary_layout_by_mechanism[group.mechanism]
    ))
    if not primary_design_group_ids:
        raise ValueError(
            "Split manifest has no design groups in the registered primary layout strata."
        )
    primary_design_group_set = set(primary_design_group_ids)
    design_curves = tuple(
        curve
        for curve in design_return_point_ledger.curves(
            split_role="design",
            policy_names=preregistration.strong_baselines,
        )
        if curve.split_group_id in primary_design_group_set
    )
    recomputed_selection = select_strongest_baseline_on_design(
        design_curves,
        candidate_order=preregistration.strong_baselines,
        split_role="design",
        return_point_ledger_sha256=design_return_point_ledger.sha256,
        design_split_group_ids=primary_design_group_ids,
    )
    if recomputed_selection.selection_rule != preregistration.payload["baselines"][
        "selection_rule"
    ]:
        raise ValueError("Frozen baseline selection rule differs from the evaluator.")
    expected_selection = baseline_selection_payload(recomputed_selection)
    selection = _mapping(
        measurements.get("baseline_selection"), "baseline_selection"
    )
    reported_selection = _measurement_core(
        selection,
        set(expected_selection),
        "baseline_selection",
    )
    if canonical_sha256(reported_selection) != canonical_sha256(expected_selection):
        raise ValueError(
            "Baseline selection does not match the raw design return-point ledger."
        )
    frozen_selection_sha256 = str(
        preregistration.payload["baselines"]["locked_selection_artifact_sha256"]
    )
    if recomputed_selection.sha256 != frozen_selection_sha256:
        raise ValueError(
            "Locked primary uses different baseline-selection content."
        )
    selected_baseline = recomputed_selection.selected_baseline
    if active_stage == "design_and_calibration_freeze":
        return PathCDecision(
            schema_version=PATH_C_DECISION_SCHEMA,
            software_conformant=True,
            instrument_valid=True,
            primary_effective=None,
            primary_estimate=None,
            primary_ci=None,
            secondary={},
            allowed_claim="instrument_valid_only",
            status="DESIGN_FROZEN_PRIMARY_NOT_RUN",
            requires_type_b_review=True,
        )
    primary = _mapping(measurements.get("primary_endpoint"), "primary_endpoint")
    if primary.get("schema_version") != "path_c_primary_endpoint_v1":
        raise ValueError("primary_endpoint has the wrong schema version.")
    primary_effective: bool | None = None
    primary_estimate: float | None = None
    primary_ci: tuple[float, float] | None = None
    if primary.get("available") is True:
        locked_return_point_ledger = _validated_return_point_ledger(
            measurements.get("locked_return_point_ledger"),
            preregistration,
            validated_policy_metrics,
            split_role="locked_audit",
        )
        if locked_return_point_ledger.factory_registry_sha256 != (
            design_return_point_ledger.factory_registry_sha256
        ):
            raise ValueError("Design and locked ledgers changed the factory registry.")
        if locked_return_point_ledger.environment_manifest_sha256 != (
            design_return_point_ledger.environment_manifest_sha256
        ):
            raise ValueError("Design and locked ledgers changed the environment manifest.")
        shared_policy_names = set(
            locked_return_point_ledger.policy_artifact_sha256_by_name
        ).intersection(design_return_point_ledger.policy_artifact_sha256_by_name)
        if any(
            locked_return_point_ledger.policy_artifact_sha256_by_name[name]
            != design_return_point_ledger.policy_artifact_sha256_by_name[name]
            for name in shared_policy_names
        ):
            raise ValueError("Design and locked ledgers changed a shared policy artifact.")
        locked_policy_names = {
            str(record.point.policy_name)
            for record in locked_return_point_ledger.points
        }
        if locked_policy_names != {
            PATH_C_PRIMARY_VARIANT,
            *preregistration.required_baselines,
        }:
            raise ValueError(
                "Locked return-point ledger must contain probing_ego and every frozen "
                "deployable baseline."
            )
        primary_layout_by_mechanism = dict(
            preregistration.split_manifest.primary_layout_strata
        )
        primary_locked_group_ids = tuple(sorted(
            group.group_id
            for group in preregistration.split_manifest.groups
            if preregistration.split_manifest.role_for(group.group_id)
            == "locked_audit"
            and group.layout_stratum
            == primary_layout_by_mechanism[group.mechanism]
        ))
        if not primary_locked_group_ids:
            raise ValueError(
                "Split manifest has no locked-audit groups in the primary layout strata."
            )
        locked_curves = tuple(
            curve
            for curve in locked_return_point_ledger.curves(
                split_role="locked_audit",
                policy_names=(PATH_C_PRIMARY_VARIANT, selected_baseline),
            )
            if curve.split_group_id in set(primary_locked_group_ids)
        )
        recomputed_primary = estimate_locked_primary_endpoint(
            [
                curve
                for curve in locked_curves
                if curve.policy_name == PATH_C_PRIMARY_VARIANT
            ],
            [
                curve
                for curve in locked_curves
                if curve.policy_name == selected_baseline
            ],
            baseline_selection=recomputed_selection,
            locked_return_point_ledger_sha256=locked_return_point_ledger.sha256,
            split_role="locked_audit",
            primary_split_group_ids=primary_locked_group_ids,
            preregistered_margin=float(
                preregistration.primary_endpoint["preregistered_margin"]
            ),
            confidence_level=float(
                preregistration.primary_endpoint["confidence_level"]
            ),
            bootstrap_iterations=int(
                preregistration.primary_endpoint["bootstrap_iterations"]
            ),
            seed=int(preregistration.payload["power_analysis"]["seed"]),
        )
        expected_primary = {
            **asdict(recomputed_primary),
            "bootstrap_iterations": int(
                preregistration.primary_endpoint["bootstrap_iterations"]
            ),
            "bootstrap_seed": int(preregistration.payload["power_analysis"]["seed"]),
        }
        reported_primary = _measurement_core(
            primary,
            set(expected_primary),
            "primary_endpoint",
        )
        if canonical_sha256(reported_primary) != canonical_sha256(expected_primary):
            raise ValueError(
                "Primary endpoint does not match the raw locked-audit return ledger "
                "and frozen estimator."
            )
        minimum_identity_clusters = int(
            preregistration.payload["split"]["minimum_groups_per_mechanism"]
        )
        if recomputed_primary.clusters < minimum_identity_clusters:
            raise ValueError(
                "Primary endpoint has fewer identity clusters than the frozen minimum."
            )
        primary_estimate = float(recomputed_primary.estimate)
        primary_ci = tuple(map(float, recomputed_primary.confidence_interval))
        primary_effective = recomputed_primary.primary_effective
    elif primary.get("available") is False:
        if measurements.get("locked_return_point_ledger") is not None:
            raise ValueError(
                "PRIMARY_NOT_RUN must not load or reference a locked-audit return ledger."
            )
        _measurement_core(
            primary,
            {"schema_version", "available"},
            "primary_endpoint",
        )
    else:
        raise ValueError("primary_endpoint.available must be boolean.")

    if primary_effective is None:
        return PathCDecision(
            schema_version=PATH_C_DECISION_SCHEMA,
            software_conformant=True,
            instrument_valid=True,
            primary_effective=None,
            primary_estimate=None,
            primary_ci=None,
            secondary={},
            allowed_claim="instrument_valid_only",
            status="PRIMARY_NOT_RUN",
            requires_type_b_review=True,
        )
    if not primary_effective:
        return PathCDecision(
            schema_version=PATH_C_DECISION_SCHEMA,
            software_conformant=True,
            instrument_valid=True,
            primary_effective=False,
            primary_estimate=primary_estimate,
            primary_ci=primary_ci,
            secondary={},
            allowed_claim="active_probing_budget_advantage_not_shown",
            status="PRIMARY_NOT_EFFECTIVE",
            requires_type_b_review=True,
        )

    if active_stage == "locked_primary_efficacy":
        return PathCDecision(
            schema_version=PATH_C_DECISION_SCHEMA,
            software_conformant=True,
            instrument_valid=True,
            primary_effective=True,
            primary_estimate=primary_estimate,
            primary_ci=primary_ci,
            secondary={},
            allowed_claim="primary_budget_advantage_pending_type_b_review",
            status="PRIMARY_EFFECTIVE_PENDING_TYPE_B_REVIEW",
            requires_type_b_review=True,
        )

    secondary = _mapping(
        measurements.get("secondary_profile"), "secondary_profile"
    )
    if secondary["schema_version"] != "path_c_secondary_profile_v1":
        raise ValueError("secondary_profile has the wrong schema version.")
    if secondary.get("available") is False:
        _require_exact_mapping_keys(
            secondary,
            {"schema_version", "available", "decision_eligible", "measurements"},
            "secondary_profile",
        )
        if secondary["decision_eligible"] is not False or secondary["measurements"] != {}:
            raise ValueError("An unavailable secondary profile must be empty.")
        secondary_measurements: Mapping[str, Any] = {}
    elif secondary.get("available") is True:
        _require_exact_mapping_keys(
            secondary,
            {
                "schema_version",
                "available",
                "decision_eligible",
                "source_measurements",
                "source_measurements_sha256",
                "measurements",
            },
            "secondary_profile",
        )
        if secondary["decision_eligible"] is not False:
            raise ValueError("Secondary profiles can never become decision gates.")
        recomputed_secondary = build_secondary_profile_v1(
            _mapping(
                secondary["source_measurements"],
                "secondary_profile.source_measurements",
            )
        )
        if canonical_sha256(secondary) != canonical_sha256(recomputed_secondary):
            raise ValueError(
                "Secondary profile was not recomputed from its content-addressed sources."
            )
        secondary_measurements = recomputed_secondary["measurements"]
    else:
        raise ValueError("secondary_profile.available must be boolean.")
    return PathCDecision(
        schema_version=PATH_C_DECISION_SCHEMA,
        software_conformant=True,
        instrument_valid=True,
        primary_effective=True,
        primary_estimate=primary_estimate,
        primary_ci=primary_ci,
        secondary=secondary_measurements,
        allowed_claim="primary_budget_advantage_pending_type_b_review",
        status="PRIMARY_EFFECTIVE_PENDING_TYPE_B_REVIEW",
        requires_type_b_review=True,
    )


def select_value_necessary_coordinates(
    coordinate_measurements: Mapping[str, Mapping[str, Any]],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, Any]:
    threshold = float(preregistration.thresholds["value_ablation_min_degradation"])
    retained = []
    rejected = []
    details = {}
    for name, item in sorted(coordinate_measurements.items()):
        raw_metrics = item.get("metrics")
        metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {"aggregate": item}
        metric_details = {}
        metric_passes = []
        for metric_name, metric_item in sorted(metrics.items()):
            if not isinstance(metric_item, Mapping):
                raise ValueError(f"coordinate {name}.{metric_name} must be a mapping.")
            value = _finite_number(
                metric_item.get("value"),
                f"coordinate {name}.{metric_name}.value",
            )
            ci_lo = _finite_number(
                metric_item.get("ci_lo", _nested_ci(metric_item, "lo")),
                f"coordinate {name}.{metric_name}.ci_lo",
            )
            passed = value >= threshold and ci_lo > threshold
            metric_passes.append(passed)
            metric_details[str(metric_name)] = {
                "value": value,
                "ci_lo": ci_lo,
                "threshold": threshold,
                "necessary": passed,
            }
        necessary = any(metric_passes)
        details[name] = {"metrics": metric_details, "necessary": necessary}
        (retained if necessary else rejected).append(name)
    return {
        "retained_coordinates": retained,
        "rejected_coordinates": rejected,
        "retained_coordinate_count": len(retained),
        "details": details,
    }


def fingerprint_admission_measurement(
    *,
    control_kind: str,
    mechanism: Sequence[Any],
    fingerprint_id: Sequence[Any],
    visibility_ci_lo: float,
    chance_accuracy: float,
    value_null_ci: Sequence[float],
    joint_rv_null_ci: Sequence[float],
    normalized_advantage_null_ci: Sequence[float],
    preregistration: FrozenPathCPreregistration,
) -> dict[str, Any]:
    mechanisms = np.asarray(mechanism).astype(str)
    fingerprints = np.asarray(fingerprint_id).astype(str)
    if mechanisms.shape[0] != fingerprints.shape[0]:
        raise ValueError("Fingerprint mechanism and id arrays must have the same length.")
    expected_kind = str(preregistration.payload["fingerprint"]["positive_control_kind"])
    mechanism_values = sorted(np.unique(mechanisms).tolist())
    fingerprint_values = sorted(np.unique(fingerprints).tolist())
    counterbalanced = bool(
        len(mechanism_values) >= 3
        and len(fingerprint_values) >= 2
        and all(
            set(fingerprints[mechanisms == mechanism_value].tolist())
            == set(fingerprint_values)
            for mechanism_value in mechanism_values
        )
    )
    thresholds = preregistration.thresholds
    visibility = float(visibility_ci_lo)
    chance = float(chance_accuracy)
    value_interval = _interval(value_null_ci)
    rv_interval = _interval(joint_rv_null_ci)
    normalized_advantage_interval = _interval(normalized_advantage_null_ci)
    admitted = bool(
        control_kind == expected_kind
        and counterbalanced
        and visibility > chance
        and _interval_inside(
            value_interval,
            -float(thresholds["nuisance_value_gain_equivalence"]),
            float(thresholds["nuisance_value_gain_equivalence"]),
        )
        and _interval_inside(
            rv_interval,
            -float(thresholds["null_equivalence_gain"]),
            float(thresholds["null_equivalence_gain"]),
        )
        and _interval_inside(
            normalized_advantage_interval,
            -float(thresholds["nuisance_value_gain_equivalence"]),
            float(thresholds["nuisance_value_gain_equivalence"]),
        )
    )
    return {
        "fingerprint_control_kind": str(control_kind),
        "expected_positive_control_kind": expected_kind,
        "fingerprint_cross_mechanism_counterbalanced": counterbalanced,
        "fingerprint_visibility_ci_lo": visibility,
        "fingerprint_chance": chance,
        "fingerprint_value_null_ci": value_interval,
        "fingerprint_joint_rv_null_ci": rv_interval,
        "fingerprint_normalized_advantage_null_ci": normalized_advantage_interval,
        "admitted": admitted,
        "available_for_synthetic_power": admitted,
    }


def build_path_c_readout_features(
    public_context: np.ndarray,
    representations: Mapping[str, np.ndarray],
    *,
    surface_identity: Sequence[Any] | None = None,
) -> dict[str, np.ndarray]:
    """Construct fair readout inputs for main and hard-baseline variants."""
    context = np.asarray(public_context, dtype=np.float32)
    if context.ndim != 2:
        raise ValueError("Path C public context must be a matrix.")
    features: dict[str, np.ndarray] = {
        "base_only": context.copy(),
        "no_admission": context.copy(),
    }
    for variant, raw in representations.items():
        representation = np.asarray(raw, dtype=np.float32)
        if representation.ndim != 2 or representation.shape[0] != context.shape[0]:
            raise ValueError(f"Path C representation {variant!r} is not row-aligned.")
        if variant in {"base_only", "no_admission"}:
            continue
        if variant == "partner_id":
            if surface_identity is None:
                raise ValueError("Partner-ID baseline requires explicit surface identity.")
            identities = np.asarray(surface_identity).astype(str)
            if identities.shape[0] != context.shape[0]:
                raise ValueError("Partner-ID surface identity is not row-aligned.")
            vocab, labels = np.unique(identities, return_inverse=True)
            identity_onehot = np.zeros((context.shape[0], len(vocab)), dtype=np.float32)
            identity_onehot[np.arange(context.shape[0]), labels] = 1.0
            features[variant] = np.concatenate([context, identity_onehot], axis=1)
        else:
            features[variant] = np.concatenate([context, representation], axis=1)
    return features


def require_formal_benchmark_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("evaluation_kind") != "formal_benchmark":
        raise ValueError("Data-collection output cannot be used as formal benchmark return.")
    if artifact.get("active_probe_collection") is not False:
        raise ValueError("Formal benchmark return must disable active probe collection.")
    if artifact.get("collection_role") != "formal_benchmark":
        raise ValueError("Formal benchmark return has an invalid collection role.")
    if artifact.get("benchmark_return_eligible") is not True:
        raise ValueError("Artifact is not eligible for benchmark-return aggregation.")


def probe_support_measurement(
    selected: Sequence[Any],
    opportunity: Sequence[Any],
    public_context: Sequence[Any],
    action_id: Sequence[Any],
) -> dict[str, Any]:
    selected_arr = np.asarray(selected).astype(bool)
    opportunity_arr = np.asarray(opportunity).astype(bool)
    contexts = np.asarray(public_context).astype(str)
    actions = np.asarray(action_id).astype(str)
    if len({selected_arr.shape[0], opportunity_arr.shape[0], contexts.shape[0], actions.shape[0]}) != 1:
        raise ValueError("Probe support arrays must have the same length.")
    if np.any(selected_arr & ~opportunity_arr):
        raise ValueError("A selected probe must also be a recorded opportunity.")
    opportunity_contexts = set(contexts[opportunity_arr].tolist())
    selected_contexts = set(contexts[selected_arr].tolist())
    opportunity_actions = set(actions[opportunity_arr].tolist())
    selected_actions = set(actions[selected_arr].tolist())
    return {
        "selected": int(np.count_nonzero(selected_arr)),
        "opportunities": int(np.count_nonzero(opportunity_arr)),
        "context_coverage": float(
            len(selected_contexts) / max(1, len(opportunity_contexts))
        ),
        "action_coverage": float(
            len(selected_actions) / max(1, len(opportunity_actions))
        ),
        "public_context_count": len(opportunity_contexts),
        "eligible_action_count": len(opportunity_actions),
    }


def make_run_id(
    *,
    checkpoint_sha256: str,
    resolved_path_c_sha256: str,
    collection_variant: str,
    partner: str,
    ego: str,
    base_seed: int,
) -> str:
    return canonical_sha256({
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_path_c_sha256": resolved_path_c_sha256,
        "collection_variant": collection_variant,
        "partner": partner,
        "ego": ego,
        "base_seed": int(base_seed),
    })[:24]


def make_episode_uid(run_id: str, base_seed: int, episode_id: int) -> str:
    return f"{run_id}:{int(base_seed)}:{int(episode_id)}"


def assign_group_disjoint_folds(
    surface_identity: Sequence[Any],
    seed_group: Sequence[Any],
    layout_style: Sequence[Any],
    *,
    n_folds: int,
) -> np.ndarray:
    """Assign connected surface/seed/layout groups to deterministic folds."""
    surfaces, seeds, layouts = (
        np.asarray(values).astype(str)
        for values in (surface_identity, seed_group, layout_style)
    )
    if len({surfaces.shape[0], seeds.shape[0], layouts.shape[0]}) != 1:
        raise ValueError("Cross-identity grouping arrays must have the same length.")
    if int(n_folds) <= 1:
        raise ValueError("Cross-identity split requires at least two folds.")
    n_rows = int(surfaces.shape[0])
    parent = list(range(n_rows))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for values in (surfaces, seeds, layouts):
        first: dict[str, int] = {}
        for index, value in enumerate(values.tolist()):
            if value in first:
                union(first[value], index)
            else:
                first[value] = index

    components: dict[int, list[int]] = {}
    for index in range(n_rows):
        components.setdefault(find(index), []).append(index)
    if len(components) < int(n_folds):
        raise ValueError(
            "Cross-identity data has fewer disjoint surface/seed/layout components than folds."
        )
    ordered = sorted(
        components.values(),
        key=lambda rows: (
            -len(rows),
            canonical_sha256({
                "surface": sorted(set(surfaces[rows].tolist())),
                "seed": sorted(set(seeds[rows].tolist())),
                "layout": sorted(set(layouts[rows].tolist())),
            }),
        ),
    )
    fold_sizes = [0] * int(n_folds)
    assignments = np.empty(n_rows, dtype=np.int16)
    for rows in ordered:
        fold = min(range(int(n_folds)), key=lambda value: (fold_sizes[value], value))
        assignments[rows] = fold
        fold_sizes[fold] += len(rows)
    return assignments


def validate_cross_identity_folds(
    fold_ids: Sequence[Any],
    surface_identity: Sequence[Any],
    seed_group: Sequence[Any],
    layout_style: Sequence[Any],
    mechanism: Sequence[Any],
) -> dict[str, Any]:
    arrays = [np.asarray(values).astype(str) for values in (
        fold_ids, surface_identity, seed_group, layout_style, mechanism
    )]
    if len({arr.shape[0] for arr in arrays}) != 1:
        raise ValueError("Cross-identity split arrays must have the same length.")
    folds, surfaces, seeds, layouts, mechanisms = arrays
    reports = []
    valid = True
    for fold in sorted(np.unique(folds).tolist()):
        test = folds == fold
        train = ~test
        overlap = {
            "surface_identity": sorted(set(surfaces[train]).intersection(surfaces[test])),
            "seed_group": sorted(set(seeds[train]).intersection(seeds[test])),
            "layout_style": sorted(set(layouts[train]).intersection(layouts[test])),
        }
        mechanism_overlap = sorted(set(mechanisms[train]).intersection(mechanisms[test]))
        fold_valid = not any(overlap.values()) and bool(mechanism_overlap)
        valid = valid and fold_valid
        reports.append({
            "fold": str(fold),
            "overlap": overlap,
            "mechanism_overlap": mechanism_overlap,
            "valid": fold_valid,
        })
    return {"valid": valid, "folds": reports}


def empirical_kernel_distance_audit(
    joint_rv: Sequence[Any],
    mechanism: Sequence[Any],
    surface_identity: Sequence[Any],
    public_context: Sequence[Any],
    episode_uid: Sequence[Any],
    preregistration: FrozenPathCPreregistration,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Describe an ecological fallback without treating it as an instrument.

    Version 3 accepts only the finite categorical outer-replica estimator in
    path_c_belief_audit for instrument validity. This compatibility function
    checks row alignment and records why an older distributional-cell table is
    secondary-only. It never emits an instrument-validity decision.
    """

    del preregistration, seed
    response = np.asarray(joint_rv)
    if response.ndim == 1:
        response = response.reshape(-1, 1)
    arrays = [
        np.asarray(values)
        for values in (mechanism, surface_identity, public_context, episode_uid)
    ]
    if any(values.shape[0] != response.shape[0] for values in arrays):
        raise ValueError("Ecological diagnostic arrays must have equal row counts.")
    return {
        "available": bool(response.shape[0]),
        "theorem_eligible": False,
        "decision_eligible": False,
        "estimator_kind": "distributional_cell_fallback",
        "reason": "requires_outer_full_state_posterior_replica_table",
        "rows": int(response.shape[0]),
        "response_columns": int(response.shape[1]),
        "mechanism_count": int(len(np.unique(arrays[0].astype(str)))),
        "identity_count": int(len(np.unique(arrays[1].astype(str)))),
        "outer_sampling_unit": None,
    }

def _unavailable_kernel_audit(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "theorem_eligible": False,
        "estimator_kind": "distributional_cell_fallback",
        "reason": reason,
        "same_mechanism_quantifier": "max_over_identity_context",
        "different_mechanism_quantifier": "min_pair_max_witness",
        "bootstrap_unit": "episode_uid",
        "simultaneous_bootstrap": True,
        "context_strata_source": "preregistered_semantic",
        "all_required_strata_available": False,
    }


def _categorical_total_variation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    vocab = sorted(set(left.astype(str).tolist()).union(right.astype(str).tolist()))
    left_counts = np.asarray([np.count_nonzero(left == value) for value in vocab], dtype=np.float64)
    right_counts = np.asarray([np.count_nonzero(right == value) for value in vocab], dtype=np.float64)
    left_probs = left_counts / left_counts.sum()
    right_probs = right_counts / right_counts.sum()
    return float(0.5 * np.abs(left_probs - right_probs).sum())


def _phase_rules(
    measurements: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
    *,
    require_contract: bool,
) -> dict[str, Any]:
    del measurements, preregistration, require_contract
    raise ValueError(
        "The version-2 aggregate gate was removed. Use evaluate_path_c_decision."
    )


def _strict_dataset_string_tuple(value: Any, name: str) -> tuple[str, ...]:
    items = _sequence(value, name)
    if not items:
        raise ValueError(f"{name} must be a non-empty list.")
    if any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in items
    ):
        raise ValueError(
            f"{name} must contain only non-empty, whitespace-trimmed strings."
        )
    values = tuple(items)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} cannot contain duplicates.")
    return values


def _validate_v3_dataset_shards(
    datasets: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
    base_dir: Path,
    *,
    required_roles: set[str] | None = None,
    allowed_roles: set[str] | None = None,
) -> None:
    if not datasets:
        raise ValueError("Path C manifest needs append-only dataset shards.")
    required_role_set = (
        {"train", "design", "calibration", "locked_audit"}
        if required_roles is None
        else set(required_roles)
    )
    allowed_role_set = (
        {"train", "design", "calibration", "locked_audit"}
        if allowed_roles is None
        else set(allowed_roles)
    )
    if not required_role_set.issubset(allowed_role_set):
        raise ValueError("Required dataset roles must be a subset of allowed roles.")
    expected_keys = {
        "schema_version",
        "collection_role",
        "storage",
        "format",
        "split_manifest_sha256",
        "split_group_ids",
        "mechanisms",
        "style_groups",
        "response_vocabulary_sha256",
        "evidence_spec_sha256",
        "effective_episodes",
        "effective_transitions",
        "audit_snapshot_manifest",
        "semantic_bindings",
        "chunks",
    }
    seen_chunks: set[Path] = set()
    seen_episode_uids: set[str] = set()
    observed_split_group_ids: set[str] = set()
    observed_roles: set[str] = set()
    split_groups = {
        group.group_id: group for group in preregistration.split_manifest.groups
    }
    for name, raw_item in datasets.items():
        item = _mapping(raw_item, f"dataset_shards.{name}")
        _require_exact_mapping_keys(item, expected_keys, f"dataset_shards.{name}")
        if item["schema_version"] != "path_c_dataset_shards_v1":
            raise ValueError(f"dataset_shards.{name} has the wrong schema version.")
        role = str(item["collection_role"])
        if role not in {"train", "design", "calibration", "locked_audit"}:
            raise ValueError(f"dataset_shards.{name} has an invalid collection role.")
        if role not in allowed_role_set:
            raise ValueError(
                f"dataset_shards.{name} exposes role {role!r} before its active stage."
            )
        observed_roles.add(role)
        if item["storage"] != "content_addressed_append_only_shards":
            raise ValueError(f"dataset_shards.{name} is not append-only/content-addressed.")
        if str(item["format"]).lower() != "parquet":
            raise ValueError(
                f"dataset_shards.{name} must use Parquet; Arrow and Zarr readers "
                "are not part of the frozen version-3 validator."
            )
        if item["split_manifest_sha256"] != preregistration.semantic_bindings["split_manifest_sha256"]:
            raise ValueError(f"dataset_shards.{name} has the wrong split manifest hash.")
        if item["response_vocabulary_sha256"] != preregistration.response_vocabulary_sha256:
            raise ValueError(f"dataset_shards.{name} has the wrong response vocabulary hash.")
        if item["evidence_spec_sha256"] != preregistration.evidence_spec_sha256:
            raise ValueError(f"dataset_shards.{name} has the wrong evidence-spec hash.")
        if item["semantic_bindings"] != preregistration.semantic_bindings:
            raise ValueError(f"dataset_shards.{name} changes frozen semantic bindings.")
        declared_group_ids = _strict_dataset_string_tuple(
            item["split_group_ids"], f"dataset_shards.{name}.split_group_ids"
        )
        declared_mechanisms = _strict_dataset_string_tuple(
            item["mechanisms"], f"dataset_shards.{name}.mechanisms"
        )
        declared_style_groups = _strict_dataset_string_tuple(
            item["style_groups"], f"dataset_shards.{name}.style_groups"
        )
        unknown_group_ids = sorted(set(declared_group_ids).difference(split_groups))
        if unknown_group_ids:
            raise ValueError(
                f"dataset_shards.{name} declares unknown split group(s): "
                + ", ".join(unknown_group_ids)
            )
        wrong_role_groups = sorted(
            group_id
            for group_id in declared_group_ids
            if preregistration.split_manifest.role_for(group_id) != role
        )
        if wrong_role_groups:
            raise ValueError(
                f"dataset_shards.{name} declares group(s) assigned to another role: "
                + ", ".join(wrong_role_groups)
            )
        expected_role_group_ids = {
            group.group_id
            for group in preregistration.split_manifest.groups_for_role(role)
        }
        if set(declared_group_ids) != expected_role_group_ids:
            raise ValueError(
                f"dataset_shards.{name} must declare every frozen group for role "
                f"{role!r}; missing="
                f"{sorted(expected_role_group_ids.difference(declared_group_ids))}, "
                f"unknown="
                f"{sorted(set(declared_group_ids).difference(expected_role_group_ids))}."
            )
        expected_mechanisms = {
            split_groups[group_id].mechanism for group_id in declared_group_ids
        }
        expected_styles = {
            split_groups[group_id].style_group for group_id in declared_group_ids
        }
        if set(declared_mechanisms) != expected_mechanisms:
            raise ValueError(
                f"dataset_shards.{name} mechanism declarations do not match its "
                "frozen split groups."
            )
        if set(declared_style_groups) != expected_styles:
            raise ValueError(
                f"dataset_shards.{name} style-group declarations do not match its "
                "frozen split groups."
            )
        episodes = _optional_int(item["effective_episodes"])
        transitions = _optional_int(item["effective_transitions"])
        if episodes is None or transitions is None or episodes <= 0 or transitions <= 0:
            raise ValueError(f"dataset_shards.{name} has invalid effective counts.")
        if role in {"train", "locked_audit"} and (
            episodes < int(preregistration.payload["budget"]["effective_episode_floor"])
            or transitions < int(preregistration.payload["budget"]["effective_transition_floor"])
        ):
            raise ValueError(f"dataset_shards.{name} is below the frozen sufficiency floor.")
        snapshot_reference = item["audit_snapshot_manifest"]
        if role == "locked_audit":
            snapshot_ref = _mapping(
                snapshot_reference,
                f"dataset_shards.{name}.audit_snapshot_manifest",
            )
            _require_exact_mapping_keys(
                snapshot_ref,
                {"path", "sha256"},
                f"dataset_shards.{name}.audit_snapshot_manifest",
            )
            _validated_file_reference(
                snapshot_ref,
                base_dir,
                f"dataset_shards.{name}.audit_snapshot_manifest",
            )
        elif snapshot_reference is not None:
            raise ValueError(
                f"dataset_shards.{name}.audit_snapshot_manifest must be null "
                "outside locked_audit."
            )
        chunks = _sequence(item["chunks"], f"dataset_shards.{name}.chunks")
        if not chunks:
            raise ValueError(f"dataset_shards.{name} contains no chunks.")
        total_rows = 0
        observed_transitions = 0
        observed_episode_uids: set[str] = set()
        observed_group_ids: set[str] = set()
        observed_mechanisms: set[str] = set()
        observed_style_groups: set[str] = set()
        observed_group_seed_pairs: set[tuple[str, int]] = set()
        episode_group: dict[str, str] = {}
        episode_seed_identity: dict[str, tuple[int, int]] = {}
        episode_seed_by_execution_seed: dict[int, int] = {}
        validated_group_seed_pairs: set[tuple[str, int]] = set()
        for index, raw_ref in enumerate(chunks):
            ref = _mapping(raw_ref, f"dataset_shards.{name}.chunks[{index}]")
            _require_exact_mapping_keys(
                ref,
                {"path", "sha256", "rows"},
                f"dataset_shards.{name}.chunks[{index}]",
            )
            path = _validated_file_reference(
                ref, base_dir, f"dataset_shards.{name}.chunks[{index}]"
            )
            if path in seen_chunks:
                raise ValueError(f"Dataset chunk is reused across manifests: {path}")
            seen_chunks.add(path)
            rows = _optional_int(ref["rows"])
            if rows is None or rows <= 0:
                raise ValueError(f"dataset_shards.{name} chunk rows must be positive.")
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError(
                    "Version-3 Path C dataset validation requires pyarrow."
                ) from exc
            required_columns = (
                "episode_uid",
                "option_transition",
                "split_group_id",
                "mechanism",
                "style_group",
                "collection_role",
                "surface_identity_key",
                "seed_group",
                "seed",
                "episode_seed",
                "execution_seed",
                "layout_style",
                "probe_selected",
                "probe_cost_per_use",
                "probe_realized_cost",
            )
            try:
                table = pq.read_table(path, columns=list(required_columns))
            except Exception as exc:
                raise ValueError(
                    f"dataset_shards.{name}.chunks[{index}] is not a readable "
                    "Parquet shard with the required columns."
                ) from exc
            if int(table.num_rows) != rows:
                raise ValueError(
                    f"dataset_shards.{name}.chunks[{index}] declared rows do not "
                    "match the Parquet row count."
                )
            required_arrow_types = {
                "seed": pa.uint64(),
                "episode_seed": pa.uint64(),
                "execution_seed": pa.uint32(),
            }
            for column_name, expected_type in required_arrow_types.items():
                observed_type = table.schema.field(column_name).type
                if observed_type != expected_type:
                    raise ValueError(
                        f"dataset_shards.{name}.chunks[{index}]."
                        f"{column_name} must use Arrow {expected_type}; observed "
                        f"{observed_type}."
                    )
            columns = table.to_pydict()
            for row_index in range(rows):
                row_name = (
                    f"dataset_shards.{name}.chunks[{index}].row[{row_index}]"
                )
                string_values: dict[str, str] = {}
                for column_name in (
                    "episode_uid",
                    "split_group_id",
                    "mechanism",
                    "style_group",
                    "collection_role",
                    "surface_identity_key",
                    "seed_group",
                    "layout_style",
                ):
                    value = columns[column_name][row_index]
                    if (
                        not isinstance(value, str)
                        or not value
                        or value != value.strip()
                    ):
                        raise ValueError(
                            f"{row_name}.{column_name} must be a non-empty, "
                            "whitespace-trimmed string."
                        )
                    string_values[column_name] = value
                episode_uid = string_values["episode_uid"]
                group_id = string_values["split_group_id"]
                mechanism = string_values["mechanism"]
                style_group = string_values["style_group"]
                row_role = string_values["collection_role"]
                if group_id not in split_groups:
                    raise ValueError(f"{row_name} uses an unknown split group.")
                group = split_groups[group_id]
                if preregistration.split_manifest.role_for(group_id) != role:
                    raise ValueError(
                        f"{row_name} split group belongs to another role."
                    )
                if row_role != role:
                    raise ValueError(
                        f"{row_name}.collection_role differs from the manifest role."
                    )
                if mechanism != group.mechanism:
                    raise ValueError(
                        f"{row_name}.mechanism differs from the frozen split group."
                    )
                if style_group != group.style_group:
                    raise ValueError(
                        f"{row_name}.style_group differs from the frozen split group."
                    )
                if string_values["surface_identity_key"] != group.identity_group:
                    raise ValueError(
                        f"{row_name}.surface_identity_key differs from the frozen "
                        "split group."
                    )
                if string_values["seed_group"] != group.seed_group:
                    raise ValueError(
                        f"{row_name}.seed_group differs from the frozen split group."
                    )
                numeric_seed = columns["seed"][row_index]
                if isinstance(numeric_seed, bool) or not isinstance(
                    numeric_seed, (int, np.integer)
                ):
                    raise TypeError(f"{row_name}.seed must be an integer.")
                group_seed_pair = (group_id, int(numeric_seed))
                if group_seed_pair not in validated_group_seed_pairs:
                    preregistration.split_manifest.validate_numeric_seed(
                        *group_seed_pair,
                    )
                    validated_group_seed_pairs.add(group_seed_pair)
                observed_group_seed_pairs.add(group_seed_pair)
                episode_seed = canonical_uint64_seed(
                    columns["episode_seed"][row_index],
                    name=f"{row_name}.episode_seed",
                )
                execution_seed = columns["execution_seed"][row_index]
                if isinstance(execution_seed, bool) or not isinstance(
                    execution_seed, (int, np.integer)
                ):
                    raise TypeError(f"{row_name}.execution_seed must be an integer.")
                expected_execution_seed = derive_ocv2_execution_seed(episode_seed)
                if int(execution_seed) != expected_execution_seed:
                    raise ValueError(
                        f"{row_name}.execution_seed does not match its canonical "
                        "episode seed."
                    )
                previous_episode_seed = episode_seed_by_execution_seed.setdefault(
                    int(execution_seed),
                    episode_seed,
                )
                if previous_episode_seed != episode_seed:
                    raise ValueError(
                        "Dataset contains an OCV2 execution-seed collision between "
                        "different canonical episode seeds."
                    )
                previous_seed_identity = episode_seed_identity.setdefault(
                    episode_uid,
                    (int(numeric_seed), episode_seed),
                )
                if previous_seed_identity != (int(numeric_seed), episode_seed):
                    raise ValueError(
                        f"{row_name}.episode_uid cannot change its numeric or "
                        "episode seed."
                    )
                if string_values["layout_style"] != group.layout_group:
                    raise ValueError(
                        f"{row_name}.layout_style differs from the frozen split group."
                    )
                selected_raw = columns["probe_selected"][row_index]
                if isinstance(selected_raw, bool):
                    probe_selected = bool(selected_raw)
                elif isinstance(selected_raw, (int, np.integer)) and int(
                    selected_raw
                ) in {0, 1}:
                    probe_selected = bool(selected_raw)
                else:
                    raise ValueError(
                        f"{row_name}.probe_selected must be boolean or 0/1."
                    )
                row_probe_cost = _finite_number(
                    columns["probe_cost_per_use"][row_index],
                    f"{row_name}.probe_cost_per_use",
                )
                row_realized_cost = _finite_number(
                    columns["probe_realized_cost"][row_index],
                    f"{row_name}.probe_realized_cost",
                )
                frozen_probe_cost = float(
                    preregistration.primary_endpoint["probe_cost_per_use"]
                )
                if row_probe_cost != frozen_probe_cost:
                    raise ValueError(
                        f"{row_name}.probe_cost_per_use differs from the frozen cost."
                    )
                expected_realized_cost = frozen_probe_cost if probe_selected else 0.0
                if row_realized_cost != expected_realized_cost:
                    raise ValueError(
                        f"{row_name}.probe_realized_cost must charge the frozen cost "
                        "exactly once for selected probes and zero otherwise."
                    )
                prior_group = episode_group.setdefault(episode_uid, group_id)
                if prior_group != group_id:
                    raise ValueError(
                        f"{row_name}.episode_uid crosses frozen split groups."
                    )
                transition = columns["option_transition"][row_index]
                if isinstance(transition, bool):
                    transition_flag = transition
                elif isinstance(transition, (int, np.integer)) and int(
                    transition
                ) in {0, 1}:
                    transition_flag = bool(transition)
                else:
                    raise ValueError(
                        f"{row_name}.option_transition must be boolean or 0/1."
                    )
                if not transition_flag:
                    raise ValueError(
                        f"{row_name}.option_transition must be true for every "
                        "version-3 decision-point row."
                    )
                observed_transitions += int(transition_flag)
                observed_episode_uids.add(episode_uid)
                observed_group_ids.add(group_id)
                observed_mechanisms.add(mechanism)
                observed_style_groups.add(style_group)
            total_rows += rows
        if total_rows <= 0:
            raise ValueError(f"dataset_shards.{name} contains no Parquet rows.")
        if observed_transitions != transitions:
            raise ValueError(
                f"dataset_shards.{name} effective transitions do not equal the "
                "transition flags read from Parquet."
            )
        if len(observed_episode_uids) != episodes:
            raise ValueError(
                f"dataset_shards.{name} effective episodes do not equal the unique "
                "episode_uid values read from Parquet."
            )
        reused_episode_uids = sorted(observed_episode_uids.intersection(seen_episode_uids))
        if reused_episode_uids:
            raise ValueError(
                "Episode UID is reused across dataset manifests: "
                + ", ".join(reused_episode_uids[:5])
            )
        seen_episode_uids.update(observed_episode_uids)
        if observed_group_ids != set(declared_group_ids):
            raise ValueError(
                f"dataset_shards.{name} split-group declarations differ from its rows."
            )
        if observed_mechanisms != set(declared_mechanisms):
            raise ValueError(
                f"dataset_shards.{name} mechanism declarations differ from its rows."
            )
        if observed_style_groups != set(declared_style_groups):
            raise ValueError(
                f"dataset_shards.{name} style-group declarations differ from its rows."
            )
        expected_group_seed_pairs = {
            (group_id, int(seed))
            for group_id in expected_role_group_ids
            for seed in preregistration.split_manifest.numeric_seeds_for_group(
                group_id
            )
        }
        if observed_group_seed_pairs != expected_group_seed_pairs:
            raise ValueError(
                f"dataset_shards.{name} group and seed schedule is incomplete; "
                f"missing="
                f"{sorted(expected_group_seed_pairs.difference(observed_group_seed_pairs))}, "
                f"unknown="
                f"{sorted(observed_group_seed_pairs.difference(expected_group_seed_pairs))}."
            )
        observed_split_group_ids.update(observed_group_ids)
    missing_roles = sorted(required_role_set.difference(observed_roles))
    if missing_roles:
        raise ValueError("Dataset shards omit split role(s): " + ", ".join(missing_roles))
    required_groups = {
        group_id
        for group_id in split_groups
        if preregistration.split_manifest.role_for(group_id) in required_role_set
    }
    missing_groups = sorted(required_groups.difference(observed_split_group_ids))
    if missing_groups:
        raise ValueError(
            "Dataset shards omit frozen split group(s): " + ", ".join(missing_groups)
        )


def _validate_v3_policy_artifacts(
    policies: Mapping[str, Any],
    preregistration: FrozenPathCPreregistration,
    base_dir: Path,
) -> None:
    required = set(
        map(
            str,
            preregistration.payload["budget"]["required_policy_keys"],
        )
    )
    missing = sorted(required.difference(policies))
    if missing:
        raise ValueError("Policy artifacts omit required policy/policies: " + ", ".join(missing))
    expected_keys = {
        "schema_version",
        "policy_name",
        "artifact",
        "oracle_baseline",
        "input_contract",
        "acting_protocol_version",
        "training_environment_steps",
        "gradient_updates",
        "evaluation_environment_step_limit",
        "evaluation_schedule_id",
        "probe_budget_grid_sha256",
        "action_support_sha256",
        "probe_cost_per_use",
        "trainable_parameters",
        "training_flops",
        "wall_clock_seconds",
        "inference_latency_ms",
        "semantic_bindings",
    }
    matched_fields = (
        "training_environment_steps",
        "gradient_updates",
        "evaluation_environment_step_limit",
        "evaluation_schedule_id",
        "probe_budget_grid_sha256",
        "action_support_sha256",
        "probe_cost_per_use",
    )
    reference: dict[str, Any] | None = None
    for name in sorted(required):
        item = _mapping(policies[name], f"policy_artifacts.{name}")
        _require_exact_mapping_keys(item, expected_keys, f"policy_artifacts.{name}")
        if item["schema_version"] != "path_c_policy_artifact_v1":
            raise ValueError(f"policy_artifacts.{name} has the wrong schema version.")
        if str(item["policy_name"]) != name:
            raise ValueError(f"policy_artifacts.{name} has a mismatched policy_name.")
        _validated_file_reference(
            _mapping(item["artifact"], f"policy_artifacts.{name}.artifact"),
            base_dir,
            f"policy_artifacts.{name}.artifact",
        )
        if item["acting_protocol_version"] != "path_c_acting_policy_v1":
            raise ValueError(f"policy_artifacts.{name} does not implement the acting protocol.")
        expected_contract = _EXPECTED_VARIANT_INPUT_CONTRACTS.get(name)
        if expected_contract is not None and item["input_contract"] != expected_contract:
            raise ValueError(f"policy_artifacts.{name} has the wrong evidence contract.")
        expected_oracle = name == "direct_information"
        if bool(item["oracle_baseline"]) != expected_oracle:
            raise ValueError(f"policy_artifacts.{name} has the wrong oracle/deployable label.")
        if item["semantic_bindings"] != preregistration.semantic_bindings:
            raise ValueError(f"policy_artifacts.{name} changes frozen semantic bindings.")
        for key in (
            "training_environment_steps",
            "gradient_updates",
            "evaluation_environment_step_limit",
            "trainable_parameters",
        ):
            value = _optional_int(item[key])
            if value is None or value < 0:
                raise ValueError(f"policy_artifacts.{name}.{key} must be non-negative.")
        for key in ("training_flops", "wall_clock_seconds", "inference_latency_ms"):
            value = _finite_number(item[key], f"policy_artifacts.{name}.{key}")
            if value < 0.0:
                raise ValueError(f"policy_artifacts.{name}.{key} must be non-negative.")
        probe_cost = _finite_number(
            item["probe_cost_per_use"],
            f"policy_artifacts.{name}.probe_cost_per_use",
        )
        if probe_cost < 0.0:
            raise ValueError(
                f"policy_artifacts.{name}.probe_cost_per_use must be non-negative."
            )
        if probe_cost != float(preregistration.primary_endpoint["probe_cost_per_use"]):
            raise ValueError(
                f"policy_artifacts.{name}.probe_cost_per_use differs from the "
                "frozen primary cost."
            )
        frozen_budget = preregistration.payload["budget"]
        expected_budget_fields = {
            "training_environment_steps": int(
                frozen_budget["training_environment_steps"]
            ),
            "gradient_updates": int(frozen_budget["training_gradient_updates"]),
            "evaluation_environment_step_limit": int(
                frozen_budget["evaluation_environment_step_limit"]
            ),
            "evaluation_schedule_id": str(frozen_budget["evaluation_schedule_id"]),
            "probe_budget_grid_sha256": str(frozen_budget["probe_budget_grid_sha256"]),
            "action_support_sha256": str(frozen_budget["action_support_sha256"]),
        }
        changed_budget_fields = [
            key for key, expected in expected_budget_fields.items()
            if item[key] != expected
        ]
        if changed_budget_fields:
            raise ValueError(
                f"policy_artifacts.{name} changes frozen budget field(s): "
                + ", ".join(changed_budget_fields)
            )
        if reference is None and name != "direct_information":
            reference = item
        if reference is not None and name != "direct_information":
            mismatched = [key for key in matched_fields if item[key] != reference[key]]
            if mismatched:
                raise ValueError(
                    f"policy_artifacts.{name} violates matched protocol fields: {mismatched}"
                )


def _load_v3_bound_artifact(
    reference: Any,
    *,
    base_dir: Path,
    artifact_name: str,
    schema_version: str,
    preregistration: FrozenPathCPreregistration,
    resolved_path_c_sha256: str,
) -> dict[str, Any]:
    ref = _mapping(reference, f"measurement_artifacts.{artifact_name}")
    _require_exact_mapping_keys(
        ref, {"path", "sha256"}, f"measurement_artifacts.{artifact_name}"
    )
    path = _validated_file_reference(
        ref, base_dir, f"measurement_artifacts.{artifact_name}"
    )
    payload = _load_mapping(path, f"Path C measurement {artifact_name}")
    for key in ("pass", "go", "status", "instrument_valid", "primary_effective", "allowed_claim"):
        if key in payload:
            raise ValueError(
                f"Measurement {artifact_name} contains derived decision field {key!r}."
            )
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"Measurement {artifact_name} has the wrong schema version.")
    if payload.get("measurement_schema_version") != PATH_C_MEASUREMENT_SCHEMA:
        raise ValueError(f"Measurement {artifact_name} has the wrong parent schema.")
    if payload.get("preregistration_sha256") != preregistration.sha256:
        raise ValueError(f"Measurement {artifact_name} has the wrong preregistration hash.")
    if payload.get("resolved_path_c_sha256") != resolved_path_c_sha256:
        raise ValueError(f"Measurement {artifact_name} has the wrong runtime hash.")
    if payload.get("semantic_bindings") != preregistration.semantic_bindings:
        raise ValueError(f"Measurement {artifact_name} changes semantic bindings.")
    result = dict(payload)
    result["artifact_sha256"] = str(ref["sha256"])
    return result


def _validated_file_reference(
    reference: Mapping[str, Any],
    base_dir: Path,
    name: str,
) -> Path:
    """Resolve one manifest file reference and verify its recorded digest."""

    if "path" not in reference or "sha256" not in reference:
        raise ValueError(f"{name} must contain path and sha256.")
    raw_path = reference["path"]
    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
        raise ValueError(f"{name}.path must be a non-empty path.")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(base_dir) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not reference a file: {path}")
    expected_sha256 = str(reference["sha256"])
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{name}.sha256 must be a lower-case SHA-256 digest.")
    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {expected_sha256}, "
            f"observed {observed_sha256}."
        )
    return path


def _require_exact_mapping_keys(
    mapping: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    observed = set(mapping)
    missing = sorted(expected.difference(observed))
    unknown = sorted(observed.difference(expected))
    if missing or unknown:
        raise ValueError(f"{name} key mismatch; missing={missing}, unknown={unknown}.")


def _reject_legacy_margin_fields(value: Any, path: str = "preregistration") -> None:
    forbidden = {"epsilon_F", "epsilon_R", "hard_fail_margin"}
    if isinstance(value, Mapping):
        found = sorted(forbidden.intersection(value))
        if found:
            raise ValueError(
                f"{path} contains removed version-2 margin field(s): {', '.join(found)}"
            )
        for key, item in value.items():
            _reject_legacy_margin_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_legacy_margin_fields(item, f"{path}[{index}]")


def _runtime_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {}
    for section_name, fields in _RUNTIME_CONTRACT_FIELDS.items():
        section = payload.get(section_name)
        if not isinstance(section, Mapping):
            section = {}
        contract[section_name] = {field: _json_scalar(section.get(field)) for field in fields}
    return contract


def _require_mapping_fields(
    mapping: Mapping[str, Any],
    fields: Sequence[str],
    name: str,
) -> None:
    missing = [field for field in fields if field not in mapping or mapping[field] is None]
    if missing:
        raise ValueError(f"{name} is missing required field(s): " + ", ".join(missing))


def _is_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _is_git_commit_sha(value: Any) -> bool:
    text = str(value)
    return len(text) == 40 and all(char in "0123456789abcdef" for char in text)


def _reject_authoritative_pass_fields(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        if "pass" in value:
            raise ValueError(f"{path} contains an authoritative 'pass' field; provide raw measurements instead.")
        for key, item in value.items():
            _reject_authoritative_pass_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_authoritative_pass_fields(item, f"{path}[{index}]")


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping: {path}")
    return payload


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list.")
    return list(value)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    values = tuple(str(item) for item in _sequence(value, name))
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must be a non-empty list without duplicates.")
    return values


def _measurement(measurements: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in measurements:
            return measurements[name]
    return None


def _aggregate_value_control(
    item: Mapping[str, Any],
    required_baselines: Sequence[str],
    name: str,
) -> dict[str, Any]:
    comparisons = _mapping(item.get("comparisons"), f"{name}.comparisons")
    missing = sorted(set(required_baselines).difference(comparisons))
    if missing:
        raise ValueError(f"{name} lacks required baseline comparison(s): " + ", ".join(missing))
    values = []
    lower_bounds = []
    normalized = {}
    for baseline in required_baselines:
        measurement = _mapping(comparisons[baseline], f"{name}.comparisons.{baseline}")
        value = _finite_number(measurement.get("value"), f"{name}.{baseline}.value")
        ci_lo = _finite_number(
            measurement.get("ci_lo", _nested_ci(measurement, "lo")),
            f"{name}.{baseline}.ci_lo",
        )
        values.append(value)
        lower_bounds.append(ci_lo)
        normalized[str(baseline)] = {"value": value, "ci_lo": ci_lo}
    return {
        "value": min(values),
        "ci_lo": min(lower_bounds),
        "comparisons": normalized,
        "aggregation": "minimum_over_required_baselines",
    }


def _nested_ci(item: Mapping[str, Any], key: str) -> Any:
    ci = item.get("ci")
    return ci.get(key) if isinstance(ci, Mapping) else None


def _optional_finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _interval(value: Any) -> list[float | None]:
    if isinstance(value, Mapping):
        return [
            _optional_finite_number(value.get("lo")),
            _optional_finite_number(value.get("hi")),
        ]
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [_optional_finite_number(value[0]), _optional_finite_number(value[1])]
    return [None, None]


def _interval_inside(
    interval: Sequence[float | None],
    lower: float,
    upper: float,
) -> bool:
    lo, hi = interval
    return lo is not None and hi is not None and lo > lower and hi < upper


def _finite_number(value: Any, name: str) -> float:
    result = _optional_finite_number(value)
    if result is None:
        raise ValueError(f"{name} must be a finite number.")
    return result


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(
        value
    ).is_integer():
        return int(value)
    return None


def _json_scalar(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_scalar(item) for item in value]
    if isinstance(value, list):
        return [_json_scalar(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    return value
