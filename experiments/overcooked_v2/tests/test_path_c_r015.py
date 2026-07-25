"""机会审计实验编号 R015 的静态合同测试。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.overcooked_v2 import path_c_r015_runtime as formal_runtime
from experiments.overcooked_v2.path_c_r015 import (
    ArtifactBinding,
    ArmOutcome,
    EffectInterval,
    R015PairedBlock,
    R015Statistics,
    _aggregate_validated_r015_blocks,
    _derive_r015_controller_key,
    _derive_r015_random_key,
    _future_random_sequence_summary,
    _load_firing_count_checkpoint_decisions,
    _load_formal_sampling_schedule,
    _validate_manifest_file_binding_mapping,
    _validate_manifest_implementation_sources,
    _validate_sequential_controller_manifest,
    clopper_pearson_upper_bound,
    decide_r015,
    evaluate_r015_firing_count_checkpoint,
    load_r015_preregistration,
    maurer_pontil_empirical_bernstein_interval,
    validate_r015_paired_block as _validate_r015_paired_block,
    validate_r015_support_report,
)
from experiments.overcooked_v2.path_c_standard_training import (
    partner_training_run_id,
)


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "path_c_r015_preregistration.yaml"
)


def test_nested_response_vocabulary_sources_are_rehashed(tmp_path: Path) -> None:
    summary_source = tmp_path / "path_c_response_summary.py"
    probe_source = tmp_path / "path_c_response_probe.py"
    summary_source.write_text("SUMMARY = 1\n", encoding="utf-8")
    probe_source.write_text("PROBE = 1\n", encoding="utf-8")
    manifest_path = tmp_path / "response_vocabulary_manifest.json"
    manifest = {
        "implementation_sources": [
            {
                "path": str(summary_source),
                "sha256": hashlib.sha256(summary_source.read_bytes()).hexdigest(),
            },
            {
                "path": str(probe_source),
                "sha256": hashlib.sha256(probe_source.read_bytes()).hexdigest(),
            },
        ]
    }
    _validate_manifest_implementation_sources(
        manifest,
        manifest_path=manifest_path,
        label="response vocabulary",
        expected_filenames=frozenset(
            {"path_c_response_summary.py", "path_c_response_probe.py"}
        ),
    )
    probe_source.write_text("PROBE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after freeze"):
        _validate_manifest_implementation_sources(
            manifest,
            manifest_path=manifest_path,
            label="response vocabulary",
            expected_filenames=frozenset(
                {"path_c_response_summary.py", "path_c_response_probe.py"}
            ),
        )


def test_named_environment_source_closure_is_rehashed(tmp_path: Path) -> None:
    source_names = frozenset({"settings", "overcooked", "layouts", "common"})
    bindings = {}
    for name in source_names:
        path = tmp_path / f"{name}.py"
        path.write_text(f"SOURCE = {name!r}\n", encoding="utf-8")
        bindings[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {"environment_source_closure": bindings}
    observed = _validate_manifest_file_binding_mapping(
        manifest,
        field="environment_source_closure",
        label="environment",
        expected_names=source_names,
    )
    assert set(observed) == source_names
    (tmp_path / "layouts.py").write_text("SOURCE = 'changed'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after freeze"):
        _validate_manifest_file_binding_mapping(
            manifest,
            field="environment_source_closure",
            label="environment",
            expected_names=source_names,
        )


def _mapping_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trace_replay_verifier(trace, context):
    groups = trace["groups"]
    probe_executed = trace["probe_step"] is not None
    group_ego_actions = {
        group: [
            (
                "up"
                if probe_executed and group == "A1" and step in {1, 2}
                else "left"
                if probe_executed and group == "A2-mask" and step == 2
                else "interact"
                if probe_executed and group == "A2-use" and step == 2
                else "stay"
            )
            for step in range(400)
        ]
        for group in groups
    }
    group_partner_actions = {
        group: ["stay"] * 400 for group in groups
    }
    group_state_writes = {
        group: [
            {
                "environment_step": step,
                "before_sha256": _mapping_sha256({"controller_state": step}),
                "after_sha256": _mapping_sha256({"controller_state": step + 1}),
            }
            for step in range(400)
        ]
        for group in groups
    }
    group_partner_state_writes = {
        group: [
            {
                "environment_step": step,
                "before_sha256": _mapping_sha256({"partner_state": step}),
                "after_sha256": _mapping_sha256({"partner_state": step + 1}),
            }
            for step in range(400)
        ]
        for group in groups
    }
    group_controller_route_summaries = {}
    for group in groups:
        if group == "A1":
            group_controller_route_summaries[group] = {
                "belief_route": "B_mask(q,x)",
                "continuation_route": "C(q,x)",
                "selected_action_id": "base",
                "selected_registered_script_actions": [],
                "selected_action_matches_replay": True,
            }
        else:
            group_controller_route_summaries[group] = {
                "belief_route": (
                    "B_use(q,x,y)"
                    if group == "A2-use" and probe_executed
                    else "B_mask(q,x)"
                ),
                "continuation_route": "C(q,x)",
                "selected_probe_actions": ["stay"] if probe_executed else [],
                "current_response_used": group == "A2-use" and probe_executed,
                "pre_probe_actions_match": True,
                "post_probe_actions_match_replay": True,
            }
    return {
        "schema_version": "path_c_r015_trace_replay_result_v2",
        "verified": True,
        "groups_sha256": _mapping_sha256(groups),
        "group_step_counts": {group: 400 for group in groups},
        "group_environment_steps_sha256": {
            group: _mapping_sha256(record["environment_steps"])
            for group, record in groups.items()
        },
        "environment_config_sha256": context["environment_config_sha256"],
        "environment_source_sha256": context["environment_source_sha256"],
        "ego_checkpoint_sha256": context["ego_checkpoint_sha256"],
        "continuation_planner_sha256": context["continuation_planner_sha256"],
        "official_history_filter_sha256": context[
            "official_history_filter_sha256"
        ],
        "response_projection_sha256": context["response_projection_sha256"],
        "ego_evidence_contract_sha256": context[
            "ego_evidence_contract_sha256"
        ],
        "response_vocabulary_sha256": context["response_vocabulary_sha256"],
        "probe_registry_sha256": context["probe_registry_sha256"],
        "probe_registry_semantic_sha256": context[
            "probe_registry_semantic_sha256"
        ],
        "random_key_derivation_sha256": context[
            "random_key_derivation_sha256"
        ],
        "partner_checkpoint_sha256": context["partner_checkpoint_sha256"],
        "partner_model_weights_sha256": context[
            "partner_model_weights_sha256"
        ],
        "partner_training_config_sha256": context[
            "partner_training_config_sha256"
        ],
        "partner_training_manifest_sha256": context[
            "partner_training_manifest_sha256"
        ],
        "partner_family_spec_sha256": context["partner_family_spec_sha256"],
        "partner_architecture_sha256": context["partner_architecture_sha256"],
        "partner_action_rule": context["partner_action_rule"],
        "support_registration_sha256": context["support_registration_sha256"],
        "support_report_sha256": context["support_report_sha256"],
        "group_ego_actions": group_ego_actions,
        "group_partner_actions": group_partner_actions,
        "group_recurrent_state_writes_sha256": {
            group: _mapping_sha256(writes)
            for group, writes in group_state_writes.items()
        },
        "group_action_state_summaries_sha256": {
            group: _mapping_sha256(
                [
                    {
                        "environment_step": write["environment_step"],
                        "ego_action": group_ego_actions[group][index],
                        "before_sha256": write["before_sha256"],
                        "after_sha256": write["after_sha256"],
                    }
                    for index, write in enumerate(writes)
                ]
            )
            for group, writes in group_state_writes.items()
        },
        "group_partner_recurrent_state_writes_sha256": {
            group: _mapping_sha256(writes)
            for group, writes in group_partner_state_writes.items()
        },
        "group_partner_action_state_summaries_sha256": {
            group: _mapping_sha256(
                [
                    {
                        "environment_step": write["environment_step"],
                        "partner_action": group_partner_actions[group][index],
                        "before_sha256": write["before_sha256"],
                        "after_sha256": write["after_sha256"],
                    }
                    for index, write in enumerate(writes)
                ]
            )
            for group, writes in group_partner_state_writes.items()
        },
        "group_controller_route_summaries": group_controller_route_summaries,
        "all_controller_actions_recomputed": True,
        "all_recurrent_state_writes_recomputed": True,
        "all_partner_actions_recomputed": True,
        "all_partner_recurrent_state_writes_recomputed": True,
    }


def _safety_branch_evidence_verifier(comparison, context):
    branches = comparison["branch_evidence"]
    branch_results = [
        {
            "branch_index": branch["branch_index"],
            "random_key": branch["random_key"],
            "wrong_delivery_detected": False,
        }
        for branch in branches
    ]
    wrong_delivery_count = 0
    return {
        "schema_version": "path_c_r015_safety_branch_verification_v2",
        "verified": True,
        "support_prototype_id": comparison["support_prototype_id"],
        "branch_count": len(branches),
        "branch_keys_sha256": _mapping_sha256(context["branch_keys"]),
        "wrong_delivery_count": wrong_delivery_count,
        "positive_posterior_support": True,
        "compatible_hidden_state_reconstructed": True,
        "all_branches_replayed": True,
        "support_checkpoint_sha256": context["support_checkpoint_sha256"],
        "support_model_weights_sha256": context[
            "support_model_weights_sha256"
        ],
        "support_architecture_sha256": context["support_architecture_sha256"],
        "support_training_config_sha256": context[
            "support_training_config_sha256"
        ],
        "support_training_manifest_sha256": context[
            "support_training_manifest_sha256"
        ],
        "support_family_spec_sha256": context["support_family_spec_sha256"],
        "support_action_rule": context["support_action_rule"],
        "support_registration_sha256": context["support_registration_sha256"],
        "support_report_sha256": context["support_report_sha256"],
        "selected_probe_id": comparison["selected_probe_id"],
        "selected_probe_actions": comparison["selected_probe_actions"],
        "probe_registry_sha256": context["probe_registry_sha256"],
        "probe_registry_semantic_sha256": context[
            "probe_registry_semantic_sha256"
        ],
        "official_history_filter_sha256": context[
            "official_history_filter_sha256"
        ],
        "response_projection_sha256": context["response_projection_sha256"],
        "ego_evidence_contract_sha256": context[
            "ego_evidence_contract_sha256"
        ],
        "response_vocabulary_sha256": context["response_vocabulary_sha256"],
        "official_history_sha256": _mapping_sha256(
            comparison["official_history"]
        ),
        "branch_evidence_sha256": _mapping_sha256(branches),
        "branch_results_sha256": _mapping_sha256(branch_results),
    }


def _decision_evidence_verifier(evidence, context):
    consultations = []
    for row in evidence["consultations"]:
        replay = row["planning_branches"][0]
        consultations.append(
            {
                "environment_step": row["environment_step"],
                "v_base": replay["v_base"],
                "candidates": replay["candidates"],
                "selected_for_safety_probe_id": replay[
                    "selected_for_safety_probe_id"
                ],
                "planning_random_key": row["planning_random_key"],
                "score_random_key": row["score_random_key"],
            }
        )
    return {
        "schema_version": "path_c_r015_decision_evidence_verification_v1",
        "verified": True,
        "consultations": consultations,
        "full_probe_registry_ids": sorted(context["probe_scripts"]),
        "first_positive_stop_valid": True,
    }


def validate_r015_paired_block(payload, *, preregistration, support_members):
    return _validate_r015_paired_block(
        payload,
        preregistration=preregistration,
        support_members=support_members,
        trace_replay_verifier=_trace_replay_verifier,
        safety_branch_evidence_verifier=_safety_branch_evidence_verifier,
        decision_evidence_verifier=_decision_evidence_verifier,
    )


def _statistics():
    return R015Statistics.from_mapping(
        {
            "partner_prototype_count": 4,
            "pilot_blocks_per_prototype": 20,
            "n_rounds_max": 2500,
            "formal_paired_block_budget": 10000,
            "formal_arm_episode_budget": 30000,
            "formal_arm_environment_step_budget": 12000000,
            "maximum_safety_comparisons": 40000,
            "planning_fork_steps_included": False,
            "delivery_reward": 20.0,
            "practical_margin_fraction": 0.25,
            "return_lower_bound": -400.0,
            "return_upper_bound": 400.0,
            "paired_difference_absolute_bound": 800.0,
            "paired_difference_width": 1600.0,
            "empirical_bernstein_formula": (
                "maurer_pontil_empirical_bernstein_bounded_v1"
            ),
            "round_sampling_contract": (
                "iid_round_vectors_ego_cook_seat_with_prefrozen_replacements_v4"
            ),
            "formal_sampling_schedule_schema": (
                "path_c_r015_formal_sampling_schedule_v4"
            ),
            "mechanical_replacement_attempts_per_coordinate": 32,
            "mechanical_replacement_attempt_indices": "0_through_31",
            "replacement_episode_seed_derivation_id": (
                "sha256_registered_coordinate_low32_allow_collisions_v1"
            ),
            "replacement_episode_seed_collision_policy": (
                "allow_value_collisions_without_redraw"
            ),
            "audit_unit_changes_across_replacement_attempts": False,
            "replacement_seed_selection_uses_outcomes": False,
            "mechanical_attempt_exhaustion_rule": (
                "terminate_entire_formal_audit"
            ),
            "ego_agent_id": "agent_1",
            "zero_count_independence_contract": (
                "independent_firing_prefixes_across_prototypes_v1"
            ),
            "alpha_total": 0.05,
            "alpha_safety": 0.025,
            "alpha_eb_net": 0.010,
            "alpha_eb_response": 0.010,
            "alpha_rho": 0.005,
            "kill_checkpoint_rounds": [200, 400, 800, 1600, 2500],
            "alpha_rho_per_checkpoint": 0.001,
            "alpha_rho_per_prototype_cp": 0.00025,
            "safety_risk_limit": 0.05,
            "safety_repeats_per_comparison": 279,
            "formal_effect_look_count": 1,
            "formal_effect_look_round": 2500,
        }
    )


def _numeric_preregistration():
    preregistration = load_r015_preregistration(CONFIG)
    artifacts = dict(preregistration.artifacts)
    artifacts["ego_evidence_contract"] = ArtifactBinding(
        name="ego_evidence_contract", path=None, sha256="a" * 64
    )
    artifacts["formal_sampling_schedule"] = ArtifactBinding(
        name="formal_sampling_schedule", path=None, sha256="f" * 64
    )
    artifacts["response_projection"] = ArtifactBinding(
        name="response_projection", path=None, sha256="b" * 64
    )
    artifacts["continuation_planner"] = ArtifactBinding(
        name="continuation_planner", path=None, sha256="1" * 64
    )
    artifacts["continuation_controller"] = ArtifactBinding(
        name="continuation_controller", path=None, sha256="e" * 64
    )
    artifacts["decision_evidence_verifier"] = ArtifactBinding(
        name="decision_evidence_verifier", path=None, sha256="0" * 64
    )
    artifacts["probe_registry"] = ArtifactBinding(
        name="probe_registry", path=None, sha256="2" * 64
    )
    artifacts["response_vocabulary"] = ArtifactBinding(
        name="response_vocabulary", path=None, sha256="3" * 64
    )
    artifacts["ego_checkpoint"] = ArtifactBinding(
        name="ego_checkpoint", path=None, sha256="4" * 64
    )
    artifacts["environment_config"] = ArtifactBinding(
        name="environment_config", path=None, sha256="5" * 64
    )
    artifacts["environment_source"] = ArtifactBinding(
        name="environment_source", path=None, sha256="6" * 64
    )
    artifacts["official_history_filter"] = ArtifactBinding(
        name="official_history_filter", path=None, sha256="7" * 64
    )
    artifacts["wrong_delivery_detector"] = ArtifactBinding(
        name="wrong_delivery_detector", path=None, sha256="8" * 64
    )
    artifacts["random_key_derivation"] = ArtifactBinding(
        name="random_key_derivation", path=None, sha256="9" * 64
    )
    artifacts["safety_branch_evidence_verifier"] = ArtifactBinding(
        name="safety_branch_evidence_verifier", path=None, sha256="d" * 64
    )
    artifacts["trace_replay_verifier"] = ArtifactBinding(
        name="trace_replay_verifier", path=None, sha256="c" * 64
    )
    return replace(
        preregistration,
        freeze_status="frozen",
        statistics=_statistics(),
        controller=replace(
            preregistration.controller,
            particles_per_prototype=64,
            resampling_timing="adaptive_ess_below_half_v1",
            resampling_ess_fraction_threshold=0.5,
            planning_branches_per_candidate=2,
        ),
        artifacts=artifacts,
        ego_model_weights_sha256="b" * 64,
    )


def _support_members(preregistration):
    return {
        candidate.candidate_id: {
            "family_id": candidate.family_id,
            "training_seed": candidate.training_seed,
            "training_run_id": hashlib.sha256(
                candidate.candidate_id.encode()
            ).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(
                f"checkpoint/{candidate.candidate_id}".encode()
            ).hexdigest(),
            "model_weights_sha256": hashlib.sha256(
                f"weights/{candidate.candidate_id}".encode()
            ).hexdigest(),
            "checkpoint_path": f"checkpoint/{candidate.candidate_id}.pt",
            "training_config_path": f"training/{candidate.candidate_id}.yaml",
            "training_config_sha256": hashlib.sha256(
                f"training-config/{candidate.candidate_id}".encode()
            ).hexdigest(),
            "training_manifest_path": f"training/{candidate.candidate_id}.json",
            "training_manifest_sha256": hashlib.sha256(
                f"training-manifest/{candidate.candidate_id}".encode()
            ).hexdigest(),
            "family_spec_sha256": hashlib.sha256(
                candidate.family_id.encode()
            ).hexdigest(),
            "architecture": {"family_id": candidate.family_id},
            "architecture_sha256": _mapping_sha256(
                {"family_id": candidate.family_id}
            ),
            "action_rule": "sample_frozen_policy",
        }
        for candidate in preregistration.support_spec.candidates
    }


def _trace_replay_context(preregistration, member):
    return {
        "environment_config_sha256": "5" * 64,
        "environment_source_sha256": "6" * 64,
        "ego_checkpoint_sha256": "4" * 64,
        "continuation_planner_sha256": "1" * 64,
        "official_history_filter_sha256": "7" * 64,
        "response_projection_sha256": "b" * 64,
        "ego_evidence_contract_sha256": "a" * 64,
        "response_vocabulary_sha256": "3" * 64,
        "probe_registry_sha256": "2" * 64,
        "probe_registry_semantic_sha256": (
            preregistration.probe_registry_semantic_sha256
        ),
        "random_key_derivation_sha256": "9" * 64,
        "partner_checkpoint_sha256": member["checkpoint_sha256"],
        "partner_model_weights_sha256": member["model_weights_sha256"],
        "partner_training_config_sha256": member["training_config_sha256"],
        "partner_training_manifest_sha256": member[
            "training_manifest_sha256"
        ],
        "partner_family_spec_sha256": member["family_spec_sha256"],
        "partner_architecture_sha256": member["architecture_sha256"],
        "partner_action_rule": member["action_rule"],
        "support_registration_sha256": preregistration.support_registration_sha256,
        "support_report_sha256": None,
    }


def _arm(
    *,
    group,
    prototype_id,
    family_id,
    training_seed,
    training_run_id,
    ego_position,
    initial_state_sha256,
    episode_seed,
    firing_indicator,
    raw_return,
    random_key,
):
    return {
        "prototype_id": prototype_id,
        "family_id": family_id,
        "training_seed": training_seed,
        "training_run_id": training_run_id,
        "ego_position": ego_position,
        "initial_state_sha256": initial_state_sha256,
        "episode_seed": episode_seed,
        "firing_indicator": firing_indicator,
        "raw_return": raw_return,
        "valid": True,
        "information_source": "official_local_history_only",
        "forbidden_fields_read": [],
        "episode_random_key": random_key,
        "controller_contract_sha256": "a" * 64,
        "ego_checkpoint_sha256": "4" * 64,
        "environment_config_sha256": "5" * 64,
        "environment_source_sha256": "6" * 64,
        "official_history_filter_sha256": "7" * 64,
    }


def _paired_block_payload(
    preregistration,
    *,
    prototype_id,
    returns=(-2.0, -1.0, 7.0),
    probe_executed=True,
):
    member = _support_members(preregistration)[prototype_id]
    family_id = member["family_id"]
    training_seed = member["training_seed"]
    training_run_id = member["training_run_id"]
    initial_state = "c" * 64
    random_key = _derive_r015_random_key(
        audit_unit_id=f"unit-{prototype_id}",
        partner_prototype_id=prototype_id,
        episode_seed=17,
        purpose="actual_episode",
        environment_step=0,
        branch_index=0,
    )
    groups = dict(zip(("A1", "A2-mask", "A2-use"), returns))
    arms = {
        group: _arm(
            group=group,
            prototype_id=prototype_id,
            family_id=family_id,
            training_seed=training_seed,
            training_run_id=training_run_id,
            ego_position=1,
            initial_state_sha256=initial_state,
            episode_seed=17,
            firing_indicator=probe_executed,
            raw_return=value,
            random_key=random_key,
        )
        for group, value in groups.items()
    }
    selected_probe_id = "stay" if probe_executed else None
    candidates = []
    for probe_id, script in preregistration.probe_scripts.items():
        j_use = 1.0 if probe_executed and probe_id == selected_probe_id else 0.0
        candidates.append(
            {
                "probe_id": probe_id,
                "registered": True,
                "script_length": len(script.primitive_actions),
                "eligible": True,
                "static_safety_pass": True,
                "j_use": j_use,
                "j_mask": 0.0,
                "v_mask": 0.0,
                "i_response": j_use,
                "c_task": 0.0,
                "s_seq": j_use,
            }
        )
    planning_key = _derive_r015_random_key(
        audit_unit_id=f"unit-{prototype_id}",
        partner_prototype_id=prototype_id,
        episode_seed=17,
        purpose="value_planning",
        environment_step=1,
        branch_index=0,
    )
    score_key = _derive_r015_random_key(
        audit_unit_id=f"unit-{prototype_id}",
        partner_prototype_id=prototype_id,
        episode_seed=17,
        purpose="score_estimation",
        environment_step=1,
        branch_index=0,
    )

    def planning_branch(canonical_slot_16):
        root_key = _derive_r015_controller_key(
            planning_key,
            "sampled_hidden_state_branches_v1",
            "canonical_slot_16",
            canonical_slot_16,
        )
        return {
            "canonical_slot_16": canonical_slot_16,
            "common_random_key": root_key,
            "source_belief_filter_algorithm_id": (
                "official_history_fully_adapted_particle_filter_v2"
            ),
            "source_belief_resampling_algorithm": (
                "strict_systematic_per_prototype_v2"
            ),
            **_future_random_sequence_summary(
                root_key=root_key,
                sequence_length=399,
            ),
            "v_base": 0.0,
            "candidates": json.loads(json.dumps(candidates)),
            "selected_for_safety_probe_id": selected_probe_id,
        }

    probe_decision = {
        "probe_registry_semantic_sha256": (
            preregistration.probe_registry_semantic_sha256
        ),
        "random_key_derivation_sha256": "9" * 64,
        "planning_random_purpose": "value_planning",
        "planning_random_stream_key": planning_key,
        "score_random_purpose": "score_estimation",
        "score_random_stream_key": score_key,
        "consultations": [
            {
                "environment_step": 1,
                "v_base": 0.0,
                "candidates": candidates,
                "selected_for_safety_probe_id": selected_probe_id,
                "planner_evidence": {
                    "schema_version": (
                        "path_c_r015_consultation_planner_evidence_v2"
                    ),
                    "official_history": [
                        {
                            "official_local_observation": {
                                "visible_state": "shared/0"
                            }
                        },
                        {
                            "official_local_observation": {
                                "visible_state": "shared/1"
                            }
                        },
                    ],
                    "planning_random_key": planning_key,
                    "score_random_key": score_key,
                    "planning_branches": [planning_branch(0), planning_branch(8)],
                    "branch_sampling_rule_id": "sampled_hidden_state_branches_v1",
                    "branch_belief_rule_id": (
                        "frozen_belief_branch_continuation_v1"
                    ),
                    "cost_accounting": {
                        "schema_version": "path_c_r015_planning_cost_v2",
                        "sample_count": 2,
                    },
                },
            }
        ],
        "executed_probe_id": selected_probe_id,
        "probe_step": 1 if probe_executed else None,
        "probe_budget_used_before": 0,
        "probe_budget_used_after": 1 if probe_executed else 0,
    }
    score_trace_sha256 = _mapping_sha256(probe_decision)
    shared_masked_value = {
        "v_base": 0.0,
        "candidate_j_mask": {
            probe_id: 0.0 for probe_id in preregistration.probe_scripts
        },
        "selectable_candidate_ids": sorted(preregistration.probe_scripts),
        "selected_action_id": "base",
        "selected_value": 0.0,
        "v_mask": 0.0,
    }
    trace_groups = {}
    for group, raw_return in groups.items():
        controller_inputs = []
        actions = []
        for step in range(400):
            state_group = group if probe_executed and step > 1 else "shared"
            controller_inputs.append(
                {
                    "official_local_observation": {
                        "visible_state": f"{state_group}/{step}"
                    }
                }
            )
            action = "stay"
            if probe_executed and group == "A1" and step in {1, 2}:
                action = "up"
            elif probe_executed and group == "A2-mask" and step == 2:
                action = "left"
            elif probe_executed and group == "A2-use" and step == 2:
                action = "interact"
            actions.append(action)
        environment_steps = []
        for step in range(400):
            after_observation = (
                controller_inputs[step + 1]["official_local_observation"]
                if step < 399
                else {
                    "visible_state": (
                        f"{group}/terminal" if probe_executed else "shared/terminal"
                    )
                }
            )
            environment_steps.append(
                {
                    "environment_step": step,
                    "ego_action": actions[step],
                    "partner_action": "stay",
                    "controller_input": controller_inputs[step],
                    "raw_team_reward": raw_return if step == 399 else 0.0,
                    "done": step == 399,
                    "environment_random_key": f"environment/shared/{step}",
                    "official_local_observation_after": after_observation,
                    "controller_recurrent_state_before_sha256": _mapping_sha256(
                        {"controller_state": step}
                    ),
                    "controller_recurrent_state_after_sha256": _mapping_sha256(
                        {"controller_state": step + 1}
                    ),
                    "partner_recurrent_state_before_sha256": _mapping_sha256(
                        {"partner_state": step}
                    ),
                    "partner_recurrent_state_after_sha256": _mapping_sha256(
                        {"partner_state": step + 1}
                    ),
                    "belief_update_mode": "online_each_environment_step",
                    "belief_before_sha256": _mapping_sha256(
                        {"belief_state": step}
                    ),
                    "belief_after_sha256": _mapping_sha256(
                        {"belief_state": step + 1}
                    ),
                }
            )
        trace_groups[group] = {
            "environment_steps": environment_steps,
            "official_history_events": controller_inputs,
            "forbidden_read_events": [],
            "raw_reward_components": [
                row["raw_team_reward"] for row in environment_steps
            ],
            "masked_value_reference": shared_masked_value,
            "replay_verification": {
                "trace_replay_verifier_sha256": "c" * 64,
                "environment_config_sha256": "5" * 64,
                "environment_source_sha256": "6" * 64,
                "ego_checkpoint_sha256": "4" * 64,
                "continuation_planner_sha256": "1" * 64,
                "official_history_filter_sha256": "7" * 64,
                "response_projection_sha256": "b" * 64,
                "step_count": 400,
                "actions_legal": True,
                "controller_actions_recomputed": True,
                "recurrent_state_writes_recomputed": True,
                "observation_chain_matches": True,
                "raw_rewards_match": True,
                "done_boundary_matches": True,
                "passed": True,
            },
        }
    shared_pre_trace = trace_groups["A1"]["official_history_events"][:2]
    if probe_executed:
        masked_history = [{"official_local_observation": {"visible_state": "after"}}]
        current_response = {"response_summary_v1_derived": "not_visible"}
        belief_common_input = {"history": masked_history, "belief": [0.25] * 4}
        continuation_common_input = {"history": masked_history, "mode": "continue"}
        recurrent_common_input = {"state": "projected_without_current_response"}
        for group in ("A1", "A2-mask", "A2-use"):
            trace_groups[group].update(
                {
                    "pre_probe_trace": shared_pre_trace,
                    "pre_probe_actions": ["stay"],
                    "pre_probe_random_keys": ["environment/shared/0"],
                    "masked_value_reference": shared_masked_value,
                }
            )
        for group in ("A2-mask", "A2-use"):
            trace_groups[group].update(
                {
                    "selected_probe_id": selected_probe_id,
                    "probe_actions": ["stay"],
                    "masked_history": masked_history,
                    "current_probe_response": current_response,
                    "belief_common_input": belief_common_input,
                    "continuation_common_input": continuation_common_input,
                    "recurrent_common_input": recurrent_common_input,
                    "continuation_direct_response": None,
                    "recurrent_direct_response": None,
                    "non_belief_response_aliases": [],
                    "probe_random_keys": ["environment/shared/1"],
                    "future_random_branch_keys": [
                        f"environment/shared/{step}" for step in range(2, 400)
                    ],
                }
            )
        trace_groups["A2-mask"]["belief_current_response_extra"] = None
        trace_groups["A2-use"][
            "belief_current_response_extra"
        ] = current_response
    else:
        for group in ("A1", "A2-mask", "A2-use"):
            trace_groups[group].update(
                {
                    "complete_trace": trace_groups[group][
                        "official_history_events"
                    ],
                    "actions": ["stay"] * 400,
                    "future_random_branch_keys": [
                        f"environment/shared/{step}" for step in range(400)
                    ],
                }
            )
    for group in ("A1", "A2-mask", "A2-use"):
        steps = trace_groups[group]["environment_steps"]
        state_writes = [
            {
                "environment_step": step["environment_step"],
                "before_sha256": step[
                    "controller_recurrent_state_before_sha256"
                ],
                "after_sha256": step[
                    "controller_recurrent_state_after_sha256"
                ],
            }
            for step in steps
        ]
        partner_state_writes = [
            {
                "environment_step": step["environment_step"],
                "before_sha256": step["partner_recurrent_state_before_sha256"],
                "after_sha256": step["partner_recurrent_state_after_sha256"],
            }
            for step in steps
        ]
        if group == "A1":
            route_summary = {
                "belief_route": "B_mask(q,x)",
                "continuation_route": "C(q,x)",
                "selected_action_id": "base",
                "selected_registered_script_actions": [],
                "selected_action_matches_replay": True,
            }
        else:
            route_summary = {
                "belief_route": (
                    "B_use(q,x,y)"
                    if group == "A2-use" and probe_executed
                    else "B_mask(q,x)"
                ),
                "continuation_route": "C(q,x)",
                "selected_probe_actions": ["stay"] if probe_executed else [],
                "current_response_used": group == "A2-use" and probe_executed,
                "pre_probe_actions_match": True,
                "post_probe_actions_match_replay": True,
            }
        trace_groups[group]["controller_replay_evidence"] = {
            "ego_actions": [step["ego_action"] for step in steps],
            "partner_actions": [step["partner_action"] for step in steps],
            "state_writes": state_writes,
            "partner_state_writes": partner_state_writes,
            "route_summary": route_summary,
        }
    trace_manifest = {
        "schema_version": "path_c_r015_trace_manifest_v1",
        "audit_unit_id": f"unit-{prototype_id}",
        "prototype_id": prototype_id,
        "family_id": family_id,
        "training_seed": training_seed,
        "training_run_id": training_run_id,
        "ego_position": 1,
        "initial_state_sha256": initial_state,
        "episode_seed": 17,
        "mechanical_attempt_index": 0,
        "episode_random_key": random_key,
        "random_key_derivation_sha256": "9" * 64,
        "probe_step": 1 if probe_executed else None,
        "groups": trace_groups,
    }
    trace_manifest["replay_verification"] = _trace_replay_verifier(
        trace_manifest,
        _trace_replay_context(preregistration, member),
    )
    trace_manifest_sha256 = _mapping_sha256(trace_manifest)
    pairing = {
        "shared_continuation_controller_sha256": "e" * 64,
        "probe_registry_sha256": "2" * 64,
        "probe_registry_semantic_sha256": (
            preregistration.probe_registry_semantic_sha256
        ),
        "response_vocabulary_sha256": "3" * 64,
        "response_projection_sha256": "b" * 64,
        "a1_registered_response_route": "masked",
        "a1_full_candidate_set": True,
        "a1_uses_masked_value_reference": True,
        "a1_score_trace_sha256": score_trace_sha256,
        "a2_score_trace_sha256": score_trace_sha256,
        "raw_official_local_observation_preserved": True,
        "future_passive_response_route": "shared",
    }
    if probe_executed:
        pairing.update(
            {
                "a1_pre_probe_trace_sha256": _mapping_sha256(shared_pre_trace),
                "a1_masked_value_reference_sha256": _mapping_sha256(
                    shared_masked_value
                ),
                "A2-mask": {
                    "selected_probe_id": selected_probe_id,
                    "pre_probe_trace_sha256": _mapping_sha256(shared_pre_trace),
                    "masked_history_sha256": _mapping_sha256(masked_history),
                    "current_probe_response_sha256": _mapping_sha256(
                        current_response
                    ),
                    "current_response_route": "masked",
                    "masked_value_reference_sha256": _mapping_sha256(
                        shared_masked_value
                    ),
                    "belief_common_input_sha256": _mapping_sha256(
                        belief_common_input
                    ),
                    "belief_current_response_extra_sha256": None,
                    "continuation_common_input_sha256": _mapping_sha256(
                        continuation_common_input
                    ),
                    "continuation_direct_response_sha256": None,
                    "recurrent_common_input_sha256": _mapping_sha256(
                        recurrent_common_input
                    ),
                    "recurrent_direct_response_sha256": None,
                    "non_belief_response_aliases": [],
                    "projection_dependency_audit_sha256": "b" * 64,
                },
                "A2-use": {
                    "selected_probe_id": selected_probe_id,
                    "pre_probe_trace_sha256": _mapping_sha256(shared_pre_trace),
                    "masked_history_sha256": _mapping_sha256(masked_history),
                    "current_probe_response_sha256": _mapping_sha256(
                        current_response
                    ),
                    "current_response_route": "belief_update_only",
                    "masked_value_reference_sha256": _mapping_sha256(
                        shared_masked_value
                    ),
                    "belief_common_input_sha256": _mapping_sha256(
                        belief_common_input
                    ),
                    "belief_current_response_extra_sha256": _mapping_sha256(
                        current_response
                    ),
                    "continuation_common_input_sha256": _mapping_sha256(
                        continuation_common_input
                    ),
                    "continuation_direct_response_sha256": None,
                    "recurrent_common_input_sha256": _mapping_sha256(
                        recurrent_common_input
                    ),
                    "recurrent_direct_response_sha256": None,
                    "non_belief_response_aliases": [],
                    "projection_dependency_audit_sha256": "b" * 64,
                },
                "a2_groups_share_pre_probe_actions": True,
                "a2_groups_share_probe_script": True,
            }
        )
        def safety_comparison(support_id):
            repetitions = preregistration.statistics.safety_repeats_per_comparison
            branch_keys = [
                _derive_r015_random_key(
                    audit_unit_id=f"unit-{prototype_id}",
                    partner_prototype_id=support_id,
                    episode_seed=17,
                    purpose="safety_validation",
                    environment_step=1,
                    branch_index=branch_index,
                )
                for branch_index in range(repetitions)
            ]
            return {
                "support_prototype_id": support_id,
                "repetitions": repetitions,
                "wrong_delivery_count": 0,
                "positive_posterior_support": True,
                "posterior_support_evidence": {"posterior_mass": 0.25},
                "compatible_hidden_state_reconstructed": True,
                "random_stream_key": f"safety/{support_id}",
                "audit_unit_id": f"unit-{prototype_id}",
                "episode_seed": 17,
                "purpose": "safety_validation",
                "environment_step": 1,
                "branch_start_index": 0,
                "branch_count": repetitions,
                "branch_keys": branch_keys,
                "branch_evidence": [
                    {
                        "branch_index": branch_index,
                        "random_key": branch_key,
                        "compatible_hidden_state_sha256": "e" * 64,
                        "environment_steps": [{"wrong_delivery": False}],
                    }
                    for branch_index, branch_key in enumerate(branch_keys)
                ],
                "branch_results": [
                    {
                        "branch_index": branch_index,
                        "random_key": branch_key,
                        "wrong_delivery_detected": False,
                    }
                    for branch_index, branch_key in enumerate(branch_keys)
                ],
                "branch_keys_sha256": _mapping_sha256(branch_keys),
                "random_key_derivation_sha256": "9" * 64,
            }

        comparisons = [
            safety_comparison(support_id)
            for support_id in _support_members(preregistration)
        ]
        safety = {
            "probe_executed": True,
            "no_probe_reason": None,
            "wrong_delivery_detector_sha256": "8" * 64,
            "recipe_indicator_cost_counted_as_safety_event": False,
            "comparisons": comparisons,
            "candidate_evaluation_count": len(candidates),
            "safety_rejection_count": 0,
            "non_positive_score_count": 0,
            "window_expired_count": 0,
        }
    else:
        pairing.update(
            {
                "a1_complete_trace_sha256": _mapping_sha256(
                    trace_groups["A1"]["official_history_events"]
                ),
                "A2-mask": {
                    "selected_probe_id": None,
                    "current_probe_response_sha256": None,
                    "current_response_route": "not_applicable",
                    "complete_trace_sha256": _mapping_sha256(
                        trace_groups["A2-mask"]["official_history_events"]
                    ),
                },
                "A2-use": {
                    "selected_probe_id": None,
                    "current_probe_response_sha256": None,
                    "current_response_route": "not_applicable",
                    "complete_trace_sha256": _mapping_sha256(
                        trace_groups["A2-use"]["official_history_events"]
                    ),
                },
                "a2_groups_share_complete_trace": True,
                "all_groups_share_complete_trace": True,
            }
        )
        safety = {
            "probe_executed": False,
            "no_probe_reason": "non_positive_score",
            "wrong_delivery_detector_sha256": "8" * 64,
            "recipe_indicator_cost_counted_as_safety_event": False,
            "comparisons": [],
            "candidate_evaluation_count": len(candidates),
            "safety_rejection_count": 0,
            "non_positive_score_count": 1,
            "window_expired_count": 0,
        }
    return {
        "schema_version": "path_c_r015_paired_block_v2",
        "phase": "formal",
        "round_index": 1,
        "audit_unit_id": f"unit-{prototype_id}",
        "prototype_id": prototype_id,
        "family_id": family_id,
        "training_seed": training_seed,
        "training_run_id": training_run_id,
        "ego_position": 1,
        "initial_state_sha256": initial_state,
        "episode_seed": 17,
        "mechanical_attempt_index": 0,
        "formal_effect_look_number": 1,
        "pilot_data": False,
        "firing_indicator": probe_executed,
        "arms": arms,
        "probe_decision": probe_decision,
        "trace_manifest": trace_manifest,
        "trace_manifest_sha256": trace_manifest_sha256,
        "pairing": pairing,
        "safety": safety,
    }


def test_static_preregistration_loads_but_cannot_be_used_for_formal_readout():
    preregistration = load_r015_preregistration(CONFIG)
    assert preregistration.document_status == "template_not_valid_for_runs"
    assert preregistration.scientific_readout_allowed is False
    assert preregistration.freeze_status == "static_registered_not_frozen"
    assert preregistration.full_state_upper_bound_status == "not_provided"
    assert preregistration.information.controller_input == "official_local_history_only"
    assert preregistration.controller.consultation_steps == tuple(range(1, 101, 5))
    assert preregistration.controller.score_tie_break == "registered_probe_order"
    assert preregistration.controller.masked_reference_tie_break == (
        "base_then_registered_probe_order"
    )
    assert preregistration.controller.particles_per_prototype is None
    assert set(preregistration.controller.prototype_prior or {}) == {
        "official_rnn_sp_ippo_v1_seed101_step29949952",
        "official_rnn_sp_ippo_v1_seed102_step29949952",
        "official_rnn_op_other_play_v1_seed201_step29999104",
        "official_rnn_op_other_play_v1_seed202_step29999104",
    }
    assert set((preregistration.controller.prototype_prior or {}).values()) == {0.25}
    assert preregistration.controller.resampling_algorithm == (
        "strict_systematic_per_prototype_v2"
    )
    assert preregistration.controller.filter_initialization_rule == (
        "independent_prior_draws_conditioned_on_opening_official_observation_v1"
    )
    assert preregistration.controller.filter_transition_update_rule == (
        "exact_six_action_marginal_then_conditional_successor_v1"
    )
    assert preregistration.controller.filter_ess_rule == (
        "pre_resample_parent_predictive_weight_ess_v1"
    )
    assert preregistration.controller.filter_zero_support_condition == (
        "sum_parent_weight_times_predictive_likelihood_equals_zero"
    )
    assert preregistration.controller.filter_device_execution_id == (
        "r015_filter_jit_scan_vmap_v2"
    )
    assert preregistration.controller.filter_device_key_contract == (
        "r015_filter_device_fold_in_keys_v2"
    )
    assert preregistration.controller.filter_microbatch_schedule_id == (
        "fixed_4096_parent_particle_slots_v1"
    )
    assert preregistration.controller.filter_parent_particle_slot_target == 4096
    assert preregistration.controller.filter_group_partner_network_by_prototype is True
    assert preregistration.controller.filter_continuation_member_forwards == 0
    assert (
        preregistration.controller.resampling_interval_environment_steps == 1
    )
    assert preregistration.controller.resampling_timing is None
    assert preregistration.controller.resampling_ess_fraction_threshold is None
    assert preregistration.controller.planning_branches_per_candidate is None
    assert preregistration.controller.planning_branch_sampling == (
        "sampled_hidden_state_branches_v1"
    )
    assert preregistration.controller.planning_branch_belief == (
        "frozen_belief_branch_continuation_v1"
    )
    assert preregistration.controller.continuation_rule_id == (
        "map_prototype_committed_cook_v1"
    )
    assert preregistration.controller.continuation_tie_fallback == "ego_seed100"
    assert preregistration.controller.continuation_parallel_recurrent_member_count == 5
    assert preregistration.controller.continuation_switch_state_rule == (
        "adopt_current_parallel_state_without_reset"
    )
    assert preregistration.controller.initialization_key_source == (
        "r015_filter_device_fold_in_keys_v2"
    )
    assert preregistration.statistics.n_rounds_max == 2500
    assert preregistration.statistics.paired_difference_absolute_bound == 800.0
    assert preregistration.statistics.paired_difference_width == 1600.0
    assert preregistration.statistics.maximum_safety_comparisons == 40_000
    assert preregistration.statistics.round_sampling_contract == (
        "iid_round_vectors_ego_cook_seat_with_prefrozen_replacements_v4"
    )
    assert (
        preregistration.statistics.mechanical_replacement_attempts_per_coordinate
        == 32
    )
    assert preregistration.statistics.ego_agent_id == "agent_1"
    assert preregistration.statistics.zero_count_independence_contract == (
        "independent_firing_prefixes_across_prototypes_v1"
    )
    assert preregistration.statistics.kill_checkpoint_rounds == (
        200,
        400,
        800,
        1600,
        2500,
    )
    assert len(preregistration.probe_registry_semantic_sha256) == 64
    assert preregistration.formal_view_record_path is None
    assert preregistration.firing_count_checkpoint_paths is None
    assert {
        "wrong_delivery_detector",
        "planning_stability_report",
        "pilot_wiring_report",
        "random_key_derivation",
        "decision_evidence_verifier",
        "safety_branch_evidence_verifier",
        "formal_sampling_schedule",
        "trace_replay_verifier",
    }.issubset(preregistration.artifacts)
    assert "evaluator_partner_action" in (
        preregistration.information.forbidden_controller_fields
    )
    assert len(preregistration.support_spec.families) == 2
    assert len(preregistration.support_spec.candidates) == 4
    with pytest.raises(ValueError, match="not frozen"):
        load_r015_preregistration(CONFIG, for_formal_decision=True)


def test_sampling_schedule_must_bind_cook_seat_rounds_and_independent_firing_prefixes(
    tmp_path,
):
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    schedule_path = tmp_path / "formal-sampling-schedule.json"
    schedule = formal_runtime.build_r015_formal_sampling_schedule(
        prototype_ids=tuple(members),
        root_seed=8115,
    )
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    artifacts = dict(preregistration.artifacts)
    artifacts["formal_sampling_schedule"] = ArtifactBinding(
        name="formal_sampling_schedule",
        path=schedule_path,
        sha256="f" * 64,
    )
    preregistration = replace(preregistration, artifacts=artifacts)
    assert len(_load_formal_sampling_schedule(preregistration, members)) == 10_000

    schedule["entries"][0]["episode_seeds_by_attempt_index"][0] ^= 1
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    with pytest.raises(ValueError, match="not derived from its coordinate"):
        _load_formal_sampling_schedule(preregistration, members)

    schedule = formal_runtime.build_r015_formal_sampling_schedule(
        prototype_ids=tuple(members),
        root_seed=8115,
    )
    schedule["entries"][0]["audit_unit_id"] = "0" * 64
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    with pytest.raises(ValueError, match="audit unit was not derived"):
        _load_formal_sampling_schedule(preregistration, members)

    schedule = formal_runtime.build_r015_formal_sampling_schedule(
        prototype_ids=tuple(members),
        root_seed=8115,
    )
    schedule["cross_prototype_shared_randomness_before_firing"] = True
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    with pytest.raises(ValueError, match="independent firing prefixes"):
        _load_formal_sampling_schedule(preregistration, members)

    schedule["cross_prototype_shared_randomness_before_firing"] = False
    schedule["ego_position_one_probability"] = 0.5
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    with pytest.raises(ValueError, match="retired random-seat field"):
        _load_formal_sampling_schedule(preregistration, members)

    schedule.pop("ego_position_one_probability")
    schedule["entries"][0]["ego_position"] = 0
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    with pytest.raises(ValueError, match="schedule row is invalid"):
        _load_formal_sampling_schedule(preregistration, members)


def test_statistics_rejects_retired_random_seat_probability() -> None:
    payload = {
        "partner_prototype_count": 4,
        "pilot_blocks_per_prototype": 20,
        "n_rounds_max": 2500,
        "formal_paired_block_budget": 10000,
        "formal_arm_episode_budget": 30000,
        "formal_arm_environment_step_budget": 12000000,
        "maximum_safety_comparisons": 40000,
        "planning_fork_steps_included": False,
        "delivery_reward": 20.0,
        "practical_margin_fraction": 0.25,
        "return_lower_bound": -400.0,
        "return_upper_bound": 400.0,
        "paired_difference_absolute_bound": 800.0,
        "paired_difference_width": 1600.0,
        "empirical_bernstein_formula": (
            "maurer_pontil_empirical_bernstein_bounded_v1"
        ),
        "round_sampling_contract": (
            "iid_round_vectors_ego_cook_seat_with_prefrozen_replacements_v4"
        ),
        "formal_sampling_schedule_schema": (
            "path_c_r015_formal_sampling_schedule_v4"
        ),
        "mechanical_replacement_attempts_per_coordinate": 32,
        "mechanical_replacement_attempt_indices": "0_through_31",
        "replacement_episode_seed_derivation_id": (
            "sha256_registered_coordinate_low32_allow_collisions_v1"
        ),
        "replacement_episode_seed_collision_policy": (
            "allow_value_collisions_without_redraw"
        ),
        "audit_unit_changes_across_replacement_attempts": False,
        "replacement_seed_selection_uses_outcomes": False,
        "mechanical_attempt_exhaustion_rule": "terminate_entire_formal_audit",
        "ego_agent_id": "agent_1",
        "zero_count_independence_contract": (
            "independent_firing_prefixes_across_prototypes_v1"
        ),
        "alpha_total": 0.05,
        "alpha_safety": 0.025,
        "alpha_eb_net": 0.010,
        "alpha_eb_response": 0.010,
        "alpha_rho": 0.005,
        "kill_checkpoint_rounds": [200, 400, 800, 1600, 2500],
        "alpha_rho_per_checkpoint": 0.001,
        "alpha_rho_per_prototype_cp": 0.00025,
        "safety_risk_limit": 0.05,
        "safety_repeats_per_comparison": 279,
        "formal_effect_look_count": 1,
        "formal_effect_look_round": 2500,
        "ego_position_one_probability": 0.5,
    }
    with pytest.raises(ValueError, match="retired random-seat field"):
        R015Statistics.from_mapping(payload)

    payload.pop("ego_position_one_probability")
    payload["round_sampling_contract"] = (
        "iid_round_vectors_equal_prototype_weights_v1"
    )
    with pytest.raises(ValueError, match="round-sampling contract changed"):
        R015Statistics.from_mapping(payload)


def test_support_registration_has_two_real_families_and_two_runs_per_family():
    support = load_r015_preregistration(CONFIG).support_spec
    assert {family.family_id for family in support.families} == {
        "official_rnn_sp_ippo_v1",
        "official_rnn_op_other_play_v1",
    }
    assert {family.model_class for family in support.families} == {
        "OfficialFlaxRecurrentActor"
    }
    assert {family.action_rule for family in support.families} == {
        "official_flax_categorical_actor_v1"
    }
    signatures = {
        (
            family.training_algorithm,
            family.training_objective,
            family.convention_generation,
        )
        for family in support.families
    }
    assert len(signatures) == 2
    for family in support.families:
        members = [
            candidate
            for candidate in support.candidates
            if candidate.family_id == family.family_id
        ]
        assert len(members) == 2
        assert len({member.training_seed for member in members}) == 2
    assert len({candidate.checkpoint_path for candidate in support.candidates}) == 4


def test_future_random_evidence_is_reconstructable_without_stepwise_keys():
    root_key = "a" * 64
    summary = _future_random_sequence_summary(
        root_key=root_key,
        sequence_length=3,
    )
    expected_sequence = [
        _derive_r015_controller_key(root_key, "future_environment", index)
        for index in range(3)
    ]
    assert summary == {
        "future_random_root_key": root_key,
        "future_random_derivation_contract_id": (
            "path_c_r015_controller_key_v1_future_environment_index_v1"
        ),
        "future_random_step_count": 3,
        "future_random_sequence_sha256": _mapping_sha256(expected_sequence),
    }
    assert "future_random_keys" not in summary


def test_formal_controller_manifest_rejects_two_action_surrogate():
    preregistration = _numeric_preregistration()
    manifest = {
        "schema_version": "path_c_r015_sequential_controller_manifest_v1",
        "controller_kind": "registered_response_sequential_branch_v1",
        "finite_prototype_two_action_surrogate_allowed": False,
        "paired_interfaces": {
            "belief_use": "B_use(q,x,y)",
            "belief_mask": "B_mask(q,x)",
            "continuation_controller": "C(q,x)",
            "common_projected_history": "x",
            "use_only_current_response": "y",
        },
        "continuation_controller_contract": {
            "rule_id": "map_prototype_committed_cook_v1",
            "action_rule": "official_flax_categorical_actor_v1",
            "routing_rule": "unique_exact_map_else_ego_seed100",
            "tie_fallback": "ego_seed100",
            "parallel_recurrent_member_count": 5,
            "switch_state_rule": "adopt_current_parallel_state_without_reset",
            "planning_branch_sampling": "sampled_hidden_state_branches_v1",
            "planning_branch_belief": "frozen_belief_branch_continuation_v1",
            "planning_filter_algorithm_id": (
                "official_history_fully_adapted_particle_filter_v2"
            ),
            "planning_resampling_algorithm": (
                "strict_systematic_per_prototype_v2"
            ),
            "planning_routing_frequency": "commit_once_at_branch_head",
            "execution_routing_frequency": "update_online_each_environment_step",
            "shared_member_selection_rule": "unique_exact_map_else_ego_seed100",
            "controller_input": "q_and_projected_official_history_only",
        },
        "return_contract": {
            "reward": "undiscounted_raw_team_reward",
            "horizon": "full_remaining_episode",
            "discount_factor": 1.0,
        },
        "paired_branch_contract": {
            "same_hidden_execution_state": True,
            "same_task_transition": True,
            "same_future_random_branch": True,
            "current_response_is_only_interface_difference": True,
        },
        "lockstep_contract": {
            "same_frozen_planner": True,
            "same_passive_masked_belief_updates": True,
            "same_episode_random_numbers": True,
            "probe_rule_firing_is_only_branch_point": True,
        },
        "hidden_state_contract": {
            "complete_hidden_state_reconstruction": True,
            "positive_posterior_support_required": True,
            "branch_sampling_rule_id": "sampled_hidden_state_branches_v1",
            "nested_branch_counts": [2, 4, 8, 16],
            "equal_weight_sample_slots": True,
        },
        "score_rule": "S_seq=J_use-V_mask",
        "tie_break_rule": "base_then_probe_registry_order",
        "consultation_steps": list(range(1, 101, 5)),
        "replay_backend": {
            "implementation_path": "path_c_r015_runtime.py",
            "implementation_sha256": "e" * 64,
            "factory_name": "build_r015_replay_backend",
        },
        "ego_action_selection_uses_full_state": False,
        "implementation_path": "registered_sequential_controller.py",
        "implementation_sha256": "c" * 64,
        "response_projection_sha256": "b" * 64,
        "official_history_filter_sha256": "7" * 64,
        "probe_registry_semantic_sha256": (
            preregistration.probe_registry_semantic_sha256
        ),
        "random_key_derivation_sha256": "9" * 64,
        "continuation_controller_sha256": "e" * 64,
    }
    _validate_sequential_controller_manifest(manifest, preregistration)

    surrogate = dict(manifest)
    surrogate["controller_kind"] = "finite_prototype_two_action_surrogate"
    surrogate["finite_prototype_two_action_surrogate_allowed"] = True
    with pytest.raises(ValueError, match="sequential branch controller|surrogate"):
        _validate_sequential_controller_manifest(surrogate, preregistration)


def test_empirical_bernstein_numeric_fixture_and_frozen_safety_count():
    statistics = _statistics()
    interval = maurer_pontil_empirical_bernstein_interval(
        [0.0] * 100,
        alpha=0.1,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    # Hand calculation: 7*2*ln(20)/(3*99) = 0.141212969123...
    assert interval.estimate == 0.0
    assert interval.sample_variance == 0.0
    assert interval.radius == pytest.approx(0.141212969123, rel=1.0e-11)
    assert interval.lower == pytest.approx(-interval.radius)
    assert interval.upper == pytest.approx(interval.radius)
    truncated = maurer_pontil_empirical_bernstein_interval(
        [1.0] * 100,
        alpha=0.1,
        lower_bound=-1.0,
        upper_bound=1.0,
    )
    assert truncated.lower == pytest.approx(1.0 - interval.radius)
    assert truncated.upper == 1.0

    repeats = statistics.safety_repeats_per_comparison
    assert repeats == 279
    comparisons = statistics.maximum_safety_comparisons
    assert comparisons == 40_000
    passed_risk = comparisons * (1.0 - statistics.safety_risk_limit) ** repeats
    previous_risk = comparisons * (
        1.0 - statistics.safety_risk_limit
    ) ** (repeats - 1)
    assert passed_risk <= statistics.alpha_safety
    assert previous_risk > statistics.alpha_safety


def test_clopper_pearson_and_zero_count_kill_fixtures():
    # For x=1,n=2, P_p(X<=1)=1-p^2=0.05, so p_U=sqrt(0.95).
    assert clopper_pearson_upper_bound(1, 2, alpha=0.05) == pytest.approx(
        0.9746794344808963,
        rel=1.0e-12,
    )
    statistics = _statistics()
    prototype_ids = ("p0", "p1", "p2", "p3")

    def zero_count_payload(checkpoint):
        return {
            "schema_version": "path_c_r015_firing_count_checkpoint_v1",
            "experiment_id": "R015",
            "checkpoint_round_count": checkpoint,
            "firing_counts_by_prototype": {
                prototype_id: 0 for prototype_id in prototype_ids
            },
        }

    at_200 = evaluate_r015_firing_count_checkpoint(
        zero_count_payload(200),
        statistics=statistics,
        expected_prototype_ids=prototype_ids,
    )
    assert at_200.verdict == "CONTINUE"
    assert at_200.rho_bar_upper == pytest.approx(-math.log(0.001) / 800)
    assert at_200.effect_absolute_upper == pytest.approx(-math.log(0.001))

    at_400 = evaluate_r015_firing_count_checkpoint(
        zero_count_payload(400),
        statistics=statistics,
        expected_prototype_ids=prototype_ids,
    )
    assert at_400.verdict == "REGISTERED_NEGATIVE"
    assert at_400.registered_negative_reason == "firing_count_upper_below_margin"
    assert at_400.full_state_upper_bound_status == "not_provided"
    assert at_400.substrate_negative_claim_allowed is False
    assert at_400.effect_absolute_upper == pytest.approx(-math.log(0.001) / 2.0)
    assert at_400.alpha_checkpoint == pytest.approx(0.001)
    threshold_blocks = 800.0 * math.log(1000.0) / 5.0
    assert threshold_blocks == pytest.approx(1105.240844637)
    assert 4 * math.ceil(threshold_blocks / 4.0) == 1108

    general = zero_count_payload(200)
    general["firing_counts_by_prototype"]["p0"] = 1
    general_result = evaluate_r015_firing_count_checkpoint(
        general,
        statistics=statistics,
        expected_prototype_ids=prototype_ids,
    )
    assert general_result.method == "per_prototype_one_sided_clopper_pearson_average"

    leaked_effect = zero_count_payload(200)
    leaked_effect["delta_net"] = 0.0
    with pytest.raises(ValueError, match="count-only"):
        evaluate_r015_firing_count_checkpoint(
            leaked_effect,
            statistics=statistics,
            expected_prototype_ids=prototype_ids,
        )

    nested_leak = zero_count_payload(200)
    nested_leak["firing_counts_by_prototype"]["delta_net"] = nested_leak[
        "firing_counts_by_prototype"
    ].pop("p3")
    with pytest.raises(ValueError, match="four frozen prototypes"):
        evaluate_r015_firing_count_checkpoint(
            nested_leak,
            statistics=statistics,
            expected_prototype_ids=prototype_ids,
        )


def test_paired_block_enforces_delta_identity_and_current_response_projection():
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    prototype_id = next(iter(members))
    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    block = validate_r015_paired_block(
        payload, preregistration=preregistration, support_members=members
    )
    assert block.delta_response == pytest.approx(8.0)
    assert block.delta_cost == pytest.approx(-1.0)
    assert block.delta_net == pytest.approx(9.0)
    assert block.delta_net == pytest.approx(block.delta_response - block.delta_cost)

    payload["pairing"]["A2-mask"]["current_response_route"] = "belief_update_only"
    with pytest.raises(ValueError, match="A2-mask"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )

    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    payload["pairing"]["a1_pre_probe_trace_sha256"] = "1" * 64
    with pytest.raises(ValueError, match="A1 and both A2"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )

    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    payload["pairing"]["A2-mask"]["non_belief_response_aliases"] = [
        "copied_response"
    ]
    payload["trace_manifest"]["groups"]["A2-mask"][
        "non_belief_response_aliases"
    ] = ["copied_response"]
    payload["trace_manifest_sha256"] = _mapping_sha256(payload["trace_manifest"])
    with pytest.raises(ValueError, match="response alias"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )


def test_probe_decision_rejects_score_or_window_changes_after_freeze():
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    prototype_id = next(iter(members))
    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    selected = next(
        candidate
        for candidate in payload["probe_decision"]["consultations"][0]["candidates"]
        if candidate["probe_id"] == "stay"
    )
    selected["s_seq"] = -1.0
    changed_trace = _mapping_sha256(payload["probe_decision"])
    payload["pairing"]["a1_score_trace_sha256"] = changed_trace
    payload["pairing"]["a2_score_trace_sha256"] = changed_trace
    with pytest.raises(ValueError, match="score decomposition"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )

    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    payload["probe_decision"]["consultations"][0]["environment_step"] = 101
    payload["probe_decision"]["probe_step"] = 101
    changed_trace = _mapping_sha256(payload["probe_decision"])
    payload["pairing"]["a1_score_trace_sha256"] = changed_trace
    payload["pairing"]["a2_score_trace_sha256"] = changed_trace
    with pytest.raises(ValueError, match="through 100"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )


def test_probe_decision_rejects_random_summary_or_v1_filter_evidence():
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    prototype_id = next(iter(members))

    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    branch = payload["probe_decision"]["consultations"][0]["planner_evidence"][
        "planning_branches"
    ][0]
    branch["future_random_sequence_sha256"] = "0" * 64
    changed_trace = _mapping_sha256(payload["probe_decision"])
    payload["pairing"]["a1_score_trace_sha256"] = changed_trace
    payload["pairing"]["a2_score_trace_sha256"] = changed_trace
    with pytest.raises(ValueError, match="compact future-random"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )

    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    branch = payload["probe_decision"]["consultations"][0]["planner_evidence"][
        "planning_branches"
    ][0]
    branch["source_belief_filter_algorithm_id"] = (
        "stratified_hidden_state_particles_v1"
    )
    changed_trace = _mapping_sha256(payload["probe_decision"])
    payload["pairing"]["a1_score_trace_sha256"] = changed_trace
    payload["pairing"]["a2_score_trace_sha256"] = changed_trace
    with pytest.raises(ValueError, match="non-v2 filter"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )

def test_decision_verifier_rejects_self_consistent_reported_value_tampering():
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    prototype_id = next(iter(members))
    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    selected = next(
        candidate
        for candidate in payload["probe_decision"]["consultations"][0]["candidates"]
        if candidate["probe_id"] == "stay"
    )
    selected["j_use"] = 2.0
    selected["i_response"] = 2.0
    selected["s_seq"] = 2.0
    changed_trace = _mapping_sha256(payload["probe_decision"])
    payload["pairing"]["a1_score_trace_sha256"] = changed_trace
    payload["pairing"]["a2_score_trace_sha256"] = changed_trace
    with pytest.raises(ValueError, match="Frozen decision verification"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )


def test_controller_replay_rejects_legal_action_and_state_self_report_tampering():
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    prototype_id = next(iter(members))
    member = members[prototype_id]
    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    trace = payload["trace_manifest"]
    record = trace["groups"]["A2-use"]
    steps = record["environment_steps"]
    changed_controller_state = _mapping_sha256({"tampered_controller_state": 3})
    changed_partner_state = _mapping_sha256({"tampered_partner_state": 3})
    steps[2]["ego_action"] = "up"
    steps[2]["partner_action"] = "up"
    steps[2]["controller_recurrent_state_after_sha256"] = changed_controller_state
    steps[3]["controller_recurrent_state_before_sha256"] = changed_controller_state
    steps[2]["partner_recurrent_state_after_sha256"] = changed_partner_state
    steps[3]["partner_recurrent_state_before_sha256"] = changed_partner_state
    self_report = record["controller_replay_evidence"]
    self_report["ego_actions"][2] = "up"
    self_report["partner_actions"][2] = "up"
    self_report["state_writes"][2]["after_sha256"] = changed_controller_state
    self_report["state_writes"][3]["before_sha256"] = changed_controller_state
    self_report["partner_state_writes"][2]["after_sha256"] = changed_partner_state
    self_report["partner_state_writes"][3]["before_sha256"] = changed_partner_state
    trace["replay_verification"] = _trace_replay_verifier(
        trace, _trace_replay_context(preregistration, member)
    )
    payload["trace_manifest_sha256"] = _mapping_sha256(trace)
    with pytest.raises(ValueError, match="controller action or state replay"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )


def test_paired_block_rejects_evaluator_only_information_and_unpaired_identity():
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    prototype_id = next(iter(members))
    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    payload["arms"]["A2-use"]["forbidden_fields_read"] = ["partner_identity"]
    with pytest.raises(ValueError, match="forbidden"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )

    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    payload["arms"]["A2-use"]["episode_seed"] = 18
    with pytest.raises(ValueError, match="share prototype"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )

    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    payload["trace_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="trace manifest digest"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )


def test_safety_certificate_requires_four_zero_event_comparisons_at_fixed_count():
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    prototype_id = next(iter(members))
    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    payload["safety"]["comparisons"][0]["repetitions"] -= 1
    payload["safety"]["comparisons"][0]["branch_count"] -= 1
    with pytest.raises(ValueError, match="repeat count"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )

    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    payload["safety"]["comparisons"][0]["branch_count"] -= 1
    with pytest.raises(ValueError, match="independent-branch"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )


def test_safety_verifier_rejects_self_consistent_wrong_delivery_tampering():
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    prototype_id = next(iter(members))
    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    comparison = payload["safety"]["comparisons"][0]
    comparison["wrong_delivery_count"] = 1
    comparison["branch_results"][0]["wrong_delivery_detected"] = True
    with pytest.raises(ValueError, match="Frozen safety verification"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )


def test_zero_probe_block_is_valid_only_when_all_formal_groups_match():
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    prototype_id = next(iter(members))
    payload = _paired_block_payload(
        preregistration,
        prototype_id=prototype_id,
        returns=(3.0, 3.0, 3.0),
        probe_executed=False,
    )
    block = validate_r015_paired_block(
        payload, preregistration=preregistration, support_members=members
    )
    assert block.delta_response == 0.0
    assert block.delta_cost == 0.0
    assert block.delta_net == 0.0

    payload["arms"]["A2-use"]["raw_return"] = 4.0
    with pytest.raises(ValueError, match="raw return|Without a probe"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )

    payload = _paired_block_payload(
        preregistration,
        prototype_id=prototype_id,
        returns=(3.0, 3.0, 3.0),
        probe_executed=False,
    )
    payload["trace_manifest"]["groups"]["A2-use"]["environment_steps"][200][
        "partner_action"
    ] = "up"
    payload["trace_manifest_sha256"] = _mapping_sha256(payload["trace_manifest"])
    with pytest.raises(ValueError, match="byte-identical full trajectories"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )


def test_fire_block_requires_shared_prefix_and_one_shared_firing_indicator():
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    prototype_id = next(iter(members))
    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    payload["trace_manifest"]["groups"]["A2-use"]["environment_steps"][0][
        "partner_action"
    ] = "up"
    payload["trace_manifest_sha256"] = _mapping_sha256(payload["trace_manifest"])
    with pytest.raises(ValueError, match="byte-identical through t_p-1"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )

    payload = _paired_block_payload(preregistration, prototype_id=prototype_id)
    payload["arms"]["A2-use"]["firing_indicator"] = False
    with pytest.raises(ValueError, match="shared block-level property"):
        validate_r015_paired_block(
            payload, preregistration=preregistration, support_members=members
        )


def test_three_state_decision_keeps_boundary_overlap_undecidable():
    response_positive = EffectInterval(2.0, 1.0, 3.0, 1.0)
    assert decide_r015(
        delta_net=EffectInterval(8.0, 6.0, 10.0, 2.0),
        delta_response=response_positive,
        practical_margin=5.0,
    ) == "GO"
    assert decide_r015(
        delta_net=EffectInterval(2.0, 0.0, 4.0, 2.0),
        delta_response=response_positive,
        practical_margin=5.0,
    ) == "REGISTERED_NEGATIVE"
    assert decide_r015(
        delta_net=EffectInterval(5.0, 4.0, 6.0, 1.0),
        delta_response=response_positive,
        practical_margin=5.0,
    ) == "UNDECIDABLE"
    assert decide_r015(
        delta_net=EffectInterval(4.0, 3.0, 5.0, 1.0),
        delta_response=response_positive,
        practical_margin=5.0,
    ) == "UNDECIDABLE"
    assert decide_r015(
        delta_net=EffectInterval(6.0, 5.0, 7.0, 1.0),
        delta_response=response_positive,
        practical_margin=5.0,
    ) == "UNDECIDABLE"
    assert decide_r015(
        delta_net=EffectInterval(8.0, 6.0, 10.0, 2.0),
        delta_response=EffectInterval(1.0, 0.0, 2.0, 1.0),
        practical_margin=5.0,
    ) == "UNDECIDABLE"


def test_internal_aggregation_uses_equal_blocks_and_accepts_seed_collisions():
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    n = preregistration.statistics.n_rounds_max
    blocks = []
    for prototype_index, (prototype_id, member) in enumerate(members.items()):
        for episode_index in range(n):
            audit_unit_id = f"unit/{prototype_index}/{episode_index}"
            episode_seed = (
                0
                if prototype_index == 0 and episode_index in (0, 1)
                else episode_index
            )
            episode_random_key = _derive_r015_random_key(
                audit_unit_id=audit_unit_id,
                partner_prototype_id=prototype_id,
                episode_seed=episode_seed,
                purpose="actual_episode",
                environment_step=0,
                branch_index=0,
            )
            arms = {
                group: ArmOutcome(
                    group=group,
                    prototype_id=prototype_id,
                    family_id=member["family_id"],
                    training_seed=member["training_seed"],
                    training_run_id=member["training_run_id"],
                    ego_position=1,
                    initial_state_sha256="c" * 64,
                    episode_seed=episode_seed,
                    firing_indicator=True,
                    raw_return=0.0,
                    valid=True,
                    information_source="official_local_history_only",
                    forbidden_fields_read=(),
                    episode_random_key=episode_random_key,
                    controller_contract_sha256="a" * 64,
                    ego_checkpoint_sha256="4" * 64,
                    environment_config_sha256="5" * 64,
                    environment_source_sha256="6" * 64,
                    official_history_filter_sha256="7" * 64,
                )
                for group in ("A1", "A2-mask", "A2-use")
            }
            blocks.append(
                R015PairedBlock(
                    round_index=episode_index + 1,
                    audit_unit_id=audit_unit_id,
                    prototype_id=prototype_id,
                    family_id=member["family_id"],
                    training_seed=member["training_seed"],
                    training_run_id=member["training_run_id"],
                    ego_position=1,
                    initial_state_sha256="c" * 64,
                    episode_seed=episode_seed,
                    mechanical_attempt_index=0,
                    formal_effect_look_number=1,
                    pilot_data=False,
                    arms=arms,
                    probe_executed=True,
                    probe_step=1,
                    no_probe_reason=None,
                    planning_random_stream_key=(
                        f"planning/{prototype_index}/{episode_index}"
                    ),
                    score_random_stream_key=(
                        f"score/{prototype_index}/{episode_index}"
                    ),
                    safety_comparisons=(),
                    candidate_evaluation_count=1,
                    safety_rejection_count=0,
                    non_positive_score_count=0,
                    window_expired_count=0,
                )
            )
    colliding_blocks = [
        block
        for block in blocks
        if block.prototype_id == next(iter(members)) and block.round_index in (1, 2)
    ]
    assert len(colliding_blocks) == 2
    assert len({block.episode_seed for block in colliding_blocks}) == 1
    assert len({block.initial_state_sha256 for block in colliding_blocks}) == 1
    assert len({block.round_index for block in colliding_blocks}) == 2
    assert len({block.audit_unit_id for block in colliding_blocks}) == 2

    kill_checkpoint_decisions = tuple(
        evaluate_r015_firing_count_checkpoint(
            {
                "schema_version": "path_c_r015_firing_count_checkpoint_v1",
                "experiment_id": "R015",
                "checkpoint_round_count": checkpoint,
                "firing_counts_by_prototype": {
                    prototype_id: checkpoint for prototype_id in members
                },
            },
            statistics=preregistration.statistics,
            expected_prototype_ids=tuple(members),
        )
        for checkpoint in preregistration.statistics.kill_checkpoint_rounds
    )
    decision = _aggregate_validated_r015_blocks(
        blocks,
        preregistration=preregistration,
        support_members=members,
        expected_sampling_schedule=frozenset(
            (
                block.round_index,
                block.audit_unit_id,
                block.prototype_id,
                (block.episode_seed, *(
                    block.episode_seed + offset for offset in range(1, 32)
                )),
                block.ego_position,
            )
            for block in blocks
        ),
        kill_checkpoint_decisions=kill_checkpoint_decisions,
    )
    assert decision.verdict == "UNDECIDABLE"
    assert decision.paired_block_count == 4 * n
    assert decision.formal_round_count == n
    assert decision.firing_counts_by_prototype == {
        prototype_id: n for prototype_id in members
    }
    assert decision.full_state_upper_bound_status == "not_provided"
    assert decision.substrate_negative_claim_allowed is False


def test_firing_count_kill_prevents_the_single_effect_look():
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    kill = evaluate_r015_firing_count_checkpoint(
        {
            "schema_version": "path_c_r015_firing_count_checkpoint_v1",
            "experiment_id": "R015",
            "checkpoint_round_count": 400,
            "firing_counts_by_prototype": {
                prototype_id: 0 for prototype_id in members
            },
        },
        statistics=preregistration.statistics,
        expected_prototype_ids=tuple(members),
    )
    assert kill.verdict == "REGISTERED_NEGATIVE"
    with pytest.raises(ValueError, match="cannot be read after"):
        _aggregate_validated_r015_blocks(
            (),
            preregistration=preregistration,
            support_members=members,
            expected_sampling_schedule=frozenset(),
            kill_checkpoint_decisions=(kill,),
        )


def test_checkpoint_loader_finishes_early_kill_without_later_count_files(tmp_path):
    preregistration = _numeric_preregistration()
    members = _support_members(preregistration)
    checkpoints = preregistration.statistics.kill_checkpoint_rounds
    checkpoint_paths = tuple(
        tmp_path / f"firing-count-{checkpoint}.json" for checkpoint in checkpoints
    )

    for checkpoint, path in zip(checkpoints[:2], checkpoint_paths[:2], strict=True):
        path.write_text(
            json.dumps(
                {
                    "schema_version": "path_c_r015_firing_count_checkpoint_v1",
                    "experiment_id": "R015",
                    "checkpoint_round_count": checkpoint,
                    "firing_counts_by_prototype": {
                        prototype_id: 0 for prototype_id in members
                    },
                }
            ),
            encoding="utf-8",
        )
    preregistration = replace(
        preregistration,
        firing_count_checkpoint_paths=checkpoint_paths,
    )
    decisions = _load_firing_count_checkpoint_decisions(
        preregistration,
        members,
    )
    assert tuple(item.checkpoint_round_count for item in decisions) == (200, 400)
    assert decisions[-1].verdict == "REGISTERED_NEGATIVE"
    assert all(not path.exists() for path in checkpoint_paths[2:])

    checkpoint_paths[2].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="later checkpoint"):
        _load_firing_count_checkpoint_decisions(preregistration, members)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_support_report_rejects_plain_text_artifacts_without_episode_evidence(tmp_path):
    preregistration = load_r015_preregistration(CONFIG)
    candidates = []
    members = {}
    for index, candidate in enumerate(preregistration.support_spec.candidates):
        checkpoint = tmp_path / f"checkpoint-{index}.pt"
        training_config = tmp_path / f"training-{index}.yaml"
        training_manifest = tmp_path / f"training-manifest-{index}.json"
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        training_config.write_text(f"seed: {candidate.training_seed}\n", encoding="utf-8")
        training_manifest.write_text(
            f'{{"seed": {candidate.training_seed}}}\n', encoding="utf-8"
        )
        candidate = replace(
            candidate,
            checkpoint_path=checkpoint,
            training_config_path=training_config,
            training_manifest_path=training_manifest,
        )
        candidates.append(candidate)
        family = preregistration.support_spec.family(candidate.family_id)
        training_config_sha256 = _sha256(training_config)
        environment_config_sha256 = "e" * 64
        training_implementation_sha256 = (
            "f" * 64
            if candidate.family_id == "official_rnn_sp_ippo_v1"
            else "d" * 64
        )
        training_run_id = partner_training_run_id(
            family_spec_sha256=family.family_spec_sha256,
            training_seed=candidate.training_seed,
            training_config_sha256=training_config_sha256,
            environment_config_sha256=environment_config_sha256,
            training_implementation_sha256=training_implementation_sha256,
        )
        members[candidate.candidate_id] = {
            "artifact_verified": True,
            "family_id": candidate.family_id,
            "family_spec_sha256": family.family_spec_sha256,
            "training_seed": candidate.training_seed,
            "snapshot_environment_steps": candidate.expected_environment_steps,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "model_weights_sha256": hashlib.sha256(
                f"weights-{index}".encode()
            ).hexdigest(),
            "training_config_path": str(training_config),
            "training_config_sha256": training_config_sha256,
            "training_manifest_path": str(training_manifest),
            "training_manifest_sha256": _sha256(training_manifest),
            "environment_config_sha256": environment_config_sha256,
            "training_implementation_sha256": training_implementation_sha256,
            "training_run_id": training_run_id,
            "admitted": True,
        }
    support_spec = replace(preregistration.support_spec, candidates=tuple(candidates))
    report = {
        "schema_version": "path_c_r015_partner_support_report_v2",
        "registration_status": "configured",
        "support_status": "admitted",
        "support_complete": True,
        "scientific_readout_allowed": False,
        "artifact_verification_status": "verified",
        "families": {
            family.family_id: {
                **family.definition(),
                "family_spec_sha256": family.family_spec_sha256,
            }
            for family in support_spec.families
        },
        "members": members,
        "qualified_candidate_ids": sorted(members),
        "members_by_family": {
            family.family_id: sorted(
                candidate.candidate_id
                for candidate in support_spec.candidates
                if candidate.family_id == family.family_id
            )
            for family in support_spec.families
        },
    }
    with pytest.raises(ValueError, match="SHA-256|episode-level evidence"):
        validate_r015_support_report(report, support_spec)
