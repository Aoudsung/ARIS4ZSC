#!/usr/bin/env python3
"""用五个真实 checkpoint 完成 R015 的 Type-A 接线回读。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from experiments.overcooked_v2.official.r015_runtime_bridge import (
    build_official_r015_production_backend,
)
from experiments.overcooked_v2.path_c_pool_admission import (
    R015PartnerSupportSpec,
    load_r015_partner_support_config,
)
from experiments.overcooked_v2.path_c_r015_design import (
    _six_coordinate_random_key,
    load_r015_design_protocol,
)
from experiments.overcooked_v2.path_c_r015_controller import (
    R015SequentialPlannerV1,
    NestedHiddenStateBranchScheduleV1,
    canonical_sha256,
    default_r015_probe_scripts,
)


EXPECTED_STABLE_ADAPTER_SHA256 = (
    "b139587a571bd187078f4a251b1427d912efdd62e42847649c719fc5293f11c8"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _real_policy_equivalence(backend: object) -> dict[str, bool]:
    """用五个真实参数树比较官方编译调用与 R015 运行包装器。"""

    import jax
    import jax.numpy as jnp

    observations, _ = backend.full_horizon_executor.adapter.reset(15071599)
    observation = observations["agent_1"]
    logits_equal = True
    recurrent_states_equal = True
    actions_equal = True
    for index, member_id in enumerate(backend.continuation_controller.member_ids):
        loaded = backend.policies[member_id]
        recurrent_state = loaded.adapter.initial_state(1)

        @jax.jit
        def reference_apply(params, state, local_observation, episode_start):
            return loaded.adapter.apply_actor(
                params,
                state,
                local_observation,
                episode_start,
            )

        reference_state, reference_logits = reference_apply(
            loaded.policy.params,
            recurrent_state,
            observation,
            True,
        )
        wrapped_state, wrapped_logits = loaded.policy.network.apply_actor(
            loaded.policy.params,
            recurrent_state,
            observation,
            True,
        )
        logits_equal = logits_equal and np.array_equal(
            np.asarray(reference_logits), np.asarray(wrapped_logits)
        )
        reference_leaves = jax.tree_util.tree_leaves(reference_state)
        wrapped_leaves = jax.tree_util.tree_leaves(wrapped_state)
        recurrent_states_equal = recurrent_states_equal and (
            len(reference_leaves) == len(wrapped_leaves)
            and all(
                np.array_equal(np.asarray(reference), np.asarray(wrapped))
                for reference, wrapped in zip(reference_leaves, wrapped_leaves)
            )
        )
        action_key = jax.random.PRNGKey(15071600 + index)
        reference_action = jax.random.categorical(
            action_key, jnp.asarray(reference_logits)
        )
        wrapped_action = jax.random.categorical(
            action_key, jnp.asarray(wrapped_logits)
        )
        actions_equal = actions_equal and np.array_equal(
            np.asarray(reference_action), np.asarray(wrapped_action)
        )
    return {
        "official_wrapper_logits_equal_all_five": logits_equal,
        "official_wrapper_recurrent_state_equal_all_five": (
            recurrent_states_equal
        ),
        "official_wrapper_action_equal_all_five": actions_equal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="核查 R015 五 checkpoint 生产接线。")
    parser.add_argument("--design-protocol", required=True, type=Path)
    parser.add_argument("--support-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--authorized", action="store_true")
    arguments = parser.parse_args()
    if not arguments.authorized:
        raise PermissionError("R015 Type-A wiring check requires --authorized.")
    if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
        raise RuntimeError("R015 Type-A wiring check requires JAX_PLATFORMS=cuda,cpu.")

    protocol = load_r015_design_protocol(arguments.design_protocol)
    support_config = load_r015_partner_support_config(arguments.support_config)
    support_spec = R015PartnerSupportSpec.from_mapping(support_config)
    backend = build_official_r015_production_backend(
        protocol,
        support_spec,
        config_base_path=arguments.design_protocol.resolve().parent,
    )
    stable_adapter = (
        Path(__file__).resolve().parents[1]
        / "official"
        / "overcooked_v2_experiments_adapter.py"
    )
    adapter_sha256 = _file_sha256(stable_adapter)
    prototype_id = backend.prototype_ids[0]
    diagnostic = backend.full_horizon_executor.run_a0_diagnostic_episode(
        partner_prototype_id=prototype_id,
        episode_seed=15071500,
        audit_unit_id="r015_type_a_a0_full_episode_20260715",
    )
    policy_equivalence = _real_policy_equivalence(backend)
    type_a_key = canonical_sha256(
        ["r015_type_a_nested_planning_20260715", "identity_free_filter_key"]
    )
    planning_history, planning_belief = (
        backend.full_horizon_executor.build_type_a_first_consultation(
            partner_prototype_id=prototype_id,
            audit_key=type_a_key,
            particles_per_prototype=64,
        )
    )
    planning_key = _six_coordinate_random_key(
        audit_unit_id="r015_type_a_first_consultation",
        partner_prototype_id=prototype_id,
        episode_seed=15071599,
        purpose="value_planning",
        environment_step=1,
    )
    score_key = _six_coordinate_random_key(
        audit_unit_id="r015_type_a_first_consultation",
        partner_prototype_id=prototype_id,
        episode_seed=15071599,
        purpose="score_estimation",
        environment_step=1,
    )
    planning_started = time.monotonic()
    doubled = R015SequentialPlannerV1(
        default_r015_probe_scripts(), 4
    ).evaluate(
        history=planning_history,
        belief=planning_belief,
        environment_step=1,
        planning_key=planning_key,
        score_key=score_key,
        executor=backend.full_horizon_executor,
    )
    lower = R015SequentialPlannerV1(
        default_r015_probe_scripts(), 2
    ).evaluate(
        history=planning_history,
        belief=planning_belief,
        environment_step=1,
        planning_key=planning_key,
        score_key=score_key,
        executor=backend.full_horizon_executor,
        cached_planning_branches=doubled.planning_branches,
    )
    planning_wall_seconds = time.monotonic() - planning_started
    planning_cost = dict(doubled.cost_accounting)
    lower_cost = dict(lower.cost_accounting)
    comparison_sample = NestedHiddenStateBranchScheduleV1.build(
        belief=planning_belief,
        planning_key=planning_key,
        sample_count=4,
    ).samples[0]
    per_branch_reference = backend.full_horizon_executor.rollout_planning_batch(
        # Repeat one branch to the same four-slot shape as the independently
        # sampled batch.  This keeps the registered GPU batch shape fixed while
        # isolating whether neighbouring hidden states affect a branch result.
        samples=(comparison_sample,) * 4,
        belief=planning_belief,
        history=planning_history,
        scripts=default_r015_probe_scripts(),
        remaining_steps=399,
    )
    if len(per_branch_reference.samples) != 4:
        raise RuntimeError("R015 repeated per-branch reference changed its batch shape.")
    reference_sample = per_branch_reference.samples[0]
    comparison_script = default_r015_probe_scripts()[0]
    reference_base = reference_sample.base
    reference_pair = reference_sample.probe_pairs[comparison_script.probe_id]
    repeated_reference_sha256 = canonical_sha256(
        [
            {
                "base": {
                    "raw_return": sample.base.raw_return,
                    "ego": sample.base.ego_action_sequence_sha256,
                    "partner": sample.base.partner_action_sequence_sha256,
                    "environment": sample.base.final_environment_state_sha256,
                    "partner_state": sample.base.final_partner_state_sha256,
                    "continuation_states": (
                        sample.base.final_continuation_states_sha256
                    ),
                },
                "probe_pairs": {
                    probe_id: {
                        label: {
                            "raw_return": rollout.raw_return,
                            "ego": rollout.ego_action_sequence_sha256,
                            "partner": rollout.partner_action_sequence_sha256,
                            "environment": rollout.final_environment_state_sha256,
                            "partner_state": rollout.final_partner_state_sha256,
                            "continuation_states": (
                                rollout.final_continuation_states_sha256
                            ),
                        }
                        for label, rollout in (
                            ("mask", pair.masked),
                            ("use", pair.used),
                        )
                    }
                    for probe_id, pair in sample.probe_pairs.items()
                },
            }
            for sample in per_branch_reference.samples
        ]
    )
    one_reference_sha256 = canonical_sha256(
        [
            {
                "base": {
                    "raw_return": reference_sample.base.raw_return,
                    "ego": reference_sample.base.ego_action_sequence_sha256,
                    "partner": reference_sample.base.partner_action_sequence_sha256,
                    "environment": reference_sample.base.final_environment_state_sha256,
                    "partner_state": reference_sample.base.final_partner_state_sha256,
                    "continuation_states": (
                        reference_sample.base.final_continuation_states_sha256
                    ),
                },
                "probe_pairs": {
                    probe_id: {
                        label: {
                            "raw_return": rollout.raw_return,
                            "ego": rollout.ego_action_sequence_sha256,
                            "partner": rollout.partner_action_sequence_sha256,
                            "environment": rollout.final_environment_state_sha256,
                            "partner_state": rollout.final_partner_state_sha256,
                            "continuation_states": (
                                rollout.final_continuation_states_sha256
                            ),
                        }
                        for label, rollout in (
                            ("mask", pair.masked),
                            ("use", pair.used),
                        )
                    }
                    for probe_id, pair in reference_sample.probe_pairs.items()
                },
            }
        ]
        * 4
    )
    batched_branch = next(
        branch
        for branch in doubled.planning_branches
        if branch["canonical_slot_16"] == comparison_sample.canonical_slot_16
    )
    batched_probe = batched_branch["candidates"][comparison_script.probe_id]
    base_summary_fields = (
        "ego_action_sequence_sha256",
        "partner_action_sequence_sha256",
        "final_environment_state_sha256",
        "final_partner_state_sha256",
        "final_continuation_states_sha256",
    )
    batch_reference_base_equal = math.isclose(
        reference_base.raw_return,
        float(batched_branch["v_base_return"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ) and all(
        getattr(reference_base, field)
        == batched_branch["base_" + field]
        for field in base_summary_fields
    )
    batch_reference_pair_equal = all(
        (
            math.isclose(
                scalar.raw_return,
                float(batched_probe[f"j_{label}_return"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and scalar.ego_action_sequence_sha256
            == batched_probe[f"{label}_ego_action_sequence_sha256"]
            and scalar.partner_action_sequence_sha256
            == batched_probe[f"{label}_partner_action_sequence_sha256"]
            and scalar.final_environment_state_sha256
            == batched_probe[f"{label}_final_environment_state_sha256"]
            and scalar.final_partner_state_sha256
            == batched_probe[f"{label}_final_partner_state_sha256"]
            and scalar.final_continuation_states_sha256
            == batched_probe[f"{label}_final_continuation_states_sha256"]
        )
        for label, scalar in (
            ("mask", reference_pair.masked),
            ("use", reference_pair.used),
        )
    )
    checks = {
        "stable_adapter_source_binding": (
            adapter_sha256 == EXPECTED_STABLE_ADAPTER_SHA256
        ),
        "five_checkpoint_members_loaded": len(backend.policies) == 5,
        "four_partner_prototypes_loaded": len(backend.prototype_ids) == 4,
        "full_horizon_executor_configured": (
            backend.full_horizon_executor is not None
        ),
        "a0_full_episode_has_400_steps": len(diagnostic["trajectory"]) == 400,
        "a0_full_episode_ends_at_step_400": (
            diagnostic["trajectory"][-1]["done"] is True
        ),
        "a0_excluded_from_formal_pairing": (
            diagnostic["formal_pairing_eligible"] is False
        ),
        "scientific_readout_closed": (
            diagnostic["scientific_readout_allowed"] is False
        ),
        **policy_equivalence,
        "r2_reconstructed_without_new_samples": (
            lower_cost["new_sample_count"] == 0
        ),
        "r4_computed_four_unique_samples": (
            planning_cost["new_sample_count"] == 4
        ),
        "r4_exact_real_transition_count": (
            planning_cost["new_real_environment_transitions"] == 20_724
        ),
        "r4_exact_branch_head_particle_transition_count": (
            planning_cost["new_branch_head_particle_transitions"] == 6_144
        ),
        "compiled_batch_call_count_recorded": (
            planning_cost["compiled_batch_calls"] == 4
        ),
        "batched_planning_has_no_stepwise_host_sync": (
            planning_cost["host_sync_inside_environment_loop"] is False
        ),
        "repeated_per_branch_reference_is_identical_in_all_slots": (
            repeated_reference_sha256 == one_reference_sha256
        ),
        "batch_per_branch_reference_base_action_return_state_equal": (
            batch_reference_base_equal
        ),
        "batch_per_branch_reference_probe_pair_action_return_state_equal": (
            batch_reference_pair_equal
        ),
    }
    report = {
        "schema_version": "path_c_r015_production_wiring_type_a_v2",
        "scientific_readout_allowed": False,
        "design_or_pilot_data_generated": False,
        "stable_adapter_path": str(stable_adapter),
        "stable_adapter_sha256": adapter_sha256,
        "member_ids": list(backend.continuation_controller.member_ids),
        "a0_diagnostic": {
            "episode_seed": diagnostic["episode_seed"],
            "trajectory_sha256": diagnostic["trajectory_sha256"],
            "correct_delivery_count": diagnostic["correct_delivery_count"],
            "wrong_delivery_count": diagnostic["wrong_delivery_count"],
        },
        "nested_planning_type_a": {
            "branch_sampling_rule_id": doubled.branch_sampling_rule_id,
            "branch_belief_rule_id": doubled.branch_belief_rule_id,
            "planning_branch_evidence_sha256": canonical_sha256(
                list(doubled.planning_branches)
            ),
            "lower_branch_evidence_sha256": canonical_sha256(
                list(lower.planning_branches)
            ),
            "cost_accounting": planning_cost,
            "lower_cost_accounting": lower_cost,
            "wall_seconds": planning_wall_seconds,
            "real_transition_throughput_per_second": (
                planning_cost["new_real_environment_transitions"]
                / planning_wall_seconds
            ),
            "value_or_decision_fields_saved": False,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(arguments.output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["passed"]:
        raise RuntimeError("R015 Type-A production wiring checks did not all pass.")


if __name__ == "__main__":
    main()
