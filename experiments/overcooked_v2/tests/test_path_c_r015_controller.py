"""R015 审计控制器与 OvercookedV2 配对接线测试。"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json

import numpy as np
import pytest

from experiments.overcooked_v2 import path_c_r015_design as design
from experiments.overcooked_v2.path_c_r015_controller import (
    ContinuationLibraryStatesV1,
    ContinuationMemberStepV1,
    CurrentResponseProjectionV1,
    FullHorizonRolloutV1,
    HiddenStateParticleV1,
    MAPPrototypeCommittedCookV1,
    NestedHiddenStateBranchScheduleV1,
    OfficialHistoryV1,
    PairedProbeRolloutV1,
    PlanningBatchRolloutsV1,
    PlanningSampleRolloutsV1,
    R015_CONTINUATION_CONTROLLER_ID,
    R015_FILTER_ALGORITHM_ID_V1,
    R015_FILTER_ALGORITHM_ID_V2,
    R015_FUTURE_RANDOM_DERIVATION_ID,
    R015_RESAMPLING_ALGORITHM_ID_V1,
    R015_RESAMPLING_ALGORITHM_ID_V2,
    R015SequentialControllerV1,
    R015SequentialPlannerV1,
    SafetyDecisionV1,
    StratifiedParticleBeliefV1,
    apply_paired_response_update,
    canonical_sha256,
    continuation_controller_manifest_payload,
    default_r015_probe_scripts,
    derive_controller_key,
)
from experiments.overcooked_v2.official.r015_runtime_bridge import (
    OCV2R015ProductionBackendV1,
    _CompiledR015ActorAdapter,
    _jax_key as _batch_jax_key,
)
from experiments.overcooked_v2.path_c_r015_runtime import (
    CallableR015ReplayBackendV1,
    OCV2SnapshotStepRuntimeV1,
    R015ExecutedSegmentV1,
    R015PairedEpisodeRunnerV1,
    build_r015_fire_trace_groups,
    build_r015_no_fire_trace_groups,
    validate_r015_lockstep_groups,
    verify_r015_probe_decision,
    verify_r015_safety_comparison,
    verify_r015_trace_manifest,
)
from experiments.overcooked_v2.path_c_r015_full_horizon import (
    OCV2R015FullHorizonExecutorV1,
    R015_RESPONSE_SPEC_PAYLOAD,
    _jax_key as _scalar_jax_key,
)
from experiments.overcooked_v2.path_c_standard import StandardEnvConfig


def _sha(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _future_random_fields(root_key: str, sequence_length: int) -> dict[str, object]:
    sequence = [
        derive_controller_key(root_key, "future_environment", index)
        for index in range(sequence_length)
    ]
    return {
        "future_random_root_key": root_key,
        "future_random_derivation_contract_id": (
            R015_FUTURE_RANDOM_DERIVATION_ID
        ),
        "future_random_step_count": sequence_length,
        "future_random_sequence_sha256": canonical_sha256(sequence),
    }


def _history(step: int = 1) -> OfficialHistoryV1:
    return OfficialHistoryV1(
        (
            {
                "official_local_observation": {"step": step},
                "ego_action_history": ["stay"],
                "raw_team_reward_history": [0.0],
                "episode_boundaries": [False],
            },
        )
    )


def _belief() -> StratifiedParticleBeliefV1:
    ids = ("ippo-101", "ippo-102", "q-201", "q-202")

    def factory(prototype_id, index, key):
        return HiddenStateParticleV1(
            prototype_id=prototype_id,
            state_sha256=canonical_sha256(
                {"prototype_id": prototype_id, "index": index, "key": key}
            ),
            state={"private": prototype_id, "key": key},
            weight=1.0,
        )

    return StratifiedParticleBeliefV1.initialize(
        prototype_ids=ids,
        particles_per_prototype=1,
        registered_prototype_prior={prototype_id: 0.25 for prototype_id in ids},
        resampling_algorithm=R015_RESAMPLING_ALGORITHM_ID_V2,
        resampling_interval_environment_steps=1,
        initialization_key="1" * 64,
        factory=factory,
    )


def test_full_horizon_candidate_status_and_response_registration_are_static():
    executor = object.__new__(OCV2R015FullHorizonExecutorV1)
    script = default_r015_probe_scripts()[0]
    assert executor.candidate_status(
        history=_history(), script=script, environment_step=1
    ) == (True, True)
    assert executor.candidate_status(
        history=_history(), script=script, environment_step=101
    ) == (False, False)
    assert R015_RESPONSE_SPEC_PAYLOAD["response_classes"] == [
        "visible",
        "unseen",
        "local_non_agent_change",
    ]
    assert R015_RESPONSE_SPEC_PAYLOAD["latency_bin_upper_bounds"] == [1]
    assert "filter_initialization_key" in inspect.signature(
        OCV2R015FullHorizonExecutorV1.run_paired_block
    ).parameters
    assert hasattr(OCV2R015FullHorizonExecutorV1, "run_a0_diagnostic_episode")


def test_scalar_and_batched_planning_use_the_same_jax_key_conversion():
    key = "0123456789abcdef" + "0" * 48
    assert np.array_equal(
        np.asarray(_scalar_jax_key(key)),
        np.asarray(_batch_jax_key(key)),
    )


def test_design_filter_advances_the_complete_particle_cloud_in_one_batch():
    source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1.update_beliefs_v1_diagnostic
    )
    assert "run_frozen_trajectory_batch" in source
    assert 'key_contract="filter_transition_v1"' in source
    assert "transition=self._particle_transition" not in source
    assert "materialize_state_hashes: bool = True" in source
    assert "parent_state_sha256" in source
    assert "copy_state=False" in source

    conversion_source = inspect.getsource(
        OCV2R015FullHorizonExecutorV1._kernel_from_batch_output
    )
    assert "copy_state: bool = True" in conversion_source
    assert "if copy_state:" in conversion_source
    assert "environment_state = output" in conversion_source


def test_design_filter_materializes_complete_particle_hashes_only_at_consultations():
    replay_source = inspect.getsource(design._replay_filter_episode)
    assert "materialize_state_hashes=False" in replay_source
    assert "materialize_belief_state_hashes" in replay_source
    assert "if retain_consultation_states:" in replay_source
    assert replay_source.index("materialize_belief_state_hashes") < replay_source.index(
        "states_by_step[environment_step]"
    )

    design_source = inspect.getsource(design.run_r015_design)
    candidate_loop = design_source.split("filter_results =", maxsplit=1)[0]
    assert "replay_filter_candidate_device" in candidate_loop
    assert "_replay_filter_episode_group(" not in candidate_loop
    assert "materialize_belief_state_hashes" not in candidate_loop
    assert "_load_or_build_selected_filter_checkpoints_v2(" in design_source
    planning_replay = inspect.getsource(
        design._load_or_build_selected_filter_checkpoints_v2
    )
    assert "replay_selected_consultation_states_v2(" in planning_replay
    assert "len(histories) * len(CONSULTATION_STEPS)" in planning_replay
    assert "PLANNING_MINIMUM_POINT_COUNT" in planning_replay
    assert "candidate" not in planning_replay.split("if manifest_path.exists():", 1)[0]


def test_filter_candidates_have_independent_atomic_outputs_before_fixed_selection():
    design_source = inspect.getsource(design.run_r015_design)
    assert 'config.get("_filter_candidate_only")' in design_source
    assert "requested_candidate not in FILTER_CANDIDATE_ORDER" in design_source
    assert "if evidence_path.exists():" in design_source
    assert "evaluate_filter_candidate" in design_source
    assert '"path_c_r015_filter_candidate_execution_v2"' in design_source
    assert design_source.index("if requested_filter_candidate is not None:") < (
        design_source.index("filter_results =")
    )


def test_filter_candidate_scans_all_history_stream_lanes_in_fixed_microbatches():
    design_source = inspect.getsource(design.run_r015_design)
    assert design.FILTER_DEVICE_EXECUTION_ID == "r015_filter_jit_scan_vmap_v2"
    assert "replay_filter_candidate_device" in design_source
    assert "grouped_pairs" in design_source
    assert "_replay_filter_episode_group(" not in design_source
    assert design.FILTER_PARENT_SLOT_TARGET == 4096
    equivalence_source = inspect.getsource(
        design.run_r015_filter_device_equivalence
    )
    assert "return_particle_weight_trace=True" in equivalence_source


def _weighted_belief(weights) -> StratifiedParticleBeliefV1:
    base = _belief()
    return StratifiedParticleBeliefV1(
        prototype_ids=base.prototype_ids,
        particles_per_prototype=1,
        registered_prototype_prior=base.registered_prototype_prior,
        resampling_algorithm=base.resampling_algorithm,
        resampling_interval_environment_steps=1,
        particles=tuple(
            HiddenStateParticleV1(
                prototype_id=particle.prototype_id,
                state_sha256=particle.state_sha256,
                state=particle.state,
                weight=weights[particle.prototype_id],
            )
            for particle in base.particles
        ),
    )


class _FixtureBranchExecutor:
    continuation_controller_id = R015_CONTINUATION_CONTROLLER_ID

    def __init__(self, *, positive: bool = True, base_tie: bool = False):
        self.positive = positive
        self.base_tie = base_tie
        self.remaining_steps = []
        self.pair_keys = []

    def candidate_status(self, *, history, script, environment_step):
        return True, True

    def rollout_base(
        self,
        *,
        particle,
        belief,
        history,
        branch_key,
        remaining_steps,
    ):
        self.remaining_steps.append(remaining_steps)
        return FullHorizonRolloutV1(
            raw_return=10.0,
            primitive_steps=remaining_steps,
            trajectory_sha256=_sha((particle.state_sha256, "base", branch_key)),
            task_transition_sha256=_sha((particle.state_sha256, "base-task")),
            **_future_random_fields(branch_key, remaining_steps),
        )

    def rollout_probe_pair(
        self,
        *,
        particle,
        belief,
        history,
        script,
        branch_key,
        remaining_steps,
    ):
        self.remaining_steps.append(remaining_steps)
        self.pair_keys.append(branch_key)
        if self.base_tie:
            masked_return = 10.0
        elif script.probe_id in {"up", "down"}:
            masked_return = 12.0
        elif script.probe_id == "right":
            masked_return = 11.0
        else:
            masked_return = 0.0
        used_return = masked_return
        if self.positive and script.probe_id in {"up", "down"}:
            used_return = 15.0
        task_hash = _sha((particle.state_sha256, script.probe_id, "task"))
        return PairedProbeRolloutV1(
            masked=FullHorizonRolloutV1(
                raw_return=masked_return,
                primitive_steps=remaining_steps,
                trajectory_sha256=_sha((branch_key, "mask")),
                task_transition_sha256=task_hash,
                **_future_random_fields(branch_key, remaining_steps),
            ),
            used=FullHorizonRolloutV1(
                raw_return=used_return,
                primitive_steps=remaining_steps,
                trajectory_sha256=_sha((branch_key, "use")),
                task_transition_sha256=task_hash,
                **_future_random_fields(branch_key, remaining_steps),
            ),
        )

    def rollout_planning_batch(
        self, *, samples, belief, history, scripts, remaining_steps
    ):
        return PlanningBatchRolloutsV1(
            samples=tuple(
                PlanningSampleRolloutsV1(
                    sample=sample,
                    base=self.rollout_base(
                        particle=sample.source_particle,
                        belief=belief,
                        history=history,
                        branch_key=sample.common_random_key,
                        remaining_steps=remaining_steps,
                    ),
                    probe_pairs={
                        script.probe_id: self.rollout_probe_pair(
                            particle=sample.source_particle,
                            belief=belief,
                            history=history,
                            script=script,
                            branch_key=sample.common_random_key,
                            remaining_steps=remaining_steps,
                        )
                        for script in scripts
                    },
                )
                for sample in samples
            ),
            compiled_batch_calls=1,
            active_batch_sizes=(len(samples),),
            host_sync_inside_environment_loop=False,
        )


def _safety_fixture(belief, *, passed: bool) -> SafetyDecisionV1:
    comparisons = []
    for index, prototype_id in enumerate(belief.prototype_ids):
        comparisons.append(
            {
                "prototype_id": prototype_id,
                "repetitions": 279,
                "wrong_delivery_count": (
                    0 if passed or index > 0 else 1
                ),
                "positive_posterior_support": True,
                "compatible_hidden_state_reconstructed": True,
            }
        )
    return SafetyDecisionV1(passed=passed, comparisons=tuple(comparisons))


def test_official_history_rejects_unknown_and_evaluator_fields():
    first = _history()
    second = OfficialHistoryV1(
        (
            {
                "official_local_observation": {"step": 1},
                "ego_action_history": ["stay"],
                "raw_team_reward_history": [0.0],
                "episode_boundaries": [False],
            },
        )
    )
    assert first.sha256 == second.sha256
    with pytest.raises(ValueError, match="unknown field"):
        OfficialHistoryV1(
            ({"official_local_observation": {}, "actual_partner_id": "q-201"},)
        )
    with pytest.raises(ValueError, match="forbidden field"):
        OfficialHistoryV1(
            (
                {
                    "official_local_observation": {
                        "partner_private_observation": [1, 2, 3]
                    }
                },
            )
        )


def test_current_response_projection_removes_aliases_and_only_use_receives_y():
    response = {
        "source_probe_id": "stay",
        "source_probe_step": 1,
        "token": "yielded",
    }
    history = OfficialHistoryV1(
        (
            {
                "official_local_observation": {"visible": "unchanged"},
                "ego_action_history": ["stay"],
                "raw_team_reward_history": [0.0],
                "episode_boundaries": [False],
                "response_summary_v1_derived": response,
                "task_progress_derived": {
                    "public_progress": 2,
                    "current_response_alias": response,
                },
                "ego_option_or_probe_history": {
                    "selected": "stay",
                    "response_state_write": response,
                },
            },
        )
    )
    projection = CurrentResponseProjectionV1(
        alias_paths=(("task_progress_derived", "current_response_alias"),),
        recurrent_state_write_paths=(
            ("ego_option_or_probe_history", "response_state_write"),
        ),
    )
    projected = projection.project(history, probe_id="stay", probe_step=1)
    record = projected.masked_history.records[0]
    assert record["official_local_observation"] == {"visible": "unchanged"}
    assert record["task_progress_derived"] == {"public_progress": 2}
    assert record["ego_option_or_probe_history"] == {"selected": "stay"}
    assert "response_summary_v1_derived" not in record

    calls = []

    def mask_updater(belief, masked_history):
        calls.append(("mask", masked_history.sha256, None))
        return {**belief, "route": "mask"}

    def use_updater(belief, masked_history, current_response):
        calls.append(("use", masked_history.sha256, dict(current_response)))
        return {**belief, "route": "use", "response": dict(current_response)}

    paired = apply_paired_response_update(
        {"prior": 1},
        projected,
        mask_updater=mask_updater,
        use_updater=use_updater,
    )
    assert calls[0] == ("mask", projected.masked_history.sha256, None)
    assert calls[1][0:2] == ("use", projected.masked_history.sha256)
    assert calls[1][2] == response
    assert paired.masked == {"prior": 1, "route": "mask"}
    assert paired.used["response"] == response


def test_stratified_filter_uses_independent_keys_and_closes_on_zero_support():
    keys = []

    def factory(prototype_id, index, key):
        keys.append(key)
        return HiddenStateParticleV1(
            prototype_id=prototype_id,
            state_sha256=_sha((prototype_id, index, key)),
            state={"hidden": prototype_id},
            weight=1.0,
        )

    belief = StratifiedParticleBeliefV1.initialize(
        prototype_ids=("p0", "p1", "p2", "p3"),
        particles_per_prototype=2,
        registered_prototype_prior={
            "p0": 0.1,
            "p1": 0.2,
            "p2": 0.3,
            "p3": 0.4,
        },
        resampling_algorithm="systematic_per_prototype_v1",
        resampling_interval_environment_steps=1,
        initialization_key="2" * 64,
        factory=factory,
    )
    assert len(keys) == len(set(keys)) == 8
    assert belief.prototype_weights == pytest.approx(
        {"p0": 0.1, "p1": 0.2, "p2": 0.3, "p3": 0.4}
    )
    assert "hidden" not in json.dumps(belief.public_summary)

    def transition(particle, official_record, key):
        return (
            HiddenStateParticleV1(
                prototype_id=particle.prototype_id,
                state_sha256=_sha((particle.state_sha256, key)),
                state={"hidden": particle.prototype_id, "updated": True},
                weight=1.0,
            ),
        )

    updated = belief.update(
        official_record={
            "official_local_observation": {"step": 2},
            "ego_action_history": ["stay"],
            "raw_team_reward_history": [0.0],
            "episode_boundaries": [False],
        },
        update_key="3" * 64,
        transition=transition,
        compatibility=lambda particle, record: 1.0,
    )
    assert len(updated.particles) == 8
    assert sum(item.weight for item in updated.particles) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="passive filtering"):
        belief.update(
            official_record={
                "official_local_observation": {"step": 2},
                "ego_action_history": ["stay"],
                "raw_team_reward_history": [0.0],
                "episode_boundaries": [False],
                "response_summary_v1_derived": {"token": "forbidden-passive-route"},
            },
            update_key="f" * 64,
            transition=transition,
            compatibility=lambda particle, record: 1.0,
        )

    with pytest.raises(ValueError, match="lost compatible support"):
        belief.update(
            official_record={
                "official_local_observation": {"step": 2},
                "ego_action_history": ["stay"],
                "raw_team_reward_history": [0.0],
                "episode_boundaries": [False],
            },
            update_key="4" * 64,
            transition=transition,
            compatibility=lambda particle, record: (
                0.0 if particle.prototype_id == "p3" else 1.0
            ),
        )
    with pytest.raises(ValueError, match="positive posterior support"):
        belief.use_current_response(
            {"token": "yielded"},
            likelihood=lambda particle, response: (
                0.0 if particle.prototype_id == "p3" else 1.0
            ),
        )


def test_filter_container_separates_prototype_mass_and_conditional_weights():
    belief = _belief()
    assert belief.filter_algorithm_id == R015_FILTER_ALGORITHM_ID_V2
    assert belief.resampling_algorithm == R015_RESAMPLING_ALGORITHM_ID_V2
    assert belief.is_formal_v2 is True
    assert belief.prototype_masses == pytest.approx(
        {prototype_id: 0.25 for prototype_id in belief.prototype_ids}
    )
    for prototype_id in belief.prototype_ids:
        assert belief.within_prototype_weights[prototype_id] == pytest.approx(
            (1.0,)
        )
    assert belief.public_summary["formal_planning_eligible"] is True

    diagnostic = StratifiedParticleBeliefV1(
        prototype_ids=belief.prototype_ids,
        particles_per_prototype=belief.particles_per_prototype,
        registered_prototype_prior=belief.registered_prototype_prior,
        resampling_algorithm=R015_RESAMPLING_ALGORITHM_ID_V1,
        resampling_interval_environment_steps=1,
        particles=belief.particles,
        filter_algorithm_id=R015_FILTER_ALGORITHM_ID_V1,
    )
    assert diagnostic.is_formal_v2 is False
    assert diagnostic.public_summary["formal_planning_eligible"] is False

    masses = {
        belief.prototype_ids[0]: 0.4,
        belief.prototype_ids[1]: 0.3,
        belief.prototype_ids[2]: 0.2,
        belief.prototype_ids[3]: 0.1,
    }
    conditional = {
        prototype_id: (0.75, 0.25) for prototype_id in belief.prototype_ids
    }
    particles = tuple(
        HiddenStateParticleV1(
            prototype_id=prototype_id,
            state_sha256=_sha((prototype_id, index)),
            state={"prototype_id": prototype_id, "index": index},
            weight=masses[prototype_id] * conditional[prototype_id][index],
        )
        for prototype_id in belief.prototype_ids
        for index in range(2)
    )
    factored = StratifiedParticleBeliefV1(
        prototype_ids=belief.prototype_ids,
        particles_per_prototype=2,
        registered_prototype_prior=belief.registered_prototype_prior,
        resampling_algorithm=R015_RESAMPLING_ALGORITHM_ID_V2,
        resampling_interval_environment_steps=1,
        particles=particles,
        prototype_masses=masses,
        within_prototype_weights=conditional,
    )
    assert factored.prototype_weights == pytest.approx(masses)
    assert factored.within_prototype_weights == conditional
    schedule = NestedHiddenStateBranchScheduleV1.build(
        belief=factored,
        planning_key="d" * 64,
        sample_count=4,
    )
    assert all(
        sample.source_particle.weight
        == masses[sample.source_particle.prototype_id]
        * conditional[sample.source_particle.prototype_id][
            sample.source_particle.state["index"]
        ]
        for sample in schedule.samples
    )


def test_filter_container_rejects_mislabeled_v1_and_inconsistent_factorization():
    belief = _belief()
    with pytest.raises(ValueError, match="identifiers disagree"):
        StratifiedParticleBeliefV1(
            prototype_ids=belief.prototype_ids,
            particles_per_prototype=belief.particles_per_prototype,
            registered_prototype_prior=belief.registered_prototype_prior,
            resampling_algorithm=R015_RESAMPLING_ALGORITHM_ID_V1,
            resampling_interval_environment_steps=1,
            particles=belief.particles,
            filter_algorithm_id=R015_FILTER_ALGORITHM_ID_V2,
        )
    with pytest.raises(ValueError, match="prototype mass differs"):
        StratifiedParticleBeliefV1(
            prototype_ids=belief.prototype_ids,
            particles_per_prototype=belief.particles_per_prototype,
            registered_prototype_prior=belief.registered_prototype_prior,
            resampling_algorithm=R015_RESAMPLING_ALGORITHM_ID_V2,
            resampling_interval_environment_steps=1,
            particles=belief.particles,
            prototype_masses={
                belief.prototype_ids[0]: 0.4,
                belief.prototype_ids[1]: 0.2,
                belief.prototype_ids[2]: 0.2,
                belief.prototype_ids[3]: 0.2,
            },
        )


def test_map_continuation_routes_unique_map_and_falls_back_on_exact_ties():
    prototype_ids = _belief().prototype_ids
    baseline_id = "ego-seed100"
    action_by_member = dict(
        zip((*prototype_ids, baseline_id), ("up", "down", "left", "right", "stay"))
    )
    calls = []

    def actor(member_id, history, recurrent_state, random_key):
        calls.append((member_id, recurrent_state, random_key, history.sha256))
        return ContinuationMemberStepV1(
            action_id=action_by_member[member_id],
            next_recurrent_state=recurrent_state + 1,
        )

    continuation = MAPPrototypeCommittedCookV1(
        prototype_ids=prototype_ids,
        baseline_member_id=baseline_id,
        member_actor=actor,
    )
    states = ContinuationLibraryStatesV1(
        {member_id: 10 for member_id in continuation.member_ids}
    )
    keys = {member_id: f"key-{member_id}" for member_id in continuation.member_ids}
    uniform = continuation.act(
        history=_history(),
        belief=_belief(),
        states=states,
        random_keys_by_member=keys,
    )
    assert uniform.unique_map_prototype_id is None
    assert uniform.selected_member_id == baseline_id
    assert uniform.action_id == "stay"
    assert len(calls) == 5

    unique = continuation.act(
        history=_history(),
        belief=_weighted_belief(
            {
                prototype_ids[0]: 0.1,
                prototype_ids[1]: 0.6,
                prototype_ids[2]: 0.2,
                prototype_ids[3]: 0.1,
            }
        ),
        states=states,
        random_keys_by_member=keys,
    )
    assert unique.unique_map_prototype_id == prototype_ids[1]
    assert unique.selected_member_id == prototype_ids[1]
    assert unique.action_id == "down"

    maximum_tie = continuation.act(
        history=_history(),
        belief=_weighted_belief(
            {
                prototype_ids[0]: 0.4,
                prototype_ids[1]: 0.4,
                prototype_ids[2]: 0.1,
                prototype_ids[3]: 0.1,
            }
        ),
        states=states,
        random_keys_by_member=keys,
    )
    assert maximum_tie.selected_member_id == baseline_id


def test_continuation_response_route_is_belief_bearing_and_states_never_reset():
    prototype_ids = _belief().prototype_ids
    baseline_id = "ego-seed100"
    observed_states = []

    def actor(member_id, history, recurrent_state, random_key):
        observed_states.append((member_id, recurrent_state))
        action = "up" if member_id == prototype_ids[0] else "down"
        return ContinuationMemberStepV1(action, recurrent_state + 1)

    continuation = MAPPrototypeCommittedCookV1(
        prototype_ids=prototype_ids,
        baseline_member_id=baseline_id,
        member_actor=actor,
    )
    controller = R015SequentialControllerV1(
        R015SequentialPlannerV1(default_r015_probe_scripts(), 2),
        continuation_controller=continuation,
    )
    initial = ContinuationLibraryStatesV1(
        {member_id: 7 for member_id in continuation.member_ids}
    )
    keys = {member_id: member_id for member_id in continuation.member_ids}
    used = controller.act(
        history=_history(),
        belief=_weighted_belief(
            {
                prototype_ids[0]: 0.7,
                prototype_ids[1]: 0.1,
                prototype_ids[2]: 0.1,
                prototype_ids[3]: 0.1,
            }
        ),
        states=initial,
        random_keys_by_member=keys,
    )
    masked = controller.act(
        history=_history(),
        belief=_weighted_belief(
            {
                prototype_ids[0]: 0.1,
                prototype_ids[1]: 0.7,
                prototype_ids[2]: 0.1,
                prototype_ids[3]: 0.1,
            }
        ),
        states=used.next_states,
        random_keys_by_member=keys,
    )
    assert used.selected_member_id == prototype_ids[0]
    assert masked.selected_member_id == prototype_ids[1]
    assert used.action_id != masked.action_id
    assert observed_states[:5] == [
        (member_id, 7) for member_id in continuation.member_ids
    ]
    assert observed_states[5:] == [
        (member_id, 8) for member_id in continuation.member_ids
    ]
    assert set(masked.next_states.by_member_id.values()) == {9}


def test_branch_interfaces_expose_full_belief_and_share_continuation_identity():
    expected = {
        "candidate_status": ("self", "history", "script", "environment_step"),
        "rollout_base": (
            "self",
            "particle",
            "belief",
            "history",
            "branch_key",
            "remaining_steps",
        ),
        "rollout_probe_pair": (
            "self",
            "particle",
            "belief",
            "history",
            "script",
            "branch_key",
            "remaining_steps",
        ),
        "rollout_planning_batch": (
            "self",
            "samples",
            "belief",
            "history",
            "scripts",
            "remaining_steps",
        ),
    }
    for method_name, parameter_names in expected.items():
        signature = inspect.signature(
            getattr(OCV2R015ProductionBackendV1, method_name)
        )
        assert tuple(signature.parameters) == parameter_names
    executor = _FixtureBranchExecutor()
    assert executor.continuation_controller_id == R015_CONTINUATION_CONTROLLER_ID


def test_continuation_manifest_binds_all_five_checkpoint_digests():
    prototype_ids = _belief().prototype_ids
    baseline_id = "ego-seed100"
    members = (*prototype_ids, baseline_id)
    payload = continuation_controller_manifest_payload(
        prototype_ids=prototype_ids,
        baseline_member_id=baseline_id,
        checkpoint_bindings={
            member_id: {
                "checkpoint_sha256": _sha((member_id, "file")),
                "model_weights_sha256": _sha((member_id, "weights")),
            }
            for member_id in members
        },
        implementation_path="path_c_r015_controller.py",
        implementation_sha256="a" * 64,
    )
    assert payload["rule_id"] == R015_CONTINUATION_CONTROLLER_ID
    assert payload["contract"]["parallel_recurrent_member_count"] == 5
    assert payload["contract"]["planning_branch_sampling"] == (
        "sampled_hidden_state_branches_v1"
    )
    assert payload["contract"]["planning_filter_algorithm_id"] == (
        R015_FILTER_ALGORITHM_ID_V2
    )
    assert payload["contract"]["planning_resampling_algorithm"] == (
        R015_RESAMPLING_ALGORITHM_ID_V2
    )
    assert payload["contract"]["planning_routing_frequency"] == (
        "commit_once_at_branch_head"
    )
    assert payload["contract"]["execution_routing_frequency"] == (
        "update_online_each_environment_step"
    )
    assert payload["members"][baseline_id]["role"] == "baseline"
    assert {
        payload["members"][prototype_id]["role"] for prototype_id in prototype_ids
    } == {"prototype"}


def test_r015_runtime_actor_builds_official_network_once_and_reuses_jit():
    import jax.numpy as jnp

    class Distribution:
        def __init__(self, logits):
            self.logits = logits

    class Network:
        def apply(self, params, recurrent_state, actor_input):
            del params
            observations, _ = actor_input
            logits = jnp.zeros(
                (*observations.shape[:2], 6), dtype=jnp.float32
            )
            value = jnp.zeros(observations.shape[:2], dtype=jnp.float32)
            return recurrent_state + 1.0, Distribution(logits), value

    class StableAdapter:
        def __init__(self):
            self.network_build_count = 0

        def _network(self):
            self.network_build_count += 1
            return Network()

        def initial_state(self, batch_size):
            return jnp.zeros((batch_size, 2), dtype=jnp.float32)

    stable = StableAdapter()
    adapter = _CompiledR015ActorAdapter(stable)
    state = adapter.initial_state(1)
    first_state, first_logits = adapter.apply_actor(
        {}, state, jnp.zeros((5, 5, 39)), True
    )
    second_state, second_logits = adapter.apply_actor(
        {}, first_state, jnp.zeros((5, 5, 39)), False
    )
    assert stable.network_build_count == 1
    assert first_logits.shape == (6,)
    assert second_logits.shape == (6,)
    assert float(second_state[0, 0]) == pytest.approx(2.0)


def test_planning_and_execution_share_member_selection_but_not_update_frequency():
    prototype_ids = _belief().prototype_ids
    baseline_id = "ego-seed100"

    def actor(member_id, history, recurrent_state, random_key):
        return ContinuationMemberStepV1("interact", recurrent_state + 1)

    continuation = MAPPrototypeCommittedCookV1(
        prototype_ids=prototype_ids,
        baseline_member_id=baseline_id,
        member_actor=actor,
    )
    belief = _weighted_belief(
        {
            prototype_ids[0]: 0.1,
            prototype_ids[1]: 0.2,
            prototype_ids[2]: 0.6,
            prototype_ids[3]: 0.1,
        }
    )
    controller = R015SequentialControllerV1(
        R015SequentialPlannerV1(default_r015_probe_scripts(), 2),
        continuation_controller=continuation,
    )
    execution = controller.act(
        history=_history(),
        belief=belief,
        states=ContinuationLibraryStatesV1(
            {member_id: 0 for member_id in continuation.member_ids}
        ),
        random_keys_by_member={
            member_id: member_id for member_id in continuation.member_ids
        },
    )
    branch_executor = _FixtureBranchExecutor()
    branch_route = continuation.unique_map_prototype(belief)
    assert branch_executor.continuation_controller_id == execution.controller_id
    assert branch_route == execution.selected_member_id


def test_full_horizon_planner_recomputes_values_and_deterministic_ties():
    planner = R015SequentialPlannerV1(
        probe_scripts=default_r015_probe_scripts(),
        branches_per_candidate=2,
    )
    executor = _FixtureBranchExecutor()
    result = planner.evaluate(
        history=_history(),
        belief=_belief(),
        environment_step=1,
        planning_key="5" * 64,
        score_key="6" * 64,
        executor=executor,
    )
    by_id = {item.probe_id: item for item in result.candidates}
    assert set(executor.remaining_steps) == {399}
    assert result.v_base == pytest.approx(10.0)
    assert result.v_mask == pytest.approx(12.0)
    assert result.masked_reference_action_id == "up"
    assert by_id["up"].j_use == pytest.approx(15.0)
    assert by_id["up"].j_mask == pytest.approx(12.0)
    assert by_id["up"].i_response == pytest.approx(3.0)
    assert by_id["up"].c_task == pytest.approx(0.0)
    assert by_id["up"].s_seq == pytest.approx(3.0)
    assert result.selected_for_safety_probe_id == "up"
    assert all(
        branch["score_evidence_key"] != branch["common_random_key"]
        for branch in result.planning_branches
    )

    base_tie = planner.evaluate(
        history=_history(),
        belief=_belief(),
        environment_step=1,
        planning_key="7" * 64,
        score_key="8" * 64,
        executor=_FixtureBranchExecutor(positive=False, base_tie=True),
    )
    assert base_tie.v_mask == pytest.approx(10.0)
    assert base_tie.masked_reference_action_id == "base"
    assert base_tie.selected_for_safety_probe_id is None


def test_nested_systematic_schedule_is_deterministic_weighted_and_recursive():
    belief = _belief()
    schedules = {
        count: NestedHiddenStateBranchScheduleV1.build(
            belief=belief,
            planning_key="9" * 64,
            sample_count=count,
        )
        for count in (2, 4, 8, 16)
    }
    repeated = NestedHiddenStateBranchScheduleV1.build(
        belief=belief,
        planning_key="9" * 64,
        sample_count=16,
    )
    assert repeated == schedules[16]
    for count, schedule in schedules.items():
        assert [sample.sample_slot for sample in schedule.samples] == list(
            range(count)
        )
        assert all(sample.estimator_weight == pytest.approx(1.0 / count) for sample in schedule.samples)
    for lower, upper in ((2, 4), (4, 8), (8, 16)):
        assert {
            sample.canonical_slot_16 for sample in schedules[lower].samples
        } == {
            sample.canonical_slot_16
            for sample in schedules[upper].lower_subset()
        }
    skewed = _weighted_belief(
        {
            "ippo-101": 0.97,
            "ippo-102": 0.01,
            "q-201": 0.01,
            "q-202": 0.01,
        }
    )
    repeated_source = NestedHiddenStateBranchScheduleV1.build(
        belief=skewed,
        planning_key="9" * 64,
        sample_count=16,
    )
    assert len({sample.source_particle_index for sample in repeated_source.samples}) < 16
    assert len({sample.sample_slot for sample in repeated_source.samples}) == 16
    assert all(
        sample.source_belief_filter_algorithm_id
        == R015_FILTER_ALGORITHM_ID_V2
        and sample.source_belief_resampling_algorithm
        == R015_RESAMPLING_ALGORITHM_ID_V2
        for sample in repeated_source.samples
    )


def test_formal_schedule_and_planner_reject_v1_diagnostic_belief():
    formal = _belief()
    diagnostic = StratifiedParticleBeliefV1(
        prototype_ids=formal.prototype_ids,
        particles_per_prototype=formal.particles_per_prototype,
        registered_prototype_prior=formal.registered_prototype_prior,
        resampling_algorithm=R015_RESAMPLING_ALGORITHM_ID_V1,
        resampling_interval_environment_steps=1,
        particles=formal.particles,
        filter_algorithm_id=R015_FILTER_ALGORITHM_ID_V1,
    )
    with pytest.raises(ValueError, match="diagnostic v1"):
        NestedHiddenStateBranchScheduleV1.build(
            belief=diagnostic,
            planning_key="9" * 64,
            sample_count=2,
        )
    with pytest.raises(ValueError, match="diagnostic v1"):
        R015SequentialPlannerV1(
            default_r015_probe_scripts(), 2
        ).evaluate(
            history=_history(),
            belief=diagnostic,
            environment_step=1,
            planning_key="5" * 64,
            score_key="6" * 64,
            executor=_FixtureBranchExecutor(),
        )
    with pytest.raises(ValueError, match="v2 filter contract"):
        R015SequentialPlannerV1(
            default_r015_probe_scripts(),
            2,
            filter_algorithm_id=R015_FILTER_ALGORITHM_ID_V1,
            resampling_algorithm=R015_RESAMPLING_ALGORITHM_ID_V1,
        )


def test_lazy_grid_reconstructs_lower_estimate_and_counts_exact_work():
    belief = _belief()
    executor = _FixtureBranchExecutor()
    high = R015SequentialPlannerV1(
        default_r015_probe_scripts(), 4
    ).evaluate(
        history=_history(),
        belief=belief,
        environment_step=1,
        planning_key="a" * 64,
        score_key="b" * 64,
        executor=executor,
    )
    calls_after_high = len(executor.remaining_steps)
    low = R015SequentialPlannerV1(
        default_r015_probe_scripts(), 2
    ).evaluate(
        history=_history(),
        belief=belief,
        environment_step=1,
        planning_key="a" * 64,
        score_key="b" * 64,
        executor=executor,
        cached_planning_branches=high.planning_branches,
    )
    assert len(executor.remaining_steps) == calls_after_high
    assert low.cost_accounting["new_sample_count"] == 0
    assert low.cost_accounting["new_real_environment_transitions"] == 0
    assert high.cost_accounting["new_sample_count"] == 4
    assert high.cost_accounting["new_real_environment_transitions"] == 4 * (
        13 * 399 - 6
    )
    assert "compiled_batch_calls" not in high.cost_accounting
    assert "active_batch_sizes" not in high.cost_accounting
    assert "host_sync_inside_environment_loop" not in high.cost_accounting
    assert high.device_execution == {
        "compiled_batch_calls": 1,
        "active_batch_sizes": (4,),
        "host_sync_inside_environment_loop": False,
    }
    assert high.cost_accounting[
        "new_paired_suffix_batched_lane_time_steps"
    ] == 4 * 7 * 399
    assert {
        branch["canonical_slot_16"] for branch in low.planning_branches
    }.issubset(
        {branch["canonical_slot_16"] for branch in high.planning_branches}
    )


def test_registered_r8_full_block_cost_fixture():
    remaining = [400 - step for step in range(1, 101, 5)]
    assert sum(8 * (13 * horizon - 6) for horizon in remaining) == 730_160
    assert sum(8 * 7 * horizon for horizon in remaining) == 393_680
    assert 20 * 8 * 6 == 960


def test_planning_batch_rejects_stepwise_host_synchronization():
    with pytest.raises(ValueError, match="synchronize the host"):
        PlanningBatchRolloutsV1(
            samples=(),
            compiled_batch_calls=1,
            active_batch_sizes=(1,),
            host_sync_inside_environment_loop=True,
        )


def test_paired_rollout_rejects_short_horizon_or_random_key_reuse_error():
    good = FullHorizonRolloutV1(
        raw_return=0.0,
        primitive_steps=399,
        trajectory_sha256="1" * 64,
        task_transition_sha256="2" * 64,
        **_future_random_fields("3" * 64, 399),
    )
    short = FullHorizonRolloutV1(
        raw_return=0.0,
        primitive_steps=398,
        trajectory_sha256="4" * 64,
        task_transition_sha256="2" * 64,
        **_future_random_fields("3" * 64, 398),
    )
    with pytest.raises(ValueError, match="remaining episode"):
        PairedProbeRolloutV1(masked=good, used=short).validate_pairing(
            expected_steps=399
        )
    changed_randomness = FullHorizonRolloutV1(
        raw_return=0.0,
        primitive_steps=399,
        trajectory_sha256="5" * 64,
        task_transition_sha256="2" * 64,
        **_future_random_fields("6" * 64, 399),
    )
    with pytest.raises(ValueError, match="future random"):
        PairedProbeRolloutV1(
            masked=good,
            used=changed_randomness,
        ).validate_pairing(expected_steps=399)


def test_planner_evidence_persists_only_future_random_summary():
    result = R015SequentialPlannerV1(
        default_r015_probe_scripts(), 2
    ).evaluate(
        history=_history(),
        belief=_belief(),
        environment_step=1,
        planning_key="5" * 64,
        score_key="6" * 64,
        executor=_FixtureBranchExecutor(),
    )
    serialized = json.dumps(result.to_payload(), sort_keys=True)
    assert "future_random_keys" not in serialized
    assert "future_random_root_key" in serialized
    assert "future_random_sequence_sha256" in serialized
    assert "compiled_batch_calls" not in result.cost_accounting


def test_first_positive_safety_pass_and_rejection_both_close_consultation():
    planner = R015SequentialPlannerV1(
        probe_scripts=default_r015_probe_scripts(),
        branches_per_candidate=2,
    )
    passed = R015SequentialControllerV1(planner)
    decision = passed.consult(
        history=_history(),
        belief=_belief(),
        environment_step=1,
        planning_key="1" * 64,
        score_key="2" * 64,
        safety_key="3" * 64,
        executor=_FixtureBranchExecutor(),
        safety_evaluator=lambda probe_id, history, belief, key: _safety_fixture(
            belief,
            passed=True,
        ),
    )
    assert decision.probe_fired is True
    assert decision.a2_probe_id == "up"
    assert decision.stop_consulting is True
    with pytest.raises(ValueError, match="cannot consult"):
        passed.consult(
            history=_history(6),
            belief=_belief(),
            environment_step=6,
            planning_key="4" * 64,
            score_key="5" * 64,
            safety_key="6" * 64,
            executor=_FixtureBranchExecutor(),
            safety_evaluator=lambda probe_id, history, belief, key: _safety_fixture(
                belief,
                passed=True,
            ),
        )

    rejected = R015SequentialControllerV1(planner)
    rejected_decision = rejected.consult(
        history=_history(),
        belief=_belief(),
        environment_step=1,
        planning_key="7" * 64,
        score_key="8" * 64,
        safety_key="9" * 64,
        executor=_FixtureBranchExecutor(),
        safety_evaluator=lambda probe_id, history, belief, key: _safety_fixture(
            belief,
            passed=False,
        ),
    )
    assert rejected_decision.probe_fired is False
    assert rejected_decision.a2_probe_id is None
    assert rejected_decision.a1_action_id == "up"
    assert rejected_decision.shared_masked_action_id == "up"
    assert rejected_decision.no_probe_reason == "safety_rejected"
    assert rejected_decision.stop_consulting is True
    with pytest.raises(ValueError, match="cannot consult"):
        rejected.consult(
            history=_history(6),
            belief=_belief(),
            environment_step=6,
            planning_key="a" * 64,
            score_key="b" * 64,
            safety_key="c" * 64,
            executor=_FixtureBranchExecutor(),
            safety_evaluator=lambda probe_id, history, belief, key: _safety_fixture(
                belief,
                passed=True,
            ),
        )

    fresh = R015SequentialControllerV1(planner)
    with pytest.raises(ValueError, match="distinct SHA-256"):
        fresh.consult(
            history=_history(),
            belief=_belief(),
            environment_step=1,
            planning_key="d" * 64,
            score_key="d" * 64,
            safety_key="e" * 64,
            executor=_FixtureBranchExecutor(),
            safety_evaluator=lambda probe_id, history, belief, key: _safety_fixture(
                belief,
                passed=True,
            ),
        )


def _step(
    index: int,
    *,
    ego_action: str = "stay",
    reward: float = 0.0,
    observation_stream: str = "shared",
):
    return {
        "environment_step": index,
        "ego_action": ego_action,
        "partner_action": "stay",
        "controller_input": {
            "official_local_observation": {
                "stream": observation_stream,
                "step": index,
            }
        },
        "raw_team_reward": reward,
        "done": index == 399,
        "environment_random_key": f"episode-key-{index}",
        "official_local_observation_after": {
            "stream": observation_stream,
            "step": index + 1,
        },
        "controller_recurrent_state_before_sha256": _sha(
            (observation_stream, "ego", index)
        ),
        "controller_recurrent_state_after_sha256": _sha(
            (observation_stream, "ego", index + 1)
        ),
        "partner_recurrent_state_before_sha256": _sha(
            (observation_stream, "partner", index)
        ),
        "partner_recurrent_state_after_sha256": _sha(
            (observation_stream, "partner", index + 1)
        ),
        "belief_update_mode": "online_each_environment_step",
        "belief_before_sha256": _sha((observation_stream, "belief", index)),
        "belief_after_sha256": _sha((observation_stream, "belief", index + 1)),
    }


def _shared_steps():
    return [_step(index) for index in range(400)]


def test_no_fire_trace_is_byte_identical_and_recomputes_zero_differences():
    groups = build_r015_no_fire_trace_groups(
        _shared_steps(),
        replay_verification_by_group={group: {} for group in ("A1", "A2-mask", "A2-use")},
        masked_value_reference={"selected_action_id": "base"},
    )
    verification = validate_r015_lockstep_groups(
        groups,
        probe_step=None,
        selected_probe_actions=(),
    )
    assert verification["differences"] == {
        "delta_net": 0.0,
        "delta_response": 0.0,
        "delta_cost": 0.0,
    }
    assert len(
        {
            canonical_sha256(groups[group]["environment_steps"])
            for group in ("A1", "A2-mask", "A2-use")
        }
    ) == 1

    tampered = copy.deepcopy(groups)
    tampered["A2-use"]["environment_steps"][20]["raw_team_reward"] = 20.0
    with pytest.raises(ValueError, match="byte-identical|zero differences"):
        validate_r015_lockstep_groups(
            tampered,
            probe_step=None,
            selected_probe_actions=(),
        )


def test_no_fire_paired_runner_executes_the_common_episode_once():
    class Executor:
        def __init__(self):
            self.calls = []

        def execute_segment(
            self,
            start_state,
            *,
            group,
            start_step,
            stop_step,
            forced_probe_actions,
        ):
            self.calls.append((group, start_step, stop_step, forced_probe_actions))
            return R015ExecutedSegmentV1(
                end_state={"step": stop_step},
                steps=tuple(_step(index) for index in range(start_step, stop_step)),
            )

    executor = Executor()
    runner = R015PairedEpisodeRunnerV1(
        executor=executor,
        clone_state=copy.deepcopy,
        state_sha256=canonical_sha256,
    )
    groups = runner.run_no_fire(
        {"step": 0},
        replay_verification_by_group={
            group: {} for group in ("A1", "A2-mask", "A2-use")
        },
        masked_value_reference={"selected_action_id": "base"},
    )
    assert executor.calls == [("shared", 0, 400, ())]
    assert len(
        {
            canonical_sha256(groups[group]["environment_steps"])
            for group in ("A1", "A2-mask", "A2-use")
        }
    ) == 1


def test_fire_runner_sends_masked_reference_to_a1_and_probe_to_both_a2_groups():
    class Executor:
        def __init__(self):
            self.calls = []

        def execute_segment(
            self,
            start_state,
            *,
            group,
            start_step,
            stop_step,
            forced_probe_actions,
        ):
            self.calls.append((group, forced_probe_actions))
            steps = []
            for index in range(start_step, stop_step):
                action = (
                    forced_probe_actions[0]
                    if index == start_step and forced_probe_actions
                    else "stay"
                )
                steps.append(_step(index, ego_action=action))
            return R015ExecutedSegmentV1(
                end_state={"step": stop_step},
                steps=tuple(steps),
            )

    executor = Executor()
    runner = R015PairedEpisodeRunnerV1(
        executor=executor,
        clone_state=copy.deepcopy,
        state_sha256=canonical_sha256,
    )
    runner.run_fire(
        {"step": 0},
        probe_step=1,
        selected_probe_id="stay",
        selected_probe_actions=("stay",),
        masked_reference_actions=("up",),
        replay_verification_by_group={
            group: {} for group in ("A1", "A2-mask", "A2-use")
        },
        masked_value_reference={"selected_action_id": "up"},
        masked_history=[{"official_local_observation": {"step": 2}}],
        current_probe_response={"token": "visible"},
        belief_common_input={"history": "masked"},
        continuation_common_input={"history": "masked"},
        recurrent_common_input={"history": "masked"},
    )
    assert executor.calls == [
        ("shared", ()),
        ("A1", ("up",)),
        ("A2-mask", ("stay",)),
        ("A2-use", ("stay",)),
    ]


def test_fire_trace_is_identical_through_t_p_minus_one_and_probe_end():
    prefix = [_step(0)]
    suffix_by_group = {}
    for group in ("A1", "A2-mask", "A2-use"):
        suffix = []
        for index in range(1, 400):
            if index == 1:
                stream = "shared"
                action = "stay" if group != "A1" else "up"
            else:
                stream = group
                action = (
                    "left"
                    if group == "A2-mask" and index == 2
                    else "interact"
                    if group == "A2-use" and index == 2
                    else "stay"
                )
            reward = 20.0 if group == "A2-use" and index == 10 else 0.0
            suffix.append(
                _step(
                    index,
                    ego_action=action,
                    reward=reward,
                    observation_stream=stream,
                )
            )
        suffix_by_group[group] = suffix
    # The first suffix record must continue the common recurrent state hashes.
    for group in suffix_by_group:
        suffix_by_group[group][0][
            "controller_recurrent_state_before_sha256"
        ] = prefix[0]["controller_recurrent_state_after_sha256"]
        suffix_by_group[group][0][
            "partner_recurrent_state_before_sha256"
        ] = prefix[0]["partner_recurrent_state_after_sha256"]
        suffix_by_group[group][0]["belief_before_sha256"] = prefix[0][
            "belief_after_sha256"
        ]
        # Record 2 reads the observation and recurrent state written by the
        # still-common probe record 1, then writes its group-specific branch.
        suffix_by_group[group][1]["controller_input"] = {
            "official_local_observation": {"stream": "shared", "step": 2}
        }
        suffix_by_group[group][1][
            "controller_recurrent_state_before_sha256"
        ] = _sha(("shared", "ego", 2))
        suffix_by_group[group][1][
            "partner_recurrent_state_before_sha256"
        ] = _sha(("shared", "partner", 2))
        suffix_by_group[group][1]["belief_before_sha256"] = _sha(
            ("shared", "belief", 2)
        )

    groups = build_r015_fire_trace_groups(
        prefix,
        suffix_by_group,
        probe_step=1,
        selected_probe_id="stay",
        selected_probe_actions=("stay",),
        replay_verification_by_group={group: {} for group in suffix_by_group},
        masked_value_reference={"selected_action_id": "base"},
        masked_history=[{"official_local_observation": {"step": 2}}],
        current_probe_response={"token": "yielded"},
        belief_common_input={"history": "masked"},
        continuation_common_input={"history": "masked"},
        recurrent_common_input={"history": "masked"},
    )
    verification = validate_r015_lockstep_groups(
        groups,
        probe_step=1,
        selected_probe_actions=("stay",),
    )
    assert verification["probe_step"] == 1
    assert verification["differences"]["delta_net"] == pytest.approx(20.0)
    assert groups["A2-mask"]["belief_current_response_extra"] is None
    assert groups["A2-use"]["belief_current_response_extra"] == {
        "token": "yielded"
    }

    bad_prefix = copy.deepcopy(groups)
    bad_prefix["A1"]["environment_steps"][0]["ego_action"] = "left"
    with pytest.raises(ValueError, match="before the probe"):
        validate_r015_lockstep_groups(
            bad_prefix,
            probe_step=1,
            selected_probe_actions=("stay",),
        )
    bad_probe = copy.deepcopy(groups)
    bad_probe["A2-use"]["environment_steps"][1]["partner_action"] = "interact"
    with pytest.raises(ValueError, match="probe script ended"):
        validate_r015_lockstep_groups(
            bad_probe,
            probe_step=1,
            selected_probe_actions=("stay",),
        )
    with pytest.raises(ValueError, match="invalid t_p"):
        validate_r015_lockstep_groups(
            groups,
            probe_step=101,
            selected_probe_actions=("stay",),
        )


def test_replay_entrypoints_use_backend_recomputation_and_reject_tampering():
    clean_trace = {"identity": "unit-1", "actions": ["stay"]}
    clean_decision = {"consultations": [{"score": 3.0}]}
    clean_safety = {"branch_results": [{"wrong_delivery_detected": False}]}
    expected_hashes = {
        "trace": canonical_sha256(clean_trace),
        "decision": canonical_sha256(clean_decision),
        "safety": canonical_sha256(clean_safety),
    }

    def strict(kind):
        def replay(payload, context):
            if canonical_sha256(payload) != expected_hashes[kind]:
                raise ValueError(f"{kind} differs from checkpoint replay")
            return {"verified": True, "kind": kind, "sha256": expected_hashes[kind]}

        return replay

    backend = CallableR015ReplayBackendV1(
        trace_replayer=strict("trace"),
        decision_replayer=strict("decision"),
        safety_replayer=strict("safety"),
    )
    context = {"_r015_replay_backend": backend}
    assert verify_r015_trace_manifest(clean_trace, context)["kind"] == "trace"
    assert verify_r015_probe_decision(clean_decision, context)["kind"] == "decision"
    assert verify_r015_safety_comparison(clean_safety, context)["kind"] == "safety"

    tampered_trace = copy.deepcopy(clean_trace)
    tampered_trace["actions"][0] = "interact"
    with pytest.raises(ValueError, match="checkpoint replay"):
        verify_r015_trace_manifest(tampered_trace, context)
    tampered_decision = copy.deepcopy(clean_decision)
    tampered_decision["consultations"][0]["score"] = 300.0
    with pytest.raises(ValueError, match="checkpoint replay"):
        verify_r015_probe_decision(tampered_decision, context)
    tampered_safety = copy.deepcopy(clean_safety)
    tampered_safety["branch_results"][0]["wrong_delivery_detected"] = True
    with pytest.raises(ValueError, match="checkpoint replay"):
        verify_r015_safety_comparison(tampered_safety, context)
    with pytest.raises(ValueError, match="continuation-planner manifest"):
        verify_r015_trace_manifest(clean_trace, {})


def test_ocv2_snapshot_runtime_is_pure_and_replayable():
    adapter = StandardEnvConfig(layout="test_time_simple").make_adapter()
    adapter.reset(20260713)
    snapshot = adapter.capture_state()
    runtime = OCV2SnapshotStepRuntimeV1(adapter=adapter)
    first = runtime.step(snapshot, ego_action="stay", partner_action="stay")
    second = runtime.step(snapshot, ego_action="stay", partner_action="stay")
    assert first.snapshot_before_sha256 == second.snapshot_before_sha256
    assert first.snapshot_after_sha256 == second.snapshot_after_sha256
    assert first.raw_team_reward == second.raw_team_reward
    assert first.done is False
