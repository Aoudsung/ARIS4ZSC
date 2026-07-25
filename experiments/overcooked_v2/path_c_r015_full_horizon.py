"""R015 的官方过滤、完整剩余回合分支、配对执行与独立重放。"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experiments.overcooked_v2.env_adapter import OCV2Adapter, OCV2AdapterSnapshot
from experiments.overcooked_v2.path_c_r015_controller import (
    ContinuationLibraryStatesV1,
    CurrentResponseProjectionV1,
    FullHorizonRolloutV1,
    HiddenStateParticleV1,
    OfficialHistoryV1,
    PairedProbeRolloutV1,
    PlanningBatchRolloutsV1,
    PlanningBranchSampleV1,
    PlanningSampleRolloutsV1,
    ProbeScriptV1,
    R015_ATOMIC_ACTIONS,
    R015_BRANCH_BELIEF_ID,
    R015_CONSULTATION_STEPS,
    R015_CONTINUATION_CONTROLLER_ID,
    R015_FUTURE_RANDOM_DERIVATION_ID,
    R015_PASSIVE_FILTER_UPDATE_FIELDS,
    R015SequentialControllerV1,
    R015SequentialPlannerV1,
    SafetyDecisionV1,
    StratifiedParticleBeliefV1,
    apply_paired_response_update,
    canonical_sha256,
    default_r015_probe_scripts,
    derive_controller_key,
)
from experiments.overcooked_v2.path_c_r015_runtime import (
    OCV2_ACTION_INDEX,
    R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT,
    R015ExecutedSegmentV1,
    R015FormalBlockProductionResultV1,
    R015FormalBlockRequestV1,
    R015MechanicalBlockInvalidError,
    R015ReplayBackend,
    _snapshot_sha256,
    build_r015_fire_trace_groups,
    build_r015_no_fire_trace_groups,
    validate_r015_lockstep_groups,
    verify_r015_probe_decision,
    verify_r015_safety_comparison,
    verify_r015_trace_manifest,
)
from experiments.overcooked_v2.path_c_response_probe import (
    REGISTERED_LOCAL_RESPONSE_SPEC,
    local_non_agent_change_from_default_observation,
    partner_visible_from_default_observation,
    response_token_from_visibility,
)


R015_FILTER_KEY_CONTRACT_V1 = "r015_filter_key_without_episode_seed_v1"
R015_FILTER_KEY_CONTRACT = R015_FILTER_KEY_CONTRACT_V1
R015_FILTER_KEY_CONTRACT_V2 = "r015_filter_device_fold_in_keys_v2"
R015_FILTER_ALGORITHM_ID_V1 = "stratified_hidden_state_particles_v1"
R015_FILTER_ALGORITHM_ID_V2 = "official_history_fully_adapted_particle_filter_v2"
R015_FILTER_DEVICE_EXECUTION_ID_V1 = "r015_filter_jit_scan_vmap_v1"
R015_FILTER_DEVICE_EXECUTION_ID_V2 = "r015_filter_jit_scan_vmap_v2"
R015_FILTER_MICROBATCH_SCHEDULE_ID = "fixed_4096_parent_particle_slots_v1"
R015_FILTER_FORMAL_PARENT_SLOT_BATCH_WIDTH = 4096
R015_FILTER_PROTOTYPE_COUNT = 4
R015_FILTER_FORMAL_PARTICLE_COUNTS = (64, 128, 256)
R015_FILTER_DESIGN_EPISODE_COUNT = 80
R015_FILTER_REPEAT_COUNT_PER_EPISODE = 2
R015_FILTER_S1_EARLY_STOP_CLOSED_EPISODE_COUNT = 2
R015_FILTER_OPENING_PREFIX_CACHE_SCHEMA = (
    "path_c_r015_conditioned_opening_particle_prefix_cache_v1"
)
R015_FILTER_OPENING_PREFIX_CACHE_CONTRACT_ID = (
    "r015_shared_conditioned_opening_256_particle_prefix_v1"
)
R015_FILTER_OPENING_PREFIX_MAXIMUM_PARTICLES_PER_PROTOTYPE = 256
R015_RANDOM_KEY_CONTRACT = "sha256_canonical_json_six_coordinates_v1"
R015_PARTICLE_LINEAGE_ID = "path_c_r015_particle_lineage_v1"
R015_RESPONSE_SPEC_PAYLOAD = {
    "schema_version": "path_c_r015_local_response_v1",
    "response_classes": ["visible", "unseen", "local_non_agent_change"],
    "latency_bin_upper_bounds": [1],
    "response_summary_sha256": REGISTERED_LOCAL_RESPONSE_SPEC.sha256,
}


class _R015OnlineFilterZeroSupportError(RuntimeError):
    """表示设备过滤器在一条具体历史上确实失去四原型正支持。"""


def _future_random_summary(
    *,
    root_key: str,
    step_keys: Sequence[str],
    step_count: int,
) -> Mapping[str, Any]:
    """把可重建的未来随机流压缩为根键、长度和完整序列摘要。"""

    keys = tuple(str(value) for value in step_keys)
    expected = tuple(
        derive_controller_key(root_key, "future_environment", index)
        for index in range(step_count)
    )
    if keys != expected:
        raise ValueError("R015 future random stream changed its registered derivation.")
    return {
        "future_random_root_key": root_key,
        "future_random_derivation_contract_id": (
            R015_FUTURE_RANDOM_DERIVATION_ID
        ),
        "future_random_step_count": step_count,
        "future_random_sequence_sha256": canonical_sha256(list(keys)),
    }


def _prepare_registered_future_random_stream(
    *,
    root_key: str,
    generated_step_count: int,
    logical_step_count: int,
) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    """一次派生固定形状设备键，并摘要实际逻辑前缀。"""

    if not _is_sha256(root_key) or not (
        1 <= logical_step_count <= generated_step_count
    ):
        raise ValueError("R015 planning future-random dimensions are invalid.")
    keys = tuple(
        derive_controller_key(root_key, "future_environment", index)
        for index in range(generated_step_count)
    )
    logical_keys = keys[:logical_step_count]
    return keys, {
        "future_random_root_key": root_key,
        "future_random_derivation_contract_id": (
            R015_FUTURE_RANDOM_DERIVATION_ID
        ),
        "future_random_step_count": logical_step_count,
        "future_random_sequence_sha256": canonical_sha256(list(logical_keys)),
    }


def _canonical_ess_fraction_for_evidence(value: Any) -> float:
    """把数值误差内的 ESS 比例上溢序列化为数学上的上界 1。"""

    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result > 1.0 + 1.0e-12:
        raise ValueError("R015 ESS/N value lies outside its mathematical range.")
    return min(result, 1.0)


def _hex_seed(value: str) -> int:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("R015 random keys must be SHA-256.")
    try:
        return int(value[:16], 16)
    except ValueError as error:
        raise ValueError("R015 random keys must be hexadecimal.") from error


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _jax_key(value: str) -> Any:
    import jax

    return jax.random.PRNGKey(_hex_seed(value) & 0xFFFFFFFF)


def _pytree_sha256(value: Any, *, domain: str) -> str:
    import jax

    leaves, tree = jax.tree_util.tree_flatten(value)
    digest = hashlib.sha256((domain + "\0").encode("utf-8"))
    tree_bytes = str(tree).encode("utf-8")
    digest.update(len(tree_bytes).to_bytes(8, "big"))
    digest.update(tree_bytes)
    for leaf in leaves:
        array = np.asarray(leaf)
        for framed in (
            str(array.dtype).encode("ascii"),
            ",".join(str(int(item)) for item in array.shape).encode("ascii"),
            np.ascontiguousarray(array).view(np.uint8).tobytes(),
        ):
            digest.update(len(framed).to_bytes(8, "big"))
            digest.update(framed)
    return digest.hexdigest()


def _planning_branch_head_projection_sha256(
    *,
    input_filter_checkpoint_sha256: str,
    official_step_record: Mapping[str, Any],
    environment_step: int,
    resampling_timing: str,
    particles_per_prototype: int,
    mode: str,
    response_token: int | None,
    prototype_ids: Sequence[str],
    prototype_masses: Any,
    closed: bool,
    committed_member_id: str,
    response_match_counts: Any | None = None,
    response_match_masses: Any | None = None,
) -> str:
    """内容寻址规划后缀实际读取的分支头信念充分投影。

    给定已摘要的输入完整粒子云、登记的一步过滤核和本步官方记录，完整输出
    粒子云可由重放唯一重建。冻结信念后缀只读取四个原型质量来选择一个成员，
    因而证据直接绑定这个充分投影，避免为摘要而把完整粒子云搬回宿主并构造
    Python 粒子对象。
    """

    if not _is_sha256(input_filter_checkpoint_sha256):
        raise ValueError("R015 branch-head projection lacks its input checkpoint hash.")
    if mode not in {"masked", "used"}:
        raise ValueError("R015 branch-head projection has an unknown response mode.")
    masses = np.ascontiguousarray(np.asarray(prototype_masses))
    if masses.dtype != np.dtype(np.float64) or masses.shape != (4,) or (
        not np.all(np.isfinite(masses))
    ):
        raise ValueError("R015 branch-head prototype masses must be four binary64 values.")
    counts_sha256 = None
    match_masses_sha256 = None
    if mode == "used":
        counts = np.ascontiguousarray(np.asarray(response_match_counts))
        match_masses = np.ascontiguousarray(np.asarray(response_match_masses))
        if counts.dtype != np.dtype(np.int32) or counts.shape != (4,) or (
            np.any(counts < 0) or np.any(counts > particles_per_prototype)
        ):
            raise ValueError("R015 response-match counts changed their exact shape.")
        if match_masses.dtype != np.dtype(np.float64) or (
            match_masses.shape != (4,)
        ) or not np.all(np.isfinite(match_masses)):
            raise ValueError("R015 response-match masses must be four binary64 values.")
        counts_sha256 = _pytree_sha256(
            counts,
            domain="path_c_r015_response_match_counts_v2",
        )
        match_masses_sha256 = _pytree_sha256(
            match_masses,
            domain="path_c_r015_response_match_masses_v2",
        )
    elif response_match_counts is not None or response_match_masses is not None:
        raise ValueError("A masked R015 branch cannot bind a response match.")
    return canonical_sha256(
        {
            "schema_version": (
                "path_c_r015_frozen_branch_head_belief_projection_v2"
            ),
            "branch_belief_rule_id": R015_BRANCH_BELIEF_ID,
            "filter_algorithm_id": R015_FILTER_ALGORITHM_ID_V2,
            "filter_key_contract": R015_FILTER_KEY_CONTRACT_V2,
            "input_filter_checkpoint_sha256": input_filter_checkpoint_sha256,
            "official_step_record_sha256": _pytree_sha256(
                official_step_record,
                domain="path_c_r015_branch_head_official_step_record_v2",
            ),
            "environment_step": int(environment_step),
            "resampling_timing": str(resampling_timing),
            "particles_per_prototype": int(particles_per_prototype),
            "mode": mode,
            "response_token": response_token,
            "prototype_ids": [str(value) for value in prototype_ids],
            "prototype_masses_sha256": _pytree_sha256(
                masses,
                domain="path_c_r015_branch_head_prototype_masses_v2",
            ),
            "closed": bool(closed),
            "committed_member_id": str(committed_member_id),
            "response_match_counts_sha256": counts_sha256,
            "response_match_masses_sha256": match_masses_sha256,
        }
    )


def _library_states_sha256(states: ContinuationLibraryStatesV1) -> str:
    return canonical_sha256(
        {
            member_id: _pytree_sha256(
                state,
                domain="path_c_r015_continuation_member_state_v1",
            )
            for member_id, state in sorted(states.by_member_id.items())
        }
    )


def _belief_sha256(belief: StratifiedParticleBeliefV1) -> str:
    return canonical_sha256(
        {
            "branch_belief_rule_id": R015_BRANCH_BELIEF_ID,
            "filter_algorithm_id": getattr(
                belief, "filter_algorithm_id", R015_FILTER_ALGORITHM_ID_V1
            ),
            "filter_key_contract": getattr(
                belief, "filter_key_contract", R015_FILTER_KEY_CONTRACT_V1
            ),
            "prototype_ids": list(belief.prototype_ids),
            "prototype_weights": dict(belief.prototype_weights),
            "particles": [
                {
                    "prototype_id": particle.prototype_id,
                    "state_sha256": particle.state_sha256,
                    "weight": particle.weight,
                }
                for particle in belief.particles
            ],
        }
    )


def _particle_state_sha256(state: "R015ParticleKernelStateV1") -> str:
    return canonical_sha256(
        {
            "snapshot_sha256": _snapshot_sha256(state.snapshot),
            "partner_prototype_id": state.partner_prototype_id,
            "partner_recurrent_state_sha256": _pytree_sha256(
                state.partner_recurrent_state,
                domain="path_c_r015_partner_recurrent_state_v1",
            ),
            "continuation_states_sha256": _library_states_sha256(
                state.continuation_states
            ),
            "predicted_response_token": state.predicted_response_token,
            "predicted_raw_team_reward": state.predicted_raw_team_reward,
            "predicted_done": state.predicted_done,
            "environment_step": state.environment_step,
        }
    )


def _clone_snapshot_with_key(snapshot: OCV2AdapterSnapshot, key: str) -> OCV2AdapterSnapshot:
    return replace(snapshot, key=_jax_key(key))


def _response_token(previous: Any, following: Any) -> int:
    before = np.asarray(previous)
    after = np.asarray(following)
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("R015 response derivation requires matching local observations.")
    visible = bool(
        partner_visible_from_default_observation(
            after[np.newaxis, ...],
            indicate_successful_delivery=True,
        )[0]
    )
    changed = bool(
        local_non_agent_change_from_default_observation(
            before[np.newaxis, np.newaxis, ...],
            after[np.newaxis, np.newaxis, ...],
            indicate_successful_delivery=True,
        )[0, 0]
    )
    return response_token_from_visibility(
        partner_action=0,
        partner_visible=visible,
        local_non_agent_change=changed,
        partner_action_channel=False,
    )


def _response_record(*, token: int, probe_id: str, probe_step: int) -> Mapping[str, Any]:
    if token not in REGISTERED_LOCAL_RESPONSE_SPEC.token_ids.values():
        raise ValueError("R015 response token is outside the registered vocabulary.")
    return {
        "schema_version": "path_c_r015_response_summary_v1",
        "response_summary_spec_sha256": REGISTERED_LOCAL_RESPONSE_SPEC.sha256,
        "token_id": int(token),
        "latency_steps": 1,
        "source_probe_id": str(probe_id),
        "source_probe_step": int(probe_step),
    }


def _passive_record(
    *, observation: Any, ego_action: str, reward: float, done: bool
) -> Mapping[str, Any]:
    return {
        "official_local_observation": np.asarray(observation),
        "ego_action_history": [str(ego_action)],
        "raw_team_reward_history": [float(reward)],
        "episode_boundaries": [bool(done)],
    }


def _history_record(
    *, observation: Any, ego_action: str | None, reward: float | None, done: bool
) -> Mapping[str, Any]:
    record: dict[str, Any] = {
        "official_local_observation": np.asarray(observation),
        "episode_boundaries": [bool(done)],
    }
    if ego_action is not None:
        record["ego_action_history"] = [str(ego_action)]
    if reward is not None:
        record["raw_team_reward_history"] = [float(reward)]
    return record


def _wrong_delivery_event(
    snapshot: OCV2AdapterSnapshot,
    next_snapshot: OCV2AdapterSnapshot,
    *,
    ego_action: str,
    partner_action: str,
) -> bool:
    """从官方状态重算当前动作是否产生错误交付。"""

    import jax
    import jax.numpy as jnp
    from jaxmarl.environments.overcooked_v2.common import (
        Actions,
        DynamicObject,
        StaticObject,
    )

    state = snapshot.state
    action_array = jnp.asarray(
        [OCV2_ACTION_INDEX[partner_action], OCV2_ACTION_INDEX[ego_action]]
    )
    forward = jax.vmap(lambda agent: agent.get_fwd_pos())(state.agents)
    cells = state.grid[forward.y, forward.x]
    deliveries = (
        (action_array == int(Actions.interact))
        & (cells[:, 0] == int(StaticObject.GOAL))
        & ((state.agents.inventory & int(DynamicObject.COOKED)) != 0)
    )
    delivery_count = int(np.asarray(jnp.sum(deliveries)))
    correct = int(np.asarray(next_snapshot.state.new_correct_delivery))
    return delivery_count > correct


@dataclass(frozen=True)
class R015ParticleKernelStateV1:
    """一个粒子私有的环境、伙伴循环状态和五成员循环状态。"""

    snapshot: OCV2AdapterSnapshot
    partner_prototype_id: str
    partner_recurrent_state: Any
    continuation_states: ContinuationLibraryStatesV1
    predicted_response_token: int | None = None
    predicted_raw_team_reward: float | None = None
    predicted_done: bool | None = None
    environment_step: int = 0


@dataclass(frozen=True)
class R015OnlineParticleBeliefV2:
    """正式执行所用的四原型条件粒子后验。

    ``prototype_masses`` 保存四个原型之间的概率，
    ``within_prototype_weights`` 保存每个原型内部的条件粒子权重。粒子对象中的
    ``weight`` 只作为两者乘积的兼容视图，供既有规划抽样和延续控制器读取；更新时不得
    从该兼容视图反推后再把四个原型混成一个重采样池。
    """

    prototype_ids: tuple[str, ...]
    particles_per_prototype: int
    registered_prototype_prior: Mapping[str, float]
    prototype_masses: Mapping[str, float]
    within_prototype_weights: Mapping[str, tuple[float, ...]]
    particles: tuple[HiddenStateParticleV1, ...]
    resampling_timing: str
    resampling_ess_fraction_threshold: float | None
    last_pre_resample_ess_fraction_by_prototype: Mapping[str, float]
    last_resampled_prototypes: tuple[str, ...]
    last_ancestor_indices_by_prototype: Mapping[str, tuple[int, ...]]
    last_predictive_mass_by_prototype: Mapping[str, float]
    opening_accepted_proposal_indices_by_prototype: Mapping[
        str, tuple[int, ...]
    ]
    opening_proposal_counts_by_prototype: Mapping[str, int]
    opening_rejected_proposal_counts_by_prototype: Mapping[str, int]
    device_filter_state: Any = field(repr=False, compare=False)
    device_state_reweighter: Callable[
        [Any, Mapping[str, float], Mapping[str, tuple[float, ...]]], Any
    ] = field(repr=False, compare=False)
    update_count: int = 0
    resampling_algorithm: str = "strict_systematic_per_prototype_v2"
    resampling_interval_environment_steps: int = 1
    filter_algorithm_id: str = R015_FILTER_ALGORITHM_ID_V2
    filter_key_contract: str = R015_FILTER_KEY_CONTRACT_V2

    def __post_init__(self) -> None:
        ids = tuple(self.prototype_ids)
        if len(ids) != R015_FILTER_PROTOTYPE_COUNT or len(set(ids)) != len(ids):
            raise ValueError("R015 v2 online filtering requires four prototypes.")
        count = self.particles_per_prototype
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("R015 v2 online filtering requires a positive particle count.")
        if len(self.particles) != len(ids) * count:
            raise ValueError("R015 v2 online belief has the wrong particle count.")
        if self.filter_algorithm_id != R015_FILTER_ALGORITHM_ID_V2:
            raise ValueError("R015 online execution cannot use the retired v1 filter.")
        if self.filter_key_contract != R015_FILTER_KEY_CONTRACT_V2:
            raise ValueError("R015 online execution uses the wrong device-key contract.")
        if self.resampling_algorithm != "strict_systematic_per_prototype_v2":
            raise ValueError("R015 online execution requires strict systematic resampling.")
        if self.resampling_interval_environment_steps != 1:
            raise ValueError("R015 v2 resampling decisions are evaluated every step.")
        if self.resampling_timing not in {
            "adaptive_ess_below_half_v1",
            "every_environment_step_v1",
        }:
            raise ValueError("R015 v2 online filtering uses an unregistered timing.")
        threshold = self.resampling_ess_fraction_threshold
        if self.resampling_timing == "adaptive_ess_below_half_v1":
            if threshold is None or not math.isclose(
                float(threshold), 0.5, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError("Adaptive R015 resampling requires ESS below 0.5P.")
        elif threshold is not None:
            raise ValueError("Every-step R015 resampling has no ESS threshold.")
        if set(self.registered_prototype_prior) != set(ids) or set(self.prototype_masses) != set(ids):
            raise ValueError("R015 v2 belief fields must cover all prototypes.")
        prior = {
            prototype_id: float(self.registered_prototype_prior[prototype_id])
            for prototype_id in ids
        }
        masses = {
            prototype_id: float(self.prototype_masses[prototype_id])
            for prototype_id in ids
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in prior.values()):
            raise ValueError("R015 registered prototype prior must be positive.")
        if any(not math.isfinite(value) or value <= 0.0 for value in masses.values()):
            raise ValueError("R015 v2 finite particle cloud lost prototype support.")
        if not math.isclose(sum(prior.values()), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("R015 registered prototype prior must sum to one.")
        if not math.isclose(sum(masses.values()), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("R015 v2 prototype masses must sum to one.")
        if set(self.within_prototype_weights) != set(ids):
            raise ValueError("R015 v2 within-prototype weights are incomplete.")
        normalized_within: dict[str, tuple[float, ...]] = {}
        for prototype_id in ids:
            values = tuple(
                float(value) for value in self.within_prototype_weights[prototype_id]
            )
            if len(values) != count or any(
                not math.isfinite(value) or value < 0.0 for value in values
            ):
                raise ValueError("R015 v2 within-prototype weights are invalid.")
            if not math.isclose(sum(values), 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
                raise ValueError("R015 v2 within-prototype weights must sum to one.")
            normalized_within[prototype_id] = values
        for prototype_index, prototype_id in enumerate(ids):
            start = prototype_index * count
            group = self.particles[start : start + count]
            if any(particle.prototype_id != prototype_id for particle in group):
                raise ValueError("R015 v2 particles are not grouped by prototype.")
            for particle, conditional_weight in zip(
                group, normalized_within[prototype_id]
            ):
                if not math.isclose(
                    particle.weight,
                    masses[prototype_id] * conditional_weight,
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-14,
                ):
                    raise ValueError(
                        "R015 v2 particle compatibility weights changed their factorization."
                    )
        diagnostics = {
            prototype_id: float(
                self.last_pre_resample_ess_fraction_by_prototype[prototype_id]
            )
            for prototype_id in self.last_pre_resample_ess_fraction_by_prototype
        }
        if diagnostics and set(diagnostics) != set(ids):
            raise ValueError("R015 v2 ESS diagnostics must cover all prototypes.")
        if any(
            not math.isfinite(value) or not 0.0 < value <= 1.0 + 1.0e-12
            for value in diagnostics.values()
        ):
            raise ValueError("R015 v2 ESS/P must lie in (0,1].")
        if set(self.last_ancestor_indices_by_prototype) not in (set(), set(ids)):
            raise ValueError("R015 v2 ancestor diagnostics are incomplete.")
        for prototype_id, indices in self.last_ancestor_indices_by_prototype.items():
            if len(indices) != count or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < count
                for index in indices
            ):
                raise ValueError("R015 v2 ancestor index is outside its prototype.")
        if set(self.last_predictive_mass_by_prototype) not in (set(), set(ids)):
            raise ValueError("R015 v2 predictive-mass diagnostics are incomplete.")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.last_predictive_mass_by_prototype.values()
        ):
            raise ValueError("R015 v2 predictive mass must remain positive.")
        if any(item not in ids for item in self.last_resampled_prototypes):
            raise ValueError("R015 v2 resampling diagnostics name an unknown prototype.")
        opening_fields = (
            self.opening_accepted_proposal_indices_by_prototype,
            self.opening_proposal_counts_by_prototype,
            self.opening_rejected_proposal_counts_by_prototype,
        )
        if any(set(field) != set(ids) for field in opening_fields):
            raise ValueError("R015 v2 opening-proposal evidence is incomplete.")
        for prototype_id in ids:
            accepted = self.opening_accepted_proposal_indices_by_prototype[
                prototype_id
            ]
            proposed = self.opening_proposal_counts_by_prototype[prototype_id]
            rejected = self.opening_rejected_proposal_counts_by_prototype[
                prototype_id
            ]
            if (
                len(accepted) != count
                or len(set(accepted)) != count
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    for index in accepted
                )
                or isinstance(proposed, bool)
                or not isinstance(proposed, int)
                or proposed < count
                or isinstance(rejected, bool)
                or not isinstance(rejected, int)
                or rejected != proposed - count
                or max(accepted) >= proposed
            ):
                raise ValueError("R015 v2 opening proposals were not independent accepts.")
        if isinstance(self.update_count, bool) or not isinstance(self.update_count, int) or self.update_count < 0:
            raise ValueError("R015 v2 update count must be non-negative.")
        if self.device_filter_state is None or not callable(self.device_state_reweighter):
            raise ValueError("R015 v2 belief requires its persistent device state.")
        object.__setattr__(self, "registered_prototype_prior", prior)
        object.__setattr__(self, "prototype_masses", masses)
        object.__setattr__(self, "within_prototype_weights", normalized_within)
        object.__setattr__(self, "last_pre_resample_ess_fraction_by_prototype", diagnostics)

    @property
    def prototype_weights(self) -> Mapping[str, float]:
        return dict(self.prototype_masses)

    @property
    def public_summary(self) -> Mapping[str, Any]:
        return {
            "schema_version": "path_c_r015_particle_belief_summary_v2",
            "filter_algorithm_id": self.filter_algorithm_id,
            "filter_key_contract": self.filter_key_contract,
            "registered_prototype_prior": dict(self.registered_prototype_prior),
            "prototype_weights": dict(self.prototype_masses),
            "particle_counts": {
                prototype_id: self.particles_per_prototype
                for prototype_id in self.prototype_ids
            },
            "resampling_algorithm": self.resampling_algorithm,
            "resampling_timing": self.resampling_timing,
            "resampling_ess_fraction_threshold": self.resampling_ess_fraction_threshold,
            "pre_resample_ess_fraction_by_prototype": dict(
                self.last_pre_resample_ess_fraction_by_prototype
            ),
            "resampled_prototypes": list(self.last_resampled_prototypes),
            "update_count": self.update_count,
        }

    def _from_total_weights(
        self,
        total_weights: Sequence[float],
    ) -> "R015OnlineParticleBeliefV2":
        values = tuple(float(value) for value in total_weights)
        if len(values) != len(self.particles) or any(
            not math.isfinite(value) or value < 0.0 for value in values
        ):
            raise ValueError("R015 v2 response update returned invalid weights.")
        total = sum(values)
        if total <= 0.0:
            raise _R015OnlineFilterZeroSupportError(
                "Current response has zero probability under all support."
            )
        normalized = tuple(value / total for value in values)
        masses: dict[str, float] = {}
        within: dict[str, tuple[float, ...]] = {}
        output_particles: list[HiddenStateParticleV1] = []
        for prototype_index, prototype_id in enumerate(self.prototype_ids):
            start = prototype_index * self.particles_per_prototype
            group_weights = normalized[start : start + self.particles_per_prototype]
            mass = sum(group_weights)
            if mass <= 0.0:
                raise _R015OnlineFilterZeroSupportError(
                    "Current response removed one prototype from support."
                )
            masses[prototype_id] = mass
            within[prototype_id] = tuple(value / mass for value in group_weights)
            output_particles.extend(
                replace(particle, weight=weight)
                for particle, weight in zip(
                    self.particles[start : start + self.particles_per_prototype],
                    group_weights,
                )
            )
        return replace(
            self,
            prototype_masses=masses,
            within_prototype_weights=within,
            particles=tuple(output_particles),
            device_filter_state=self.device_state_reweighter(
                self.device_filter_state,
                masses,
                within,
            ),
        )

    def use_current_response(
        self,
        current_response: Mapping[str, Any],
        *,
        likelihood: Callable[[HiddenStateParticleV1, Mapping[str, Any]], float],
    ) -> "R015OnlineParticleBeliefV2":
        if not isinstance(current_response, Mapping) or not current_response:
            raise ValueError("R015 response update requires one registered response.")
        return self._from_total_weights(
            tuple(
                particle.weight * float(likelihood(particle, current_response))
                for particle in self.particles
            )
        )


def _materialize_online_particle_belief_v2(
    *,
    production_backend: Any,
    adapter: OCV2Adapter,
    filter_state: Mapping[str, Any],
    continuation_states: ContinuationLibraryStatesV1,
    resampling_timing: str,
    template_belief: R015OnlineParticleBeliefV2 | None = None,
    previous_official_observation: Any | None = None,
    predicted_raw_team_reward: float | None = None,
    predicted_done: bool | None = None,
) -> R015OnlineParticleBeliefV2:
    """在设备段边界把一条 v2 过滤车道绑定回完整粒子对象。

    设备过滤状态只保存环境和伙伴循环状态。五个延续策略成员读取的是同一份
    官方主体历史，因此它们的当前循环状态在一个真实执行组内共享；这里把段末
    的五成员状态重新绑定到每个粒子，避免沿用段首的旧状态。
    """

    if not isinstance(continuation_states, ContinuationLibraryStatesV1):
        raise TypeError("R015 v2 materialization requires five continuation states.")
    materialize = getattr(production_backend, "materialize_online_filter_v2", None)
    reweight = getattr(production_backend, "reweight_online_filter_v2_response", None)
    if not callable(materialize) or not callable(reweight):
        raise RuntimeError("R015 v2 materialization requires the shared filter bridge.")
    host = dict(materialize(filter_state=filter_state))
    masses_array = np.asarray(host["prototype_masses"], dtype=np.float64)
    within_array = np.asarray(
        host["within_prototype_weights"], dtype=np.float64
    )
    if masses_array.ndim != 2 or masses_array.shape[0] != 1 or (
        within_array.ndim != 3
        or within_array.shape[:2] != masses_array.shape
    ):
        raise ValueError("R015 v2 materialization requires exactly one filter lane.")
    prototype_ids = tuple(production_backend.prototype_ids)
    if masses_array.shape[1] != len(prototype_ids) or len(prototype_ids) != 4:
        raise ValueError("R015 v2 materialization changed the four prototypes.")
    particles_per_prototype = int(within_array.shape[2])
    closed = np.asarray(host["closed"], dtype=np.bool_)
    if closed.shape != (1,):
        raise ValueError("R015 v2 filter changed its closed-indicator lane shape.")
    if bool(closed[0]):
        raise _R015OnlineFilterZeroSupportError(
            "R015 v2 filter lost support before materialization."
        )
    update_count_array = np.asarray(host["update_count"], dtype=np.int64)
    if update_count_array.shape != (1,):
        raise ValueError("R015 v2 filter update count changed its lane shape.")
    environment_step = int(update_count_array[0])

    accepted_sources = np.asarray(
        host["opening_accepted_source_indices"], dtype=np.int64
    )
    proposal_counts = np.asarray(host["opening_proposal_counts"], dtype=np.int64)
    accepted_counts = np.asarray(host["opening_accepted_counts"], dtype=np.int64)
    ess_array = np.asarray(
        host["pre_resample_ess_fraction"], dtype=np.float64
    )
    response_tokens = np.asarray(
        host["last_predicted_response_tokens"], dtype=np.int32
    )
    expected_grid = (1, len(prototype_ids), particles_per_prototype)
    if accepted_sources.shape != expected_grid or accepted_counts.shape != (
        1,
        len(prototype_ids),
    ) or proposal_counts.shape != accepted_counts.shape or ess_array.shape != (
        1,
        len(prototype_ids),
    ) or response_tokens.shape != expected_grid:
        raise ValueError("R015 v2 opening evidence changed its registered shape.")
    if np.any(accepted_counts[0] != particles_per_prototype):
        raise ValueError("R015 v2 opening conditioning did not fill every stratum.")

    import jax

    environment_grid = host["environment_state"]
    observation_grid = dict(host["raw_observation"])
    snapshot_keys = np.asarray(host["snapshot_keys"])
    partner_states = tuple(host["partner_recurrent_states"])
    if len(partner_states) != len(prototype_ids):
        raise ValueError("R015 v2 materialization changed the partner-state strata.")

    def slice_particle(value: Any, prototype_index: int, particle_index: int) -> Any:
        return jax.tree_util.tree_map(
            lambda item: np.asarray(item)[0, prototype_index, particle_index],
            value,
        )

    def slice_partner(value: Any, particle_index: int) -> Any:
        return jax.tree_util.tree_map(
            lambda item: np.asarray(item)[0, particle_index : particle_index + 1],
            value,
        )

    masses = {
        prototype_id: float(masses_array[0, prototype_index])
        for prototype_index, prototype_id in enumerate(prototype_ids)
    }
    within = {
        prototype_id: tuple(
            float(value) for value in within_array[0, prototype_index]
        )
        for prototype_index, prototype_id in enumerate(prototype_ids)
    }
    particles: list[HiddenStateParticleV1] = []
    for prototype_index, prototype_id in enumerate(prototype_ids):
        for particle_index in range(particles_per_prototype):
            raw_obs = {
                agent_id: np.asarray(value)[
                    0, prototype_index, particle_index
                ]
                for agent_id, value in observation_grid.items()
            }
            kernel = R015ParticleKernelStateV1(
                snapshot=OCV2AdapterSnapshot(
                    layout_name=adapter.layout_name,
                    max_steps=int(adapter.max_steps),
                    key=np.asarray(snapshot_keys)[
                        0, prototype_index, particle_index
                    ],
                    state=slice_particle(
                        environment_grid, prototype_index, particle_index
                    ),
                    raw_obs=raw_obs,
                ),
                partner_prototype_id=prototype_id,
                partner_recurrent_state=slice_partner(
                    partner_states[prototype_index], particle_index
                ),
                # 五个成员由同一份官方历史推进，当前循环状态对所有兼容粒子
                # 完全相同。粒子只共享这个不可变包装；真正分支时再深拷贝，
                # 避免每个咨询边界按四倍于每原型粒子数复制相同数组。
                continuation_states=continuation_states,
                predicted_response_token=(
                    None
                    if int(
                        response_tokens[
                            0, prototype_index, particle_index
                        ]
                    )
                    < 0
                    else int(
                        response_tokens[
                            0, prototype_index, particle_index
                        ]
                    )
                ),
                predicted_raw_team_reward=predicted_raw_team_reward,
                predicted_done=predicted_done,
                environment_step=environment_step,
            )
            particles.append(
                HiddenStateParticleV1(
                    prototype_id=prototype_id,
                    state_sha256=_particle_state_sha256(kernel),
                    state=kernel,
                    weight=(
                        masses[prototype_id]
                        * within[prototype_id][particle_index]
                    ),
                )
            )

    opening_indices = {
        prototype_id: tuple(
            int(value) for value in accepted_sources[0, prototype_index]
        )
        for prototype_index, prototype_id in enumerate(prototype_ids)
    }
    proposals = {
        prototype_id: int(proposal_counts[0, prototype_index])
        for prototype_index, prototype_id in enumerate(prototype_ids)
    }
    rejected = {
        prototype_id: proposals[prototype_id] - particles_per_prototype
        for prototype_id in prototype_ids
    }
    prior = (
        dict(template_belief.registered_prototype_prior)
        if template_belief is not None
        else {prototype_id: 0.25 for prototype_id in prototype_ids}
    )

    def reweight_device_state(
        state: Any,
        new_masses: Mapping[str, float],
        new_within: Mapping[str, tuple[float, ...]],
    ) -> Any:
        return reweight(
            filter_state=state,
            prototype_masses=np.asarray(
                [[new_masses[prototype_id] for prototype_id in prototype_ids]],
                dtype=np.float64,
            ),
            within_prototype_weights=np.asarray(
                [[new_within[prototype_id] for prototype_id in prototype_ids]],
                dtype=np.float64,
            ),
        )

    return R015OnlineParticleBeliefV2(
        prototype_ids=prototype_ids,
        particles_per_prototype=particles_per_prototype,
        registered_prototype_prior=prior,
        prototype_masses=masses,
        within_prototype_weights=within,
        particles=tuple(particles),
        resampling_timing=resampling_timing,
        resampling_ess_fraction_threshold=(
            0.5 if resampling_timing == "adaptive_ess_below_half_v1" else None
        ),
        last_pre_resample_ess_fraction_by_prototype={
            prototype_id: _canonical_ess_fraction_for_evidence(
                ess_array[0, prototype_index]
            )
            for prototype_index, prototype_id in enumerate(prototype_ids)
        },
        last_resampled_prototypes=(),
        last_ancestor_indices_by_prototype={},
        last_predictive_mass_by_prototype={},
        opening_accepted_proposal_indices_by_prototype=opening_indices,
        opening_proposal_counts_by_prototype=proposals,
        opening_rejected_proposal_counts_by_prototype=rejected,
        device_filter_state=filter_state,
        device_state_reweighter=reweight_device_state,
        update_count=environment_step,
    )


@dataclass(frozen=True)
class R015BranchStateV1:
    kernel: R015ParticleKernelStateV1
    belief: StratifiedParticleBeliefV1 | R015OnlineParticleBeliefV2
    history: OfficialHistoryV1
    environment_step: int
    belief_evidence_sha256: str | None = None


@dataclass(frozen=True)
class _BranchExecutionV1:
    state: R015BranchStateV1
    raw_return: float
    trajectory: tuple[Mapping[str, Any], ...]
    future_random_keys: tuple[str, ...]
    wrong_delivery_detected: bool
    throughput: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class R015PlanningPointBatchRequestV2:
    """一个咨询点尚未计算的规划样本及其完整 v2 信念状态。"""

    consultation_id: str
    filter_checkpoint_sha256: str
    belief: R015OnlineParticleBeliefV2
    history: OfficialHistoryV1
    samples: tuple[PlanningBranchSampleV1, ...]
    remaining_steps: int

    def __post_init__(self) -> None:
        if not _is_sha256(self.consultation_id) or not _is_sha256(
            self.filter_checkpoint_sha256
        ):
            raise ValueError("R015 planning point requires content-addressed ids.")
        if not isinstance(self.belief, R015OnlineParticleBeliefV2):
            raise TypeError("R015 planning point requires the selected v2 filter.")
        if not isinstance(self.history, OfficialHistoryV1):
            raise TypeError("R015 planning point requires official history.")
        if not self.samples or len(
            {sample.canonical_slot_16 for sample in self.samples}
        ) != len(self.samples):
            raise ValueError("R015 planning point has no unique missing samples.")
        if not 300 <= self.remaining_steps <= 399:
            raise ValueError("R015 planning point lies outside the registered window.")


@dataclass(frozen=True)
class R015PlanningLengthBucketResultV2:
    """同一剩余长度的跨咨询点规划输出及一次真实设备计数。"""

    remaining_steps: int
    rollouts_by_consultation_id: Mapping[str, PlanningBatchRolloutsV1]
    device_execution: Mapping[str, Any]


def _execution_state_summaries(
    execution: _BranchExecutionV1,
    *,
    ego_action_prefix: Sequence[int] = (),
    partner_action_prefix: Sequence[int] = (),
) -> Mapping[str, str]:
    return {
        "ego_action_sequence_sha256": canonical_sha256(
            [
                *ego_action_prefix,
                *[
                    OCV2_ACTION_INDEX[step["ego_action"]]
                    for step in execution.trajectory
                ],
            ]
        ),
        "partner_action_sequence_sha256": canonical_sha256(
            [
                *partner_action_prefix,
                *[
                    OCV2_ACTION_INDEX[step["partner_action"]]
                    for step in execution.trajectory
                ],
            ]
        ),
        "final_environment_state_sha256": _pytree_sha256(
            execution.state.kernel.snapshot.state,
            domain="path_c_r015_batch_final_environment_state_v1",
        ),
        "final_partner_state_sha256": _pytree_sha256(
            execution.state.kernel.partner_recurrent_state,
            domain="path_c_r015_batch_final_partner_state_v1",
        ),
        "final_continuation_states_sha256": _pytree_sha256(
            execution.state.kernel.continuation_states.by_member_id,
            domain="path_c_r015_batch_final_continuation_states_v1",
        ),
    }


@dataclass
class OCV2R015FullHorizonExecutorV1:
    """使用官方环境和五个真实 checkpoint 执行完整剩余回合。"""

    adapter: OCV2Adapter
    production_backend: Any
    response_projection: CurrentResponseProjectionV1 = CurrentResponseProjectionV1(
        alias_paths=(),
        recurrent_state_write_paths=(),
    )
    continuation_controller_id: str = R015_CONTINUATION_CONTROLLER_ID
    formal_freeze_identity_verifier: Callable[[], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _conditioned_opening_prefix_caches: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _prepared_planning_batches: dict[
        tuple[str, str, int, tuple[tuple[int, str], ...]],
        PlanningBatchRolloutsV1,
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.adapter.max_steps) != 400:
            raise ValueError("R015 full-horizon execution requires 400 steps.")
        if self.production_backend.continuation_controller_id != (
            self.continuation_controller_id
        ):
            raise ValueError("R015 executor and policy bridge use different C(q,x).")

    @property
    def prototype_ids(self) -> tuple[str, ...]:
        return tuple(self.production_backend.prototype_ids)

    def candidate_status(
        self,
        *,
        history: OfficialHistoryV1,
        script: ProbeScriptV1,
        environment_step: int,
    ) -> tuple[bool, bool]:
        if not isinstance(history, OfficialHistoryV1):
            raise TypeError("R015 candidate status requires official history.")
        eligible = (
            environment_step in R015_CONSULTATION_STEPS
            and script.probe_id in R015_ATOMIC_ACTIONS
            and tuple(script.primitive_actions) == (script.probe_id,)
        )
        return eligible, eligible

    @staticmethod
    def _validate_conditioned_opening_prefix_cache_binding(
        binding: Mapping[str, Any],
        *,
        history_stream_pairs: Sequence[tuple[Mapping[str, Any], str]],
    ) -> str:
        """核对缓存只绑定当前源码、配置、初始化键和完整历史。

        这里不从磁盘读取粒子状态。进程崩溃后缓存为空并重新生成；恢复时传入的
        绑定必须由设计入口从当前文件重新计算，不能仅凭旧缓存文件名继续使用。
        """

        normalized = dict(binding)
        digest = normalized.pop("cache_binding_sha256", None)
        if not _is_sha256(digest) or digest != canonical_sha256(normalized):
            raise ValueError("R015 conditioned-opening cache digest is invalid.")
        if (
            normalized.get("schema_version")
            != R015_FILTER_OPENING_PREFIX_CACHE_SCHEMA
            or normalized.get("cache_contract_id")
            != R015_FILTER_OPENING_PREFIX_CACHE_CONTRACT_ID
            or normalized.get("filter_algorithm_id")
            != R015_FILTER_ALGORITHM_ID_V2
            or normalized.get("filter_key_contract")
            != R015_FILTER_KEY_CONTRACT_V2
            or normalized.get("maximum_particles_per_prototype")
            != R015_FILTER_OPENING_PREFIX_MAXIMUM_PARTICLES_PER_PROTOTYPE
            or normalized.get("stored_state_scope")
            != "conditioned_opening_particles_only"
            or normalized.get("post_resampling_state_reuse_allowed") is not False
        ):
            raise ValueError("R015 conditioned-opening cache changed its scope.")
        for field_name in (
            "protocol_file_sha256",
            "protocol_payload_sha256",
            "design_histories_file_sha256",
            "design_histories_payload_sha256",
            "source_closure_sha256",
        ):
            if not _is_sha256(normalized.get(field_name)):
                raise ValueError("R015 conditioned-opening cache lacks a source binding.")
        source_files = normalized.get("source_file_sha256")
        if not isinstance(source_files, Mapping) or not source_files or any(
            not _is_sha256(value) for value in source_files.values()
        ) or normalized["source_closure_sha256"] != canonical_sha256(
            dict(source_files)
        ):
            raise ValueError("R015 conditioned-opening source closure is invalid.")
        expected_lanes = tuple(
            {
                "design_episode_id": str(history["design_episode_id"]),
                "design_sequence_index": int(history["design_sequence_index"]),
                "filter_stream_id": str(stream_id),
                "filter_initialization_key": canonical_sha256(
                    [
                        R015_FILTER_KEY_CONTRACT_V2,
                        str(stream_id),
                        int(history["design_sequence_index"]),
                    ]
                ),
                "official_history_sha256": canonical_sha256(
                    tuple(history.get("official_history", ()))
                ),
            }
            for history, stream_id in history_stream_pairs
        )
        realized_lanes = normalized.get("lane_bindings")
        if not isinstance(realized_lanes, Sequence) or isinstance(
            realized_lanes, (str, bytes, bytearray)
        ) or tuple(realized_lanes) != expected_lanes:
            raise ValueError("R015 conditioned-opening cache changed its lanes.")
        return str(digest)

    def _conditioned_opening_prefix_cache(
        self,
        binding: Mapping[str, Any],
        *,
        history_stream_pairs: Sequence[tuple[Mapping[str, Any], str]],
    ) -> tuple[str, dict[str, Any]]:
        """返回只在当前进程内复用的已条件化开局粒子前缀。"""

        binding_sha256 = self._validate_conditioned_opening_prefix_cache_binding(
            binding,
            history_stream_pairs=history_stream_pairs,
        )
        cache = self._conditioned_opening_prefix_caches.get(binding_sha256)
        if cache is None:
            cache = {
                "binding": copy.deepcopy(dict(binding)),
                "conditioned_opening_prefixes": {},
            }
            self._conditioned_opening_prefix_caches[binding_sha256] = cache
        elif cache.get("binding") != dict(binding):
            raise RuntimeError("R015 conditioned-opening cache binding changed in memory.")
        prefixes = cache.get("conditioned_opening_prefixes")
        if not isinstance(prefixes, dict):
            raise RuntimeError("R015 conditioned-opening cache payload is malformed.")
        return binding_sha256, prefixes

    def release_conditioned_opening_prefix_cache(self, binding_sha256: str) -> int:
        """过滤数值选择完成后释放开局粒子，不让它占用后续规划显存。"""

        if not _is_sha256(binding_sha256):
            raise ValueError("R015 opening-prefix cache release requires SHA-256.")
        cache = self._conditioned_opening_prefix_caches.pop(binding_sha256, None)
        if cache is None:
            return 0
        prefixes = cache.get("conditioned_opening_prefixes", {})
        if not isinstance(prefixes, Mapping):
            raise RuntimeError("R015 opening-prefix cache payload changed before release.")
        return len(prefixes)

    def _initial_kernel(self, prototype_id: str, seed_key: str) -> R015ParticleKernelStateV1:
        self.adapter.reset(_hex_seed(seed_key))
        return R015ParticleKernelStateV1(
            snapshot=self.adapter.capture_state(),
            partner_prototype_id=prototype_id,
            partner_recurrent_state=self.production_backend.initial_partner_state(
                prototype_id
            ),
            continuation_states=self.production_backend.initial_continuation_states(),
            environment_step=0,
        )

    def initialize_belief(
        self,
        *,
        official_initial_observation: Any,
        particles_per_prototype: int,
        resampling_timing: str,
        initialization_key: str,
    ) -> R015OnlineParticleBeliefV2:
        """从官方开局观测直接建立设备常驻的正式 v2 粒子后验。"""

        if (
            isinstance(particles_per_prototype, bool)
            or not isinstance(particles_per_prototype, int)
            or particles_per_prototype <= 0
        ):
            raise ValueError("R015 v2 initialization requires a positive P.")
        if resampling_timing not in {
            "adaptive_ess_below_half_v1",
            "every_environment_step_v1",
        }:
            raise ValueError("R015 v2 initialization uses an unregistered timing.")
        if not _is_sha256(initialization_key):
            raise ValueError("R015 v2 initialization requires an identity-free SHA-256 key.")
        initializer = getattr(
            self.production_backend,
            "initialize_online_filter_v2",
            None,
        )
        if not callable(initializer):
            raise RuntimeError("R015 formal execution lacks the v2 filter initializer.")
        opening_record = {
            "official_local_observation": np.asarray(
                official_initial_observation
            ),
            "episode_boundaries": [True],
        }
        OfficialHistoryV1((opening_record,))
        device_state = initializer(
            adapter=self.adapter,
            official_opening_records=(opening_record,),
            initialization_keys=(initialization_key,),
            particles_per_prototype=particles_per_prototype,
        )
        return _materialize_online_particle_belief_v2(
            production_backend=self.production_backend,
            adapter=self.adapter,
            filter_state=device_state,
            continuation_states=(
                self.production_backend.initial_continuation_states()
            ),
            resampling_timing=resampling_timing,
        )

    def update_belief(
        self,
        belief: R015OnlineParticleBeliefV2,
        record: Mapping[str, Any],
        *,
        key: str,
        continuation_states: ContinuationLibraryStatesV1,
    ) -> R015OnlineParticleBeliefV2:
        """在诊断单步边界调用与设备扫描相同的 v2 一步核。"""

        if not isinstance(belief, R015OnlineParticleBeliefV2):
            raise TypeError("R015 formal belief update refuses the retired v1 filter.")
        OfficialHistoryV1((record,))
        if set(record) != R015_PASSIVE_FILTER_UPDATE_FIELDS:
            raise ValueError(
                "R015 passive filtering accepts only official observation, ego action, "
                "raw team reward, and episode boundary fields."
            )
        actions = record.get("ego_action_history")
        rewards = record.get("raw_team_reward_history")
        boundaries = record.get("episode_boundaries")
        if not isinstance(actions, Sequence) or len(actions) != 1 or str(
            actions[0]
        ) not in OCV2_ACTION_INDEX:
            raise ValueError("R015 passive filtering requires one primitive ego action.")
        if not isinstance(rewards, Sequence) or len(rewards) != 1 or not isinstance(
            boundaries, Sequence
        ) or len(boundaries) != 1:
            raise ValueError("R015 passive filtering requires one reward and boundary.")
        if not _is_sha256(key):
            raise ValueError("R015 v2 update requires a SHA-256 key.")
        # ``key`` 只证明调用方给出了规范步骤坐标；过滤随机数必须由设备状态中
        # 持久的初始化根键和完成后的环境步派生，不能读取真实 episode seed。
        advance = getattr(self.production_backend, "advance_online_filter_v2", None)
        if not callable(advance):
            raise RuntimeError("R015 formal execution lacks the v2 filter update kernel.")
        import jax.numpy as jnp

        advanced = advance(
            adapter=self.adapter,
            filter_state=belief.device_filter_state,
            official_step_records={
                "official_local_observation": jnp.asarray(
                    np.asarray(record["official_local_observation"])[None, ...]
                ),
                "ego_action_index": jnp.asarray(
                    [OCV2_ACTION_INDEX[str(actions[0])]], dtype=jnp.int32
                ),
                "raw_team_reward": jnp.asarray(
                    [float(rewards[0])], dtype=jnp.float32
                ),
                "episode_boundary": jnp.asarray(
                    [bool(boundaries[0])], dtype=jnp.bool_
                ),
            },
            # 设备过滤坐标按“完成后的环境步”计数；开局后的第一条记录是 1，
            # 与离线 1,...,400 扫描及正式真实执行保持同一坐标。
            environment_step=jnp.asarray(
                [belief.update_count + 1], dtype=jnp.int32
            ),
            resampling_timing=belief.resampling_timing,
            return_diagnostics=True,
        )
        first_state = belief.particles[0].state
        if not isinstance(first_state, R015ParticleKernelStateV1):
            raise TypeError("R015 v2 particle contains another kernel state.")
        return _materialize_online_particle_belief_v2(
            production_backend=self.production_backend,
            adapter=self.adapter,
            filter_state=advanced["filter_state"],
            continuation_states=continuation_states,
            resampling_timing=belief.resampling_timing,
            template_belief=belief,
            previous_official_observation=(
                first_state.snapshot.raw_obs["agent_1"]
            ),
            predicted_raw_team_reward=float(rewards[0]),
            predicted_done=bool(boundaries[0]),
        )

    def initialize_belief_v1_diagnostic(
        self,
        *,
        official_initial_observation: Any,
        particles_per_prototype: int,
        resampling_timing: str,
        initialization_key: str,
    ) -> StratifiedParticleBeliefV1:
        if particles_per_prototype <= 0:
            raise ValueError("R015 particle count must be positive.")
        prior = {prototype_id: 0.25 for prototype_id in self.prototype_ids}
        particles: list[HiddenStateParticleV1] = []
        for prototype_id in self.prototype_ids:
            candidates: list[tuple[R015ParticleKernelStateV1, bool]] = []
            for index in range(particles_per_prototype):
                key = derive_controller_key(
                    initialization_key,
                    R015_FILTER_KEY_CONTRACT,
                    prototype_id,
                    index,
                )
                state = self._initial_kernel(prototype_id, key)
                compatible = np.array_equal(
                    np.asarray(state.snapshot.raw_obs["agent_1"]),
                    np.asarray(official_initial_observation),
                )
                candidates.append((state, compatible))
            compatible_count = sum(item[1] for item in candidates)
            if compatible_count == 0:
                raise ValueError(
                    "R015 filter initialization lost compatible support for a prototype."
                )
            for state, compatible in candidates:
                particles.append(
                    HiddenStateParticleV1(
                        prototype_id=prototype_id,
                        state_sha256=_particle_state_sha256(state),
                        state=state,
                        weight=(0.25 / compatible_count if compatible else 0.0),
                    )
                )
        return StratifiedParticleBeliefV1(
            prototype_ids=self.prototype_ids,
            particles_per_prototype=particles_per_prototype,
            registered_prototype_prior=prior,
            resampling_algorithm="systematic_per_prototype_v1",
            resampling_interval_environment_steps=1,
            particles=tuple(particles),
            resampling_timing=resampling_timing,
            resampling_ess_fraction_threshold=(
                0.5 if resampling_timing == "adaptive_ess_below_half_v1" else None
            ),
        )

    def _advance_library(
        self,
        state: R015ParticleKernelStateV1,
        *,
        observation: Any,
        episode_start: bool,
        key: str,
    ) -> ContinuationLibraryStatesV1:
        next_states: dict[str, Any] = {}
        for member_id in self.production_backend.continuation_controller.member_ids:
            step = self.production_backend.act_policy_member(
                member_id,
                observation,
                state.continuation_states.by_member_id[member_id],
                derive_controller_key(key, "continuation_member", member_id),
                episode_start=episode_start,
            )
            next_states[member_id] = step.next_recurrent_state
        return ContinuationLibraryStatesV1(next_states)

    def _particle_transition(
        self,
        particle: HiddenStateParticleV1,
        official_record: Mapping[str, Any],
        key: str,
    ) -> Sequence[HiddenStateParticleV1]:
        state = particle.state
        if not isinstance(state, R015ParticleKernelStateV1):
            raise TypeError("R015 particle contains another kernel state.")
        ego_actions = official_record["ego_action_history"]
        if not isinstance(ego_actions, Sequence) or len(ego_actions) != 1:
            raise ValueError("R015 passive filter requires one ego action.")
        ego_action = str(ego_actions[0])
        current_observation = state.snapshot.raw_obs["agent_1"]
        partner_step = self.production_backend.act_policy_member(
            state.partner_prototype_id,
            state.snapshot.raw_obs["agent_0"],
            state.partner_recurrent_state,
            derive_controller_key(key, "partner_action"),
            episode_start=state.environment_step == 0,
        )
        environment_key = derive_controller_key(key, "particle_environment")
        source_snapshot = _clone_snapshot_with_key(state.snapshot, environment_key)
        result = self.adapter.step_joint_from_state(
            source_snapshot,
            agent_0_action=OCV2_ACTION_INDEX[partner_step.action_id],
            agent_1_action=OCV2_ACTION_INDEX[ego_action],
        )
        next_state = R015ParticleKernelStateV1(
            snapshot=result.snapshot,
            partner_prototype_id=state.partner_prototype_id,
            partner_recurrent_state=partner_step.next_recurrent_state,
            continuation_states=self._advance_library(
                state,
                observation=current_observation,
                episode_start=state.environment_step == 0,
                key=key,
            ),
            predicted_response_token=_response_token(
                current_observation,
                result.step.obs["agent_1"],
            ),
            predicted_raw_team_reward=float(result.step.rewards["agent_0"]),
            predicted_done=bool(result.step.dones["__all__"]),
            environment_step=state.environment_step + 1,
        )
        return (
            HiddenStateParticleV1(
                prototype_id=particle.prototype_id,
                state_sha256=_particle_state_sha256(next_state),
                state=next_state,
                weight=1.0,
            ),
        )

    @staticmethod
    def _particle_compatibility(
        particle: HiddenStateParticleV1,
        official_record: Mapping[str, Any],
    ) -> float:
        state = particle.state
        if not isinstance(state, R015ParticleKernelStateV1):
            return 0.0
        if state.predicted_raw_team_reward is None or state.predicted_done is None:
            return 0.0
        expected_observation = official_record["official_local_observation"]
        rewards = official_record["raw_team_reward_history"]
        boundaries = official_record["episode_boundaries"]
        if len(rewards) != 1 or len(boundaries) != 1:
            return 0.0
        observation_match = np.array_equal(
            np.asarray(state.snapshot.raw_obs["agent_1"]),
            np.asarray(expected_observation),
        )
        reward_match = math.isclose(
            float(state.predicted_raw_team_reward),
            float(rewards[0]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        boundary_match = bool(state.predicted_done) is bool(boundaries[0])
        return 1.0 if observation_match and reward_match and boundary_match else 0.0

    def update_belief_v1_diagnostic(
        self,
        belief: StratifiedParticleBeliefV1,
        record: Mapping[str, Any],
        *,
        key: str,
        materialize_state_hashes: bool = True,
    ) -> StratifiedParticleBeliefV1:
        """只为第一版归档诊断推进完整粒子云。

        第二版设计、试点和正式执行均不调用本方法。归档诊断可只在登记咨询点计算完整
        摘要；咨询点之间的摘要只记录父状态与本步随机键的确定性继承关系，不进入过滤
        权重、相容性、重采样或控制器输入。
        """

        result = self.update_beliefs_v1_diagnostic(
            (belief,),
            (record,),
            keys=(key,),
            materialize_state_hashes=materialize_state_hashes,
            close_zero_support=False,
        )[0]
        if result is None:
            raise RuntimeError("R015 single-belief update closed without an exception.")
        return result

    def update_beliefs_v1_diagnostic(
        self,
        beliefs: Sequence[StratifiedParticleBeliefV1],
        records: Sequence[Mapping[str, Any]],
        *,
        keys: Sequence[str],
        materialize_state_hashes: bool = True,
        close_zero_support: bool = False,
    ) -> tuple[StratifiedParticleBeliefV1 | None, ...]:
        """Advance independent filters in one device batch, then update each separately."""

        if not beliefs or not (len(beliefs) == len(records) == len(keys)):
            raise ValueError("R015 batched filter inputs have different lengths.")
        kernels = []
        transition_keys = []
        forced_actions = []
        committed_members = []
        particles_by_belief: list[tuple[HiddenStateParticleV1, ...]] = []
        transition_maps: list[dict[str, HiddenStateParticleV1]] = []
        for belief, record, key in zip(beliefs, records, keys):
            OfficialHistoryV1((record,))
            if set(record) != R015_PASSIVE_FILTER_UPDATE_FIELDS:
                raise ValueError(
                    "R015 passive filtering accepts only observation, ego action, "
                    "raw team reward, and episode boundary fields."
                )
            ego_actions = record["ego_action_history"]
            if not isinstance(ego_actions, Sequence) or len(ego_actions) != 1:
                raise ValueError("R015 passive filter requires one ego action.")
            ego_action = str(ego_actions[0])
            particles = tuple(belief.particles)
            particles_by_belief.append(particles)
            transition_maps.append({})
            committed_member = self._committed_member_id(belief)
            for particle_index, particle in enumerate(particles):
                if not isinstance(particle.state, R015ParticleKernelStateV1):
                    raise TypeError("R015 particle contains another kernel state.")
                kernels.append(particle.state)
                transition_keys.append(
                    derive_controller_key(
                        key,
                        "filter_transition",
                        particle.prototype_id,
                        particle_index,
                    )
                )
                forced_actions.append(ego_action)
                committed_members.append(committed_member)
        outputs = self.production_backend.run_frozen_trajectory_batch(
            adapter=self.adapter,
            kernels=tuple(kernels),
            branch_keys=tuple(transition_keys),
            total_steps=1,
            key_offset=0,
            forced_first_actions=tuple(forced_actions),
            committed_member_ids=tuple(committed_members),
            key_contract="filter_transition_v1",
        )
        output_cursor = 0
        for belief_index, particles in enumerate(particles_by_belief):
            transitioned = transition_maps[belief_index]
            for particle in particles:
                transition_key = transition_keys[output_cursor]
                output = outputs[output_cursor]
                output_cursor += 1
                next_kernel = self._kernel_from_batch_output(
                    particle.state,
                    output,
                    environment_step=particle.state.environment_step + 1,
                    copy_state=False,
                )
                state_sha256 = (
                    _particle_state_sha256(next_kernel)
                    if materialize_state_hashes
                    else canonical_sha256(
                        {
                            "schema_version": R015_PARTICLE_LINEAGE_ID,
                            "parent_state_sha256": particle.state_sha256,
                            "transition_key": transition_key,
                        }
                    )
                )
                transitioned[transition_key] = HiddenStateParticleV1(
                    prototype_id=particle.prototype_id,
                    state_sha256=state_sha256,
                    state=next_kernel,
                    weight=1.0,
                )
        results: list[StratifiedParticleBeliefV1 | None] = []
        for belief, record, key, transitioned in zip(
            beliefs,
            records,
            keys,
            transition_maps,
        ):
            try:
                results.append(
                    belief.update(
                        official_record=record,
                        update_key=key,
                        transition=lambda particle, official_record, transition_key: (
                            transitioned[transition_key],
                        ),
                        compatibility=self._particle_compatibility,
                    )
                )
            except ValueError as error:
                if not close_zero_support or "compatible support" not in str(error):
                    raise
                results.append(None)
        return tuple(results)

    def replay_filter_candidate_device(
        self,
        *,
        history_stream_pairs: Sequence[tuple[Mapping[str, Any], str]],
        particles_per_prototype: int,
        resampling_timing: str,
        return_particle_weight_trace: bool = False,
        environment_step_limit: int = 400,
        return_state_diagnostics: bool = False,
        type_a_parent_slot_batch_width: int | None = None,
        conditioned_opening_prefix_cache_binding: Mapping[str, Any] | None = None,
        stop_at_irreversible_s1_failure: bool = False,
    ) -> Mapping[str, Any]:
        """按固定父粒子槽日程执行正式过滤；小批宽只供核查软件接线的 Type-A 检查。"""

        return self.replay_filter_candidate_device_v2(
            history_stream_pairs=history_stream_pairs,
            particles_per_prototype=particles_per_prototype,
            resampling_timing=resampling_timing,
            return_particle_weight_trace=return_particle_weight_trace,
            environment_step_limit=environment_step_limit,
            return_state_diagnostics=return_state_diagnostics,
            type_a_parent_slot_batch_width=type_a_parent_slot_batch_width,
            conditioned_opening_prefix_cache_binding=(
                conditioned_opening_prefix_cache_binding
            ),
            stop_at_irreversible_s1_failure=stop_at_irreversible_s1_failure,
        )

    def replay_selected_consultation_states_v2(
        self,
        *,
        history_records: Sequence[Mapping[str, Any]],
        particles_per_prototype: int,
        resampling_timing: str,
        stream_id: str,
        consultation_steps: Sequence[int] = R015_CONSULTATION_STEPS,
        conditioned_opening_prefix_cache_binding_sha256: str | None = None,
    ) -> Mapping[str, Any]:
        """机械选中后一次恢复完整 v2 咨询点，而非重跑候选网格。

        过滤候选只保存用于 S1--S3 的摘要。这个独立设备程序同时推进被选中的
        v2 过滤状态和五个延续策略循环状态，并只在登记咨询边界返回完整状态。
        """

        histories = tuple(history_records)
        steps = tuple(int(value) for value in consultation_steps)
        if not histories or steps != R015_CONSULTATION_STEPS:
            raise ValueError("R015 selected-filter replay changed its point schedule.")
        official_histories = []
        initialization_keys = []
        for history in histories:
            records = tuple(history.get("official_history", ()))
            if len(records) != 401:
                raise ValueError("R015 selected-filter replay requires 400-step histories.")
            official_histories.append(records)
            initialization_keys.append(
                canonical_sha256(
                    [
                        R015_FILTER_KEY_CONTRACT_V2,
                        str(stream_id),
                        int(history["design_sequence_index"]),
                    ]
                )
            )
        replay = getattr(
            self.production_backend,
            "replay_selected_consultation_states_v2",
            None,
        )
        if not callable(replay):
            raise RuntimeError(
                "R015 production backend lacks selected v2 consultation replay."
            )
        selected_opening_prefixes = None
        selected_opening_prefix_binding_sha256 = None
        selected_opening_cache_mode = "cold_rebuild_from_registered_roots"
        if conditioned_opening_prefix_cache_binding_sha256 is not None:
            if not _is_sha256(conditioned_opening_prefix_cache_binding_sha256):
                raise ValueError("R015 selected-filter opening-cache binding is invalid.")
            cache_record = self._conditioned_opening_prefix_caches.get(
                conditioned_opening_prefix_cache_binding_sha256
            )
            # 缓存只用于同一进程内提速。候选证据可能刚完成发布，进程便退出，
            # 因此恢复必须能用相同的冻结初始化键重建选中粒子序列。缓存存在时
            # 仍严格核验绑定；缓存不存在时向后端传入两个 None，使其确定性重建。
            if cache_record is not None:
                if not isinstance(cache_record, Mapping):
                    raise ValueError("R015 selected-filter opening cache is malformed.")
                cache_binding = cache_record.get("binding")
                if not isinstance(cache_binding, Mapping):
                    raise ValueError(
                        "R015 selected-filter opening cache no longer binds its histories."
                    )
                normalized_cache_binding = dict(cache_binding)
                recorded_cache_binding_sha256 = normalized_cache_binding.pop(
                    "cache_binding_sha256", None
                )
                lane_bindings = normalized_cache_binding.get("lane_bindings")
                expected_selected_lane_bindings = tuple(
                    {
                        "design_episode_id": str(history["design_episode_id"]),
                        "design_sequence_index": int(
                            history["design_sequence_index"]
                        ),
                        "filter_stream_id": str(stream_id),
                        "filter_initialization_key": initialization_key,
                        "official_history_sha256": canonical_sha256(
                            tuple(history.get("official_history", ()))
                        ),
                    }
                    for history, initialization_key in zip(
                        histories,
                        initialization_keys,
                    )
                )
                if (
                    recorded_cache_binding_sha256
                    != conditioned_opening_prefix_cache_binding_sha256
                    or recorded_cache_binding_sha256
                    != canonical_sha256(normalized_cache_binding)
                    or not isinstance(lane_bindings, Sequence)
                    or isinstance(lane_bindings, (str, bytes, bytearray))
                    or any(
                        expected not in tuple(lane_bindings)
                        for expected in expected_selected_lane_bindings
                    )
                ):
                    raise ValueError(
                        "R015 selected-filter opening cache no longer binds its histories."
                    )
                complete_prefixes = cache_record.get("conditioned_opening_prefixes")
                if not isinstance(complete_prefixes, Mapping):
                    raise RuntimeError("R015 selected-filter opening cache is malformed.")
                requested_count_key = str(int(particles_per_prototype))

                def owns_complete_selected_prefix(initialization_key: str) -> bool:
                    entry = complete_prefixes.get(initialization_key)
                    if not isinstance(entry, Mapping):
                        return False
                    materialized = entry.get(
                        "materialized_particles_per_prototype"
                    )
                    prefix_digests = entry.get(
                        "prefix_sha256_by_particle_count"
                    )
                    return (
                        entry.get("binding_sha256")
                        == conditioned_opening_prefix_cache_binding_sha256
                        and not isinstance(materialized, bool)
                        and isinstance(materialized, int)
                        and materialized >= int(particles_per_prototype)
                        and isinstance(entry.get("payload"), Mapping)
                        and isinstance(prefix_digests, Mapping)
                        and _is_sha256(prefix_digests.get(requested_count_key))
                    )

                # 选中候选只能复用每条车道都完整拥有的 P 粒子前缀。任一车道
                # 缺失或只生成了部分粒子时，整批从登记随机根确定性重建，不能
                # 把失败的开局尝试当成可复用证据，也不能混用新旧粒子序列。
                if all(
                    owns_complete_selected_prefix(initialization_key)
                    for initialization_key in initialization_keys
                ):
                    selected_opening_prefixes = {
                        key: complete_prefixes[key] for key in initialization_keys
                    }
                    selected_opening_prefix_binding_sha256 = (
                        conditioned_opening_prefix_cache_binding_sha256
                    )
                    selected_opening_cache_mode = "reused_complete_prefix"
        scanned = replay(
            adapter=self.adapter,
            official_histories=tuple(official_histories),
            initialization_keys=tuple(initialization_keys),
            particles_per_prototype=int(particles_per_prototype),
            resampling_timing=str(resampling_timing),
            consultation_steps=steps,
            conditioned_opening_prefix_cache=selected_opening_prefixes,
            conditioned_opening_prefix_cache_binding_sha256=(
                selected_opening_prefix_binding_sha256
            ),
        )
        raw_points = tuple(scanned.get("consultation_states", ()))
        expected_point_count = len(histories) * len(steps)
        if len(raw_points) != expected_point_count:
            raise RuntimeError("R015 selected-filter replay omitted consultation states.")
        points = []
        import jax

        # 后端通常已经在固定 microbatch 边界返回宿主数组；这一条批量边界也
        # 覆盖设备数组实现，禁止在 200 个咨询点上逐点同步两次。
        host_points = tuple(jax.device_get(raw_points))

        for raw in host_points:
            if not isinstance(raw, Mapping):
                raise TypeError("R015 selected-filter replay returned a malformed state.")
            lane_index = int(raw.get("lane_index", -1))
            environment_step = int(raw.get("environment_step", -1))
            if not 0 <= lane_index < len(histories) or environment_step not in steps:
                raise ValueError("R015 selected-filter replay changed a point coordinate.")
            if bool(raw.get("closed_for_zero_support", True)):
                raise ValueError("The selected R015 filter closed during point replay.")
            continuation = raw.get("continuation_states_by_member_id")
            if not isinstance(continuation, Mapping) or set(continuation) != set(
                self.production_backend.continuation_controller.member_ids
            ):
                raise ValueError("R015 selected-filter replay changed the five members.")
            points.append(
                {
                    "design_episode_id": str(
                        histories[lane_index]["design_episode_id"]
                    ),
                    "environment_step": environment_step,
                    "device_filter_state": raw["device_filter_state"],
                    "continuation_states_by_member_id": dict(continuation),
                }
            )
        expected_coordinates = tuple(
            (str(history["design_episode_id"]), environment_step)
            for history in histories
            for environment_step in steps
        )
        if tuple(
            (point["design_episode_id"], point["environment_step"])
            for point in points
        ) != expected_coordinates:
            raise RuntimeError("R015 selected-filter replay changed the point order.")
        device_execution = dict(scanned.get("device_execution", {}))
        device_execution.update(
            {
                "requested_conditioned_opening_prefix_cache_binding_sha256": (
                    conditioned_opening_prefix_cache_binding_sha256
                ),
                "selected_replay_opening_cache_mode": selected_opening_cache_mode,
            }
        )
        return {
            "schema_version": "path_c_r015_selected_filter_state_scan_v2",
            "consultation_states": tuple(points),
            "device_execution": device_execution,
        }

    def materialize_selected_consultation_state_v2(
        self,
        *,
        device_filter_state: Mapping[str, Any],
        continuation_states_by_member_id: Mapping[str, Any],
        resampling_timing: str,
        official_history_records: Sequence[Mapping[str, Any]],
    ) -> tuple[R015OnlineParticleBeliefV2, OfficialHistoryV1]:
        """从完整 checkpoint 恢复正式规划唯一允许的 v2 信念。"""

        history = OfficialHistoryV1(tuple(official_history_records))
        continuation = ContinuationLibraryStatesV1(
            dict(continuation_states_by_member_id)
        )
        belief = _materialize_online_particle_belief_v2(
            production_backend=self.production_backend,
            adapter=self.adapter,
            filter_state=device_filter_state,
            continuation_states=continuation,
            resampling_timing=str(resampling_timing),
        )
        if belief.update_count != len(history.records) - 1:
            raise ValueError("R015 selected-filter checkpoint changed its step count.")
        return belief, history

    def replay_filter_candidate_device_v1(
        self,
        *,
        history_stream_pairs: Sequence[tuple[Mapping[str, Any], str]],
        particles_per_prototype: int,
        resampling_timing: str,
        return_particle_weight_trace: bool = False,
        environment_step_limit: int = 400,
        return_state_diagnostics: bool = False,
    ) -> Mapping[str, Any]:
        """保留旧单批扫描，只允许复现历史诊断，不进入正式设计选择。"""

        result = dict(
            self._replay_filter_candidate_device_single_batch(
                history_stream_pairs=history_stream_pairs,
                particles_per_prototype=particles_per_prototype,
                resampling_timing=resampling_timing,
                return_particle_weight_trace=return_particle_weight_trace,
                environment_step_limit=environment_step_limit,
                return_state_diagnostics=return_state_diagnostics,
                filter_algorithm_id=R015_FILTER_ALGORITHM_ID_V1,
                filter_key_contract=R015_FILTER_KEY_CONTRACT,
                backend_method_name="run_offline_filter_scan_v1_diagnostic",
            )
        )
        result["filter_algorithm_id"] = R015_FILTER_ALGORITHM_ID_V1
        result["filter_key_contract"] = R015_FILTER_KEY_CONTRACT
        result["device_execution_id"] = R015_FILTER_DEVICE_EXECUTION_ID_V1
        result["diagnostic_only"] = True
        return result

    @staticmethod
    def _filter_parent_slot_schedule(
        *,
        particles_per_prototype: int,
        type_a_parent_slot_batch_width: int | None,
    ) -> tuple[int, int, bool]:
        """返回父粒子槽宽、每批车道数及是否为正式固定日程。"""

        if (
            isinstance(particles_per_prototype, bool)
            or not isinstance(particles_per_prototype, int)
            or particles_per_prototype <= 0
        ):
            raise ValueError("R015 filter particle count must be positive.")
        slots_per_lane = R015_FILTER_PROTOTYPE_COUNT * particles_per_prototype
        formal_schedule = type_a_parent_slot_batch_width is None
        if formal_schedule:
            if particles_per_prototype not in R015_FILTER_FORMAL_PARTICLE_COUNTS:
                raise ValueError(
                    "R015 formal filter uses only the registered 64/128/256 grid."
                )
            parent_slot_width = R015_FILTER_FORMAL_PARENT_SLOT_BATCH_WIDTH
        else:
            if (
                isinstance(type_a_parent_slot_batch_width, bool)
                or not isinstance(type_a_parent_slot_batch_width, int)
                or type_a_parent_slot_batch_width <= 0
                or type_a_parent_slot_batch_width
                > R015_FILTER_FORMAL_PARENT_SLOT_BATCH_WIDTH
            ):
                raise ValueError(
                    "R015 Type-A parent-slot batch width must lie in 1,...,4096."
                )
            parent_slot_width = type_a_parent_slot_batch_width
        if parent_slot_width % slots_per_lane != 0:
            raise ValueError(
                "R015 parent-slot batch width must contain complete four-prototype lanes."
            )
        lanes_per_batch = parent_slot_width // slots_per_lane
        if lanes_per_batch <= 0:
            raise ValueError("R015 parent-slot batch width cannot hold one history lane.")
        return parent_slot_width, lanes_per_batch, formal_schedule

    @staticmethod
    def _merge_filter_trace_results(
        batches: Sequence[Mapping[str, Any]],
        *,
        particles_per_prototype: int,
        parent_slot_batch_width: int,
        lanes_per_batch: int,
        formal_schedule: bool,
        expected_lane_count: int,
        return_particle_weight_trace: bool,
        return_state_diagnostics: bool,
        conditioned_opening_prefix_cache_binding_sha256: str,
    ) -> Mapping[str, Any]:
        """按输入顺序合并完整设备扫描，不在环境步骤之间读取设备结果。"""

        if not batches:
            raise ValueError("R015 filter microbatch merge requires at least one batch.")
        lane_results = tuple(
            lane
            for batch in batches
            for lane in tuple(batch["lane_results"])
        )
        if len(lane_results) != expected_lane_count:
            raise RuntimeError("R015 filter microbatch merge changed the lane count.")

        throughputs = tuple(dict(batch["throughput"]) for batch in batches)
        active_lane_counts = tuple(
            int(value["active_lane_count"]) for value in throughputs
        )
        realized_batch_lane_counts = tuple(
            len(tuple(batch["lane_results"])) for batch in batches
        )
        if active_lane_counts != realized_batch_lane_counts:
            raise RuntimeError(
                "R015 filter microbatch throughput disagrees with returned lanes."
            )
        active_parent_slot_widths = tuple(
            int(value["active_particle_batch_width"]) for value in throughputs
        )
        if sum(active_lane_counts) != expected_lane_count:
            raise RuntimeError("R015 filter microbatch throughput changed lane accounting.")
        expected_widths = tuple(
            count * R015_FILTER_PROTOTYPE_COUNT * int(particles_per_prototype)
            for count in active_lane_counts
        )
        if active_parent_slot_widths != expected_widths:
            raise RuntimeError("R015 filter microbatch parent-slot accounting is inconsistent.")
        if formal_schedule and any(
            count != lanes_per_batch for count in active_lane_counts
        ):
            raise RuntimeError("R015 formal filter emitted a partial parent-slot batch.")
        if formal_schedule and any(
            width != parent_slot_batch_width
            for width in active_parent_slot_widths
        ):
            raise RuntimeError("R015 formal filter changed the fixed 4096-slot width.")
        if any(
            "host_sync_inside_environment_loop" not in value
            or bool(value["host_sync_inside_environment_loop"])
            for value in throughputs
        ):
            raise RuntimeError("R015 filter synchronized with the host inside a device scan.")
        if any(int(value["compiled_batch_calls"]) != 1 for value in throughputs):
            raise RuntimeError("R015 filter microbatch did not use one device scan call.")
        if any(
            value.get("conditioned_opening_prefix_cache_contract_id")
            != R015_FILTER_OPENING_PREFIX_CACHE_CONTRACT_ID
            or value.get("conditioned_opening_prefix_cache_binding_sha256")
            != conditioned_opening_prefix_cache_binding_sha256
            or not 0
            <= int(
                value.get(
                    "conditioned_opening_prefix_particles_used_per_prototype",
                    -1,
                )
            )
            <= int(particles_per_prototype)
            or int(value.get("post_resampling_state_reuse_count", -1)) != 0
            for value in throughputs
        ):
            raise RuntimeError("R015 filter microbatch changed the opening-prefix cache.")

        cached_particles_by_microbatch = [
            int(
                value[
                    "conditioned_opening_prefix_particles_used_per_prototype"
                ]
            )
            for value in throughputs
        ]

        true_transitions = sum(
            int(value["true_particle_environment_transitions"])
            for value in throughputs
        )
        wall_seconds = sum(float(value["wall_seconds"]) for value in throughputs)
        throughput = {
            "filter_algorithm_id": R015_FILTER_ALGORITHM_ID_V2,
            "filter_key_contract": R015_FILTER_KEY_CONTRACT_V2,
            "device_execution_id": R015_FILTER_DEVICE_EXECUTION_ID_V2,
            "microbatch_schedule_id": R015_FILTER_MICROBATCH_SCHEDULE_ID,
            "formal_fixed_parent_slot_schedule": bool(formal_schedule),
            "configured_parent_slot_batch_width": int(parent_slot_batch_width),
            "lanes_per_microbatch": int(lanes_per_batch),
            "microbatch_count": len(batches),
            "microbatch_active_lane_counts": list(active_lane_counts),
            "microbatch_active_parent_slot_widths": list(
                active_parent_slot_widths
            ),
            "microbatch_wall_seconds": [
                float(value["wall_seconds"]) for value in throughputs
            ],
            "microbatch_true_transitions_per_second": [
                value.get("true_transitions_per_second") for value in throughputs
            ],
            "true_particle_environment_transitions": true_transitions,
            "compiled_batch_calls": sum(
                int(value["compiled_batch_calls"]) for value in throughputs
            ),
            "jit_compilations": sum(
                int(value["jit_compilations"]) for value in throughputs
            ),
            "active_lane_count": expected_lane_count,
            "active_particle_batch_width": max(active_parent_slot_widths),
            "policy_actor_lane_batch_width": max(
                int(value.get("policy_actor_lane_batch_width", 0))
                for value in throughputs
            ),
            "wall_seconds": wall_seconds,
            "true_transitions_per_second": (
                true_transitions / wall_seconds if wall_seconds > 0.0 else None
            ),
            "host_sync_inside_environment_loop": False,
            "conditioned_opening_prefix_cache_contract_id": (
                R015_FILTER_OPENING_PREFIX_CACHE_CONTRACT_ID
            ),
            "conditioned_opening_prefix_cache_binding_sha256": (
                conditioned_opening_prefix_cache_binding_sha256
            ),
            "conditioned_opening_prefix_particles_used_per_prototype": int(
                min(cached_particles_by_microbatch, default=0)
            ),
            "microbatch_conditioned_opening_prefix_particles_used_per_prototype": (
                cached_particles_by_microbatch
            ),
            "conditioned_opening_prefix_generated_particle_slots": sum(
                int(
                    value.get(
                        "conditioned_opening_prefix_generated_particle_slots",
                        0,
                    )
                )
                for value in throughputs
            ),
            "conditioned_opening_prefix_reused_particle_slots": sum(
                int(
                    value.get(
                        "conditioned_opening_prefix_reused_particle_slots",
                        0,
                    )
                )
                for value in throughputs
            ),
            "post_resampling_state_reuse_count": 0,
        }
        result: dict[str, Any] = {
            "filter_algorithm_id": R015_FILTER_ALGORITHM_ID_V2,
            "filter_key_contract": R015_FILTER_KEY_CONTRACT_V2,
            "device_execution_id": R015_FILTER_DEVICE_EXECUTION_ID_V2,
            "microbatch_schedule_id": R015_FILTER_MICROBATCH_SCHEDULE_ID,
            "formal_fixed_parent_slot_schedule": bool(formal_schedule),
            "lane_results": lane_results,
            "throughput": throughput,
        }
        if return_particle_weight_trace:
            lane_trace_names = (
                "particle_weight_trace",
                "posterior_trace",
                "ess_trace",
                "closed_trace",
                "partner_action_trace",
                "resampling_due_trace",
                "selected_source_index_trace",
            )
            for name in lane_trace_names:
                result[name] = np.concatenate(
                    tuple(np.asarray(batch[name]) for batch in batches),
                    axis=1,
                )
            if return_state_diagnostics:
                state_trace_names = (
                    "partner_recurrent_state_trace",
                    "partner_recurrent_state_before_trace",
                    "agent_0_observation_trace",
                    "agent_0_observation_before_trace",
                )
                for name in state_trace_names:
                    result[name] = np.concatenate(
                        tuple(np.asarray(batch[name]) for batch in batches),
                        axis=1,
                    )
        return result

    def replay_filter_candidate_device_v2(
        self,
        *,
        history_stream_pairs: Sequence[tuple[Mapping[str, Any], str]],
        particles_per_prototype: int,
        resampling_timing: str,
        return_particle_weight_trace: bool = False,
        environment_step_limit: int = 400,
        return_state_diagnostics: bool = False,
        type_a_parent_slot_batch_width: int | None = None,
        conditioned_opening_prefix_cache_binding: Mapping[str, Any] | None = None,
        stop_at_irreversible_s1_failure: bool = False,
    ) -> Mapping[str, Any]:
        """用固定 4096 个父粒子槽的微批依次扫描过滤车道。

        正式网格的每个车道含四个原型，因此 P=64、128、256 时每批分别放入
        16、8、4 条车道。每个微批在设备内完成整段 ``lax.scan`` 后才回到宿主；
        输入车道只按相邻切片分批，合并时恢复原顺序。显式小批宽只供 Type-A
        接线核查，不能进入正式设计证据。
        """

        if not history_stream_pairs:
            raise ValueError("R015 device filter requires at least one history lane.")
        if len(self.prototype_ids) != R015_FILTER_PROTOTYPE_COUNT:
            raise ValueError("R015 formal filter requires exactly four prototypes.")
        if return_state_diagnostics and not return_particle_weight_trace:
            raise ValueError("R015 state diagnostics require the particle trace.")
        if type(stop_at_irreversible_s1_failure) is not bool:
            raise TypeError("R015 S1 early-stop marker must be Boolean.")
        if stop_at_irreversible_s1_failure and (
            return_particle_weight_trace or return_state_diagnostics
        ):
            raise ValueError(
                "R015 S1 early stop cannot truncate an equivalence diagnostic trace."
            )
        if not isinstance(conditioned_opening_prefix_cache_binding, Mapping):
            raise ValueError("R015 v2 filter requires a bound opening-prefix cache.")
        (
            opening_cache_binding_sha256,
            conditioned_opening_prefixes,
        ) = self._conditioned_opening_prefix_cache(
            conditioned_opening_prefix_cache_binding,
            history_stream_pairs=history_stream_pairs,
        )
        (
            parent_slot_batch_width,
            lanes_per_batch,
            formal_schedule,
        ) = self._filter_parent_slot_schedule(
            particles_per_prototype=particles_per_prototype,
            type_a_parent_slot_batch_width=type_a_parent_slot_batch_width,
        )
        if stop_at_irreversible_s1_failure and not formal_schedule:
            raise ValueError("R015 S1 early stop requires the fixed 4096-slot schedule.")
        if formal_schedule and int(environment_step_limit) != 400:
            raise ValueError("R015 formal filter microbatches must scan all 400 steps.")
        if formal_schedule and len(history_stream_pairs) % lanes_per_batch != 0:
            raise ValueError(
                "R015 formal filter lane count must fill every 4096-slot microbatch."
            )

        canonical_episode_ids: tuple[str, ...] = ()
        if stop_at_irreversible_s1_failure:
            episode_lane_counts: dict[str, int] = {}
            ordered_episode_ids: list[str] = []
            previous_episode_id: str | None = None
            for history, _ in history_stream_pairs:
                episode_id = str(history.get("design_episode_id", ""))
                if not episode_id:
                    raise ValueError("R015 S1 early stop requires design episode ids.")
                episode_lane_counts[episode_id] = episode_lane_counts.get(episode_id, 0) + 1
                if episode_id != previous_episode_id:
                    if episode_id in ordered_episode_ids:
                        raise ValueError(
                            "R015 S1 early stop requires adjacent repeat lanes per episode."
                        )
                    ordered_episode_ids.append(episode_id)
                    previous_episode_id = episode_id
            if len(ordered_episode_ids) != R015_FILTER_DESIGN_EPISODE_COUNT or any(
                count != R015_FILTER_REPEAT_COUNT_PER_EPISODE
                for count in episode_lane_counts.values()
            ):
                raise ValueError(
                    "R015 S1 early stop requires 80 episodes with two repeat lanes each."
                )
            if lanes_per_batch % R015_FILTER_REPEAT_COUNT_PER_EPISODE != 0:
                raise RuntimeError(
                    "R015 filter microbatches must not split an episode's repeat lanes."
                )
            canonical_episode_ids = tuple(ordered_episode_ids)

        batches = []
        processed_episode_ids: list[str] = []
        closed_episode_ids: set[str] = set()
        for start in range(0, len(history_stream_pairs), lanes_per_batch):
            batch_pairs = tuple(
                history_stream_pairs[start : start + lanes_per_batch]
            )
            batch = self._replay_filter_candidate_device_single_batch(
                history_stream_pairs=batch_pairs,
                particles_per_prototype=particles_per_prototype,
                resampling_timing=resampling_timing,
                return_particle_weight_trace=return_particle_weight_trace,
                environment_step_limit=environment_step_limit,
                return_state_diagnostics=return_state_diagnostics,
                filter_algorithm_id=R015_FILTER_ALGORITHM_ID_V2,
                filter_key_contract=R015_FILTER_KEY_CONTRACT_V2,
                backend_method_name="run_offline_filter_scan_v2",
                conditioned_opening_prefix_cache=conditioned_opening_prefixes,
                conditioned_opening_prefix_cache_binding_sha256=(
                    opening_cache_binding_sha256
                ),
            )
            batches.append(batch)
            if stop_at_irreversible_s1_failure:
                lanes_by_episode: dict[str, list[Mapping[str, Any]]] = {}
                for lane in tuple(batch["lane_results"]):
                    episode_id = str(lane.get("design_episode_id", ""))
                    lanes_by_episode.setdefault(episode_id, []).append(lane)
                expected_batch_episode_ids = tuple(
                    dict.fromkeys(
                        str(history["design_episode_id"])
                        for history, _ in batch_pairs
                    )
                )
                if tuple(lanes_by_episode) != expected_batch_episode_ids or any(
                    len(lanes) != R015_FILTER_REPEAT_COUNT_PER_EPISODE
                    for lanes in lanes_by_episode.values()
                ):
                    raise RuntimeError(
                        "R015 filter microbatch changed the two-repeat episode grouping."
                    )
                for episode_id, lanes in lanes_by_episode.items():
                    processed_episode_ids.append(episode_id)
                    if any(bool(lane["closed_for_zero_support"]) for lane in lanes):
                        closed_episode_ids.add(episode_id)
                if len(closed_episode_ids) >= (
                    R015_FILTER_S1_EARLY_STOP_CLOSED_EPISODE_COUNT
                ):
                    break
        processed_lane_count = sum(len(tuple(batch["lane_results"])) for batch in batches)
        merged = self._merge_filter_trace_results(
            batches,
            particles_per_prototype=particles_per_prototype,
            parent_slot_batch_width=parent_slot_batch_width,
            lanes_per_batch=lanes_per_batch,
            formal_schedule=formal_schedule,
            expected_lane_count=processed_lane_count,
            return_particle_weight_trace=return_particle_weight_trace,
            return_state_diagnostics=return_state_diagnostics,
            conditioned_opening_prefix_cache_binding_sha256=(
                opening_cache_binding_sha256
            ),
        )
        expected_lane_order = tuple(
            (
                str(history["design_episode_id"]),
                str(stream_id),
                canonical_sha256(
                    [
                        R015_FILTER_KEY_CONTRACT_V2,
                        str(stream_id),
                        int(history["design_sequence_index"]),
                    ]
                ),
            )
            for history, stream_id in history_stream_pairs[:processed_lane_count]
        )
        realized_lane_order = tuple(
            (
                str(lane["design_episode_id"]),
                str(lane["filter_stream_id"]),
                str(lane["filter_initialization_key"]),
            )
            for lane in merged["lane_results"]
        )
        if realized_lane_order != expected_lane_order:
            raise RuntimeError("R015 filter microbatch merge changed the input lane order.")
        if stop_at_irreversible_s1_failure:
            if tuple(processed_episode_ids) != canonical_episode_ids[
                : len(processed_episode_ids)
            ]:
                raise RuntimeError("R015 S1 early stop changed the episode prefix order.")
            complete = len(processed_episode_ids) == R015_FILTER_DESIGN_EPISODE_COUNT
            irreversible_failure = len(closed_episode_ids) >= (
                R015_FILTER_S1_EARLY_STOP_CLOSED_EPISODE_COUNT
            )
            status = (
                "complete"
                if complete
                else "s1_irreversible_failure"
            )
            if not complete and not irreversible_failure:
                raise RuntimeError("R015 filter stopped before S1 became irreversible.")
            merged = {
                **dict(merged),
                "candidate_progress": {
                    "status": status,
                    "total_episode_count": R015_FILTER_DESIGN_EPISODE_COUNT,
                    "processed_episode_count": len(processed_episode_ids),
                    "processed_lane_count": processed_lane_count,
                    "zero_support_close_count": len(closed_episode_ids),
                    "s1_close_rate_lower_bound": (
                        len(closed_episode_ids) / R015_FILTER_DESIGN_EPISODE_COUNT
                    ),
                    "stop_candidate_at_closed_episode_count": (
                        R015_FILTER_S1_EARLY_STOP_CLOSED_EPISODE_COUNT
                    ),
                },
            }
        prefix_digest_by_particle_count: dict[str, str] = {}
        attempt_digest_by_particle_count: dict[str, str] = {}
        ordered_initialization_keys = tuple(item[2] for item in expected_lane_order)

        def cached_prefix_digest(initialization_key: str, count_key: str) -> Any:
            entry = conditioned_opening_prefixes.get(initialization_key)
            if not isinstance(entry, Mapping):
                return None
            digests = entry.get("prefix_sha256_by_particle_count")
            if not isinstance(digests, Mapping):
                return None
            return digests.get(count_key)

        def cached_attempt_digest(initialization_key: str, count_key: str) -> Any:
            entry = conditioned_opening_prefixes.get(initialization_key)
            if not isinstance(entry, Mapping):
                return None
            digests = entry.get("opening_attempt_sha256_by_particle_count")
            if not isinstance(digests, Mapping):
                return None
            return digests.get(count_key)

        for particle_count in R015_FILTER_FORMAL_PARTICLE_COUNTS:
            count_key = str(particle_count)
            if all(
                _is_sha256(cached_prefix_digest(initialization_key, count_key))
                for initialization_key in ordered_initialization_keys
            ):
                prefix_digest_by_particle_count[count_key] = canonical_sha256(
                    [
                        {
                            "filter_initialization_key": initialization_key,
                            "prefix_sha256": cached_prefix_digest(
                                initialization_key, count_key
                            ),
                        }
                        for initialization_key in ordered_initialization_keys
                    ]
                )
            if all(
                _is_sha256(cached_attempt_digest(initialization_key, count_key))
                for initialization_key in ordered_initialization_keys
            ):
                attempt_digest_by_particle_count[count_key] = canonical_sha256(
                    [
                        {
                            "filter_initialization_key": initialization_key,
                            "opening_attempt_sha256": cached_attempt_digest(
                                initialization_key, count_key
                            ),
                        }
                        for initialization_key in ordered_initialization_keys
                    ]
                )
        requested_count_key = str(int(particles_per_prototype))
        if requested_count_key not in attempt_digest_by_particle_count:
            raise RuntimeError("R015 filter omitted its conditioned opening-attempt hash.")
        merged = {
            **dict(merged),
            "conditioned_opening_prefix_sha256_by_particle_count": (
                prefix_digest_by_particle_count
            ),
            "conditioned_opening_attempt_sha256_by_particle_count": (
                attempt_digest_by_particle_count
            ),
        }
        return merged

    def _replay_filter_candidate_device_single_batch(
        self,
        *,
        history_stream_pairs: Sequence[tuple[Mapping[str, Any], str]],
        particles_per_prototype: int,
        resampling_timing: str,
        return_particle_weight_trace: bool = False,
        environment_step_limit: int = 400,
        return_state_diagnostics: bool = False,
        filter_algorithm_id: str,
        filter_key_contract: str,
        backend_method_name: str,
        conditioned_opening_prefix_cache: dict[str, Any] | None = None,
        conditioned_opening_prefix_cache_binding_sha256: str | None = None,
    ) -> Mapping[str, Any]:
        """一次编译后在设备内扫描完整设计候选。

        每个车道是一条设计历史和一条过滤随机流。设备程序返回的四原型后验、
        重采样前有效样本量和关闭标志直接用于原有证据结构。旧诊断保留旧键；
        新路径的初始化键只绑定过滤流和设计序号，使全部候选共享开局提议序列。
        """

        if not history_stream_pairs:
            raise ValueError("R015 device filter requires at least one history lane.")
        histories = []
        initialization_keys = []
        lane_metadata = []
        for history_record, stream_id in history_stream_pairs:
            records = tuple(history_record.get("official_history", ()))
            if len(records) != 401:
                raise ValueError("R015 design history must contain 400 completed steps.")
            for opening_index, record in enumerate(records):
                expected_fields = (
                    {"official_local_observation", "episode_boundaries"}
                    if opening_index == 0
                    else R015_PASSIVE_FILTER_UPDATE_FIELDS
                )
                if set(record) != expected_fields:
                    raise ValueError("R015 device filter history exposes an unknown field.")
                OfficialHistoryV1((record,))
            if (
                filter_algorithm_id == R015_FILTER_ALGORITHM_ID_V1
                and filter_key_contract == R015_FILTER_KEY_CONTRACT
            ):
                initial_key = canonical_sha256(
                    [
                        str(stream_id),
                        int(particles_per_prototype),
                        str(resampling_timing),
                        int(history_record["design_sequence_index"]),
                    ]
                )
            elif (
                filter_algorithm_id == R015_FILTER_ALGORITHM_ID_V2
                and filter_key_contract == R015_FILTER_KEY_CONTRACT_V2
            ):
                initial_key = canonical_sha256(
                    [
                        R015_FILTER_KEY_CONTRACT_V2,
                        str(stream_id),
                        int(history_record["design_sequence_index"]),
                    ]
                )
            else:
                raise ValueError("R015 filter algorithm and key contract do not match.")
            histories.append(records)
            initialization_keys.append(initial_key)
            lane_metadata.append(
                {
                    "design_episode_id": str(history_record["design_episode_id"]),
                    "filter_stream_id": str(stream_id),
                    "filter_initialization_key": initial_key,
                }
            )
        backend_method = getattr(self.production_backend, backend_method_name, None)
        if not callable(backend_method):
            raise TypeError(
                f"R015 production backend lacks {backend_method_name}()."
            )
        backend_arguments: dict[str, Any] = {
            "adapter": self.adapter,
            "official_histories": tuple(histories),
            "initialization_keys": tuple(initialization_keys),
            "particles_per_prototype": int(particles_per_prototype),
            "resampling_timing": str(resampling_timing),
            "return_particle_weight_trace": bool(return_particle_weight_trace),
            "environment_step_limit": int(environment_step_limit),
            "return_state_diagnostics": bool(return_state_diagnostics),
        }
        cached_particle_counts_before: dict[str, int] = {}
        cached_prefix_digests_before: dict[str, dict[str, str]] = {}
        cached_attempt_digests_before: dict[str, dict[str, str]] = {}
        if filter_algorithm_id == R015_FILTER_ALGORITHM_ID_V2:
            if not isinstance(conditioned_opening_prefix_cache, dict) or not (
                _is_sha256(conditioned_opening_prefix_cache_binding_sha256)
            ):
                raise ValueError("R015 v2 filter lacks its opening-prefix cache context.")
            for cache_key, cache_entry in conditioned_opening_prefix_cache.items():
                if not isinstance(cache_entry, Mapping):
                    raise RuntimeError("R015 opening-prefix cache entry is malformed.")
                materialized = cache_entry.get(
                    "materialized_particles_per_prototype"
                )
                if (
                    isinstance(materialized, bool)
                    or not isinstance(materialized, int)
                    or not 0 <= materialized
                    <= R015_FILTER_OPENING_PREFIX_MAXIMUM_PARTICLES_PER_PROTOTYPE
                    or (
                        materialized > 0
                        and not isinstance(cache_entry.get("payload"), Mapping)
                    )
                ):
                    raise RuntimeError("R015 opening-prefix cache count is invalid.")
                cached_particle_counts_before[str(cache_key)] = int(materialized)
                raw_digests = cache_entry.get("prefix_sha256_by_particle_count")
                if not isinstance(raw_digests, Mapping) or any(
                    str(count) not in {"64", "128", "256"}
                    or not _is_sha256(digest)
                    for count, digest in raw_digests.items()
                ):
                    raise RuntimeError("R015 opening-prefix cache digest map is invalid.")
                cached_prefix_digests_before[str(cache_key)] = {
                    str(count): str(digest)
                    for count, digest in raw_digests.items()
                }
                raw_attempt_digests = cache_entry.get(
                    "opening_attempt_sha256_by_particle_count"
                )
                if not isinstance(raw_attempt_digests, Mapping) or any(
                    str(count) not in {"64", "128", "256"}
                    or not _is_sha256(digest)
                    for count, digest in raw_attempt_digests.items()
                ):
                    raise RuntimeError("R015 opening-attempt digest map is invalid.")
                cached_attempt_digests_before[str(cache_key)] = {
                    str(count): str(digest)
                    for count, digest in raw_attempt_digests.items()
                }
            backend_arguments.update(
                {
                    "conditioned_opening_prefix_cache": (
                        conditioned_opening_prefix_cache
                    ),
                    "conditioned_opening_prefix_cache_binding_sha256": (
                        conditioned_opening_prefix_cache_binding_sha256
                    ),
                }
            )
        scanned = backend_method(
            **backend_arguments,
        )
        if filter_algorithm_id == R015_FILTER_ALGORITHM_ID_V2:
            expected_cache_keys = tuple(initialization_keys)
            if len(set(expected_cache_keys)) != len(expected_cache_keys):
                raise RuntimeError("R015 v2 filter initialization keys are not unique.")
            scanned_throughput = dict(scanned.get("throughput", {}))
            common_cached_count = scanned_throughput.get(
                "conditioned_opening_prefix_particles_used_per_prototype"
            )
            expected_common_cached_count = (
                min(
                    int(particles_per_prototype),
                    min(
                        cached_particle_counts_before.get(cache_key, 0)
                        for cache_key in expected_cache_keys
                    ),
                )
                if expected_cache_keys
                else 0
            )
            if (
                isinstance(common_cached_count, bool)
                or not isinstance(common_cached_count, int)
                or not 0 <= common_cached_count <= int(particles_per_prototype)
                or common_cached_count != expected_common_cached_count
            ):
                raise RuntimeError(
                    "R015 v2 backend returned an invalid common cached-prefix count."
                )
            opening_diagnostics = scanned.get("opening_diagnostics")
            if not isinstance(opening_diagnostics, Mapping):
                raise RuntimeError("R015 v2 backend omitted opening diagnostics.")
            opening_accepted_counts = np.asarray(
                opening_diagnostics.get("opening_accepted_counts"),
                dtype=np.int64,
            )
            if opening_accepted_counts.shape != (
                len(expected_cache_keys),
                R015_FILTER_PROTOTYPE_COUNT,
            ):
                raise RuntimeError(
                    "R015 v2 backend changed opening-acceptance lane order or shape."
                )
            generated_particle_slots = 0
            reused_particle_slots = (
                common_cached_count
                * len(expected_cache_keys)
                * R015_FILTER_PROTOTYPE_COUNT
            )
            for lane_index, cache_key in enumerate(expected_cache_keys):
                cache_entry = conditioned_opening_prefix_cache.get(cache_key)
                if not isinstance(cache_entry, Mapping):
                    raise RuntimeError(
                        "R015 v2 backend did not return every conditioned opening prefix."
                    )
                if cache_entry.get("binding_sha256") != (
                    conditioned_opening_prefix_cache_binding_sha256
                ):
                    raise RuntimeError(
                        "R015 v2 backend changed an opening-cache binding."
                    )
                materialized = cache_entry.get(
                    "materialized_particles_per_prototype"
                )
                if (
                    isinstance(materialized, bool)
                    or not isinstance(materialized, int)
                    or not 0 <= materialized
                    <= R015_FILTER_OPENING_PREFIX_MAXIMUM_PARTICLES_PER_PROTOTYPE
                    or (
                        materialized > 0
                        and not isinstance(cache_entry.get("payload"), Mapping)
                    )
                ):
                    raise RuntimeError(
                        "R015 v2 backend returned an invalid reusable opening prefix."
                    )
                before = cached_particle_counts_before.get(cache_key, 0)
                if materialized < before:
                    raise RuntimeError("R015 v2 backend shortened an opening prefix.")
                raw_digests = cache_entry.get("prefix_sha256_by_particle_count")
                if not isinstance(raw_digests, Mapping):
                    raise RuntimeError(
                        "R015 v2 backend returned a malformed opening-prefix digest map."
                    )
                requested_count_key = str(int(particles_per_prototype))
                if materialized >= int(particles_per_prototype) and not _is_sha256(
                    raw_digests.get(requested_count_key)
                ):
                    raise RuntimeError(
                        "A complete R015 opening prefix lacks its requested digest."
                    )
                before_digests = cached_prefix_digests_before.get(cache_key, {})
                if any(raw_digests.get(count) != digest for count, digest in before_digests.items()):
                    raise RuntimeError("R015 v2 backend changed an existing opening prefix.")
                raw_attempt_digests = cache_entry.get(
                    "opening_attempt_sha256_by_particle_count"
                )
                if not isinstance(raw_attempt_digests, Mapping) or not _is_sha256(
                    raw_attempt_digests.get(requested_count_key)
                ):
                    raise RuntimeError(
                        "R015 v2 backend omitted the requested opening-attempt digest."
                    )
                before_attempt_digests = cached_attempt_digests_before.get(
                    cache_key, {}
                )
                if any(
                    raw_attempt_digests.get(count) != digest
                    for count, digest in before_attempt_digests.items()
                ):
                    raise RuntimeError("R015 v2 backend changed an opening attempt.")
                lane_accepted_counts = opening_accepted_counts[lane_index]
                if np.any(lane_accepted_counts < common_cached_count) or np.any(
                    lane_accepted_counts > int(particles_per_prototype)
                ):
                    raise RuntimeError(
                        "R015 v2 backend opening counts disagree with its common prefix."
                    )
                completed_requested_prefix = bool(
                    np.all(lane_accepted_counts >= int(particles_per_prototype))
                )
                if completed_requested_prefix != (
                    materialized >= int(particles_per_prototype)
                    and _is_sha256(raw_digests.get(requested_count_key))
                ):
                    raise RuntimeError(
                        "R015 v2 cache completion disagrees with opening acceptance."
                    )
                generated_particle_slots += int(
                    np.sum(lane_accepted_counts - common_cached_count)
                )
            expected_cache_key_set = set(expected_cache_keys)
            if set(conditioned_opening_prefix_cache) != (
                set(cached_particle_counts_before) | expected_cache_key_set
            ):
                raise RuntimeError(
                    "R015 v2 backend added or removed an unrelated cache lane."
                )
            for cache_key, before_count in cached_particle_counts_before.items():
                if cache_key in expected_cache_key_set:
                    continue
                cache_entry = conditioned_opening_prefix_cache.get(cache_key)
                if not isinstance(cache_entry, Mapping) or cache_entry.get(
                    "materialized_particles_per_prototype"
                ) != before_count or cache_entry.get("binding_sha256") != (
                    conditioned_opening_prefix_cache_binding_sha256
                ) or dict(
                    cache_entry.get("prefix_sha256_by_particle_count", {})
                ) != cached_prefix_digests_before[cache_key] or dict(
                    cache_entry.get(
                        "opening_attempt_sha256_by_particle_count", {}
                    )
                ) != cached_attempt_digests_before[cache_key]:
                    raise RuntimeError(
                        "R015 v2 backend changed a cache lane outside its microbatch."
                    )
            scanned_throughput.update(
                {
                    "conditioned_opening_prefix_cache_contract_id": (
                        R015_FILTER_OPENING_PREFIX_CACHE_CONTRACT_ID
                    ),
                    "conditioned_opening_prefix_cache_binding_sha256": (
                        conditioned_opening_prefix_cache_binding_sha256
                    ),
                    "conditioned_opening_prefix_particles_used_per_prototype": min(
                        common_cached_count,
                        int(particles_per_prototype),
                    ),
                    "conditioned_opening_prefix_generated_particle_slots": (
                        generated_particle_slots
                    ),
                    "conditioned_opening_prefix_reused_particle_slots": (
                        reused_particle_slots
                    ),
                    "post_resampling_state_reuse_count": 0,
                }
            )
            scanned = {**dict(scanned), "throughput": scanned_throughput}
        trace = scanned["trace"]
        posterior = np.asarray(trace["posterior"])
        ess = np.asarray(trace["ess"])
        closed = np.asarray(trace["closed"], dtype=np.bool_)
        expected_shape = (int(environment_step_limit), len(histories), 4)
        if posterior.shape != expected_shape or ess.shape != expected_shape or (
            closed.shape != expected_shape[:2]
        ):
            raise RuntimeError("R015 device filter returned an unexpected trace shape.")
        lane_results = []
        for lane, metadata in enumerate(lane_metadata):
            consultations = []
            for environment_step in R015_CONSULTATION_STEPS:
                if environment_step > int(environment_step_limit):
                    continue
                trace_index = environment_step - 1
                if bool(closed[trace_index, lane]):
                    continue
                consultations.append(
                    {
                        "environment_step": environment_step,
                        "prototype_posterior": {
                            prototype_id: float(posterior[trace_index, lane, index])
                            for index, prototype_id in enumerate(self.prototype_ids)
                        },
                        "pre_resample_ess_fraction_by_prototype": {
                            prototype_id: _canonical_ess_fraction_for_evidence(
                                ess[trace_index, lane, index]
                            )
                            for index, prototype_id in enumerate(self.prototype_ids)
                        },
                    }
                )
            lane_results.append(
                {
                    **metadata,
                    "closed_for_zero_support": bool(closed[-1, lane]),
                    "consultations": consultations,
                }
            )
        result: dict[str, Any] = {
            "lane_results": tuple(lane_results),
            "throughput": dict(scanned["throughput"]),
        }
        if return_particle_weight_trace:
            weights = np.asarray(trace.get("particle_weights"))
            expected_weights = (
                int(environment_step_limit),
                len(histories),
                4,
                int(particles_per_prototype),
            )
            if weights.shape != expected_weights:
                raise RuntimeError(
                    "R015 device filter returned an unexpected particle-weight trace."
                )
            partner_actions = np.asarray(trace.get("partner_actions"))
            resampling_due = np.asarray(
                trace.get("resampling_due"),
                dtype=np.bool_,
            )
            selected_source_indices = np.asarray(
                trace.get("selected_source_indices"),
                dtype=np.int32,
            )
            if (
                partner_actions.shape != expected_weights
                or resampling_due.shape != expected_shape
                or selected_source_indices.shape != expected_weights
            ):
                raise RuntimeError(
                    "R015 device filter returned an unexpected diagnostic trace."
                )
            result["particle_weight_trace"] = weights
            result["posterior_trace"] = posterior
            result["ess_trace"] = ess
            result["closed_trace"] = closed
            result["partner_action_trace"] = partner_actions
            result["resampling_due_trace"] = resampling_due
            result["selected_source_index_trace"] = selected_source_indices
            if return_state_diagnostics:
                total_slots = (
                    len(histories) * 4 * int(particles_per_prototype)
                )
                partner_state = np.asarray(
                    trace.get("partner_recurrent_state")
                )
                partner_state_before = np.asarray(
                    trace.get("partner_recurrent_state_before")
                )
                agent_0_observation = np.asarray(
                    trace.get("agent_0_observation")
                )
                agent_0_observation_before = np.asarray(
                    trace.get("agent_0_observation_before")
                )
                if (
                    partner_state.shape[:2]
                    != (int(environment_step_limit), total_slots)
                    or partner_state_before.shape[:2]
                    != (int(environment_step_limit), total_slots)
                    or agent_0_observation.shape[:2]
                    != (int(environment_step_limit), total_slots)
                    or agent_0_observation_before.shape[:2]
                    != (int(environment_step_limit), total_slots)
                ):
                    raise RuntimeError(
                        "R015 device filter returned malformed state diagnostics."
                    )
                result["partner_recurrent_state_trace"] = partner_state.reshape(
                    (
                        int(environment_step_limit),
                        len(histories),
                        4,
                        int(particles_per_prototype),
                    )
                    + partner_state.shape[2:]
                )
                result["partner_recurrent_state_before_trace"] = (
                    partner_state_before.reshape(
                        (
                            int(environment_step_limit),
                            len(histories),
                            4,
                            int(particles_per_prototype),
                        )
                        + partner_state_before.shape[2:]
                    )
                )
                result["agent_0_observation_trace"] = (
                    agent_0_observation.reshape(
                        (
                            int(environment_step_limit),
                            len(histories),
                            4,
                            int(particles_per_prototype),
                        )
                        + agent_0_observation.shape[2:]
                    )
                )
                result["agent_0_observation_before_trace"] = (
                    agent_0_observation_before.reshape(
                        (
                            int(environment_step_limit),
                            len(histories),
                            4,
                            int(particles_per_prototype),
                        )
                        + agent_0_observation_before.shape[2:]
                    )
                )
        return result

    @staticmethod
    def materialize_belief_state_hashes(
        belief: StratifiedParticleBeliefV1,
    ) -> StratifiedParticleBeliefV1:
        """Replace design-replay inheritance hashes with complete state hashes."""

        return replace(
            belief,
            particles=tuple(
                replace(
                    particle,
                    state_sha256=_particle_state_sha256(particle.state),
                )
                for particle in belief.particles
            ),
        )

    @staticmethod
    def _response_likelihood(
        particle: HiddenStateParticleV1,
        response: Mapping[str, Any],
    ) -> float:
        state = particle.state
        return float(
            isinstance(state, R015ParticleKernelStateV1)
            and state.predicted_response_token == int(response.get("token_id", -1))
        )

    def _step_true_state(
        self,
        branch: R015BranchStateV1,
        *,
        ego_action: str,
        step_key: str,
        next_library: ContinuationLibraryStatesV1,
        update_belief_online: bool = True,
    ) -> tuple[R015BranchStateV1, Mapping[str, Any], bool]:
        state = branch.kernel
        current_observation = state.snapshot.raw_obs["agent_1"]
        partner_step = self.production_backend.act_policy_member(
            state.partner_prototype_id,
            state.snapshot.raw_obs["agent_0"],
            state.partner_recurrent_state,
            derive_controller_key(step_key, "partner_action"),
            episode_start=branch.environment_step == 0,
        )
        source_snapshot = _clone_snapshot_with_key(state.snapshot, step_key)
        before_controller_sha = _library_states_sha256(state.continuation_states)
        before_partner_sha = _pytree_sha256(
            state.partner_recurrent_state,
            domain="path_c_r015_partner_recurrent_state_v1",
        )
        result = self.adapter.step_joint_from_state(
            source_snapshot,
            agent_0_action=OCV2_ACTION_INDEX[partner_step.action_id],
            agent_1_action=OCV2_ACTION_INDEX[ego_action],
        )
        reward_0 = float(result.step.rewards["agent_0"])
        reward_1 = float(result.step.rewards["agent_1"])
        if not math.isclose(reward_0, reward_1, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("R015 requires one shared raw team reward.")
        done = bool(result.step.dones["__all__"])
        passive = _passive_record(
            observation=result.step.obs["agent_1"],
            ego_action=ego_action,
            reward=reward_0,
            done=done,
        )
        next_belief = (
            self.update_belief(
                branch.belief,
                passive,
                key=derive_controller_key(step_key, "passive_filter"),
                continuation_states=next_library,
            )
            if update_belief_online
            else branch.belief
        )
        next_history = branch.history.append(
            _history_record(
                observation=result.step.obs["agent_1"],
                ego_action=ego_action,
                reward=reward_0,
                done=done,
            )
        )
        next_kernel = R015ParticleKernelStateV1(
            snapshot=result.snapshot,
            partner_prototype_id=state.partner_prototype_id,
            partner_recurrent_state=partner_step.next_recurrent_state,
            continuation_states=next_library,
            predicted_response_token=_response_token(
                current_observation,
                result.step.obs["agent_1"],
            ),
            predicted_raw_team_reward=reward_0,
            predicted_done=done,
            environment_step=branch.environment_step + 1,
        )
        trace = {
            "environment_step": branch.environment_step,
            "ego_action": ego_action,
            "partner_action": partner_step.action_id,
            "controller_input": copy.deepcopy(branch.history.records[-1]),
            "raw_team_reward": reward_0,
            "done": done,
            "environment_random_key": step_key,
            "official_local_observation_after": np.asarray(
                result.step.obs["agent_1"]
            ),
            "controller_recurrent_state_before_sha256": before_controller_sha,
            "controller_recurrent_state_after_sha256": _library_states_sha256(
                next_library
            ),
            "partner_recurrent_state_before_sha256": before_partner_sha,
            "partner_recurrent_state_after_sha256": _pytree_sha256(
                partner_step.next_recurrent_state,
                domain="path_c_r015_partner_recurrent_state_v1",
            ),
            "belief_update_mode": (
                "online_each_environment_step"
                if update_belief_online
                else R015_BRANCH_BELIEF_ID
            ),
            "belief_before_sha256": _belief_sha256(branch.belief),
            "belief_after_sha256": _belief_sha256(next_belief),
        }
        wrong = _wrong_delivery_event(
            source_snapshot,
            result.snapshot,
            ego_action=ego_action,
            partner_action=partner_step.action_id,
        )
        return (
            R015BranchStateV1(
                kernel=next_kernel,
                belief=next_belief,
                history=next_history,
                environment_step=branch.environment_step + 1,
            ),
            trace,
            wrong,
        )

    def _continuation_step(
        self,
        branch: R015BranchStateV1,
        *,
        key: str,
    ) -> tuple[str, ContinuationLibraryStatesV1]:
        action = self.production_backend.continuation_controller.act(
            history=branch.history,
            belief=branch.belief,
            states=branch.kernel.continuation_states,
            random_keys_by_member={
                member_id: derive_controller_key(key, "continuation", member_id)
                for member_id in self.production_backend.continuation_controller.member_ids
            },
        )
        return action.action_id, action.next_states

    def _committed_member_id(
        self,
        belief: StratifiedParticleBeliefV1,
    ) -> str:
        selected = self.production_backend.continuation_controller.unique_map_prototype(
            belief
        )
        return (
            self.production_backend.baseline_member_id
            if selected is None
            else selected
        )

    def _continuation_step_frozen(
        self,
        branch: R015BranchStateV1,
        *,
        key: str,
        committed_member_id: str,
    ) -> tuple[str, ContinuationLibraryStatesV1]:
        if committed_member_id not in branch.kernel.continuation_states.by_member_id:
            raise ValueError("R015 planning committed an unknown continuation member.")
        action = self.production_backend.continuation_controller.act(
            history=branch.history,
            belief=branch.belief,
            states=branch.kernel.continuation_states,
            random_keys_by_member={
                member_id: derive_controller_key(key, "continuation", member_id)
                for member_id in self.production_backend.continuation_controller.member_ids
            },
        )
        if action.selected_member_id != committed_member_id:
            raise ValueError("R015 frozen branch changed its committed member.")
        return action.action_id, action.next_states

    def _run(
        self,
        branch: R015BranchStateV1,
        *,
        branch_key: str,
        total_steps: int,
        key_offset: int = 0,
        first_forced_action: str | None = None,
        frozen_member_id: str | None = None,
        update_belief_online: bool = True,
        explicit_step_keys: Sequence[str] | None = None,
        _precomputed_online_batch: Mapping[str, Any] | None = None,
        _online_lane_index: int = 0,
        _account_precomputed_batch_cost: bool = True,
    ) -> _BranchExecutionV1:
        """用一个设备扫描推进真实执行段；冻结规划只保留诊断参照。"""

        if frozen_member_id is not None or not update_belief_online:
            return self._run_host_reference(
                branch,
                branch_key=branch_key,
                total_steps=total_steps,
                key_offset=key_offset,
                first_forced_action=first_forced_action,
                frozen_member_id=frozen_member_id,
                update_belief_online=update_belief_online,
                explicit_step_keys=explicit_step_keys,
            )
        if not isinstance(branch.belief, R015OnlineParticleBeliefV2):
            raise TypeError("R015 real execution requires the online v2 belief.")
        if branch.belief.update_count != branch.environment_step:
            raise ValueError(
                "R015 true state and persistent online filter are on different steps."
            )
        stack_filter_states = getattr(
            self.production_backend,
            "stack_online_filter_states_v2",
            None,
        )
        unstack_filter_states = getattr(
            self.production_backend,
            "unstack_online_filter_states_v2",
            None,
        )
        materialize_filter_state = getattr(
            self.production_backend,
            "materialize_online_filter_state_v2",
            None,
        )
        if not all(
            callable(value)
            for value in (
                stack_filter_states,
                unstack_filter_states,
                materialize_filter_state,
            )
        ):
            raise RuntimeError(
                "R015 real execution requires v2 filter stack, unstack, and materialization."
            )
        step_keys = (
            tuple(explicit_step_keys)
            if explicit_step_keys is not None
            else tuple(
                derive_controller_key(
                    branch_key,
                    "future_environment",
                    key_offset + offset,
                )
                for offset in range(total_steps)
            )
        )
        if len(step_keys) != total_steps:
            raise ValueError("R015 real execution keys changed the segment length.")
        result = (
            dict(_precomputed_online_batch)
            if _precomputed_online_batch is not None
            else self.production_backend.run_online_trajectory_batch(
                adapter=self.adapter,
                kernels=(copy.deepcopy(branch.kernel),),
                filter_state=stack_filter_states(
                    (branch.belief.device_filter_state,)
                ),
                branch_keys=(branch_key,),
                total_steps=total_steps,
                key_offset=key_offset,
                forced_first_actions=(first_forced_action,),
                resampling_timing=branch.belief.resampling_timing,
                active_steps=(total_steps,),
                explicit_step_keys=(step_keys,),
                return_recurrent_state_trace=True,
            )
        )
        import jax

        trace = dict(result["trace"])
        active = np.asarray(trace["active"], dtype=np.bool_)
        if active.ndim != 2 or active.shape[0] != total_steps or not (
            0 <= _online_lane_index < active.shape[1]
        ):
            raise ValueError("R015 real execution returned an invalid active mask.")
        if not bool(np.all(active[:, _online_lane_index])):
            raise R015MechanicalBlockInvalidError(
                "R015 real execution closed before the requested segment boundary."
            )
        closed = np.asarray(trace["filter_closed"], dtype=np.bool_)
        if closed.shape != active.shape:
            raise ValueError("R015 real execution changed the filter-closed shape.")
        if bool(np.any(closed[:, _online_lane_index])):
            raise R015MechanicalBlockInvalidError(
                "R015 online filter lost support during real execution."
            )
        final_filter_states = tuple(
            unstack_filter_states(result["final_filter_state"])
        )
        if len(final_filter_states) != active.shape[1]:
            raise RuntimeError("R015 online execution changed its filter lane count.")

        def slice_lane(value: Any, *, keep_batch: bool) -> Any:
            return jax.tree_util.tree_map(
                lambda item: np.asarray(item)[
                    _online_lane_index : _online_lane_index + 1
                ]
                if keep_batch
                else np.asarray(item)[_online_lane_index],
                value,
            )

        final_environment_state = slice_lane(
            result["final_environment_state"], keep_batch=False
        )
        final_observation = {
            agent_id: np.asarray(value)[_online_lane_index]
            for agent_id, value in dict(result["final_observation"]).items()
        }
        final_partner_state = slice_lane(
            result["final_partner_state"], keep_batch=True
        )
        final_continuation_states = ContinuationLibraryStatesV1(
            {
                member_id: slice_lane(member_state, keep_batch=True)
                for member_id, member_state in zip(
                    self.production_backend.continuation_controller.member_ids,
                    tuple(result["final_continuation_states"]),
                )
            }
        )
        ego_actions = np.asarray(trace["ego_actions"], dtype=np.int32)[
            :, _online_lane_index
        ]
        partner_actions = np.asarray(
            trace["partner_actions"], dtype=np.int32
        )[:, _online_lane_index]
        rewards = np.asarray(trace["raw_team_rewards"], dtype=np.float64)[
            :, _online_lane_index
        ]
        dones = np.asarray(trace["done"], dtype=np.bool_)[
            :, _online_lane_index
        ]
        observations_before = np.asarray(
            trace["agent_1_observation_before"]
        )[:, _online_lane_index]
        observations_after = np.asarray(
            trace["agent_1_observation_after"]
        )[:, _online_lane_index]
        posterior_after = np.asarray(
            trace["prototype_posterior_after"], dtype=np.float64
        )[:, _online_lane_index]
        ess = np.asarray(
            trace["filter_pre_resample_ess_fraction"], dtype=np.float64
        )[:, _online_lane_index]
        partner_after = trace["partner_state_after"]
        continuation_after = tuple(trace["continuation_states_after"])
        final_belief = materialize_filter_state(
            adapter=self.adapter,
            filter_state=final_filter_states[_online_lane_index],
            template_belief=branch.belief,
            continuation_states=final_continuation_states,
            previous_official_observation=observations_before[-1],
            predicted_raw_team_reward=float(rewards[-1]),
            predicted_done=bool(dones[-1]),
        )
        if not isinstance(final_belief, R015OnlineParticleBeliefV2):
            raise TypeError("R015 v2 filter materialization returned another belief type.")
        current_history = branch.history
        belief_digest = (
            branch.belief_evidence_sha256
            if branch.belief_evidence_sha256 is not None
            else _belief_sha256(branch.belief)
        )
        trajectory: list[Mapping[str, Any]] = []
        # lax.scan 的下一步 carry 就是本步 selected_*_state，因此相邻步骤的
        # before/after 循环状态逐字节相同。每段只摘要一次初始状态，之后复用
        # 上一步 after 摘要作为下一步 before，避免把同一批设备数据重复哈希。
        controller_before_sha256 = _library_states_sha256(
            branch.kernel.continuation_states
        )
        partner_before_sha256 = _pytree_sha256(
            branch.kernel.partner_recurrent_state,
            domain="path_c_r015_partner_recurrent_state_v1",
        )
        wrong_by_step = np.asarray(
            trace["wrong_delivery"], dtype=np.bool_
        )[:, _online_lane_index]
        action_ids_by_index = tuple(
            action_id
            for action_id, _index in sorted(
                OCV2_ACTION_INDEX.items(), key=lambda item: item[1]
            )
        )
        for offset in range(total_steps):
            before_digest = belief_digest
            belief_digest = canonical_sha256(
                {
                    "schema_version": "path_c_r015_online_belief_transition_v2",
                    "before_sha256": before_digest,
                    "filter_algorithm_id": R015_FILTER_ALGORITHM_ID_V2,
                    "filter_key_contract": R015_FILTER_KEY_CONTRACT_V2,
                    "filter_key_source": "persistent_filter_root_key_words",
                    "completed_environment_step": (
                        branch.environment_step + offset + 1
                    ),
                    "prototype_posterior": posterior_after[offset].tolist(),
                    "pre_resample_ess_fraction": ess[offset].tolist(),
                }
            )
            controller_after = ContinuationLibraryStatesV1(
                {
                    member_id: jax.tree_util.tree_map(
                        lambda item: np.asarray(item)[
                            offset,
                            _online_lane_index : _online_lane_index + 1,
                        ],
                        member_state,
                    )
                    for member_id, member_state in zip(
                        self.production_backend.continuation_controller.member_ids,
                        continuation_after,
                    )
                }
            )
            partner_state_after = jax.tree_util.tree_map(
                lambda item: np.asarray(item)[
                    offset,
                    _online_lane_index : _online_lane_index + 1,
                ],
                partner_after,
            )
            controller_after_sha256 = _library_states_sha256(controller_after)
            partner_after_sha256 = _pytree_sha256(
                partner_state_after,
                domain="path_c_r015_partner_recurrent_state_v1",
            )
            ego_action = action_ids_by_index[int(ego_actions[offset])]
            partner_action = action_ids_by_index[int(partner_actions[offset])]
            trajectory.append(
                {
                    "environment_step": branch.environment_step + offset,
                    "ego_action": ego_action,
                    "partner_action": partner_action,
                    "controller_input": copy.deepcopy(
                        current_history.records[-1]
                    ),
                    "raw_team_reward": float(rewards[offset]),
                    "done": bool(dones[offset]),
                    "environment_random_key": step_keys[offset],
                    "official_local_observation_after": observations_after[offset],
                    "controller_recurrent_state_before_sha256": (
                        controller_before_sha256
                    ),
                    "controller_recurrent_state_after_sha256": (
                        controller_after_sha256
                    ),
                    "partner_recurrent_state_before_sha256": (
                        partner_before_sha256
                    ),
                    "partner_recurrent_state_after_sha256": (
                        partner_after_sha256
                    ),
                    "belief_update_mode": "online_each_environment_step",
                    "belief_before_sha256": before_digest,
                    "belief_after_sha256": belief_digest,
                }
            )
            controller_before_sha256 = controller_after_sha256
            partner_before_sha256 = partner_after_sha256
            current_history = current_history.append(
                _history_record(
                    observation=observations_after[offset],
                    ego_action=ego_action,
                    reward=float(rewards[offset]),
                    done=bool(dones[offset]),
                )
            )
        next_snapshot = OCV2AdapterSnapshot(
            layout_name=branch.kernel.snapshot.layout_name,
            max_steps=branch.kernel.snapshot.max_steps,
            key=np.asarray(result["final_snapshot_keys"])[_online_lane_index],
            state=final_environment_state,
            raw_obs=final_observation,
        )
        next_kernel = R015ParticleKernelStateV1(
            snapshot=next_snapshot,
            partner_prototype_id=branch.kernel.partner_prototype_id,
            partner_recurrent_state=final_partner_state,
            continuation_states=final_continuation_states,
            predicted_response_token=_response_token(
                observations_before[-1], observations_after[-1]
            ),
            predicted_raw_team_reward=float(rewards[-1]),
            predicted_done=bool(dones[-1]),
            environment_step=branch.environment_step + total_steps,
        )
        throughput = dict(result["throughput"])
        if _precomputed_online_batch is not None and not (
            _account_precomputed_batch_cost
        ):
            throughput = {
                **throughput,
                "true_environment_transitions": 0,
                "computed_environment_transitions_including_masked_padding": 0,
                "filter_particle_environment_transitions": 0,
                "computed_filter_particle_environment_transitions": 0,
                "compiled_batch_calls": 0,
                "jit_compilations": 0,
                "wall_seconds": 0.0,
                "true_transitions_per_second": None,
                "computed_transitions_per_second": None,
                "shared_batch_cost_accounted_by_another_lane": True,
            }
        return _BranchExecutionV1(
            state=R015BranchStateV1(
                kernel=next_kernel,
                belief=final_belief,
                history=current_history,
                environment_step=branch.environment_step + total_steps,
                belief_evidence_sha256=belief_digest,
            ),
            raw_return=float(math.fsum(float(value) for value in rewards)),
            trajectory=tuple(trajectory),
            future_random_keys=step_keys,
            wrong_delivery_detected=bool(np.any(wrong_by_step)),
            throughput=throughput,
        )

    def _run_host_reference(
        self,
        branch: R015BranchStateV1,
        *,
        branch_key: str,
        total_steps: int,
        key_offset: int = 0,
        first_forced_action: str | None = None,
        frozen_member_id: str | None = None,
        update_belief_online: bool = True,
        explicit_step_keys: Sequence[str] | None = None,
    ) -> _BranchExecutionV1:
        if explicit_step_keys is not None and len(explicit_step_keys) != total_steps:
            raise ValueError("R015 explicit execution keys changed the segment length.")
        trajectory: list[Mapping[str, Any]] = []
        started = time.perf_counter()
        keys: list[str] = []
        raw_return = 0.0
        wrong = False
        current = copy.deepcopy(branch)
        for offset in range(total_steps):
            future_index = key_offset + offset
            step_key = (
                str(explicit_step_keys[offset])
                if explicit_step_keys is not None
                else derive_controller_key(
                    branch_key,
                    "future_environment",
                    future_index,
                )
            )
            if not _is_sha256(step_key):
                raise ValueError("R015 execution step keys must be SHA-256 values.")
            keys.append(step_key)
            if frozen_member_id is None:
                continuation_action, next_library = self._continuation_step(
                    current,
                    key=step_key,
                )
            else:
                continuation_action, next_library = self._continuation_step_frozen(
                    current,
                    key=step_key,
                    committed_member_id=frozen_member_id,
                )
            ego_action = (
                first_forced_action
                if offset == 0 and first_forced_action is not None
                else continuation_action
            )
            current, trace, step_wrong = self._step_true_state(
                current,
                ego_action=ego_action,
                step_key=step_key,
                next_library=next_library,
                update_belief_online=update_belief_online,
            )
            trajectory.append(trace)
            raw_return += float(trace["raw_team_reward"])
            wrong = wrong or step_wrong
        return _BranchExecutionV1(
            state=current,
            raw_return=raw_return,
            trajectory=tuple(trajectory),
            future_random_keys=tuple(keys),
            wrong_delivery_detected=wrong,
            throughput={
                "execution_mode": "host_step_reference_v1",
                "true_environment_transitions": total_steps,
                "computed_environment_transitions_including_masked_padding": (
                    total_steps
                ),
                "compiled_batch_calls": total_steps,
                "active_lane_batch_width": 1,
                "wall_seconds": time.perf_counter() - started,
                "host_sync_inside_environment_loop": True,
            },
        )

    @staticmethod
    def _particle_branch(
        particle: HiddenStateParticleV1,
        belief: StratifiedParticleBeliefV1,
        history: OfficialHistoryV1,
        remaining_steps: int,
    ) -> R015BranchStateV1:
        if not isinstance(particle.state, R015ParticleKernelStateV1):
            raise TypeError("R015 planner received another particle state.")
        environment_step = 400 - int(remaining_steps)
        if environment_step < 0 or environment_step > 399:
            raise ValueError("R015 remaining horizon is invalid.")
        return R015BranchStateV1(
            kernel=copy.deepcopy(particle.state),
            # 信念和官方历史都是不可变值对象；分支更新会返回新对象，不会原地
            # 写入。这里只复制私有隐藏状态，避免每个规划样本复制整片粒子云。
            belief=belief,
            history=history,
            environment_step=environment_step,
        )

    def rollout_base(
        self,
        *,
        particle: HiddenStateParticleV1,
        belief: StratifiedParticleBeliefV1,
        history: OfficialHistoryV1,
        branch_key: str,
        remaining_steps: int,
    ) -> FullHorizonRolloutV1:
        committed_member_id = self._committed_member_id(belief)
        frozen_belief_sha256 = _belief_sha256(belief)
        execution = self._run(
            self._particle_branch(particle, belief, history, remaining_steps),
            branch_key=branch_key,
            total_steps=remaining_steps,
            frozen_member_id=committed_member_id,
            update_belief_online=False,
        )
        if any(
            step["belief_before_sha256"] != frozen_belief_sha256
            or step["belief_after_sha256"] != frozen_belief_sha256
            or step["belief_update_mode"] != R015_BRANCH_BELIEF_ID
            for step in execution.trajectory
        ):
            raise ValueError("R015 base planning branch changed its frozen belief.")
        transition_sha = canonical_sha256(
            execution.trajectory[:1] if execution.trajectory else []
        )
        return FullHorizonRolloutV1(
            raw_return=execution.raw_return,
            primitive_steps=remaining_steps,
            trajectory_sha256=canonical_sha256(list(execution.trajectory)),
            task_transition_sha256=transition_sha,
            **_future_random_summary(
                root_key=branch_key,
                step_keys=execution.future_random_keys,
                step_count=remaining_steps,
            ),
            committed_member_id=committed_member_id,
            frozen_belief_sha256=frozen_belief_sha256,
            **_execution_state_summaries(execution),
        )

    def rollout_probe_pair(
        self,
        *,
        particle: HiddenStateParticleV1,
        belief: StratifiedParticleBeliefV1,
        history: OfficialHistoryV1,
        script: ProbeScriptV1,
        branch_key: str,
        remaining_steps: int,
    ) -> PairedProbeRolloutV1:
        if tuple(script.primitive_actions) != (script.probe_id,):
            raise ValueError("R015 production supports the six registered one-step scripts.")
        initial = self._particle_branch(particle, belief, history, remaining_steps)
        shared = self._run(
            initial,
            branch_key=branch_key,
            total_steps=1,
            first_forced_action=script.probe_id,
            update_belief_online=True,
        )
        probe_step = initial.environment_step
        token = shared.state.kernel.predicted_response_token
        if token is None:
            raise ValueError("R015 probe transition produced no response token.")
        response = _response_record(
            token=token,
            probe_id=script.probe_id,
            probe_step=probe_step,
        )
        full_history = OfficialHistoryV1(
            (
                *shared.state.history.records[:-1],
                {
                    **shared.state.history.records[-1],
                    "response_summary_v1_derived": response,
                },
            )
        )
        projected = self.response_projection.project(
            full_history,
            probe_id=script.probe_id,
            probe_step=probe_step,
        )
        paired = apply_paired_response_update(
            shared.state.belief,
            projected,
            mask_updater=lambda current, masked_history: current,
            use_updater=lambda current, masked_history, current_response: (
                current.use_current_response(
                    current_response,
                    likelihood=self._response_likelihood,
                )
            ),
        )
        masked_start = replace(
            shared.state,
            belief=paired.masked,
            history=projected.masked_history,
        )
        used_start = replace(
            shared.state,
            belief=paired.used,
            history=projected.masked_history,
        )
        masked_belief_sha256 = _belief_sha256(paired.masked)
        used_belief_sha256 = _belief_sha256(paired.used)
        masked_member_id = self._committed_member_id(paired.masked)
        used_member_id = self._committed_member_id(paired.used)
        suffix_steps = remaining_steps - 1
        masked_suffix = self._run(
            masked_start,
            branch_key=branch_key,
            total_steps=suffix_steps,
            key_offset=1,
            frozen_member_id=masked_member_id,
            update_belief_online=False,
        )
        used_suffix = self._run(
            used_start,
            branch_key=branch_key,
            total_steps=suffix_steps,
            key_offset=1,
            frozen_member_id=used_member_id,
            update_belief_online=False,
        )
        if any(
            step["belief_before_sha256"] != masked_belief_sha256
            or step["belief_after_sha256"] != masked_belief_sha256
            or step["belief_update_mode"] != R015_BRANCH_BELIEF_ID
            for step in masked_suffix.trajectory
        ) or any(
            step["belief_before_sha256"] != used_belief_sha256
            or step["belief_after_sha256"] != used_belief_sha256
            or step["belief_update_mode"] != R015_BRANCH_BELIEF_ID
            for step in used_suffix.trajectory
        ):
            raise ValueError("R015 probe planning suffix changed its frozen belief.")
        shared_trace = shared.trajectory
        masked_trace = (*shared_trace, *masked_suffix.trajectory)
        used_trace = (*shared_trace, *used_suffix.trajectory)
        future_keys = (
            *shared.future_random_keys,
            *masked_suffix.future_random_keys,
        )
        if future_keys != (
            *shared.future_random_keys,
            *used_suffix.future_random_keys,
        ):
            raise ValueError("R015 paired probe branches changed future random keys.")
        task_transition_sha = canonical_sha256(list(shared_trace))
        return PairedProbeRolloutV1(
            masked=FullHorizonRolloutV1(
                raw_return=shared.raw_return + masked_suffix.raw_return,
                primitive_steps=remaining_steps,
                trajectory_sha256=canonical_sha256(list(masked_trace)),
                task_transition_sha256=task_transition_sha,
                **_future_random_summary(
                    root_key=branch_key,
                    step_keys=future_keys,
                    step_count=remaining_steps,
                ),
                committed_member_id=masked_member_id,
                frozen_belief_sha256=masked_belief_sha256,
                **_execution_state_summaries(
                    masked_suffix,
                    ego_action_prefix=(
                        OCV2_ACTION_INDEX[shared_trace[0]["ego_action"]],
                    ),
                    partner_action_prefix=(
                        OCV2_ACTION_INDEX[shared_trace[0]["partner_action"]],
                    ),
                ),
            ),
            used=FullHorizonRolloutV1(
                raw_return=shared.raw_return + used_suffix.raw_return,
                primitive_steps=remaining_steps,
                trajectory_sha256=canonical_sha256(list(used_trace)),
                task_transition_sha256=task_transition_sha,
                **_future_random_summary(
                    root_key=branch_key,
                    step_keys=future_keys,
                    step_count=remaining_steps,
                ),
                committed_member_id=used_member_id,
                frozen_belief_sha256=used_belief_sha256,
                **_execution_state_summaries(
                    used_suffix,
                    ego_action_prefix=(
                        OCV2_ACTION_INDEX[shared_trace[0]["ego_action"]],
                    ),
                    partner_action_prefix=(
                        OCV2_ACTION_INDEX[shared_trace[0]["partner_action"]],
                    ),
                ),
            ),
            masked_branch_head_belief_sha256=masked_belief_sha256,
            used_branch_head_belief_sha256=used_belief_sha256,
            branch_head_particle_transitions=len(belief.particles),
        )

    @staticmethod
    def _batch_output_sha256(output: Mapping[str, Any]) -> str:
        return canonical_sha256(
            {
                "raw_team_rewards": np.asarray(output["raw_team_rewards"]).tolist(),
                "partner_actions": np.asarray(output["partner_actions"]).tolist(),
                "ego_actions": np.asarray(output["ego_actions"]).tolist(),
                "done": np.asarray(output["done"]).tolist(),
                "agent_1_observation_sequence": np.asarray(
                    output["agent_1_observation_sequence"]
                ).tolist(),
                "final_environment_state_sha256": _pytree_sha256(
                    output["environment_state"],
                    domain="path_c_r015_batch_final_environment_state_v1",
                ),
                "final_partner_state_sha256": _pytree_sha256(
                    output["partner_recurrent_state"],
                    domain="path_c_r015_batch_final_partner_state_v1",
                ),
                "final_continuation_states_sha256": _pytree_sha256(
                    output["continuation_states"],
                    domain="path_c_r015_batch_final_continuation_states_v1",
                ),
            }
        )

    @staticmethod
    def _batch_state_summaries(
        output: Mapping[str, Any],
        *,
        ego_action_prefix: Sequence[int] = (),
        partner_action_prefix: Sequence[int] = (),
    ) -> Mapping[str, str]:
        return {
            "ego_action_sequence_sha256": canonical_sha256(
                [
                    *ego_action_prefix,
                    *np.asarray(output["ego_actions"]).astype(int).tolist(),
                ]
            ),
            "partner_action_sequence_sha256": canonical_sha256(
                [
                    *partner_action_prefix,
                    *np.asarray(output["partner_actions"]).astype(int).tolist(),
                ]
            ),
            "final_environment_state_sha256": _pytree_sha256(
                output["environment_state"],
                domain="path_c_r015_batch_final_environment_state_v1",
            ),
            "final_partner_state_sha256": _pytree_sha256(
                output["partner_recurrent_state"],
                domain="path_c_r015_batch_final_partner_state_v1",
            ),
            "final_continuation_states_sha256": _pytree_sha256(
                output["continuation_states"],
                domain="path_c_r015_batch_final_continuation_states_v1",
            ),
        }

    def _kernel_from_batch_output(
        self,
        source: R015ParticleKernelStateV1,
        output: Mapping[str, Any],
        *,
        environment_step: int,
        copy_state: bool = True,
    ) -> R015ParticleKernelStateV1:
        rewards = np.asarray(output["raw_team_rewards"])
        dones = np.asarray(output["done"])
        observations = np.asarray(output["agent_1_observation_sequence"])
        previous = np.asarray(source.snapshot.raw_obs["agent_1"])
        token = None
        if len(observations):
            token = _response_token(previous, observations[0])
        if copy_state:
            snapshot_key = copy.deepcopy(output["snapshot_key"])
            environment_state = copy.deepcopy(output["environment_state"])
            raw_observation = {
                key: np.asarray(value).copy()
                for key, value in output["raw_observation"].items()
            }
            partner_recurrent_state = copy.deepcopy(
                output["partner_recurrent_state"]
            )
            continuation_states = copy.deepcopy(
                dict(output["continuation_states"])
            )
        else:
            # 过滤批次的输出由新的宿主数组独占，后续环境调用也不原地修改这些数组。
            # 保留只读视图避免每一步为全部粒子重复深拷贝；咨询点仍会计算完整状态摘要。
            snapshot_key = output["snapshot_key"]
            environment_state = output["environment_state"]
            raw_observation = {
                key: np.asarray(value)
                for key, value in output["raw_observation"].items()
            }
            partner_recurrent_state = output["partner_recurrent_state"]
            continuation_states = dict(output["continuation_states"])
        return R015ParticleKernelStateV1(
            snapshot=OCV2AdapterSnapshot(
                layout_name=source.snapshot.layout_name,
                max_steps=source.snapshot.max_steps,
                key=snapshot_key,
                state=environment_state,
                raw_obs=raw_observation,
            ),
            partner_prototype_id=source.partner_prototype_id,
            partner_recurrent_state=partner_recurrent_state,
            continuation_states=ContinuationLibraryStatesV1(
                continuation_states
            ),
            predicted_response_token=token,
            predicted_raw_team_reward=(
                None if rewards.size == 0 else float(rewards[-1])
            ),
            predicted_done=None if dones.size == 0 else bool(dones[-1]),
            environment_step=environment_step,
        )

    def _rollout_planning_batch_v1_diagnostic_retired(
        self,
        *,
        samples: Sequence[PlanningBranchSampleV1],
        belief: StratifiedParticleBeliefV1,
        history: OfficialHistoryV1,
        scripts: Sequence[ProbeScriptV1],
        remaining_steps: int,
    ) -> PlanningBatchRolloutsV1:
        """已退役的单咨询点硬相容更新，只保留历史诊断读取。"""

        sample_tuple = tuple(samples)
        script_tuple = tuple(scripts)
        if not sample_tuple or tuple(script.probe_id for script in script_tuple) != (
            R015_ATOMIC_ACTIONS
        ):
            raise ValueError("R015 planning batch requires samples and six probes.")
        initial_kernels = tuple(
            self._particle_branch(
                sample.source_particle, belief, history, remaining_steps
            ).kernel
            for sample in sample_tuple
        )
        base_member = self._committed_member_id(belief)
        base_outputs = self.production_backend.run_frozen_trajectory_batch(
            adapter=self.adapter,
            kernels=initial_kernels,
            branch_keys=tuple(sample.common_random_key for sample in sample_tuple),
            total_steps=remaining_steps,
            key_offset=0,
            forced_first_actions=(None,) * len(sample_tuple),
            committed_member_ids=(base_member,) * len(sample_tuple),
        )
        shared_kernels = []
        shared_keys = []
        shared_forced_actions = []
        shared_coordinates = []
        for sample_index, sample in enumerate(sample_tuple):
            for script in script_tuple:
                shared_kernels.append(initial_kernels[sample_index])
                shared_keys.append(sample.common_random_key)
                shared_forced_actions.append(script.probe_id)
                shared_coordinates.append((sample_index, script.probe_id))
        shared_outputs = self.production_backend.run_frozen_trajectory_batch(
            adapter=self.adapter,
            kernels=tuple(shared_kernels),
            branch_keys=tuple(shared_keys),
            total_steps=1,
            key_offset=0,
            forced_first_actions=tuple(shared_forced_actions),
            committed_member_ids=(base_member,) * len(shared_kernels),
        )
        paired_heads: dict[tuple[int, str], Mapping[str, Any]] = {}
        suffix_kernels = []
        suffix_keys = []
        suffix_members = []
        suffix_coordinates = []
        head_material: dict[tuple[int, str], Mapping[str, Any]] = {}
        filter_kernels = []
        filter_transition_keys = []
        filter_forced_actions = []
        filter_coordinates = []
        for coordinate, output in zip(shared_coordinates, shared_outputs):
            sample_index, probe_id = coordinate
            initial_kernel = initial_kernels[sample_index]
            shared_kernel = self._kernel_from_batch_output(
                initial_kernel,
                output,
                environment_step=400 - remaining_steps + 1,
            )
            reward = float(np.asarray(output["raw_team_rewards"])[0])
            done = bool(np.asarray(output["done"])[0])
            ego_observation = np.asarray(output["raw_observation"]["agent_1"])
            passive = _passive_record(
                observation=ego_observation,
                ego_action=probe_id,
                reward=reward,
                done=done,
            )
            step_key = derive_controller_key(
                sample_tuple[sample_index].common_random_key,
                "future_environment",
                0,
            )
            update_key = derive_controller_key(step_key, "passive_filter")
            head_material[coordinate] = {
                "output": output,
                "shared_kernel": shared_kernel,
                "passive": passive,
                "update_key": update_key,
                "reward": reward,
                "done": done,
                "ego_observation": ego_observation,
            }
            for particle_index, particle in enumerate(belief.particles):
                if not isinstance(particle.state, R015ParticleKernelStateV1):
                    raise TypeError("R015 planning belief contains another kernel state.")
                transition_key = derive_controller_key(
                    update_key,
                    "filter_transition",
                    particle.prototype_id,
                    particle_index,
                )
                filter_kernels.append(particle.state)
                filter_transition_keys.append(transition_key)
                filter_forced_actions.append(probe_id)
                filter_coordinates.append(
                    (coordinate, transition_key, particle)
                )
        filter_outputs = self.production_backend.run_frozen_trajectory_batch(
            adapter=self.adapter,
            kernels=tuple(filter_kernels),
            branch_keys=tuple(filter_transition_keys),
            total_steps=1,
            key_offset=0,
            forced_first_actions=tuple(filter_forced_actions),
            committed_member_ids=(base_member,) * len(filter_kernels),
            key_contract="filter_transition_v1",
        )
        transitioned_particles: dict[tuple[tuple[int, str], str], HiddenStateParticleV1] = {}
        for (coordinate, transition_key, particle), output in zip(
            filter_coordinates, filter_outputs
        ):
            next_kernel = self._kernel_from_batch_output(
                particle.state,
                output,
                environment_step=particle.state.environment_step + 1,
            )
            transitioned_particles[(coordinate, transition_key)] = HiddenStateParticleV1(
                prototype_id=particle.prototype_id,
                state_sha256=_particle_state_sha256(next_kernel),
                state=next_kernel,
                weight=1.0,
            )
        for coordinate in shared_coordinates:
            sample_index, probe_id = coordinate
            material = head_material[coordinate]
            output = material["output"]
            shared_kernel = material["shared_kernel"]
            passive = material["passive"]
            reward = material["reward"]
            done = material["done"]
            ego_observation = material["ego_observation"]
            update_key = str(material["update_key"])
            masked_belief = belief.update(
                official_record=passive,
                update_key=update_key,
                transition=lambda particle, record, transition_key, coordinate=coordinate: (
                    transitioned_particles[(coordinate, transition_key)],
                ),
                compatibility=self._particle_compatibility,
            )
            response = _response_record(
                token=int(shared_kernel.predicted_response_token),
                probe_id=probe_id,
                probe_step=400 - remaining_steps,
            )
            full_history = OfficialHistoryV1(
                (
                    *history.records,
                    {
                        **_history_record(
                            observation=ego_observation,
                            ego_action=probe_id,
                            reward=reward,
                            done=done,
                        ),
                        "response_summary_v1_derived": response,
                    },
                )
            )
            projected = self.response_projection.project(
                full_history,
                probe_id=probe_id,
                probe_step=400 - remaining_steps,
            )
            used_belief = masked_belief.use_current_response(
                response,
                likelihood=self._response_likelihood,
            )
            masked_member = self._committed_member_id(masked_belief)
            used_member = self._committed_member_id(used_belief)
            paired_heads[coordinate] = {
                "shared_output": output,
                "masked_belief": masked_belief,
                "used_belief": used_belief,
                "masked_member": masked_member,
                "used_member": used_member,
            }
            for label, member in (("masked", masked_member), ("used", used_member)):
                suffix_kernels.append(shared_kernel)
                suffix_keys.append(sample_tuple[sample_index].common_random_key)
                suffix_members.append(member)
                suffix_coordinates.append((sample_index, probe_id, label))
        suffix_outputs = self.production_backend.run_frozen_trajectory_batch(
            adapter=self.adapter,
            kernels=tuple(suffix_kernels),
            branch_keys=tuple(suffix_keys),
            total_steps=remaining_steps - 1,
            key_offset=1,
            forced_first_actions=(None,) * len(suffix_kernels),
            committed_member_ids=tuple(suffix_members),
        )
        suffix_by_coordinate = dict(zip(suffix_coordinates, suffix_outputs))
        output_samples = []
        for sample_index, sample in enumerate(sample_tuple):
            base_output = base_outputs[sample_index]
            base = FullHorizonRolloutV1(
                raw_return=float(np.sum(base_output["raw_team_rewards"])),
                primitive_steps=remaining_steps,
                trajectory_sha256=self._batch_output_sha256(base_output),
                task_transition_sha256=canonical_sha256(
                    {
                        "ego_action": int(base_output["ego_actions"][0]),
                        "partner_action": int(base_output["partner_actions"][0]),
                        "reward": float(base_output["raw_team_rewards"][0]),
                    }
                ),
                **_future_random_summary(
                    root_key=sample.common_random_key,
                    step_keys=base_output["future_random_keys"],
                    step_count=remaining_steps,
                ),
                committed_member_id=base_member,
                frozen_belief_sha256=_belief_sha256(belief),
                **self._batch_state_summaries(base_output),
            )
            pairs = {}
            for script in script_tuple:
                head = paired_heads[(sample_index, script.probe_id)]
                shared_output = head["shared_output"]
                masked_output = suffix_by_coordinate[
                    (sample_index, script.probe_id, "masked")
                ]
                used_output = suffix_by_coordinate[
                    (sample_index, script.probe_id, "used")
                ]
                shared_return = float(shared_output["raw_team_rewards"][0])
                shared_transition_sha = canonical_sha256(
                    {
                        "ego_action": int(shared_output["ego_actions"][0]),
                        "partner_action": int(shared_output["partner_actions"][0]),
                        "reward": shared_return,
                        "observation": np.asarray(
                            shared_output["raw_observation"]["agent_1"]
                        ).tolist(),
                    }
                )
                future_keys = (
                    shared_output["future_random_keys"][0],
                    *masked_output["future_random_keys"],
                )
                if future_keys != (
                    shared_output["future_random_keys"][0],
                    *used_output["future_random_keys"],
                ):
                    raise ValueError("R015 batched probe suffixes changed random keys.")
                masked_belief_sha = _belief_sha256(head["masked_belief"])
                used_belief_sha = _belief_sha256(head["used_belief"])
                pairs[script.probe_id] = PairedProbeRolloutV1(
                    masked=FullHorizonRolloutV1(
                        raw_return=shared_return
                        + float(np.sum(masked_output["raw_team_rewards"])),
                        primitive_steps=remaining_steps,
                        trajectory_sha256=canonical_sha256(
                            [
                                self._batch_output_sha256(shared_output),
                                self._batch_output_sha256(masked_output),
                            ]
                        ),
                        task_transition_sha256=shared_transition_sha,
                        **_future_random_summary(
                            root_key=sample.common_random_key,
                            step_keys=future_keys,
                            step_count=remaining_steps,
                        ),
                        committed_member_id=str(head["masked_member"]),
                        frozen_belief_sha256=masked_belief_sha,
                        **self._batch_state_summaries(
                            masked_output,
                            ego_action_prefix=(
                                int(shared_output["ego_actions"][0]),
                            ),
                            partner_action_prefix=(
                                int(shared_output["partner_actions"][0]),
                            ),
                        ),
                    ),
                    used=FullHorizonRolloutV1(
                        raw_return=shared_return
                        + float(np.sum(used_output["raw_team_rewards"])),
                        primitive_steps=remaining_steps,
                        trajectory_sha256=canonical_sha256(
                            [
                                self._batch_output_sha256(shared_output),
                                self._batch_output_sha256(used_output),
                            ]
                        ),
                        task_transition_sha256=shared_transition_sha,
                        **_future_random_summary(
                            root_key=sample.common_random_key,
                            step_keys=future_keys,
                            step_count=remaining_steps,
                        ),
                        committed_member_id=str(head["used_member"]),
                        frozen_belief_sha256=used_belief_sha,
                        **self._batch_state_summaries(
                            used_output,
                            ego_action_prefix=(
                                int(shared_output["ego_actions"][0]),
                            ),
                            partner_action_prefix=(
                                int(shared_output["partner_actions"][0]),
                            ),
                        ),
                    ),
                    masked_branch_head_belief_sha256=masked_belief_sha,
                    used_branch_head_belief_sha256=used_belief_sha,
                    branch_head_particle_transitions=len(belief.particles),
                )
            output_samples.append(
                PlanningSampleRolloutsV1(
                    sample=sample,
                    base=base,
                    probe_pairs=pairs,
                )
            )
        return PlanningBatchRolloutsV1(
            samples=tuple(output_samples),
            compiled_batch_calls=4,
            active_batch_sizes=(
                len(sample_tuple),
                len(sample_tuple) * len(script_tuple),
                len(sample_tuple) * len(script_tuple) * len(belief.particles),
                len(sample_tuple) * len(script_tuple) * 2,
            ),
            host_sync_inside_environment_loop=False,
        )

    @staticmethod
    def _planning_batch_cache_key(
        *,
        belief_sha256: str,
        history_sha256: str,
        samples: Sequence[PlanningBranchSampleV1],
        remaining_steps: int,
    ) -> tuple[str, str, int, tuple[tuple[int, str], ...]]:
        if not _is_sha256(belief_sha256) or not _is_sha256(history_sha256):
            raise ValueError("R015 planning cache requires content digests.")
        return (
            belief_sha256,
            history_sha256,
            int(remaining_steps),
            tuple(
                (int(sample.canonical_slot_16), str(sample.common_random_key))
                for sample in samples
            ),
        )

    def rollout_planning_length_bucket_v2(
        self,
        *,
        requests: Sequence[R015PlanningPointBatchRequestV2],
        scripts: Sequence[ProbeScriptV1],
        remaining_steps: int,
    ) -> R015PlanningLengthBucketResultV2:
        """把同一剩余长度的咨询点合并为四个设备阶段。

        四个阶段依次是基线完整后缀、六个共享探查头、一次完整粒子云的
        v2 分支头更新，以及屏蔽/使用两组冻结成员后缀。宿主只在这四个因果边界
        读取结果，环境步循环始终留在设备扫描中。
        """

        request_tuple = tuple(requests)
        script_tuple = tuple(scripts)
        if not request_tuple or tuple(
            script.probe_id for script in script_tuple
        ) != R015_ATOMIC_ACTIONS:
            raise ValueError("R015 length bucket requires points and six probes.")
        if any(request.remaining_steps != remaining_steps for request in request_tuple):
            raise ValueError("R015 length bucket mixed different horizons.")
        if len({request.consultation_id for request in request_tuple}) != len(
            request_tuple
        ):
            raise ValueError("R015 length bucket duplicated a consultation.")
        timings = {request.belief.resampling_timing for request in request_tuple}
        particle_counts = {
            request.belief.particles_per_prototype for request in request_tuple
        }
        if len(timings) != 1 or len(particle_counts) != 1:
            raise ValueError("R015 length bucket mixed filter configurations.")
        if any(
            request.belief.filter_algorithm_id != R015_FILTER_ALGORITHM_ID_V2
            or request.belief.resampling_algorithm
            != "strict_systematic_per_prototype_v2"
            for request in request_tuple
        ):
            raise ValueError("R015 planning requires the selected formal v2 filter.")
        # 完整粒子信念的摘要与样本数成正比。每个咨询点只计算一次并核对
        # checkpoint 绑定，之后所有样本证据和设备结果缓存都复用同一摘要。
        request_belief_sha256 = tuple(
            _belief_sha256(request.belief) for request in request_tuple
        )
        if any(
            request.filter_checkpoint_sha256 != belief_sha256
            for request, belief_sha256 in zip(
                request_tuple, request_belief_sha256
            )
        ):
            raise ValueError("R015 planning filter checkpoint changed its belief.")
        request_history_sha256 = tuple(
            canonical_sha256(list(request.history.records))
            for request in request_tuple
        )

        stack_filter_states = getattr(
            self.production_backend, "stack_online_filter_states_v2", None
        )
        run_branch_head_batch_v2 = getattr(
            self.production_backend, "run_planning_branch_head_batch_v2", None
        )
        if not all(
            callable(value)
            for value in (
                stack_filter_states,
                run_branch_head_batch_v2,
            )
        ):
            raise RuntimeError("R015 planning lacks the shared v2 filter API.")

        flat_coordinates: list[tuple[int, int, PlanningBranchSampleV1]] = []
        initial_kernels: list[R015ParticleKernelStateV1] = []
        base_members: list[str] = []
        branch_keys: list[str] = []
        for request_index, request in enumerate(request_tuple):
            member = self._committed_member_id(request.belief)
            for sample_index, sample in enumerate(request.samples):
                flat_coordinates.append((request_index, sample_index, sample))
                initial_kernels.append(
                    self._particle_branch(
                        sample.source_particle,
                        request.belief,
                        request.history,
                        remaining_steps,
                    ).kernel
                )
                base_members.append(member)
                branch_keys.append(sample.common_random_key)
        sample_lane_count = len(flat_coordinates)
        if sample_lane_count <= 0:
            raise ValueError("R015 length bucket contains no missing sample.")

        # 二十个登记咨询长度共享 399 步静态图。每个隐藏样本只派生一次完整
        # 未来随机流；基线、六个探查头和十二个后缀都引用这个同一字节序列。
        static_base_steps = 399
        static_suffix_steps = 398
        prepared_future_streams = []
        future_random_summaries = []
        for branch_key in branch_keys:
            stream, summary = _prepare_registered_future_random_stream(
                root_key=branch_key,
                generated_step_count=static_base_steps,
                logical_step_count=remaining_steps,
            )
            prepared_future_streams.append(stream)
            future_random_summaries.append(summary)

        started = time.perf_counter()
        base_outputs = self.production_backend.run_frozen_trajectory_batch(
            adapter=self.adapter,
            kernels=tuple(initial_kernels),
            branch_keys=tuple(branch_keys),
            total_steps=static_base_steps,
            key_offset=0,
            forced_first_actions=(None,) * sample_lane_count,
            committed_member_ids=tuple(base_members),
            active_steps=(remaining_steps,) * sample_lane_count,
            explicit_step_keys=tuple(prepared_future_streams),
        )

        shared_kernels: list[R015ParticleKernelStateV1] = []
        shared_keys: list[str] = []
        shared_actions: list[str] = []
        shared_members: list[str] = []
        shared_coordinates: list[tuple[int, str]] = []
        for flat_index, (_request_index, _sample_index, sample) in enumerate(
            flat_coordinates
        ):
            for script in script_tuple:
                shared_kernels.append(initial_kernels[flat_index])
                shared_keys.append(sample.common_random_key)
                shared_actions.append(script.probe_id)
                shared_members.append(base_members[flat_index])
                shared_coordinates.append((flat_index, script.probe_id))
        shared_outputs = self.production_backend.run_frozen_trajectory_batch(
            adapter=self.adapter,
            kernels=tuple(shared_kernels),
            branch_keys=tuple(shared_keys),
            total_steps=1,
            key_offset=0,
            forced_first_actions=tuple(shared_actions),
            committed_member_ids=tuple(shared_members),
            active_steps=(1,) * len(shared_kernels),
            explicit_step_keys=tuple(
                (prepared_future_streams[flat_index][0],)
                for flat_index, _probe_id in shared_coordinates
            ),
        )

        branch_head_states = []
        branch_head_observations = []
        branch_head_actions = []
        branch_head_rewards = []
        branch_head_boundaries = []
        branch_head_tokens = []
        branch_head_material: list[Mapping[str, Any]] = []
        for shared_index, ((flat_index, probe_id), output) in enumerate(
            zip(shared_coordinates, shared_outputs)
        ):
            request_index, _sample_index, sample = flat_coordinates[flat_index]
            request = request_tuple[request_index]
            shared_kernel = self._kernel_from_batch_output(
                initial_kernels[flat_index],
                output,
                environment_step=400 - remaining_steps + 1,
            )
            token = shared_kernel.predicted_response_token
            if token is None:
                raise ValueError("R015 planning probe produced no response token.")
            reward = float(np.asarray(output["raw_team_rewards"])[0])
            done = bool(np.asarray(output["done"])[0])
            observation = np.asarray(output["raw_observation"]["agent_1"])
            response = _response_record(
                token=int(token),
                probe_id=probe_id,
                probe_step=400 - remaining_steps,
            )
            full_history = OfficialHistoryV1(
                (
                    *request.history.records,
                    {
                        **_history_record(
                            observation=observation,
                            ego_action=probe_id,
                            reward=reward,
                            done=done,
                        ),
                        "response_summary_v1_derived": response,
                    },
                )
            )
            projected = self.response_projection.project(
                full_history,
                probe_id=probe_id,
                probe_step=400 - remaining_steps,
            )
            if projected.current_response != response:
                raise ValueError("R015 response projection changed the observed token.")
            branch_head_states.append(request.belief.device_filter_state)
            branch_head_observations.append(observation)
            branch_head_actions.append(OCV2_ACTION_INDEX[probe_id])
            branch_head_rewards.append(reward)
            branch_head_boundaries.append(done)
            branch_head_tokens.append(int(token))
            branch_head_material.append(
                {
                    "shared_output": output,
                    "shared_kernel": shared_kernel,
                    "request": request,
                    "response": response,
                }
            )

        stacked_heads = stack_filter_states(tuple(branch_head_states))
        branch_heads = run_branch_head_batch_v2(
            adapter=self.adapter,
            filter_state=stacked_heads,
            official_step_records={
                "official_local_observation": np.stack(branch_head_observations),
                "ego_action_index": np.asarray(branch_head_actions, dtype=np.int32),
                "raw_team_reward": np.asarray(
                    branch_head_rewards, dtype=np.float32
                ),
                "episode_boundary": np.asarray(
                    branch_head_boundaries, dtype=np.bool_
                ),
            },
            environment_step=400 - remaining_steps + 1,
            resampling_timing=next(iter(timings)),
            response_tokens=np.asarray(branch_head_tokens, dtype=np.int32),
        )
        if branch_heads.get("compiled_batch_calls") != 1 or (
            branch_heads.get("host_sync_inside_environment_loop") is not False
        ):
            raise RuntimeError("R015 branch-head filter changed its one compiled call.")
        compact_field_dtypes = {
            "masked_prototype_masses": np.float64,
            "used_prototype_masses": np.float64,
            "masked_closed": np.bool_,
            "used_closed": np.bool_,
            "masked_committed_member_indices": np.int32,
            "used_committed_member_indices": np.int32,
            "response_match_counts_by_prototype": np.int32,
            "response_match_masses_by_prototype": np.float64,
        }
        compact = {
            name: np.ascontiguousarray(np.asarray(branch_heads[name], dtype=dtype))
            for name, dtype in compact_field_dtypes.items()
        }
        lane_count = len(shared_coordinates)
        expected_shapes = {
            "masked_prototype_masses": (lane_count, 4),
            "used_prototype_masses": (lane_count, 4),
            "masked_closed": (lane_count,),
            "used_closed": (lane_count,),
            "masked_committed_member_indices": (lane_count,),
            "used_committed_member_indices": (lane_count,),
            "response_match_counts_by_prototype": (lane_count, 4),
            "response_match_masses_by_prototype": (lane_count, 4),
        }
        if any(compact[name].shape != shape for name, shape in expected_shapes.items()):
            raise RuntimeError("R015 branch-head compact projection changed shape.")
        member_ids = tuple(branch_heads.get("continuation_member_ids", ()))
        expected_member_ids = tuple(
            self.production_backend.continuation_controller.member_ids
        )
        if member_ids != expected_member_ids:
            raise RuntimeError("R015 branch-head compact projection changed member order.")
        if np.any(compact["masked_closed"]) or np.any(compact["used_closed"]):
            raise _R015OnlineFilterZeroSupportError(
                "R015 planning branch lost one prototype's support."
            )

        baseline_index = member_ids.index(self.production_backend.baseline_member_id)
        prototype_member_indices = np.asarray(
            [member_ids.index(value) for value in request_tuple[0].belief.prototype_ids],
            dtype=np.int32,
        )
        for mode in ("masked", "used"):
            masses = compact[f"{mode}_prototype_masses"]
            maxima = np.max(masses, axis=-1, keepdims=True)
            winners = masses == maxima
            expected_indices = np.where(
                np.sum(winners, axis=-1) == 1,
                prototype_member_indices[np.argmax(masses, axis=-1)],
                baseline_index,
            ).astype(np.int32)
            if not np.array_equal(
                expected_indices,
                compact[f"{mode}_committed_member_indices"],
            ):
                raise RuntimeError("R015 branch-head compact MAP routing changed.")

        paired_heads: dict[tuple[int, str], Mapping[str, Any]] = {}
        suffix_kernels: list[R015ParticleKernelStateV1] = []
        suffix_keys: list[str] = []
        suffix_members: list[str] = []
        suffix_coordinates: list[tuple[int, str, str]] = []
        for shared_index, coordinate in enumerate(shared_coordinates):
            flat_index, probe_id = coordinate
            material = branch_head_material[shared_index]
            request = material["request"]
            shared_kernel = material["shared_kernel"]
            official_step_record = {
                "official_local_observation": branch_head_observations[shared_index],
                "ego_action_index": np.int32(branch_head_actions[shared_index]),
                "raw_team_reward": np.float32(branch_head_rewards[shared_index]),
                "episode_boundary": np.bool_(branch_head_boundaries[shared_index]),
            }
            masked_member = member_ids[
                int(compact["masked_committed_member_indices"][shared_index])
            ]
            used_member = member_ids[
                int(compact["used_committed_member_indices"][shared_index])
            ]
            masked_sha256 = _planning_branch_head_projection_sha256(
                input_filter_checkpoint_sha256=request.filter_checkpoint_sha256,
                official_step_record=official_step_record,
                environment_step=400 - remaining_steps + 1,
                resampling_timing=next(iter(timings)),
                particles_per_prototype=request.belief.particles_per_prototype,
                mode="masked",
                response_token=None,
                prototype_ids=request.belief.prototype_ids,
                prototype_masses=compact["masked_prototype_masses"][shared_index],
                closed=bool(compact["masked_closed"][shared_index]),
                committed_member_id=masked_member,
            )
            used_sha256 = _planning_branch_head_projection_sha256(
                input_filter_checkpoint_sha256=request.filter_checkpoint_sha256,
                official_step_record=official_step_record,
                environment_step=400 - remaining_steps + 1,
                resampling_timing=next(iter(timings)),
                particles_per_prototype=request.belief.particles_per_prototype,
                mode="used",
                response_token=int(branch_head_tokens[shared_index]),
                prototype_ids=request.belief.prototype_ids,
                prototype_masses=compact["used_prototype_masses"][shared_index],
                closed=bool(compact["used_closed"][shared_index]),
                committed_member_id=used_member,
                response_match_counts=compact[
                    "response_match_counts_by_prototype"
                ][shared_index],
                response_match_masses=compact[
                    "response_match_masses_by_prototype"
                ][shared_index],
            )
            paired_heads[coordinate] = {
                **material,
                "masked_member": masked_member,
                "used_member": used_member,
                "masked_belief_sha256": masked_sha256,
                "used_belief_sha256": used_sha256,
            }
            for label, member in (
                ("masked", masked_member),
                ("used", used_member),
            ):
                suffix_kernels.append(shared_kernel)
                suffix_keys.append(shared_keys[shared_index])
                suffix_members.append(member)
                suffix_coordinates.append((flat_index, probe_id, label))

        suffix_outputs = self.production_backend.run_frozen_trajectory_batch(
            adapter=self.adapter,
            kernels=tuple(suffix_kernels),
            branch_keys=tuple(suffix_keys),
            total_steps=static_suffix_steps,
            key_offset=1,
            forced_first_actions=(None,) * len(suffix_kernels),
            committed_member_ids=tuple(suffix_members),
            active_steps=(remaining_steps - 1,) * len(suffix_kernels),
            explicit_step_keys=tuple(
                prepared_future_streams[flat_index][1:]
                for flat_index, _probe_id, _label in suffix_coordinates
            ),
        )
        suffix_by_coordinate = dict(zip(suffix_coordinates, suffix_outputs))

        output_by_request: dict[int, list[PlanningSampleRolloutsV1]] = {
            index: [] for index in range(len(request_tuple))
        }
        for flat_index, (request_index, _sample_index, sample) in enumerate(
            flat_coordinates
        ):
            request = request_tuple[request_index]
            base_output = base_outputs[flat_index]
            base = FullHorizonRolloutV1(
                raw_return=float(np.sum(base_output["raw_team_rewards"])),
                primitive_steps=remaining_steps,
                trajectory_sha256=self._batch_output_sha256(base_output),
                task_transition_sha256=canonical_sha256(
                    {
                        "ego_action": int(base_output["ego_actions"][0]),
                        "partner_action": int(base_output["partner_actions"][0]),
                        "reward": float(base_output["raw_team_rewards"][0]),
                    }
                ),
                **future_random_summaries[flat_index],
                committed_member_id=base_members[flat_index],
                frozen_belief_sha256=request_belief_sha256[request_index],
                **self._batch_state_summaries(base_output),
            )
            pairs: dict[str, PairedProbeRolloutV1] = {}
            for script in script_tuple:
                head = paired_heads[(flat_index, script.probe_id)]
                shared_output = head["shared_output"]
                masked_output = suffix_by_coordinate[
                    (flat_index, script.probe_id, "masked")
                ]
                used_output = suffix_by_coordinate[
                    (flat_index, script.probe_id, "used")
                ]
                shared_return = float(shared_output["raw_team_rewards"][0])
                shared_transition_sha256 = canonical_sha256(
                    {
                        "ego_action": int(shared_output["ego_actions"][0]),
                        "partner_action": int(shared_output["partner_actions"][0]),
                        "reward": shared_return,
                        "observation": np.asarray(
                            shared_output["raw_observation"]["agent_1"]
                        ).tolist(),
                    }
                )
                future_keys = (
                    shared_output["future_random_keys"][0],
                    *masked_output["future_random_keys"],
                )
                if future_keys != (
                    shared_output["future_random_keys"][0],
                    *used_output["future_random_keys"],
                ):
                    raise ValueError("R015 paired suffix random streams differ.")
                if future_keys != prepared_future_streams[flat_index][
                    :remaining_steps
                ] or tuple(base_output["future_random_keys"]) != future_keys:
                    raise ValueError("R015 planning did not consume its prepared key stream.")
                masked_sha256 = str(head["masked_belief_sha256"])
                used_sha256 = str(head["used_belief_sha256"])
                pairs[script.probe_id] = PairedProbeRolloutV1(
                    masked=FullHorizonRolloutV1(
                        raw_return=shared_return
                        + float(np.sum(masked_output["raw_team_rewards"])),
                        primitive_steps=remaining_steps,
                        trajectory_sha256=canonical_sha256(
                            [
                                self._batch_output_sha256(shared_output),
                                self._batch_output_sha256(masked_output),
                            ]
                        ),
                        task_transition_sha256=shared_transition_sha256,
                        **future_random_summaries[flat_index],
                        committed_member_id=str(head["masked_member"]),
                        frozen_belief_sha256=masked_sha256,
                        **self._batch_state_summaries(
                            masked_output,
                            ego_action_prefix=(int(shared_output["ego_actions"][0]),),
                            partner_action_prefix=(
                                int(shared_output["partner_actions"][0]),
                            ),
                        ),
                    ),
                    used=FullHorizonRolloutV1(
                        raw_return=shared_return
                        + float(np.sum(used_output["raw_team_rewards"])),
                        primitive_steps=remaining_steps,
                        trajectory_sha256=canonical_sha256(
                            [
                                self._batch_output_sha256(shared_output),
                                self._batch_output_sha256(used_output),
                            ]
                        ),
                        task_transition_sha256=shared_transition_sha256,
                        **future_random_summaries[flat_index],
                        committed_member_id=str(head["used_member"]),
                        frozen_belief_sha256=used_sha256,
                        **self._batch_state_summaries(
                            used_output,
                            ego_action_prefix=(int(shared_output["ego_actions"][0]),),
                            partner_action_prefix=(
                                int(shared_output["partner_actions"][0]),
                            ),
                        ),
                    ),
                    masked_branch_head_belief_sha256=masked_sha256,
                    used_branch_head_belief_sha256=used_sha256,
                    branch_head_particle_transitions=len(request.belief.particles),
                )
            output_by_request[request_index].append(
                PlanningSampleRolloutsV1(
                    sample=sample,
                    base=base,
                    probe_pairs=pairs,
                )
            )

        zero_cost_by_consultation: dict[str, PlanningBatchRolloutsV1] = {}
        for request_index, request in enumerate(request_tuple):
            batch = PlanningBatchRolloutsV1(
                samples=tuple(output_by_request[request_index]),
                compiled_batch_calls=0,
                active_batch_sizes=(),
                host_sync_inside_environment_loop=False,
                prepared_view=True,
            )
            cache_key = self._planning_batch_cache_key(
                belief_sha256=request_belief_sha256[request_index],
                history_sha256=request_history_sha256[request_index],
                samples=request.samples,
                remaining_steps=remaining_steps,
            )
            if cache_key in self._prepared_planning_batches:
                raise RuntimeError("R015 length bucket attempted duplicate device work.")
            self._prepared_planning_batches[cache_key] = batch
            zero_cost_by_consultation[request.consultation_id] = batch

        wall_seconds = time.perf_counter() - started
        particle_count = 4 * next(iter(particle_counts))
        true_environment_transitions = sample_lane_count * (
            13 * remaining_steps - 6
        )
        computed_environment_transitions_including_padding = sample_lane_count * (
            static_base_steps + 6 + 12 * static_suffix_steps
        )
        branch_head_particle_transitions = (
            6 * sample_lane_count * particle_count
        )
        key_derivation_stages = (
            base_outputs[0]["future_random_key_derivation"],
            shared_outputs[0]["future_random_key_derivation"],
            suffix_outputs[0]["future_random_key_derivation"],
        )
        derived_step_key_count = sum(
            int(value["derived_step_key_count"])
            for value in key_derivation_stages
        )
        requested_step_key_slot_count = sum(
            int(value["requested_step_key_slot_count"])
            for value in key_derivation_stages
        )
        device_execution = {
            "schema_version": "path_c_r015_planning_length_bucket_execution_v2",
            "remaining_environment_steps": remaining_steps,
            "consultation_point_count": len(request_tuple),
            "new_sample_count": sample_lane_count,
            "true_environment_transitions": true_environment_transitions,
            "computed_environment_transitions_including_padding": (
                computed_environment_transitions_including_padding
            ),
            "static_scan_step_counts": {
                "base": static_base_steps,
                "shared_probe_head": 1,
                "paired_suffix": static_suffix_steps,
            },
            "branch_head_particle_transitions": (
                branch_head_particle_transitions
            ),
            "compiled_batch_calls": 4,
            "active_batch_sizes": (
                sample_lane_count,
                6 * sample_lane_count,
                6 * sample_lane_count * particle_count,
                12 * sample_lane_count,
            ),
            "future_random_key_derivation": {
                "derivation_contract_id": R015_FUTURE_RANDOM_DERIVATION_ID,
                "unique_root_count": sample_lane_count,
                "derived_step_key_count": derived_step_key_count,
                "requested_step_key_slot_count": requested_step_key_slot_count,
                "reused_step_key_slot_count": (
                    requested_step_key_slot_count - derived_step_key_count
                ),
            },
            "wall_seconds": wall_seconds,
            "true_transitions_per_second": (
                true_environment_transitions / wall_seconds
                if wall_seconds > 0.0
                else None
            ),
            "host_sync_inside_environment_loop": False,
        }
        return R015PlanningLengthBucketResultV2(
            remaining_steps=remaining_steps,
            rollouts_by_consultation_id=zero_cost_by_consultation,
            device_execution=device_execution,
        )

    def rollout_planning_batch(
        self,
        *,
        samples: Sequence[PlanningBranchSampleV1],
        belief: StratifiedParticleBeliefV1 | R015OnlineParticleBeliefV2,
        history: OfficialHistoryV1,
        scripts: Sequence[ProbeScriptV1],
        remaining_steps: int,
    ) -> PlanningBatchRolloutsV1:
        """正式规划只消费跨咨询点长度分桶的 v2 结果。"""

        if not isinstance(belief, R015OnlineParticleBeliefV2):
            raise TypeError("R015 formal planning rejects the retired v1 belief.")
        # 显式绑定公共批量入口，防止规划层重新出现第三套过滤语义。
        if not callable(
            getattr(
                self.production_backend,
                "run_planning_branch_head_batch_v2",
                None,
            )
        ):
            raise RuntimeError("R015 formal planning lacks the shared v2 filter.")
        sample_tuple = tuple(samples)
        belief_sha256 = _belief_sha256(belief)
        history_sha256 = canonical_sha256(list(history.records))
        cache_key = self._planning_batch_cache_key(
            belief_sha256=belief_sha256,
            history_sha256=history_sha256,
            samples=sample_tuple,
            remaining_steps=remaining_steps,
        )
        prepared = self._prepared_planning_batches.pop(cache_key, None)
        if prepared is not None:
            return prepared
        consultation_id = canonical_sha256(
            [
                "path_c_r015_direct_planning_point_v2",
                belief_sha256,
                history_sha256,
                remaining_steps,
            ]
        )
        grouped = self.rollout_planning_length_bucket_v2(
            requests=(
                R015PlanningPointBatchRequestV2(
                    consultation_id=consultation_id,
                    filter_checkpoint_sha256=belief_sha256,
                    belief=belief,
                    history=history,
                    samples=sample_tuple,
                    remaining_steps=remaining_steps,
                ),
            ),
            scripts=scripts,
            remaining_steps=remaining_steps,
        )
        cached = grouped.rollouts_by_consultation_id[consultation_id]
        stored = self._prepared_planning_batches.pop(cache_key, None)
        if stored is not cached:
            raise RuntimeError("R015 direct planning lost its prepared device result.")
        device = grouped.device_execution
        return PlanningBatchRolloutsV1(
            samples=cached.samples,
            compiled_batch_calls=int(device["compiled_batch_calls"]),
            active_batch_sizes=tuple(device["active_batch_sizes"]),
            host_sync_inside_environment_loop=False,
        )

    def _split_probe_response(
        self,
        shared_state: R015BranchStateV1,
        *,
        probe_id: str,
        probe_step: int,
    ) -> tuple[R015BranchStateV1, R015BranchStateV1, Mapping[str, Any], OfficialHistoryV1]:
        token = shared_state.kernel.predicted_response_token
        if token is None:
            raise ValueError("R015 probe execution produced no registered response.")
        response = _response_record(
            token=token,
            probe_id=probe_id,
            probe_step=probe_step,
        )
        full_history = OfficialHistoryV1(
            (
                *shared_state.history.records[:-1],
                {
                    **shared_state.history.records[-1],
                    "response_summary_v1_derived": response,
                },
            )
        )
        projected = self.response_projection.project(
            full_history,
            probe_id=probe_id,
            probe_step=probe_step,
        )
        paired = apply_paired_response_update(
            shared_state.belief,
            projected,
            mask_updater=lambda current, masked_history: current,
            use_updater=lambda current, masked_history, current_response: (
                current.use_current_response(
                    current_response,
                    likelihood=self._response_likelihood,
                )
            ),
        )
        masked = replace(
            shared_state,
            belief=paired.masked,
            history=projected.masked_history,
        )
        prior_belief_evidence_sha256 = (
            shared_state.belief_evidence_sha256
            if shared_state.belief_evidence_sha256 is not None
            else _belief_sha256(shared_state.belief)
        )
        used = replace(
            shared_state,
            belief=paired.used,
            history=projected.masked_history,
            belief_evidence_sha256=canonical_sha256(
                {
                    "schema_version": "path_c_r015_response_belief_update_v2",
                    "before_sha256": prior_belief_evidence_sha256,
                    "probe_id": probe_id,
                    "probe_step": probe_step,
                    "response": response,
                    "masked_history_sha256": projected.masked_history.sha256,
                    "after_public_summary": paired.used.public_summary,
                }
            ),
        )
        return masked, used, response, projected.masked_history

    def run_a0_diagnostic_episode(
        self,
        *,
        partner_prototype_id: str,
        episode_seed: int,
        audit_unit_id: str,
    ) -> Mapping[str, Any]:
        """单独运行 seed-100 原始策略；该轨迹不得进入正式三组配对差。"""

        observations, _ = self.adapter.reset(episode_seed)
        snapshot = self.adapter.capture_state()
        partner_state = self.production_backend.initial_partner_state(
            partner_prototype_id
        )
        library = self.production_backend.initial_continuation_states()
        root_key = canonical_sha256(
            ["r015_a0_diagnostic_v1", audit_unit_id, episode_seed]
        )
        trajectory: list[Mapping[str, Any]] = []
        raw_return = 0.0
        correct_deliveries = 0
        wrong_deliveries = 0
        for environment_step in range(400):
            step_key = derive_controller_key(
                root_key,
                "a0_environment",
                environment_step,
            )
            next_library: dict[str, Any] = {}
            member_actions: dict[str, str] = {}
            for member_id in self.production_backend.continuation_controller.member_ids:
                member = self.production_backend.act_policy_member(
                    member_id,
                    snapshot.raw_obs["agent_1"],
                    library.by_member_id[member_id],
                    derive_controller_key(step_key, "continuation", member_id),
                    episode_start=environment_step == 0,
                )
                next_library[member_id] = member.next_recurrent_state
                member_actions[member_id] = member.action_id
            partner = self.production_backend.act_policy_member(
                partner_prototype_id,
                snapshot.raw_obs["agent_0"],
                partner_state,
                derive_controller_key(step_key, "partner_action"),
                episode_start=environment_step == 0,
            )
            ego_action = member_actions[self.production_backend.baseline_member_id]
            source_snapshot = _clone_snapshot_with_key(snapshot, step_key)
            result = self.adapter.step_joint_from_state(
                source_snapshot,
                agent_0_action=OCV2_ACTION_INDEX[partner.action_id],
                agent_1_action=OCV2_ACTION_INDEX[ego_action],
            )
            reward = float(result.step.rewards["agent_0"])
            if not math.isclose(
                reward,
                float(result.step.rewards["agent_1"]),
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError("R015 A0 diagnostic lacks one shared team reward.")
            correct = int(np.asarray(result.snapshot.state.new_correct_delivery))
            wrong = int(
                _wrong_delivery_event(
                    source_snapshot,
                    result.snapshot,
                    ego_action=ego_action,
                    partner_action=partner.action_id,
                )
            )
            correct_deliveries += correct
            wrong_deliveries += wrong
            raw_return += reward
            trajectory.append(
                {
                    "environment_step": environment_step,
                    "ego_action": ego_action,
                    "partner_action": partner.action_id,
                    "raw_team_reward": reward,
                    "done": bool(result.step.dones["__all__"]),
                    "environment_random_key": step_key,
                    "continuation_member_actions": member_actions,
                    "continuation_state_before_sha256": _library_states_sha256(
                        library
                    ),
                    "continuation_state_after_sha256": _library_states_sha256(
                        ContinuationLibraryStatesV1(next_library)
                    ),
                }
            )
            snapshot = result.snapshot
            partner_state = partner.next_recurrent_state
            library = ContinuationLibraryStatesV1(next_library)
        if trajectory[-1]["done"] is not True:
            raise ValueError("R015 A0 diagnostic did not end at environment step 400.")
        return {
            "schema_version": "path_c_r015_a0_diagnostic_episode_v1",
            "scientific_readout_allowed": False,
            "formal_pairing_eligible": False,
            "audit_unit_id": audit_unit_id,
            "partner_prototype_id": partner_prototype_id,
            "episode_seed": episode_seed,
            "raw_episode_return": raw_return,
            "correct_delivery_count": correct_deliveries,
            "wrong_delivery_count": wrong_deliveries,
            "trajectory": trajectory,
            "trajectory_sha256": canonical_sha256(trajectory),
        }

    def build_type_a_first_consultation(
        self,
        *,
        partner_prototype_id: str,
        audit_key: str,
        particles_per_prototype: int = 64,
    ) -> tuple[OfficialHistoryV1, StratifiedParticleBeliefV1]:
        """构造不进入设计证据的第一处真实咨询状态。"""

        true_kernel = self._initial_kernel(
            partner_prototype_id,
            derive_controller_key(audit_key, "true_episode"),
        )
        belief = self.initialize_belief(
            official_initial_observation=true_kernel.snapshot.raw_obs["agent_1"],
            particles_per_prototype=particles_per_prototype,
            resampling_timing="adaptive_ess_below_half_v1",
            initialization_key=derive_controller_key(audit_key, "filter"),
        )
        history = OfficialHistoryV1(
            (
                _history_record(
                    observation=true_kernel.snapshot.raw_obs["agent_1"],
                    ego_action=None,
                    reward=0.0,
                    done=True,
                ),
            )
        )
        branch = R015BranchStateV1(
            kernel=true_kernel,
            belief=belief,
            history=history,
            environment_step=0,
        )
        first = self._run(
            branch,
            branch_key=derive_controller_key(audit_key, "first_step"),
            total_steps=1,
            update_belief_online=True,
        )
        return first.state.history, first.state.belief

    def run_paired_block(
        self,
        *,
        partner_prototype_id: str,
        episode_seed: int,
        audit_unit_id: str,
        filter_initialization_key: str,
        particles_per_prototype: int,
        resampling_timing: str,
        planning_branches: int,
        episode_random_root_key: str | None = None,
        formal_random_coordinates: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """运行一个三组配对块；A0 只可由调用方另行诊断。"""

        # 规划输出缓存只服务设计阶段显式的 R/2R 重建。真实块与其独立重放
        # 必须各自重新执行规划；这里只清结果，不清 JIT executable 缓存。
        self._prepared_planning_batches.clear()

        observations, _ = self.adapter.reset(episode_seed)
        initial_snapshot = self.adapter.capture_state()
        initial_state_sha256 = _snapshot_sha256(initial_snapshot)
        true_kernel = R015ParticleKernelStateV1(
            snapshot=initial_snapshot,
            partner_prototype_id=partner_prototype_id,
            partner_recurrent_state=self.production_backend.initial_partner_state(
                partner_prototype_id
            ),
            continuation_states=self.production_backend.initial_continuation_states(),
            environment_step=0,
        )
        if not isinstance(filter_initialization_key, str) or len(
            filter_initialization_key
        ) != 64:
            raise ValueError("R015 paired block requires an identity-free filter key.")
        filter_key = derive_controller_key(
            filter_initialization_key,
            "actual_filter_initialization",
        )
        try:
            belief = self.initialize_belief(
                official_initial_observation=observations["agent_1"],
                particles_per_prototype=particles_per_prototype,
                resampling_timing=resampling_timing,
                initialization_key=filter_key,
            )
        except _R015OnlineFilterZeroSupportError as error:
            raise R015MechanicalBlockInvalidError(
                "R015 online filter lost support at the true episode opening."
            ) from error
        history = OfficialHistoryV1(
            (
                {
                    "official_local_observation": observations["agent_1"],
                    "episode_boundaries": [True],
                },
            )
        )
        branch = R015BranchStateV1(
            kernel=true_kernel,
            belief=belief,
            history=history,
            environment_step=0,
        )
        controller = R015SequentialControllerV1(
            planner=R015SequentialPlannerV1(
                probe_scripts=default_r015_probe_scripts(),
                branches_per_candidate=planning_branches,
            ),
            continuation_controller=self.production_backend.continuation_controller,
        )
        root_key = (
            canonical_sha256(
                [
                    "r015_paired_block_v2",
                    audit_unit_id,
                    episode_seed,
                    partner_prototype_id,
                ]
            )
            if episode_random_root_key is None
            else str(episode_random_root_key)
        )
        if not _is_sha256(root_key):
            raise ValueError("R015 paired execution requires a SHA-256 episode root key.")

        def controller_key(
            purpose: str,
            environment_step: int,
            *,
            partner_id: str = partner_prototype_id,
            branch_index: int = 0,
        ) -> str:
            if formal_random_coordinates is None:
                return derive_controller_key(root_key, purpose, environment_step)
            from experiments.overcooked_v2.path_c_r015 import (
                _derive_r015_random_key,
            )

            if formal_random_coordinates.get("audit_unit_id") != audit_unit_id or (
                int(formal_random_coordinates.get("episode_seed", -1))
                != episode_seed
            ):
                raise ValueError("R015 formal random coordinates changed the block identity.")
            return _derive_r015_random_key(
                audit_unit_id=audit_unit_id,
                partner_prototype_id=partner_id,
                episode_seed=episode_seed,
                purpose=purpose,
                environment_step=environment_step,
                branch_index=branch_index,
            )
        prefix: list[Mapping[str, Any]] = []
        consultations: list[Mapping[str, Any]] = []
        safety_evidence: list[Mapping[str, Any]] = []
        planning_fork_steps = 0
        planning_branch_head_particle_transitions = 0
        planning_paired_suffix_batched_lane_time_steps = 0
        planning_compiled_batch_calls = 0
        planning_active_batch_sizes: list[int] = []
        execution_throughput_segments: list[Mapping[str, Any]] = []
        no_probe_counts = {
            "candidate_evaluation_count": 0,
            "safety_rejection_count": 0,
            "nonpositive_score_count": 0,
            "window_expiration_count": 0,
        }
        firing_decision = None
        final_no_probe_reason: str | None = None
        last_masked_reference = "base"
        while branch.environment_step < 400:
            environment_step = branch.environment_step
            forced_action: str | None = None
            if environment_step in R015_CONSULTATION_STEPS and not (
                controller.probe_used or controller.consultation_closed
            ):
                try:
                    decision = controller.consult(
                        history=branch.history,
                        belief=branch.belief,
                        environment_step=environment_step,
                        planning_key=controller_key(
                            "value_planning", environment_step
                        ),
                        score_key=controller_key(
                            "score_estimation", environment_step
                        ),
                        safety_key=controller_key(
                            "safety_validation", environment_step
                        ),
                        executor=self,
                        safety_evaluator=lambda probe_id, current_history, current_belief, key: self.evaluate_safety(
                            probe_id,
                            current_history,
                            current_belief,
                            key,
                            formal_random_coordinates=(
                                None
                                if formal_random_coordinates is None
                                else {
                                    "audit_unit_id": audit_unit_id,
                                    "episode_seed": episode_seed,
                                    "environment_step": environment_step,
                                }
                            ),
                        ),
                    )
                except _R015OnlineFilterZeroSupportError as error:
                    raise R015MechanicalBlockInvalidError(
                        "R015 planning branch lost finite-particle support."
                    ) from error
                consultations.append(decision.consultation.to_payload())
                final_no_probe_reason = decision.no_probe_reason
                last_masked_reference = decision.a1_action_id
                no_probe_counts["candidate_evaluation_count"] += 6
                planning_cost = decision.consultation.cost_accounting
                planning_fork_steps += int(
                    planning_cost["new_real_environment_transitions"]
                )
                planning_branch_head_particle_transitions += int(
                    planning_cost["new_branch_head_particle_transitions"]
                )
                planning_paired_suffix_batched_lane_time_steps += int(
                    planning_cost[
                        "new_paired_suffix_batched_lane_time_steps"
                    ]
                )
                planning_device = decision.consultation.device_execution
                if planning_device.get(
                    "host_sync_inside_environment_loop"
                ) is not False:
                    raise RuntimeError(
                        "R015 planning synchronized inside an environment scan."
                    )
                planning_compiled_batch_calls += int(
                    planning_device["compiled_batch_calls"]
                )
                planning_active_batch_sizes.extend(
                    int(value) for value in planning_device["active_batch_sizes"]
                )
                if decision.safety is not None:
                    safety_evidence.extend(decision.safety.comparisons)
                if decision.probe_fired:
                    firing_decision = decision
                    break
                if decision.no_probe_reason == "safety_rejected":
                    no_probe_counts["safety_rejection_count"] += 1
                elif decision.no_probe_reason == "support_incompatible":
                    # 后验支持不足和安全分支拒绝都属于同一类“未能获得安全
                    # 放行”的解释；合同没有为支持不足另设第五类计数。
                    no_probe_counts["safety_rejection_count"] += 1
                elif decision.no_probe_reason == "no_safe_candidate":
                    no_probe_counts["safety_rejection_count"] += 1
                elif decision.no_probe_reason == "non_positive_score":
                    no_probe_counts["nonpositive_score_count"] += 1
                elif decision.no_probe_reason == "window_expired":
                    no_probe_counts["window_expiration_count"] += 1
                if decision.shared_masked_action_id != "base":
                    forced_action = decision.shared_masked_action_id
            future_consultations = tuple(
                step
                for step in R015_CONSULTATION_STEPS
                if step > environment_step
            )
            stop_step = (
                future_consultations[0]
                if future_consultations
                and not controller.probe_used
                and not controller.consultation_closed
                else 400
            )
            segment_steps = stop_step - environment_step
            if segment_steps <= 0:
                raise RuntimeError("R015 execution segment did not advance time.")
            one_step = self._run(
                branch,
                branch_key=derive_controller_key(
                    root_key, "actual_episode", environment_step
                ),
                total_steps=segment_steps,
                first_forced_action=forced_action,
                explicit_step_keys=tuple(
                    derive_controller_key(
                        derive_controller_key(
                            root_key,
                            "actual_episode",
                            step,
                        ),
                        "future_environment",
                        0,
                    )
                    for step in range(environment_step, stop_step)
                ),
            )
            prefix.extend(one_step.trajectory)
            execution_throughput_segments.append(one_step.throughput)
            branch = one_step.state

        replay_by_group = {
            group: {"verification_status": "pending_independent_replay"}
            for group in ("A1", "A2-mask", "A2-use")
        }
        if firing_decision is None:
            if len(prefix) != 400:
                raise ValueError("R015 no-fire block did not complete 400 steps.")
            groups = build_r015_no_fire_trace_groups(
                prefix,
                replay_verification_by_group=replay_by_group,
                masked_value_reference={"selected_action_id": last_masked_reference},
            )
            lockstep = validate_r015_lockstep_groups(
                groups,
                probe_step=None,
                selected_probe_actions=(),
            )
            probe_step = None
            selected_probe_id = None
        else:
            probe_step = branch.environment_step
            selected_probe_id = str(firing_decision.a2_probe_id)
            reference = firing_decision.a1_action_id
            post_key = derive_controller_key(root_key, "post_fire", probe_step)
            remaining = 400 - probe_step
            if not isinstance(branch.belief, R015OnlineParticleBeliefV2):
                raise TypeError("R015 paired execution requires the online v2 belief.")
            stack_filter_states = getattr(
                self.production_backend,
                "stack_online_filter_states_v2",
                None,
            )
            if not callable(stack_filter_states):
                raise RuntimeError("R015 paired execution lacks the v2 filter stacker.")
            first_step_key = derive_controller_key(
                post_key, "future_environment", 0
            )
            head_batch = self.production_backend.run_online_trajectory_batch(
                adapter=self.adapter,
                kernels=(
                    copy.deepcopy(branch.kernel),
                    copy.deepcopy(branch.kernel),
                ),
                filter_state=stack_filter_states(
                    (
                        branch.belief.device_filter_state,
                        branch.belief.device_filter_state,
                    )
                ),
                branch_keys=(post_key, post_key),
                total_steps=1,
                key_offset=0,
                forced_first_actions=(
                    None if reference == "base" else reference,
                    selected_probe_id,
                ),
                resampling_timing=branch.belief.resampling_timing,
                active_steps=(1, 1),
                explicit_step_keys=((first_step_key,), (first_step_key,)),
                return_recurrent_state_trace=True,
            )
            a1_head = self._run(
                copy.deepcopy(branch),
                branch_key=post_key,
                total_steps=1,
                first_forced_action=(None if reference == "base" else reference),
                explicit_step_keys=(first_step_key,),
                _precomputed_online_batch=head_batch,
                _online_lane_index=0,
                _account_precomputed_batch_cost=True,
            )
            a2_shared = self._run(
                copy.deepcopy(branch),
                branch_key=post_key,
                total_steps=1,
                first_forced_action=selected_probe_id,
                explicit_step_keys=(first_step_key,),
                _precomputed_online_batch=head_batch,
                _online_lane_index=1,
                _account_precomputed_batch_cost=False,
            )
            try:
                (
                    masked_start,
                    used_start,
                    response,
                    masked_history,
                ) = self._split_probe_response(
                    a2_shared.state,
                    probe_id=selected_probe_id,
                    probe_step=probe_step,
                )
            except _R015OnlineFilterZeroSupportError as error:
                raise R015MechanicalBlockInvalidError(
                    "R015 current probe response lost finite-particle support."
                ) from error
            suffix_steps = remaining - 1
            suffix_keys = tuple(
                derive_controller_key(
                    post_key,
                    "future_environment",
                    offset,
                )
                for offset in range(1, remaining)
            )
            suffix_starts = (a1_head.state, masked_start, used_start)
            suffix_batch = self.production_backend.run_online_trajectory_batch(
                adapter=self.adapter,
                kernels=tuple(
                    copy.deepcopy(start.kernel) for start in suffix_starts
                ),
                filter_state=stack_filter_states(
                    tuple(
                        start.belief.device_filter_state
                        for start in suffix_starts
                    )
                ),
                branch_keys=(post_key, post_key, post_key),
                total_steps=suffix_steps,
                key_offset=1,
                forced_first_actions=(None, None, None),
                resampling_timing=branch.belief.resampling_timing,
                active_steps=(suffix_steps, suffix_steps, suffix_steps),
                explicit_step_keys=(suffix_keys, suffix_keys, suffix_keys),
                return_recurrent_state_trace=True,
            )
            suffix_results = tuple(
                self._run(
                    start,
                    branch_key=post_key,
                    total_steps=suffix_steps,
                    key_offset=1,
                    explicit_step_keys=suffix_keys,
                    _precomputed_online_batch=suffix_batch,
                    _online_lane_index=lane,
                    _account_precomputed_batch_cost=lane == 0,
                )
                for lane, start in enumerate(suffix_starts)
            )
            a1_suffix, masked, used = suffix_results

            def combined_throughput(
                first: Mapping[str, Any],
                second: Mapping[str, Any],
            ) -> Mapping[str, Any]:
                first_payload = dict(first)
                second_payload = dict(second)
                wall = float(first_payload.get("wall_seconds", 0.0)) + float(
                    second_payload.get("wall_seconds", 0.0)
                )
                transitions = int(
                    first_payload.get("true_environment_transitions", 0)
                ) + int(second_payload.get("true_environment_transitions", 0))
                computed = int(
                    first_payload.get(
                        "computed_environment_transitions_including_masked_padding",
                        0,
                    )
                ) + int(
                    second_payload.get(
                        "computed_environment_transitions_including_masked_padding",
                        0,
                    )
                )
                filter_particle_transitions = int(
                    first_payload.get(
                        "filter_particle_environment_transitions", 0
                    )
                ) + int(
                    second_payload.get(
                        "filter_particle_environment_transitions", 0
                    )
                )
                computed_filter_particle_transitions = int(
                    first_payload.get(
                        "computed_filter_particle_environment_transitions", 0
                    )
                ) + int(
                    second_payload.get(
                        "computed_filter_particle_environment_transitions", 0
                    )
                )
                return {
                    "execution_mode": "paired_online_device_scan_v2",
                    "true_environment_transitions": transitions,
                    "computed_environment_transitions_including_masked_padding": (
                        computed
                    ),
                    "filter_particle_environment_transitions": (
                        filter_particle_transitions
                    ),
                    "computed_filter_particle_environment_transitions": (
                        computed_filter_particle_transitions
                    ),
                    "compiled_batch_calls": int(
                        first_payload.get("compiled_batch_calls", 0)
                    )
                    + int(second_payload.get("compiled_batch_calls", 0)),
                    "jit_compilations": int(
                        first_payload.get("jit_compilations", 0)
                    )
                    + int(second_payload.get("jit_compilations", 0)),
                    "active_lane_batch_width": 3,
                    "wall_seconds": wall,
                    "true_transitions_per_second": (
                        transitions / wall if wall > 0.0 else None
                    ),
                    "host_sync_inside_environment_loop": False,
                }

            a1 = _BranchExecutionV1(
                state=a1_suffix.state,
                raw_return=a1_head.raw_return + a1_suffix.raw_return,
                trajectory=(*a1_head.trajectory, *a1_suffix.trajectory),
                future_random_keys=(
                    *a1_head.future_random_keys,
                    *a1_suffix.future_random_keys,
                ),
                wrong_delivery_detected=(
                    a1_head.wrong_delivery_detected
                    or a1_suffix.wrong_delivery_detected
                ),
                throughput=combined_throughput(
                    a1_head.throughput,
                    a1_suffix.throughput,
                ),
            )
            suffixes = {
                "A1": a1.trajectory,
                "A2-mask": (*a2_shared.trajectory, *masked.trajectory),
                "A2-use": (*a2_shared.trajectory, *used.trajectory),
            }
            execution_throughput_segments.extend(
                (
                    a1.throughput,
                    a2_shared.throughput,
                    masked.throughput,
                    used.throughput,
                )
            )
            groups = build_r015_fire_trace_groups(
                prefix,
                suffixes,
                probe_step=probe_step,
                selected_probe_id=selected_probe_id,
                selected_probe_actions=(selected_probe_id,),
                replay_verification_by_group=replay_by_group,
                masked_value_reference={"selected_action_id": reference},
                masked_history=masked_history.to_payload(),
                current_probe_response=response,
                belief_common_input={"masked_history_sha256": masked_history.sha256},
                continuation_common_input={
                    "controller_id": self.continuation_controller_id,
                    "masked_history_sha256": masked_history.sha256,
                },
                recurrent_common_input={
                    "continuation_state_before_sha256": _library_states_sha256(
                        a2_shared.state.kernel.continuation_states
                    )
                },
            )
            lockstep = validate_r015_lockstep_groups(
                groups,
                probe_step=probe_step,
                selected_probe_actions=(selected_probe_id,),
            )
        total_execution_wall_seconds = math.fsum(
            float(item.get("wall_seconds", 0.0))
            for item in execution_throughput_segments
        )
        total_execution_transitions = sum(
            int(item.get("true_environment_transitions", 0))
            for item in execution_throughput_segments
        )
        computed_execution_transitions = sum(
            int(
                item.get(
                    "computed_environment_transitions_including_masked_padding",
                    item.get("true_environment_transitions", 0),
                )
            )
            for item in execution_throughput_segments
        )
        execution_compiled_calls = sum(
            int(item.get("compiled_batch_calls", 0))
            for item in execution_throughput_segments
        )
        execution_jit_compilations = sum(
            int(item.get("jit_compilations", 0))
            for item in execution_throughput_segments
        )
        execution_filter_particle_transitions = sum(
            int(item.get("filter_particle_environment_transitions", 0))
            for item in execution_throughput_segments
        )
        computed_execution_filter_particle_transitions = sum(
            int(
                item.get(
                    "computed_filter_particle_environment_transitions", 0
                )
            )
            for item in execution_throughput_segments
        )
        block_payload = {
            "schema_version": "path_c_r015_executed_paired_block_v2",
            "scientific_readout_allowed": False,
            "evidence_stage": "executed_requires_independent_replay",
            "formal_pairing_eligible": False,
            "filter_algorithm_id": R015_FILTER_ALGORITHM_ID_V2,
            "filter_key_contract": R015_FILTER_KEY_CONTRACT_V2,
            "continuation_controller_id": self.continuation_controller_id,
            "audit_unit_id": audit_unit_id,
            "partner_prototype_id": partner_prototype_id,
            "episode_seed": episode_seed,
            "initial_state_sha256": initial_state_sha256,
            "episode_random_root_key": root_key,
            "probe_fired": firing_decision is not None,
            "no_probe_reason": (
                None if firing_decision is not None else final_no_probe_reason
            ),
            "probe_step": probe_step,
            "selected_probe_id": selected_probe_id,
            "groups": groups,
            "lockstep_verification": lockstep,
            "consultations": consultations,
            "consultations_sha256": canonical_sha256(consultations),
            "safety_comparisons": safety_evidence,
            "safety_comparisons_sha256": canonical_sha256(safety_evidence),
            "safety_branch_count": sum(
                len(tuple(item.get("branches", ())))
                for item in safety_evidence
            ),
            "firing_indicator_binding": {
                "computed_from_shared_prefix": True,
                "identical_across_groups": True,
                "probe_fired": firing_decision is not None,
                "probe_step": probe_step,
                "shared_prefix_sha256": canonical_sha256(
                    prefix if firing_decision is not None else tuple(prefix)
                ),
            },
            "planning_fork_environment_steps": planning_fork_steps,
            "planning_real_environment_transitions": planning_fork_steps,
            "planning_branch_head_particle_transitions": (
                planning_branch_head_particle_transitions
            ),
            "planning_paired_suffix_batched_lane_time_steps": (
                planning_paired_suffix_batched_lane_time_steps
            ),
            "planning_compiled_batch_calls": planning_compiled_batch_calls,
            "planning_active_batch_sizes": planning_active_batch_sizes,
            "execution_throughput": {
                "true_environment_transitions": total_execution_transitions,
                "computed_environment_transitions_including_masked_padding": (
                    computed_execution_transitions
                ),
                "compiled_batch_calls": execution_compiled_calls,
                "jit_compilations": execution_jit_compilations,
                "filter_particle_environment_transitions": (
                    execution_filter_particle_transitions
                ),
                "computed_filter_particle_environment_transitions": (
                    computed_execution_filter_particle_transitions
                ),
                "segment_count": len(execution_throughput_segments),
                "active_lane_batch_widths": tuple(
                    int(item.get("active_lane_batch_width", 1))
                    for item in execution_throughput_segments
                ),
                "wall_seconds": total_execution_wall_seconds,
                "true_transitions_per_second": (
                    total_execution_transitions / total_execution_wall_seconds
                    if total_execution_wall_seconds > 0.0
                    else None
                ),
                "host_sync_inside_environment_loop": any(
                    bool(item.get("host_sync_inside_environment_loop", True))
                    for item in execution_throughput_segments
                ),
            },
            "zero_probe_interpretability_counts": no_probe_counts,
            "formal_telemetry_projection": {
                "candidate_evaluation_count": no_probe_counts[
                    "candidate_evaluation_count"
                ],
                "safety_rejection_count": no_probe_counts[
                    "safety_rejection_count"
                ],
                "non_positive_score_count": no_probe_counts[
                    "nonpositive_score_count"
                ],
                "window_expired_count": no_probe_counts[
                    "window_expiration_count"
                ],
            },
        }
        return {
            **block_payload,
            "executed_block_sha256": canonical_sha256(block_payload),
        }

    def produce_formal_paired_block(
        self,
        *,
        request: R015FormalBlockRequestV1,
        preregistration: Any,
    ) -> R015FormalBlockProductionResultV1:
        """生成、独立重放并裁决一个正式 v2 配对块。"""

        from experiments.overcooked_v2.path_c_r015 import (
            _derive_r015_random_key,
            validate_r015_paired_block,
        )

        if not isinstance(request, R015FormalBlockRequestV1):
            raise TypeError("R015 formal production requires a frozen block request.")
        if request.ego_position != 1:
            raise ValueError("R015 formal production fixes the ego cook at agent_1.")
        if isinstance(request.mechanical_attempt_index, bool) or not (
            0
            <= int(request.mechanical_attempt_index)
            < R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT
        ):
            raise ValueError("R015 formal request uses an unregistered replacement attempt.")
        support_members = preregistration._require_frozen_support()
        if request.prototype_id not in support_members:
            raise ValueError("R015 formal request uses a prototype outside frozen support.")
        prototype_order = tuple(
            candidate.candidate_id
            for candidate in preregistration.support_spec.candidates
        )
        if set(prototype_order) != set(support_members):
            raise ValueError("R015 formal prototype order differs from frozen support.")
        sequence_index = (request.round_index - 1) * len(prototype_order) + (
            prototype_order.index(request.prototype_id) + 1
        )
        filter_initialization_key = canonical_sha256(
            [
                "r015_formal_filter_initialization_v4",
                sequence_index,
                request.episode_seed,
            ]
        )
        episode_random_key = _derive_r015_random_key(
            audit_unit_id=request.audit_unit_id,
            partner_prototype_id=request.prototype_id,
            episode_seed=request.episode_seed,
            purpose="actual_episode",
            environment_step=0,
            branch_index=0,
        )
        arguments = {
            "partner_prototype_id": request.prototype_id,
            "episode_seed": request.episode_seed,
            "audit_unit_id": request.audit_unit_id,
            "filter_initialization_key": filter_initialization_key,
            "particles_per_prototype": (
                preregistration.controller.particles_per_prototype
            ),
            "resampling_timing": preregistration.controller.resampling_timing,
            "planning_branches": (
                preregistration.controller.planning_branches_per_candidate
            ),
            "episode_random_root_key": episode_random_key,
            "formal_random_coordinates": {
                "audit_unit_id": request.audit_unit_id,
                "episode_seed": request.episode_seed,
            },
        }
        if any(
            arguments[field] is None
            for field in (
                "particles_per_prototype",
                "resampling_timing",
                "planning_branches",
            )
        ):
            raise ValueError("R015 formal controller numerics are not frozen.")
        actual_execution_completed = False
        try:
            execution_started = time.monotonic()
            executed = self.run_paired_block(**arguments)
            actual_execution_completed = True
            execution_wall_seconds = max(
                time.monotonic() - execution_started, 1.0e-12
            )
            if self.formal_freeze_identity_verifier is None:
                raise ValueError(
                    "R015 formal independent replay lacks frozen identity verification."
                )
            self.formal_freeze_identity_verifier()
            replay_started = time.monotonic()
            independently_replayed = self.run_paired_block(**arguments)
            replay_wall_seconds = max(time.monotonic() - replay_started, 1.0e-12)
            actual_components = _formal_components_from_executed(
                executed,
                request=request,
                preregistration=preregistration,
                support_members=support_members,
            )
            replay_components = _formal_components_from_executed(
                independently_replayed,
                request=request,
                preregistration=preregistration,
                support_members=support_members,
            )
            replay_backend = build_r015_replay_backend(
                {
                    "_r015_production_backend": self.production_backend,
                    "_r015_recomputed_formal_components": replay_components,
                    "_r015_preregistration": preregistration,
                    "_r015_support_members": support_members,
                }
            )

            def trace_verifier(trace, context):
                return verify_r015_trace_manifest(
                    trace,
                    {**dict(context), "_r015_replay_backend": replay_backend},
                )

            def decision_verifier(evidence, context):
                return verify_r015_probe_decision(
                    evidence,
                    {**dict(context), "_r015_replay_backend": replay_backend},
                )

            def safety_verifier(comparison, context):
                return verify_r015_safety_comparison(
                    comparison,
                    {**dict(context), "_r015_replay_backend": replay_backend},
                )

            trace_manifest = copy.deepcopy(actual_components["trace_manifest"])
            trace_manifest["replay_verification"] = trace_verifier(
                trace_manifest,
                actual_components["trace_replay_context"],
            )
            paired_block = {
                **copy.deepcopy(actual_components["paired_block_without_trace"]),
                "trace_manifest": trace_manifest,
                "trace_manifest_sha256": canonical_sha256(trace_manifest),
                "mechanical_execution_telemetry": {
                    "schema_version": (
                        "path_c_r015_formal_production_telemetry_v1"
                    ),
                    "actual_execution_wall_seconds": execution_wall_seconds,
                    "independent_replay_wall_seconds": replay_wall_seconds,
                    "independent_replay_reused_compiled_device_programs": True,
                    "actual_execution": copy.deepcopy(
                        dict(executed.get("execution_throughput", {}))
                    ),
                    "independent_replay_execution": copy.deepcopy(
                        dict(
                            independently_replayed.get(
                                "execution_throughput", {}
                            )
                        )
                    ),
                    "actual_planning_real_environment_transitions": int(
                        executed["planning_real_environment_transitions"]
                    ),
                    "independent_replay_planning_real_environment_transitions": int(
                        independently_replayed[
                            "planning_real_environment_transitions"
                        ]
                    ),
                    "effect_values_in_telemetry": False,
                },
            }
            validate_r015_paired_block(
                paired_block,
                preregistration=preregistration,
                support_members=support_members,
                trace_replay_verifier=trace_verifier,
                safety_branch_evidence_verifier=safety_verifier,
                decision_evidence_verifier=decision_verifier,
            )
        except R015MechanicalBlockInvalidError as error:
            if actual_execution_completed:
                # 真执行已经完成后，专用异常只能来自独立重放。此时它证明
                # 两次确定性执行不一致，必须停止正式链，不能请求替换这一块。
                raise
            return R015FormalBlockProductionResultV1(
                mechanically_valid=False,
                paired_block=None,
                invalid_reason=f"{type(error).__name__}: {error}",
            )
        return R015FormalBlockProductionResultV1(
            mechanically_valid=True,
            paired_block=paired_block,
            invalid_reason=None,
        )

    def evaluate_safety(
        self,
        probe_id: str,
        history: OfficialHistoryV1,
        belief: R015OnlineParticleBeliefV2,
        safety_key: str,
        *,
        repetitions: int = 279,
        formal_random_coordinates: Mapping[str, Any] | None = None,
    ) -> SafetyDecisionV1:
        """按支持原型用固定父粒子槽微批运行全部独立安全后缀。

        每条安全车道都携带四个原型的完整在线粒子后验。设备批宽按 4096 个
        父粒子槽切分；一个微批仍在单个 ``lax.scan`` 中跑完整后缀，只有完整
        后缀结束后才回到宿主。切分不改变分支键、粒子轮换顺序或判定次数。
        """

        if not isinstance(belief, R015OnlineParticleBeliefV2):
            raise TypeError("R015 formal safety requires the online v2 belief.")
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or not 1 <= repetitions <= 279
        ):
            raise ValueError("R015 safety repetitions must lie in 1,...,279.")
        stack_filter_states = getattr(
            self.production_backend,
            "stack_online_filter_states_v2",
            None,
        )
        if not callable(stack_filter_states):
            raise RuntimeError("R015 safety requires the shared v2 filter stacker.")
        if len(self.prototype_ids) != R015_FILTER_PROTOTYPE_COUNT:
            raise ValueError("R015 safety requires exactly four support prototypes.")
        (
            parent_slot_batch_width,
            lanes_per_microbatch,
            formal_schedule,
        ) = self._filter_parent_slot_schedule(
            particles_per_prototype=belief.particles_per_prototype,
            type_a_parent_slot_batch_width=None,
        )
        if not formal_schedule or (
            parent_slot_batch_width
            != R015_FILTER_FORMAL_PARENT_SLOT_BATCH_WIDTH
        ):
            raise RuntimeError("R015 safety changed its fixed parent-slot schedule.")
        parent_slots_per_lane = (
            R015_FILTER_PROTOTYPE_COUNT * belief.particles_per_prototype
        )
        comparisons: list[Mapping[str, Any]] = []
        remaining_steps = 400 - (len(history.records) - 1)
        if not 1 <= remaining_steps <= 400:
            raise ValueError("R015 safety history is outside the 400-step episode.")
        for prototype_id in self.prototype_ids:
            support = tuple(
                particle
                for particle in belief.particles
                if particle.prototype_id == prototype_id and particle.weight > 0.0
            )
            branch_evidence: list[Mapping[str, Any]] = []
            wrong_count = 0
            filter_closed_count = 0
            throughput: Mapping[str, Any] = {
                "device_execution_id": "r015_safety_jit_scan_vmap_v2",
                "microbatch_schedule_id": R015_FILTER_MICROBATCH_SCHEDULE_ID,
                "configured_parent_slot_batch_width": (
                    parent_slot_batch_width
                ),
                "parent_particle_slots_per_lane": parent_slots_per_lane,
                "lanes_per_full_microbatch": lanes_per_microbatch,
                "microbatch_count": 0,
                "microbatch_active_lane_counts": [],
                "microbatch_active_parent_particle_slot_widths": [],
                "microbatch_wall_seconds": [],
                "true_environment_transitions": 0,
                "computed_environment_transitions_including_masked_padding": 0,
                "filter_particle_environment_transitions": 0,
                "computed_filter_particle_environment_transitions": 0,
                "compiled_batch_calls": 0,
                "jit_compilations": 0,
                "active_lane_batch_width": 0,
                "wall_seconds": 0.0,
                "true_transitions_per_second": None,
                "computed_transitions_per_second": None,
                "host_sync_inside_environment_loop": False,
            }
            if support:
                selected_particles = tuple(
                    support[repetition % len(support)]
                    for repetition in range(repetitions)
                )
                if formal_random_coordinates is None:
                    branch_keys = tuple(
                        derive_controller_key(
                            safety_key,
                            "safety_branch",
                            prototype_id,
                            repetition,
                        )
                        for repetition in range(repetitions)
                    )
                else:
                    from experiments.overcooked_v2.path_c_r015 import (
                        _derive_r015_random_key,
                    )

                    branch_keys = tuple(
                        _derive_r015_random_key(
                            audit_unit_id=str(
                                formal_random_coordinates["audit_unit_id"]
                            ),
                            partner_prototype_id=prototype_id,
                            episode_seed=int(
                                formal_random_coordinates["episode_seed"]
                            ),
                            purpose="safety_validation",
                            environment_step=int(
                                formal_random_coordinates["environment_step"]
                            ),
                            branch_index=repetition,
                        )
                        for repetition in range(repetitions)
                    )
                trace_batches: list[Mapping[str, Any]] = []
                microbatch_throughputs: list[Mapping[str, Any]] = []
                for microbatch_start in range(
                    0, repetitions, lanes_per_microbatch
                ):
                    microbatch_stop = min(
                        repetitions,
                        microbatch_start + lanes_per_microbatch,
                    )
                    current_particles = selected_particles[
                        microbatch_start:microbatch_stop
                    ]
                    current_branch_keys = branch_keys[
                        microbatch_start:microbatch_stop
                    ]
                    active_lane_count = len(current_particles)
                    stacked_filter_state = stack_filter_states(
                        tuple(
                            belief.device_filter_state
                            for _ in current_particles
                        )
                    )
                    batch = self.production_backend.run_safety_trajectory_batch(
                        adapter=self.adapter,
                        kernels=tuple(
                            copy.deepcopy(particle.state)
                            for particle in current_particles
                        ),
                        filter_state=stacked_filter_state,
                        branch_keys=current_branch_keys,
                        remaining_steps=remaining_steps,
                        probe_id=probe_id,
                        resampling_timing=belief.resampling_timing,
                        repetitions=active_lane_count,
                    )
                    batch_throughput = dict(batch["throughput"])
                    if bool(
                        batch_throughput.get(
                            "host_sync_inside_environment_loop", True
                        )
                    ):
                        raise RuntimeError(
                            "R015 safety synchronized with the host inside a suffix."
                        )
                    if int(
                        batch_throughput.get("active_lane_batch_width", -1)
                    ) != active_lane_count:
                        raise RuntimeError(
                            "R015 safety microbatch changed its active lane width."
                        )
                    if int(
                        batch_throughput.get("compiled_batch_calls", -1)
                    ) != 1:
                        raise RuntimeError(
                            "R015 safety microbatch did not use one device scan."
                        )
                    trace_batches.append(dict(batch["trace"]))
                    microbatch_throughputs.append(batch_throughput)
                required_trace_fields = (
                    "active",
                    "ego_actions",
                    "partner_actions",
                    "raw_team_rewards",
                    "done",
                    "wrong_delivery",
                    "filter_closed",
                    "prototype_posterior_before",
                    "prototype_posterior_after",
                )
                if any(
                    any(field not in item for field in required_trace_fields)
                    for item in trace_batches
                ):
                    raise RuntimeError("R015 safety microbatch lacks replay evidence.")
                trace = {
                    field: np.concatenate(
                        tuple(np.asarray(item[field]) for item in trace_batches),
                        axis=1,
                    )
                    for field in required_trace_fields
                }
                wrong_by_step = np.asarray(
                    trace["wrong_delivery"], dtype=np.bool_
                )
                closed_by_step = np.asarray(
                    trace["filter_closed"], dtype=np.bool_
                )
                expected_shape = (remaining_steps, repetitions)
                if wrong_by_step.shape != expected_shape or (
                    closed_by_step.shape != expected_shape
                ):
                    raise RuntimeError("R015 safety batch changed its registered shape.")
                wrong_by_branch = np.any(wrong_by_step, axis=0)
                closed_by_branch = np.any(closed_by_step, axis=0)
                wrong_count = int(np.sum(wrong_by_branch))
                filter_closed_count = int(np.sum(closed_by_branch))
                active_lane_counts = tuple(
                    int(item["active_lane_batch_width"])
                    for item in microbatch_throughputs
                )
                parent_slot_widths = tuple(
                    count * parent_slots_per_lane
                    for count in active_lane_counts
                )
                if sum(active_lane_counts) != repetitions or any(
                    width > parent_slot_batch_width
                    for width in parent_slot_widths
                ):
                    raise RuntimeError(
                        "R015 safety microbatch schedule changed branch accounting."
                    )
                wall_seconds = math.fsum(
                    float(item.get("wall_seconds", 0.0))
                    for item in microbatch_throughputs
                )
                true_transitions = sum(
                    int(item.get("true_environment_transitions", 0))
                    for item in microbatch_throughputs
                )
                computed_transitions = sum(
                    int(
                        item.get(
                            "computed_environment_transitions_including_masked_padding",
                            0,
                        )
                    )
                    for item in microbatch_throughputs
                )
                filter_particle_transitions = sum(
                    int(
                        item.get(
                            "filter_particle_environment_transitions", 0
                        )
                    )
                    for item in microbatch_throughputs
                )
                computed_filter_particle_transitions = sum(
                    int(
                        item.get(
                            "computed_filter_particle_environment_transitions",
                            0,
                        )
                    )
                    for item in microbatch_throughputs
                )
                throughput = {
                    "device_execution_id": "r015_safety_jit_scan_vmap_v2",
                    "microbatch_schedule_id": (
                        R015_FILTER_MICROBATCH_SCHEDULE_ID
                    ),
                    "configured_parent_slot_batch_width": (
                        parent_slot_batch_width
                    ),
                    "parent_particle_slots_per_lane": parent_slots_per_lane,
                    "lanes_per_full_microbatch": lanes_per_microbatch,
                    "microbatch_count": len(microbatch_throughputs),
                    "microbatch_active_lane_counts": list(active_lane_counts),
                    "microbatch_active_parent_particle_slot_widths": list(
                        parent_slot_widths
                    ),
                    "microbatch_wall_seconds": [
                        float(item.get("wall_seconds", 0.0))
                        for item in microbatch_throughputs
                    ],
                    "true_environment_transitions": true_transitions,
                    "computed_environment_transitions_including_masked_padding": (
                        computed_transitions
                    ),
                    "filter_particle_environment_transitions": (
                        filter_particle_transitions
                    ),
                    "computed_filter_particle_environment_transitions": (
                        computed_filter_particle_transitions
                    ),
                    "compiled_batch_calls": sum(
                        int(item.get("compiled_batch_calls", 0))
                        for item in microbatch_throughputs
                    ),
                    "jit_compilations": sum(
                        int(item.get("jit_compilations", 0))
                        for item in microbatch_throughputs
                    ),
                    "active_lane_batch_width": max(active_lane_counts),
                    "wall_seconds": wall_seconds,
                    "true_transitions_per_second": (
                        true_transitions / wall_seconds
                        if wall_seconds > 0.0
                        else None
                    ),
                    "computed_transitions_per_second": (
                        computed_transitions / wall_seconds
                        if wall_seconds > 0.0
                        else None
                    ),
                    "host_sync_inside_environment_loop": False,
                }
                active = np.asarray(trace["active"], dtype=np.bool_)
                ego_actions = np.asarray(trace["ego_actions"])
                partner_actions = np.asarray(trace["partner_actions"])
                rewards = np.asarray(trace["raw_team_rewards"])
                dones = np.asarray(trace["done"], dtype=np.bool_)
                posterior_before = np.asarray(
                    trace["prototype_posterior_before"]
                )
                posterior_after = np.asarray(
                    trace["prototype_posterior_after"]
                )
                for repetition, (particle, branch_key) in enumerate(
                    zip(selected_particles, branch_keys)
                ):
                    trajectory_sha256 = canonical_sha256(
                        {
                            "active": active[:, repetition].tolist(),
                            "ego_actions": ego_actions[:, repetition].tolist(),
                            "partner_actions": partner_actions[
                                :, repetition
                            ].tolist(),
                            "raw_team_rewards": rewards[:, repetition].tolist(),
                            "done": dones[:, repetition].tolist(),
                            "wrong_delivery": wrong_by_step[
                                :, repetition
                            ].tolist(),
                            "prototype_posterior_before": posterior_before[
                                :, repetition
                            ].tolist(),
                            "prototype_posterior_after": posterior_after[
                                :, repetition
                            ].tolist(),
                        }
                    )
                    # 正式裁决器会用 checkpoint、分支键和官方历史独立重跑整个
                    # 后缀。块内只保留一个内容寻址的完整后缀描述，避免把每条
                    # 279×4 安全分支再展开成至多 400 个 Python 字典。这个描述
                    # 不是用摘要替代重跑；第二次设备执行仍由正式生产入口完成。
                    replayable_steps = [
                        {
                            "environment_step": len(history.records) - 1,
                            "environment_step_count": remaining_steps,
                            "complete_suffix_sha256": trajectory_sha256,
                            "branch_random_key": branch_key,
                        }
                    ]
                    branch_payload = {
                        "schema_version": "path_c_r015_safety_branch_evidence_v2",
                        "prototype_id": prototype_id,
                        "repetition_index": repetition,
                        "probe_id": probe_id,
                        "branch_key": branch_key,
                        "source_particle_state_sha256": particle.state_sha256,
                        "remaining_steps": remaining_steps,
                        "wrong_delivery_detected": bool(
                            wrong_by_branch[repetition]
                        ),
                        "filter_closed_for_zero_support": bool(
                            closed_by_branch[repetition]
                        ),
                        "active_step_count": int(np.sum(active[:, repetition])),
                        "trajectory_sha256": trajectory_sha256,
                        "replayable_environment_steps": replayable_steps,
                    }
                    branch_evidence.append(
                        {
                            **branch_payload,
                            "branch_evidence_sha256": canonical_sha256(
                                branch_payload
                            ),
                        }
                    )
                if tuple(
                    int(item["repetition_index"]) for item in branch_evidence
                ) != tuple(range(repetitions)):
                    raise RuntimeError(
                        "R015 safety microbatch merge changed branch order."
                    )
            comparisons.append(
                {
                    "schema_version": "path_c_r015_safety_comparison_v2",
                    "prototype_id": prototype_id,
                    "repetitions": repetitions,
                    "wrong_delivery_count": wrong_count,
                    "positive_posterior_support": bool(support),
                    "compatible_hidden_state_reconstructed": bool(support)
                    and filter_closed_count == 0,
                    "filter_closed_count": filter_closed_count,
                    "safety_key": derive_controller_key(
                        safety_key,
                        "safety_branch_family",
                        prototype_id,
                    ),
                    "branches": tuple(branch_evidence),
                    "throughput": throughput,
                }
            )
        passed = all(
            item["positive_posterior_support"]
            and item["compatible_hidden_state_reconstructed"]
            and item["wrong_delivery_count"] == 0
            for item in comparisons
        )
        return SafetyDecisionV1(passed=passed, comparisons=tuple(comparisons))


def _artifact_sha256(preregistration: Any, name: str) -> str:
    binding = preregistration.artifacts[name]
    if not _is_sha256(binding.sha256):
        raise ValueError(f"R015 frozen artifact {name} lacks a SHA-256 digest.")
    return str(binding.sha256)


def _support_report_sha256(preregistration: Any) -> str | None:
    path = preregistration.support_report_path
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _formal_group_replay_record(preregistration: Any) -> Mapping[str, Any]:
    return {
        "trace_replay_verifier_sha256": _artifact_sha256(
            preregistration, "trace_replay_verifier"
        ),
        "environment_config_sha256": _artifact_sha256(
            preregistration, "environment_config"
        ),
        "environment_source_sha256": _artifact_sha256(
            preregistration, "environment_source"
        ),
        "ego_checkpoint_sha256": _artifact_sha256(
            preregistration, "ego_checkpoint"
        ),
        "continuation_planner_sha256": _artifact_sha256(
            preregistration, "continuation_planner"
        ),
        "official_history_filter_sha256": _artifact_sha256(
            preregistration, "official_history_filter"
        ),
        "response_projection_sha256": _artifact_sha256(
            preregistration, "response_projection"
        ),
        "step_count": 400,
        "actions_legal": True,
        "controller_actions_recomputed": True,
        "recurrent_state_writes_recomputed": True,
        "observation_chain_matches": True,
        "raw_rewards_match": True,
        "done_boundary_matches": True,
        "passed": True,
    }


def _formal_masked_reference(
    consultation: Mapping[str, Any],
) -> Mapping[str, Any]:
    candidates = tuple(consultation["candidates"])
    candidate_j_mask = {
        str(candidate["probe_id"]): float(candidate["j_mask"])
        for candidate in candidates
    }
    safe_ids = sorted(
        str(candidate["probe_id"])
        for candidate in candidates
        if candidate["eligible"] is True
        and candidate["static_safety_pass"] is True
    )
    selected = str(consultation["masked_reference_action_id"])
    selected_value = (
        float(consultation["v_base"])
        if selected == "base"
        else candidate_j_mask[selected]
    )
    return {
        "v_base": float(consultation["v_base"]),
        "candidate_j_mask": candidate_j_mask,
        "selectable_candidate_ids": safe_ids,
        "selected_action_id": selected,
        "selected_value": selected_value,
        "v_mask": float(consultation["v_mask"]),
    }


def _formal_trace_replay_context(
    preregistration: Any,
    member: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "environment_config_sha256": _artifact_sha256(
            preregistration, "environment_config"
        ),
        "environment_source_sha256": _artifact_sha256(
            preregistration, "environment_source"
        ),
        "ego_checkpoint_sha256": _artifact_sha256(
            preregistration, "ego_checkpoint"
        ),
        "continuation_planner_sha256": _artifact_sha256(
            preregistration, "continuation_planner"
        ),
        "official_history_filter_sha256": _artifact_sha256(
            preregistration, "official_history_filter"
        ),
        "response_projection_sha256": _artifact_sha256(
            preregistration, "response_projection"
        ),
        "ego_evidence_contract_sha256": _artifact_sha256(
            preregistration, "ego_evidence_contract"
        ),
        "response_vocabulary_sha256": _artifact_sha256(
            preregistration, "response_vocabulary"
        ),
        "probe_registry_sha256": _artifact_sha256(
            preregistration, "probe_registry"
        ),
        "probe_registry_semantic_sha256": (
            preregistration.probe_registry_semantic_sha256
        ),
        "random_key_derivation_sha256": _artifact_sha256(
            preregistration, "random_key_derivation"
        ),
        "partner_checkpoint_sha256": member["checkpoint_sha256"],
        "partner_model_weights_sha256": member["model_weights_sha256"],
        "partner_training_config_sha256": member["training_config_sha256"],
        "partner_training_manifest_sha256": member[
            "training_manifest_sha256"
        ],
        "partner_family_spec_sha256": member["family_spec_sha256"],
        "partner_architecture_sha256": member["architecture_sha256"],
        "partner_action_rule": member["action_rule"],
        "support_registration_sha256": (
            preregistration.support_registration_sha256
        ),
        "support_report_sha256": _support_report_sha256(preregistration),
    }


def _formal_safety_comparisons(
    executed: Mapping[str, Any],
    *,
    preregistration: Any,
    request: R015FormalBlockRequestV1,
) -> tuple[Mapping[str, Any], ...]:
    probe_fired = executed["probe_fired"] is True
    no_probe_reason = executed.get("no_probe_reason")
    if not probe_fired and no_probe_reason != "safety_rejected":
        return ()
    if probe_fired:
        probe_step = int(executed["probe_step"])
    else:
        selected_rows = [
            row
            for row in executed["consultations"]
            if row.get("selected_for_safety_probe_id") is not None
        ]
        if len(selected_rows) != 1:
            raise ValueError(
                "R015 safety rejection requires one selected consultation."
            )
        probe_step = int(selected_rows[0]["environment_step"])
    comparisons: list[Mapping[str, Any]] = []
    for raw in executed["safety_comparisons"]:
        support_id = str(raw["prototype_id"])
        branches = tuple(raw["branches"])
        branch_keys = tuple(str(branch["branch_key"]) for branch in branches)
        branch_evidence = tuple(
            {
                "branch_index": index,
                "random_key": branch_keys[index],
                "compatible_hidden_state_sha256": str(
                    branch["source_particle_state_sha256"]
                ),
                "environment_steps": copy.deepcopy(
                    list(branch["replayable_environment_steps"])
                ),
            }
            for index, branch in enumerate(branches)
        )
        branch_results = tuple(
            {
                "branch_index": index,
                "random_key": branch_keys[index],
                "wrong_delivery_detected": bool(
                    branch["wrong_delivery_detected"]
                ),
            }
            for index, branch in enumerate(branches)
        )
        comparison = {
            "support_prototype_id": support_id,
            "repetitions": int(raw["repetitions"]),
            "wrong_delivery_count": int(raw["wrong_delivery_count"]),
            "positive_posterior_support": bool(
                raw["positive_posterior_support"]
            ),
            "posterior_support_evidence": {
                "positive_support": bool(raw["positive_posterior_support"]),
                "compatible_branch_count": len(branches),
            },
            "compatible_hidden_state_reconstructed": bool(
                raw["compatible_hidden_state_reconstructed"]
            ),
            "random_stream_key": canonical_sha256(
                [
                    "r015_formal_safety_stream_v1",
                    request.audit_unit_id,
                    support_id,
                    probe_step,
                ]
            ),
            "audit_unit_id": request.audit_unit_id,
            "episode_seed": request.episode_seed,
            "purpose": "safety_validation",
            "environment_step": probe_step,
            "branch_start_index": 0,
            "branch_count": len(branches),
            "branch_keys": list(branch_keys),
            "branch_evidence": list(branch_evidence),
            "branch_results": list(branch_results),
            "branch_keys_sha256": canonical_sha256(list(branch_keys)),
            "random_key_derivation_sha256": _artifact_sha256(
                preregistration, "random_key_derivation"
            ),
        }
        comparisons.append(comparison)
    return tuple(comparisons)


def _formal_components_from_executed(
    executed: Mapping[str, Any],
    *,
    request: R015FormalBlockRequestV1,
    preregistration: Any,
    support_members: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    if executed.get("schema_version") != "path_c_r015_executed_paired_block_v2":
        raise ValueError("R015 formal conversion requires an executed v2 block.")
    if (
        executed.get("audit_unit_id") != request.audit_unit_id
        or executed.get("partner_prototype_id") != request.prototype_id
        or int(executed.get("episode_seed", -1)) != request.episode_seed
    ):
        raise ValueError("R015 executed block changed its frozen coordinate.")
    member = support_members[request.prototype_id]
    consultations = copy.deepcopy(list(executed["consultations"]))
    if not consultations:
        raise ValueError("R015 formal block has no consultation evidence.")
    final_consultation = consultations[-1]
    masked_reference = _formal_masked_reference(final_consultation)
    group_replay = _formal_group_replay_record(preregistration)
    groups = copy.deepcopy(dict(executed["groups"]))
    for group in ("A1", "A2-mask", "A2-use"):
        groups[group]["masked_value_reference"] = copy.deepcopy(masked_reference)
        groups[group]["replay_verification"] = copy.deepcopy(group_replay)
    probe_fired = executed["probe_fired"] is True
    probe_step = executed.get("probe_step")
    selected_probe_id = executed.get("selected_probe_id")
    episode_random_key = str(executed["episode_random_root_key"])
    arms = {}
    for group in ("A1", "A2-mask", "A2-use"):
        raw_return = math.fsum(
            float(step["raw_team_reward"])
            for step in groups[group]["environment_steps"]
        )
        arms[group] = {
            "prototype_id": request.prototype_id,
            "family_id": member["family_id"],
            "training_seed": int(member["training_seed"]),
            "training_run_id": member["training_run_id"],
            "ego_position": 1,
            "initial_state_sha256": executed["initial_state_sha256"],
            "episode_seed": request.episode_seed,
            "firing_indicator": probe_fired,
            "raw_return": raw_return,
            "valid": True,
            "information_source": "official_local_history_only",
            "forbidden_fields_read": [],
            "episode_random_key": episode_random_key,
            "controller_contract_sha256": _artifact_sha256(
                preregistration, "ego_evidence_contract"
            ),
            "ego_checkpoint_sha256": _artifact_sha256(
                preregistration, "ego_checkpoint"
            ),
            "environment_config_sha256": _artifact_sha256(
                preregistration, "environment_config"
            ),
            "environment_source_sha256": _artifact_sha256(
                preregistration, "environment_source"
            ),
            "official_history_filter_sha256": _artifact_sha256(
                preregistration, "official_history_filter"
            ),
        }
    final_evidence = final_consultation["planner_evidence"]
    probe_decision = {
        "probe_registry_semantic_sha256": (
            preregistration.probe_registry_semantic_sha256
        ),
        "random_key_derivation_sha256": _artifact_sha256(
            preregistration, "random_key_derivation"
        ),
        "planning_random_purpose": "value_planning",
        "planning_random_stream_key": final_evidence["planning_random_key"],
        "score_random_purpose": "score_estimation",
        "score_random_stream_key": final_evidence["score_random_key"],
        "consultations": consultations,
        "executed_probe_id": selected_probe_id if probe_fired else None,
        "probe_step": probe_step if probe_fired else None,
        "probe_budget_used_before": 0,
        "probe_budget_used_after": 1 if probe_fired else 0,
    }
    trace_manifest = {
        "schema_version": "path_c_r015_trace_manifest_v1",
        "audit_unit_id": request.audit_unit_id,
        "prototype_id": request.prototype_id,
        "family_id": member["family_id"],
        "training_seed": int(member["training_seed"]),
        "training_run_id": member["training_run_id"],
        "ego_position": 1,
        "initial_state_sha256": executed["initial_state_sha256"],
        "episode_seed": request.episode_seed,
        "mechanical_attempt_index": request.mechanical_attempt_index,
        "episode_random_key": episode_random_key,
        "random_key_derivation_sha256": _artifact_sha256(
            preregistration, "random_key_derivation"
        ),
        "probe_step": probe_step if probe_fired else None,
        "groups": groups,
    }
    pairing = {
        "shared_continuation_controller_sha256": _artifact_sha256(
            preregistration, "continuation_controller"
        ),
        "probe_registry_sha256": _artifact_sha256(
            preregistration, "probe_registry"
        ),
        "probe_registry_semantic_sha256": (
            preregistration.probe_registry_semantic_sha256
        ),
        "response_vocabulary_sha256": _artifact_sha256(
            preregistration, "response_vocabulary"
        ),
        "response_projection_sha256": _artifact_sha256(
            preregistration, "response_projection"
        ),
        "a1_registered_response_route": "masked",
        "a1_full_candidate_set": True,
        "a1_uses_masked_value_reference": True,
        "a1_score_trace_sha256": canonical_sha256(probe_decision),
        "a2_score_trace_sha256": canonical_sha256(probe_decision),
        "raw_official_local_observation_preserved": True,
        "future_passive_response_route": "shared",
    }
    if probe_fired:
        pairing.update(
            {
                "a1_pre_probe_trace_sha256": canonical_sha256(
                    groups["A1"]["pre_probe_trace"]
                ),
                "a1_masked_value_reference_sha256": canonical_sha256(
                    masked_reference
                ),
                "a2_groups_share_pre_probe_actions": True,
                "a2_groups_share_probe_script": True,
            }
        )
        for group, route in (
            ("A2-mask", "masked"),
            ("A2-use", "belief_update_only"),
        ):
            record = groups[group]
            response_extra = record.get("belief_current_response_extra")
            pairing[group] = {
                "selected_probe_id": selected_probe_id,
                "pre_probe_trace_sha256": canonical_sha256(
                    record["pre_probe_trace"]
                ),
                "masked_history_sha256": canonical_sha256(
                    record["masked_history"]
                ),
                "current_probe_response_sha256": canonical_sha256(
                    record["current_probe_response"]
                ),
                "current_response_route": route,
                "masked_value_reference_sha256": canonical_sha256(
                    masked_reference
                ),
                "belief_common_input_sha256": canonical_sha256(
                    record["belief_common_input"]
                ),
                "belief_current_response_extra_sha256": (
                    None
                    if response_extra is None
                    else canonical_sha256(response_extra)
                ),
                "continuation_common_input_sha256": canonical_sha256(
                    record["continuation_common_input"]
                ),
                "continuation_direct_response_sha256": None,
                "recurrent_common_input_sha256": canonical_sha256(
                    record["recurrent_common_input"]
                ),
                "recurrent_direct_response_sha256": None,
                "non_belief_response_aliases": [],
                "projection_dependency_audit_sha256": _artifact_sha256(
                    preregistration, "response_projection"
                ),
            }
    else:
        complete_sha = canonical_sha256(groups["A1"]["complete_trace"])
        pairing.update(
            {
                "a1_complete_trace_sha256": complete_sha,
                "A2-mask": {
                    "selected_probe_id": None,
                    "current_probe_response_sha256": None,
                    "current_response_route": "not_applicable",
                    "complete_trace_sha256": complete_sha,
                },
                "A2-use": {
                    "selected_probe_id": None,
                    "current_probe_response_sha256": None,
                    "current_response_route": "not_applicable",
                    "complete_trace_sha256": complete_sha,
                },
                "a2_groups_share_complete_trace": True,
                "all_groups_share_complete_trace": True,
            }
        )
    comparisons = _formal_safety_comparisons(
        executed,
        preregistration=preregistration,
        request=request,
    )
    no_probe_reason = None if probe_fired else executed.get("no_probe_reason")
    safety_probe_id = selected_probe_id
    if not probe_fired and no_probe_reason == "safety_rejected":
        selected_for_safety = {
            str(row["selected_for_safety_probe_id"])
            for row in consultations
            if row.get("selected_for_safety_probe_id") is not None
        }
        if len(selected_for_safety) != 1:
            raise ValueError(
                "R015 safety rejection requires one selected probe script."
            )
        safety_probe_id = next(iter(selected_for_safety))
    telemetry = dict(executed["formal_telemetry_projection"])
    safety = {
        "probe_executed": probe_fired,
        "no_probe_reason": no_probe_reason,
        "wrong_delivery_detector_sha256": _artifact_sha256(
            preregistration, "wrong_delivery_detector"
        ),
        "recipe_indicator_cost_counted_as_safety_event": False,
        "comparisons": list(comparisons),
        "candidate_evaluation_count": int(
            telemetry["candidate_evaluation_count"]
        ),
        "safety_rejection_count": int(telemetry["safety_rejection_count"]),
        "non_positive_score_count": int(
            telemetry["non_positive_score_count"]
        ),
        "window_expired_count": int(telemetry["window_expired_count"]),
    }
    paired_without_trace = {
        "schema_version": "path_c_r015_paired_block_v2",
        "phase": "formal",
        "round_index": request.round_index,
        "audit_unit_id": request.audit_unit_id,
        "prototype_id": request.prototype_id,
        "family_id": member["family_id"],
        "training_seed": int(member["training_seed"]),
        "training_run_id": member["training_run_id"],
        "ego_position": 1,
        "initial_state_sha256": executed["initial_state_sha256"],
        "episode_seed": request.episode_seed,
        "mechanical_attempt_index": request.mechanical_attempt_index,
        "formal_effect_look_number": 1,
        "pilot_data": False,
        "firing_indicator": probe_fired,
        "arms": arms,
        "probe_decision": probe_decision,
        "pairing": pairing,
        "safety": safety,
    }
    decision_verifier_input = {
        "consultations": [
            {
                "environment_step": row["environment_step"],
                "official_history": copy.deepcopy(
                    row["planner_evidence"]["official_history"]
                ),
                "planning_random_key": row["planner_evidence"][
                    "planning_random_key"
                ],
                "score_random_key": row["planner_evidence"]["score_random_key"],
                "planning_branches": copy.deepcopy(
                    row["planner_evidence"]["planning_branches"]
                ),
                "branch_sampling_rule_id": row["planner_evidence"][
                    "branch_sampling_rule_id"
                ],
                "branch_belief_rule_id": row["planner_evidence"][
                    "branch_belief_rule_id"
                ],
                "cost_accounting": copy.deepcopy(
                    row["planner_evidence"]["cost_accounting"]
                ),
            }
            for row in consultations
        ]
    }
    safety_inputs = {
        comparison["support_prototype_id"]: {
            "support_prototype_id": comparison["support_prototype_id"],
            "posterior_support_evidence": copy.deepcopy(
                comparison["posterior_support_evidence"]
            ),
            "branch_evidence": copy.deepcopy(comparison["branch_evidence"]),
            "official_history": copy.deepcopy(
                groups["A1"]["official_history_events"][
                    : int(comparison["environment_step"]) + 1
                ]
            ),
            "selected_probe_id": safety_probe_id,
            "selected_probe_actions": list(
                preregistration.probe_scripts[
                    str(safety_probe_id)
                ].primitive_actions
            ),
        }
        for comparison in comparisons
    }
    return {
        "paired_block_without_trace": paired_without_trace,
        "trace_manifest": trace_manifest,
        "trace_replay_context": _formal_trace_replay_context(
            preregistration, member
        ),
        "decision_verifier_input": decision_verifier_input,
        "safety_verifier_inputs": safety_inputs,
        "safety_comparisons": {
            comparison["support_prototype_id"]: comparison
            for comparison in comparisons
        },
    }


@dataclass(frozen=True)
class OCV2R015SegmentExecutorV1:
    """从完整执行状态推进一个连续片段。"""

    full_horizon: OCV2R015FullHorizonExecutorV1

    def execute_segment(
        self,
        start_state: Any,
        *,
        group: str,
        start_step: int,
        stop_step: int,
        forced_probe_actions: tuple[str, ...],
    ) -> R015ExecutedSegmentV1:
        if not isinstance(start_state, R015BranchStateV1):
            raise TypeError("R015 segment execution requires a complete branch state.")
        if start_state.environment_step != start_step or not start_step <= stop_step <= 400:
            raise ValueError("R015 segment bounds differ from its complete state.")
        if len(forced_probe_actions) > 1:
            raise ValueError("R015 registered scripts contain one primitive action.")
        key = derive_controller_key(
            canonical_sha256(
                {
                    "history": start_state.history.to_payload(),
                    "group": group,
                    "start_step": start_step,
                    "stop_step": stop_step,
                }
            ),
            "segment_execution",
        )
        result = self.full_horizon._run(
            start_state,
            branch_key=key,
            total_steps=stop_step - start_step,
            first_forced_action=(
                forced_probe_actions[0] if forced_probe_actions else None
            ),
        )
        return R015ExecutedSegmentV1(
            end_state=result.state,
            steps=result.trajectory,
        )


@dataclass(frozen=True)
class OCV2R015ReplayBackendV1:
    """从冻结 checkpoint 和登记随机键重新生成三类证据。"""

    executor: OCV2R015FullHorizonExecutorV1
    recomputed_formal_components: Mapping[str, Any] | None = None
    preregistration: Any | None = None
    support_members: Mapping[str, Mapping[str, Any]] | None = None

    def _callback_recomputed(
        self,
        payload: Mapping[str, Any],
        context: Mapping[str, Any],
        *,
        kind: str,
    ) -> Mapping[str, Any]:
        callback = context.get(f"_r015_{kind}_recompute")
        if not callable(callback):
            raise ValueError(f"R015 {kind} replay lacks a recomputation callback.")
        recomputed = callback(self.executor, copy.deepcopy(dict(context)))
        if not isinstance(recomputed, Mapping) or canonical_sha256(
            payload
        ) != canonical_sha256(recomputed):
            raise ValueError(f"R015 {kind} differs from checkpoint replay.")
        return {
            "schema_version": f"path_c_r015_{kind}_replay_verification_v1",
            "verified": True,
            "recomputed_sha256": canonical_sha256(recomputed),
        }

    def _components(self) -> Mapping[str, Any]:
        if (
            self.recomputed_formal_components is None
            or self.preregistration is None
            or self.support_members is None
        ):
            raise ValueError("R015 formal replay lacks its independent full-block replay.")
        return self.recomputed_formal_components

    def replay_trace(self, trace, context):
        if self.recomputed_formal_components is None:
            return self._callback_recomputed(trace, context, kind="trace")
        components = self._components()
        recomputed = copy.deepcopy(dict(components["trace_manifest"]))
        reported = copy.deepcopy(dict(trace))
        recomputed.pop("replay_verification", None)
        reported.pop("replay_verification", None)
        if canonical_sha256(reported) != canonical_sha256(recomputed):
            raise ValueError("R015 trace differs from independent checkpoint replay.")
        raw_groups = dict(trace["groups"])
        group_ego_actions = {
            group: [step["ego_action"] for step in raw_groups[group]["environment_steps"]]
            for group in ("A1", "A2-mask", "A2-use")
        }
        group_partner_actions = {
            group: [
                step["partner_action"]
                for step in raw_groups[group]["environment_steps"]
            ]
            for group in ("A1", "A2-mask", "A2-use")
        }
        group_state_writes = {
            group: [
                {
                    "environment_step": step["environment_step"],
                    "before_sha256": step[
                        "controller_recurrent_state_before_sha256"
                    ],
                    "after_sha256": step[
                        "controller_recurrent_state_after_sha256"
                    ],
                }
                for step in raw_groups[group]["environment_steps"]
            ]
            for group in ("A1", "A2-mask", "A2-use")
        }
        group_partner_state_writes = {
            group: [
                {
                    "environment_step": step["environment_step"],
                    "before_sha256": step[
                        "partner_recurrent_state_before_sha256"
                    ],
                    "after_sha256": step[
                        "partner_recurrent_state_after_sha256"
                    ],
                }
                for step in raw_groups[group]["environment_steps"]
            ]
            for group in ("A1", "A2-mask", "A2-use")
        }
        action_state_hashes = {
            group: canonical_sha256(
                [
                    {
                        "environment_step": write["environment_step"],
                        "ego_action": group_ego_actions[group][index],
                        "before_sha256": write["before_sha256"],
                        "after_sha256": write["after_sha256"],
                    }
                    for index, write in enumerate(group_state_writes[group])
                ]
            )
            for group in ("A1", "A2-mask", "A2-use")
        }
        partner_action_state_hashes = {
            group: canonical_sha256(
                [
                    {
                        "environment_step": write["environment_step"],
                        "partner_action": group_partner_actions[group][index],
                        "before_sha256": write["before_sha256"],
                        "after_sha256": write["after_sha256"],
                    }
                    for index, write in enumerate(
                        group_partner_state_writes[group]
                    )
                ]
            )
            for group in ("A1", "A2-mask", "A2-use")
        }
        selected_action = str(
            raw_groups["A1"]["masked_value_reference"]["selected_action_id"]
        )
        selected_script = (
            list(
                self.preregistration.probe_scripts[
                    selected_action
                ].primitive_actions
            )
            if selected_action in self.preregistration.probe_scripts
            else []
        )
        probe_id = trace.get("groups", {}).get("A2-use", {}).get(
            "selected_probe_id"
        )
        probe_actions = (
            list(self.preregistration.probe_scripts[str(probe_id)].primitive_actions)
            if probe_id in self.preregistration.probe_scripts
            else []
        )
        probe_executed = trace.get("probe_step") is not None
        routes = {
            "A1": {
                "belief_route": "B_mask(q,x)",
                "continuation_route": "C(q,x)",
                "selected_action_id": selected_action,
                "selected_registered_script_actions": selected_script,
                "selected_action_matches_replay": True,
            },
            "A2-mask": {
                "belief_route": "B_mask(q,x)",
                "continuation_route": "C(q,x)",
                "selected_probe_actions": probe_actions,
                "current_response_used": False,
                "pre_probe_actions_match": True,
                "post_probe_actions_match_replay": True,
            },
            "A2-use": {
                "belief_route": (
                    "B_use(q,x,y)" if probe_executed else "B_mask(q,x)"
                ),
                "continuation_route": "C(q,x)",
                "selected_probe_actions": probe_actions,
                "current_response_used": probe_executed,
                "pre_probe_actions_match": True,
                "post_probe_actions_match_replay": True,
            },
        }
        return {
            "schema_version": "path_c_r015_trace_replay_result_v2",
            "verified": True,
            "groups_sha256": canonical_sha256(raw_groups),
            "group_step_counts": {
                group: 400 for group in ("A1", "A2-mask", "A2-use")
            },
            "group_environment_steps_sha256": {
                group: canonical_sha256(raw_groups[group]["environment_steps"])
                for group in ("A1", "A2-mask", "A2-use")
            },
            "environment_config_sha256": context["environment_config_sha256"],
            "environment_source_sha256": context["environment_source_sha256"],
            "ego_checkpoint_sha256": context["ego_checkpoint_sha256"],
            "continuation_planner_sha256": context[
                "continuation_planner_sha256"
            ],
            "official_history_filter_sha256": context[
                "official_history_filter_sha256"
            ],
            "response_projection_sha256": context["response_projection_sha256"],
            "ego_evidence_contract_sha256": context[
                "ego_evidence_contract_sha256"
            ],
            "response_vocabulary_sha256": context[
                "response_vocabulary_sha256"
            ],
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
            "partner_family_spec_sha256": context[
                "partner_family_spec_sha256"
            ],
            "partner_architecture_sha256": context[
                "partner_architecture_sha256"
            ],
            "partner_action_rule": context["partner_action_rule"],
            "support_registration_sha256": context[
                "support_registration_sha256"
            ],
            "support_report_sha256": context["support_report_sha256"],
            "group_ego_actions": group_ego_actions,
            "group_partner_actions": group_partner_actions,
            "group_recurrent_state_writes_sha256": {
                group: canonical_sha256(group_state_writes[group])
                for group in ("A1", "A2-mask", "A2-use")
            },
            "group_action_state_summaries_sha256": action_state_hashes,
            "group_partner_recurrent_state_writes_sha256": {
                group: canonical_sha256(group_partner_state_writes[group])
                for group in ("A1", "A2-mask", "A2-use")
            },
            "group_partner_action_state_summaries_sha256": (
                partner_action_state_hashes
            ),
            "group_controller_route_summaries": routes,
            "all_controller_actions_recomputed": True,
            "all_recurrent_state_writes_recomputed": True,
            "all_partner_actions_recomputed": True,
            "all_partner_recurrent_state_writes_recomputed": True,
        }

    def replay_probe_decision(self, evidence, context):
        if self.recomputed_formal_components is None:
            return self._callback_recomputed(
                evidence, context, kind="probe_decision"
            )
        components = self._components()
        expected_input = components["decision_verifier_input"]
        if canonical_sha256(evidence) != canonical_sha256(expected_input):
            raise ValueError("R015 decision evidence differs from independent replay.")
        probe_decision = components["paired_block_without_trace"]["probe_decision"]
        consultations = [
            {
                "environment_step": row["environment_step"],
                "v_base": row["v_base"],
                "candidates": copy.deepcopy(list(row["candidates"])),
                "selected_for_safety_probe_id": row[
                    "selected_for_safety_probe_id"
                ],
                "planning_random_key": row["planner_evidence"][
                    "planning_random_key"
                ],
                "score_random_key": row["planner_evidence"]["score_random_key"],
            }
            for row in probe_decision["consultations"]
        ]
        return {
            "schema_version": "path_c_r015_decision_evidence_verification_v1",
            "verified": True,
            "consultations": consultations,
            "full_probe_registry_ids": sorted(self.preregistration.probe_scripts),
            "first_positive_stop_valid": True,
        }

    def replay_safety_comparison(self, comparison, context):
        if self.recomputed_formal_components is None:
            return self._callback_recomputed(
                comparison, context, kind="safety_comparison"
            )
        components = self._components()
        support_id = str(comparison.get("support_prototype_id", ""))
        expected_input = components["safety_verifier_inputs"].get(support_id)
        if expected_input is None or canonical_sha256(comparison) != canonical_sha256(
            expected_input
        ):
            raise ValueError("R015 safety evidence differs from independent replay.")
        full = components["safety_comparisons"][support_id]
        member = self.support_members[support_id]
        return {
            "schema_version": "path_c_r015_safety_branch_verification_v2",
            "verified": True,
            "support_prototype_id": support_id,
            "branch_count": int(full["branch_count"]),
            "branch_keys_sha256": full["branch_keys_sha256"],
            "wrong_delivery_count": int(full["wrong_delivery_count"]),
            "positive_posterior_support": bool(
                full["positive_posterior_support"]
            ),
            "compatible_hidden_state_reconstructed": bool(
                full["compatible_hidden_state_reconstructed"]
            ),
            "all_branches_replayed": True,
            "support_checkpoint_sha256": member["checkpoint_sha256"],
            "support_model_weights_sha256": member["model_weights_sha256"],
            "support_architecture_sha256": member["architecture_sha256"],
            "support_training_config_sha256": member["training_config_sha256"],
            "support_training_manifest_sha256": member[
                "training_manifest_sha256"
            ],
            "support_family_spec_sha256": member["family_spec_sha256"],
            "support_action_rule": member["action_rule"],
            "support_registration_sha256": (
                self.preregistration.support_registration_sha256
            ),
            "support_report_sha256": _support_report_sha256(
                self.preregistration
            ),
            "selected_probe_id": comparison["selected_probe_id"],
            "selected_probe_actions": copy.deepcopy(
                list(comparison["selected_probe_actions"])
            ),
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
            "response_vocabulary_sha256": context[
                "response_vocabulary_sha256"
            ],
            "official_history_sha256": canonical_sha256(
                comparison["official_history"]
            ),
            "branch_evidence_sha256": canonical_sha256(
                comparison["branch_evidence"]
            ),
            "branch_results_sha256": canonical_sha256(
                full["branch_results"]
            ),
        }


def _environment_from_backend(backend: Any) -> OCV2Adapter:
    config = backend.policies[backend.baseline_member_id].config
    kwargs = dict(config["env"]["ENV_KWARGS"])
    kwargs.pop("op_ingredient_permutations", None)
    observation_type = kwargs.get("observation_type", "default")
    if hasattr(observation_type, "value"):
        observation_type = observation_type.value
    kwargs["observation_type"] = str(observation_type).lower()
    kwargs["max_steps"] = 400
    return OCV2Adapter(**kwargs)


def build_r015_full_horizon_executor(
    config: Mapping[str, Any],
    production_backend: Any,
) -> OCV2R015FullHorizonExecutorV1:
    del config
    return OCV2R015FullHorizonExecutorV1(
        adapter=_environment_from_backend(production_backend),
        production_backend=production_backend,
    )


def build_r015_replay_backend(context: Mapping[str, Any]) -> R015ReplayBackend:
    injected = context.get("_r015_production_backend")
    if injected is None or getattr(injected, "full_horizon_executor", None) is None:
        raise ValueError("R015 replay requires the frozen five-checkpoint backend.")
    backend = OCV2R015ReplayBackendV1(
        injected.full_horizon_executor,
        recomputed_formal_components=context.get(
            "_r015_recomputed_formal_components"
        ),
        preregistration=context.get("_r015_preregistration"),
        support_members=context.get("_r015_support_members"),
    )
    if not isinstance(backend, R015ReplayBackend):
        raise TypeError("R015 replay backend does not satisfy its registered interface.")
    return backend


__all__ = [
    "OCV2R015FullHorizonExecutorV1",
    "OCV2R015ReplayBackendV1",
    "OCV2R015SegmentExecutorV1",
    "R015BranchStateV1",
    "R015OnlineParticleBeliefV2",
    "R015ParticleKernelStateV1",
    "R015PlanningLengthBucketResultV2",
    "R015PlanningPointBatchRequestV2",
    "R015_RESPONSE_SPEC_PAYLOAD",
    "build_r015_full_horizon_executor",
    "build_r015_replay_backend",
]
