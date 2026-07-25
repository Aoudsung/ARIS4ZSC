"""R015 配对轨迹、独立重放和正式轮次的可恢复执行入口。

本模块实现显式 R015 审计的完整运行路径，但不实现或开启一千万步学习版适应控制器。
正式执行仍要求最终预登记已经绑定粒子数、规划分支数、checkpoint、清单和内容摘要。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from experiments.overcooked_v2.path_c_flax_policy import OfficialFlaxPolicy
from experiments.overcooked_v2.path_c_r015_controller import (
    R015_ATOMIC_ACTIONS,
    canonical_json_bytes,
    canonical_sha256,
)


R015_FORMAL_GROUPS = ("A1", "A2-mask", "A2-use")
R015_EGO_AGENT_ID = "agent_1"
R015_PARTNER_AGENT_ID = "agent_0"
R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT = 32
R015_FORMAL_SAMPLING_SCHEDULE_SCHEMA = (
    "path_c_r015_formal_sampling_schedule_v4"
)
R015_FORMAL_ROUND_SAMPLING_CONTRACT = (
    "iid_round_vectors_ego_cook_seat_with_prefrozen_replacements_v4"
)
R015_FORMAL_REPLACEMENT_SEED_DERIVATION_ID = (
    "sha256_registered_coordinate_low32_allow_collisions_v1"
)
R015_TRACE_STEP_FIELDS_V1 = frozenset(
    {
        "environment_step",
        "ego_action",
        "partner_action",
        "controller_input",
        "raw_team_reward",
        "done",
        "environment_random_key",
        "official_local_observation_after",
        "controller_recurrent_state_before_sha256",
        "controller_recurrent_state_after_sha256",
        "partner_recurrent_state_before_sha256",
        "partner_recurrent_state_after_sha256",
    }
)
R015_TRACE_STEP_FIELDS = frozenset(
    {
        *R015_TRACE_STEP_FIELDS_V1,
        "belief_update_mode",
        "belief_before_sha256",
        "belief_after_sha256",
    }
)
OCV2_ACTION_INDEX = {
    "right": 0,
    "down": 1,
    "left": 2,
    "up": 3,
    "stay": 4,
    "interact": 5,
}
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class R015OfficialPolicyStatesV1:
    ego: Any
    partner: Any


@dataclass(frozen=True)
class R015OfficialPolicyRouterV1:
    """Route the official Flax wrappers to the two role-fixed agent slots."""

    ego_policy: OfficialFlaxPolicy
    partner_policy: OfficialFlaxPolicy
    ego_agent_id: str = R015_EGO_AGENT_ID
    partner_agent_id: str = R015_PARTNER_AGENT_ID

    def __post_init__(self) -> None:
        if not isinstance(self.ego_policy, OfficialFlaxPolicy) or not isinstance(
            self.partner_policy, OfficialFlaxPolicy
        ):
            raise TypeError("R015 runtime policies must use the official Flax wrapper.")
        if self.ego_agent_id != R015_EGO_AGENT_ID or (
            self.partner_agent_id != R015_PARTNER_AGENT_ID
        ):
            raise ValueError("R015 fixes the ego cook at agent_1 and partner at agent_0.")

    def initial_states(self, batch_size: int) -> R015OfficialPolicyStatesV1:
        return R015OfficialPolicyStatesV1(
            ego=self.ego_policy.initial_state(batch_size),
            partner=self.partner_policy.initial_state(batch_size),
        )

    def act_joint(
        self,
        observations: Mapping[str, Any],
        episode_starts: Mapping[str, Any],
        random_keys: Mapping[str, Any],
        states: R015OfficialPolicyStatesV1,
    ) -> tuple[Mapping[str, Any], R015OfficialPolicyStatesV1]:
        expected = {R015_EGO_AGENT_ID, R015_PARTNER_AGENT_ID}
        if set(observations) != expected or set(episode_starts) != expected or set(
            random_keys
        ) != expected:
            raise ValueError("R015 official policy routing requires both fixed agent slots.")
        ego_step = self.ego_policy.act(
            observations[R015_EGO_AGENT_ID],
            episode_starts[R015_EGO_AGENT_ID],
            states.ego,
            random_keys[R015_EGO_AGENT_ID],
        )
        partner_step = self.partner_policy.act(
            observations[R015_PARTNER_AGENT_ID],
            episode_starts[R015_PARTNER_AGENT_ID],
            states.partner,
            random_keys[R015_PARTNER_AGENT_ID],
        )
        return (
            {
                R015_EGO_AGENT_ID: ego_step.action,
                R015_PARTNER_AGENT_ID: partner_step.action,
            },
            R015OfficialPolicyStatesV1(
                ego=ego_step.next_recurrent_state,
                partner=partner_step.next_recurrent_state,
            ),
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX)
    )


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{field} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite.")
    return number


def _normalize_step(
    raw_step: Mapping[str, Any],
    *,
    index: int,
    allow_legacy_v1: bool = False,
) -> Mapping[str, Any]:
    expected_fields = (
        R015_TRACE_STEP_FIELDS_V1 if allow_legacy_v1 else R015_TRACE_STEP_FIELDS
    )
    if not isinstance(raw_step, Mapping) or set(raw_step) != expected_fields:
        raise ValueError("An R015 trace step has the wrong schema.")
    if isinstance(raw_step.get("environment_step"), bool) or int(
        raw_step.get("environment_step", -1)
    ) != index:
        raise ValueError("An R015 trace step has the wrong environment-step index.")
    if raw_step.get("ego_action") not in R015_ATOMIC_ACTIONS or raw_step.get(
        "partner_action"
    ) not in R015_ATOMIC_ACTIONS:
        raise ValueError("An R015 trace step contains an unknown primitive action.")
    if not isinstance(raw_step.get("controller_input"), Mapping):
        raise TypeError("An R015 trace step requires a controller-input mapping.")
    _finite(raw_step.get("raw_team_reward"), field="raw_team_reward")
    if raw_step.get("done") is not (index == 399):
        raise ValueError("An R015 trace must end exactly at environment step 399.")
    if not str(raw_step.get("environment_random_key", "")):
        raise ValueError("An R015 trace step lacks its environment random key.")
    for field in (
        "controller_recurrent_state_before_sha256",
        "controller_recurrent_state_after_sha256",
        "partner_recurrent_state_before_sha256",
        "partner_recurrent_state_after_sha256",
    ):
        if not _is_sha256(raw_step.get(field)):
            raise ValueError(f"An R015 trace step lacks {field}.")
    if not allow_legacy_v1:
        if raw_step.get("belief_update_mode") not in {
            "online_each_environment_step",
            "frozen_belief_branch_continuation_v1",
        }:
            raise ValueError("An R015 trace step has an unknown belief-update mode.")
        for field in ("belief_before_sha256", "belief_after_sha256"):
            if not _is_sha256(raw_step.get(field)):
                raise ValueError(f"An R015 trace step lacks {field}.")
    return json.loads(canonical_json_bytes(raw_step).decode("utf-8"))


def _normalize_trajectory(
    raw_steps: Sequence[Mapping[str, Any]],
    *,
    start_index: int = 0,
    allow_legacy_v1: bool = False,
    allowed_belief_discontinuity_indices: Sequence[int] = (),
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(raw_steps, (str, bytes)):
        raise TypeError("R015 trajectory steps must be a sequence of mappings.")
    normalized = tuple(
        _normalize_step(
            step,
            index=start_index + offset,
            allow_legacy_v1=allow_legacy_v1,
        )
        for offset, step in enumerate(raw_steps)
    )
    for index in range(1, len(normalized)):
        previous = normalized[index - 1]
        current = normalized[index]
        if current["controller_recurrent_state_before_sha256"] != previous[
            "controller_recurrent_state_after_sha256"
        ]:
            raise ValueError("R015 controller recurrent-state writes are not contiguous.")
        if current["partner_recurrent_state_before_sha256"] != previous[
            "partner_recurrent_state_after_sha256"
        ]:
            raise ValueError("R015 partner recurrent-state writes are not contiguous.")
        if current["controller_input"].get(
            "official_local_observation"
        ) != previous["official_local_observation_after"]:
            raise ValueError("R015 official observations are not contiguous.")
        if (
            not allow_legacy_v1
            and current["environment_step"]
            not in set(allowed_belief_discontinuity_indices)
            and current["belief_before_sha256"]
            != previous["belief_after_sha256"]
        ):
            raise ValueError("R015 online belief-state writes are not contiguous.")
    return normalized


def normalize_r015_legacy_diagnostic_trajectory_v1(
    raw_steps: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """读取没有信念摘要的历史诊断轨迹；正式数据不得调用此入口。"""

    return _normalize_trajectory(raw_steps, allow_legacy_v1=True)


def _trajectory_bytes(steps: Sequence[Mapping[str, Any]]) -> bytes:
    return canonical_json_bytes(list(steps))


def _return_from_steps(steps: Sequence[Mapping[str, Any]]) -> float:
    return math.fsum(float(step["raw_team_reward"]) for step in steps)


def _base_group_payload(
    steps: tuple[Mapping[str, Any], ...],
    *,
    replay_verification: Mapping[str, Any],
    masked_value_reference: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "environment_steps": list(steps),
        "official_history_events": [
            copy.deepcopy(step["controller_input"]) for step in steps
        ],
        "forbidden_read_events": [],
        "raw_reward_components": [step["raw_team_reward"] for step in steps],
        "replay_verification": copy.deepcopy(dict(replay_verification)),
        "masked_value_reference": copy.deepcopy(dict(masked_value_reference)),
    }


def build_r015_no_fire_trace_groups(
    shared_trajectory: Sequence[Mapping[str, Any]],
    *,
    replay_verification_by_group: Mapping[str, Mapping[str, Any]],
    masked_value_reference: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    """Normalize one common episode once, then byte-copy it to all three groups."""

    if set(replay_verification_by_group) != set(R015_FORMAL_GROUPS):
        raise ValueError("R015 replay records must cover all three formal groups.")
    normalized = _normalize_trajectory(shared_trajectory)
    if len(normalized) != 400:
        raise ValueError("An R015 formal trajectory must contain exactly 400 steps.")
    canonical = _trajectory_bytes(normalized)
    groups: dict[str, Mapping[str, Any]] = {}
    for group in R015_FORMAL_GROUPS:
        copied_steps = tuple(json.loads(canonical.decode("utf-8")))
        payload = _base_group_payload(
            copied_steps,
            replay_verification=replay_verification_by_group[group],
            masked_value_reference=masked_value_reference,
        )
        payload.update(
            {
                "complete_trace": [
                    copy.deepcopy(step["controller_input"]) for step in copied_steps
                ],
                "actions": [step["ego_action"] for step in copied_steps],
                "future_random_branch_keys": [
                    step["environment_random_key"] for step in copied_steps
                ],
            }
        )
        groups[group] = payload
    validate_r015_lockstep_groups(groups, probe_step=None, selected_probe_actions=())
    return groups


def build_r015_fire_trace_groups(
    shared_prefix: Sequence[Mapping[str, Any]],
    suffix_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    probe_step: int,
    selected_probe_id: str,
    selected_probe_actions: Sequence[str],
    replay_verification_by_group: Mapping[str, Mapping[str, Any]],
    masked_value_reference: Mapping[str, Any],
    masked_history: Sequence[Mapping[str, Any]],
    current_probe_response: Mapping[str, Any],
    belief_common_input: Mapping[str, Any],
    continuation_common_input: Mapping[str, Any],
    recurrent_common_input: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    """Build three 400-step groups whose only branch starts after ``t_p``."""

    if isinstance(probe_step, bool) or not 1 <= int(probe_step) <= 100:
        raise ValueError("R015 probe_step must lie from 1 through 100.")
    if not selected_probe_id or not selected_probe_actions or any(
        action not in R015_ATOMIC_ACTIONS for action in selected_probe_actions
    ):
        raise ValueError("R015 firing trace requires one registered probe script.")
    if len(selected_probe_actions) > 2:
        raise ValueError("R015 registered probe scripts contain at most two actions.")
    if set(suffix_by_group) != set(R015_FORMAL_GROUPS) or set(
        replay_verification_by_group
    ) != set(R015_FORMAL_GROUPS):
        raise ValueError("R015 firing data must cover all three formal groups.")
    prefix = _normalize_trajectory(shared_prefix)
    if len(prefix) != probe_step:
        raise ValueError("R015 shared prefix must contain records 0 through t_p-1.")
    prefix_bytes = _trajectory_bytes(prefix)
    groups: dict[str, Mapping[str, Any]] = {}
    for group in R015_FORMAL_GROUPS:
        copied_prefix = tuple(json.loads(prefix_bytes.decode("utf-8")))
        suffix = _normalize_trajectory(
            suffix_by_group[group],
            start_index=probe_step,
            allowed_belief_discontinuity_indices=(
                (probe_step + len(selected_probe_actions),)
                if group == "A2-use"
                else ()
            ),
        )
        steps = (*copied_prefix, *suffix)
        if len(steps) != 400:
            raise ValueError("Every R015 firing trajectory must contain 400 steps.")
        if suffix and copied_prefix:
            first = suffix[0]
            last = copied_prefix[-1]
            if first["controller_recurrent_state_before_sha256"] != last[
                "controller_recurrent_state_after_sha256"
            ] or first["partner_recurrent_state_before_sha256"] != last[
                "partner_recurrent_state_after_sha256"
            ]:
                raise ValueError("An R015 suffix does not continue the shared prefix state.")
            if first["controller_input"].get(
                "official_local_observation"
            ) != last["official_local_observation_after"]:
                raise ValueError("An R015 suffix does not continue the shared observation.")
        payload = _base_group_payload(
            tuple(steps),
            replay_verification=replay_verification_by_group[group],
            masked_value_reference=masked_value_reference,
        )
        payload.update(
            {
                "pre_probe_trace": [
                    copy.deepcopy(step["controller_input"])
                    for step in steps[: probe_step + 1]
                ],
                "pre_probe_actions": [
                    step["ego_action"] for step in steps[:probe_step]
                ],
                "pre_probe_random_keys": [
                    step["environment_random_key"] for step in steps[:probe_step]
                ],
            }
        )
        if group in {"A2-mask", "A2-use"}:
            probe_end = probe_step + len(selected_probe_actions)
            payload.update(
                {
                    "selected_probe_id": selected_probe_id,
                    "probe_actions": list(selected_probe_actions),
                    "masked_history": copy.deepcopy(list(masked_history)),
                    "current_probe_response": copy.deepcopy(
                        dict(current_probe_response)
                    ),
                    "belief_common_input": copy.deepcopy(dict(belief_common_input)),
                    "continuation_common_input": copy.deepcopy(
                        dict(continuation_common_input)
                    ),
                    "recurrent_common_input": copy.deepcopy(
                        dict(recurrent_common_input)
                    ),
                    "probe_random_keys": [
                        step["environment_random_key"]
                        for step in steps[probe_step:probe_end]
                    ],
                    "future_random_branch_keys": [
                        step["environment_random_key"] for step in steps[probe_end:]
                    ],
                    "belief_current_response_extra": (
                        None
                        if group == "A2-mask"
                        else copy.deepcopy(dict(current_probe_response))
                    ),
                    "continuation_direct_response": None,
                    "recurrent_direct_response": None,
                    "non_belief_response_aliases": [],
                }
            )
        groups[group] = payload
    validate_r015_lockstep_groups(
        groups,
        probe_step=probe_step,
        selected_probe_actions=tuple(selected_probe_actions),
    )
    return groups


def validate_r015_lockstep_groups(
    groups: Mapping[str, Mapping[str, Any]],
    *,
    probe_step: int | None,
    selected_probe_actions: Sequence[str],
) -> Mapping[str, Any]:
    """Recompute the structural zero or shared-prefix lockstep invariant."""

    if set(groups) != set(R015_FORMAL_GROUPS):
        raise ValueError("R015 lockstep validation requires exactly three groups.")
    trajectories: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for group in R015_FORMAL_GROUPS:
        record = groups[group]
        raw_steps = record.get("environment_steps")
        if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
            raise TypeError("R015 group environment_steps must be a sequence.")
        steps = _normalize_trajectory(
            raw_steps,
            allowed_belief_discontinuity_indices=(
                (probe_step + len(selected_probe_actions),)
                if group == "A2-use" and probe_step is not None
                else ()
            ),
        )
        if len(steps) != 400:
            raise ValueError("R015 lockstep validation requires 400-step episodes.")
        trajectories[group] = steps
    returns = {
        group: _return_from_steps(trajectories[group]) for group in R015_FORMAL_GROUPS
    }
    differences = {
        "delta_net": returns["A2-use"] - returns["A1"],
        "delta_response": returns["A2-use"] - returns["A2-mask"],
        "delta_cost": returns["A1"] - returns["A2-mask"],
    }
    if probe_step is None:
        hashes = {
            canonical_sha256(list(trajectories[group]))
            for group in R015_FORMAL_GROUPS
        }
        if len(hashes) != 1 or any(value != 0.0 for value in differences.values()):
            raise ValueError(
                "A no-fire R015 block must be byte-identical with zero differences."
            )
        prefix_sha256 = next(iter(hashes))
    else:
        if isinstance(probe_step, bool) or not 1 <= int(probe_step) <= 100:
            raise ValueError("A firing R015 block has an invalid t_p.")
        prefix_hashes = {
            canonical_sha256(list(trajectories[group][:probe_step]))
            for group in R015_FORMAL_GROUPS
        }
        if len(prefix_hashes) != 1:
            raise ValueError("R015 trajectories differ before the probe branch point.")
        prefix_sha256 = next(iter(prefix_hashes))
        actions = tuple(str(action) for action in selected_probe_actions)
        if not actions or any(action not in R015_ATOMIC_ACTIONS for action in actions):
            raise ValueError("A firing R015 block lacks the registered probe actions.")
        probe_end = probe_step + len(actions)
        for group in ("A2-mask", "A2-use"):
            if tuple(
                step["ego_action"]
                for step in trajectories[group][probe_step:probe_end]
            ) != actions:
                raise ValueError("An A2 group executed a different probe script.")
        if canonical_sha256(
            list(trajectories["A2-mask"][:probe_end])
        ) != canonical_sha256(list(trajectories["A2-use"][:probe_end])):
            raise ValueError("The two A2 groups diverged before the probe script ended.")
        for step in range(400):
            random_keys = {
                trajectories[group][step]["environment_random_key"]
                for group in R015_FORMAL_GROUPS
            }
            if len(random_keys) != 1:
                raise ValueError("The three R015 groups do not share episode random numbers.")
    if not math.isclose(
        differences["delta_net"],
        differences["delta_response"] - differences["delta_cost"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("R015 paired differences violate their exact identity.")
    return {
        "schema_version": "path_c_r015_lockstep_verification_v1",
        "verified": True,
        "probe_fired": probe_step is not None,
        "probe_step": probe_step,
        "shared_prefix_sha256": prefix_sha256,
        "returns": returns,
        "differences": differences,
    }


@dataclass(frozen=True)
class R015ExecutedSegmentV1:
    """One actually executed, contiguous portion of an R015 episode."""

    end_state: Any
    steps: tuple[Mapping[str, Any], ...]


@runtime_checkable
class R015SegmentExecutor(Protocol):
    """Run a requested group from an injected complete execution state."""

    def execute_segment(
        self,
        start_state: Any,
        *,
        group: str,
        start_step: int,
        stop_step: int,
        forced_probe_actions: tuple[str, ...],
    ) -> R015ExecutedSegmentV1:
        ...


@dataclass(frozen=True)
class R015PairedEpisodeRunnerV1:
    """Execute a common prefix once and fork only when the probe rule fires."""

    executor: R015SegmentExecutor
    clone_state: Callable[[Any], Any]
    state_sha256: Callable[[Any], str]

    def __post_init__(self) -> None:
        if not isinstance(self.executor, R015SegmentExecutor):
            raise TypeError("R015 paired execution requires a segment executor.")
        if not callable(self.clone_state) or not callable(self.state_sha256):
            raise TypeError("R015 paired execution requires state clone and hash functions.")

    def run_no_fire(
        self,
        initial_state: Any,
        *,
        replay_verification_by_group: Mapping[str, Mapping[str, Any]],
        masked_value_reference: Mapping[str, Any],
    ) -> Mapping[str, Mapping[str, Any]]:
        shared = self.executor.execute_segment(
            self.clone_state(initial_state),
            group="shared",
            start_step=0,
            stop_step=400,
            forced_probe_actions=(),
        )
        if not isinstance(shared, R015ExecutedSegmentV1):
            raise TypeError("R015 segment executor returned the wrong result type.")
        return build_r015_no_fire_trace_groups(
            shared.steps,
            replay_verification_by_group=replay_verification_by_group,
            masked_value_reference=masked_value_reference,
        )

    def run_fire(
        self,
        initial_state: Any,
        *,
        probe_step: int,
        selected_probe_id: str,
        selected_probe_actions: Sequence[str],
        replay_verification_by_group: Mapping[str, Mapping[str, Any]],
        masked_value_reference: Mapping[str, Any],
        masked_history: Sequence[Mapping[str, Any]],
        current_probe_response: Mapping[str, Any],
        belief_common_input: Mapping[str, Any],
        continuation_common_input: Mapping[str, Any],
        recurrent_common_input: Mapping[str, Any],
        masked_reference_actions: Sequence[str] | None = None,
    ) -> Mapping[str, Mapping[str, Any]]:
        if isinstance(probe_step, bool) or not 1 <= int(probe_step) <= 100:
            raise ValueError("R015 paired runner received an invalid probe step.")
        actions = tuple(str(action) for action in selected_probe_actions)
        if masked_reference_actions is None:
            selected_reference = masked_value_reference.get("selected_action_id")
            reference_actions = (
                ()
                if selected_reference in {None, "base"}
                else (str(selected_reference),)
            )
        else:
            reference_actions = tuple(str(action) for action in masked_reference_actions)
        if any(action not in R015_ATOMIC_ACTIONS for action in reference_actions) or (
            len(reference_actions) > 1
        ):
            raise ValueError("R015 A1 received an invalid masked-reference script.")
        shared = self.executor.execute_segment(
            self.clone_state(initial_state),
            group="shared",
            start_step=0,
            stop_step=probe_step,
            forced_probe_actions=(),
        )
        if not isinstance(shared, R015ExecutedSegmentV1):
            raise TypeError("R015 segment executor returned the wrong shared prefix.")
        branch_states = {
            group: self.clone_state(shared.end_state) for group in R015_FORMAL_GROUPS
        }
        branch_hashes = {self.state_sha256(state) for state in branch_states.values()}
        if len(branch_hashes) != 1 or not all(_is_sha256(item) for item in branch_hashes):
            raise ValueError("R015 groups did not fork from one complete execution state.")
        suffix_by_group: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for group in R015_FORMAL_GROUPS:
            suffix = self.executor.execute_segment(
                branch_states[group],
                group=group,
                start_step=probe_step,
                stop_step=400,
                forced_probe_actions=(
                    actions
                    if group in {"A2-mask", "A2-use"}
                    else reference_actions
                ),
            )
            if not isinstance(suffix, R015ExecutedSegmentV1):
                raise TypeError("R015 segment executor returned the wrong suffix.")
            suffix_by_group[group] = suffix.steps
        return build_r015_fire_trace_groups(
            shared.steps,
            suffix_by_group,
            probe_step=probe_step,
            selected_probe_id=selected_probe_id,
            selected_probe_actions=actions,
            replay_verification_by_group=replay_verification_by_group,
            masked_value_reference=masked_value_reference,
            masked_history=masked_history,
            current_probe_response=current_probe_response,
            belief_common_input=belief_common_input,
            continuation_common_input=continuation_common_input,
            recurrent_common_input=recurrent_common_input,
        )


def _snapshot_sha256(snapshot: Any) -> str:
    """Hash an OCV2 snapshot pytree without relying on object repr addresses."""

    try:
        import jax
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("OCV2 snapshot hashing requires JAX.") from exc
    leaves, tree = jax.tree_util.tree_flatten(
        (snapshot.key, snapshot.state, snapshot.raw_obs)
    )
    digest = hashlib.sha256(b"path_c_r015_ocv2_snapshot_v1\x00")
    tree_bytes = str(tree).encode("utf-8")
    digest.update(len(tree_bytes).to_bytes(8, "big"))
    digest.update(tree_bytes)
    for leaf in leaves:
        value = np.asarray(leaf)
        for framed in (
            str(value.dtype).encode("ascii"),
            ",".join(str(int(item)) for item in value.shape).encode("ascii"),
            np.ascontiguousarray(value).view(np.uint8).tobytes(),
        ):
            digest.update(len(framed).to_bytes(8, "big"))
            digest.update(framed)
    return digest.hexdigest()


@dataclass(frozen=True)
class OCV2PrimitiveTransitionV1:
    next_snapshot: Any
    official_observation_after: Mapping[str, Any]
    raw_team_reward: float
    done: bool
    snapshot_before_sha256: str
    snapshot_after_sha256: str


@dataclass(frozen=True)
class OCV2SnapshotStepRuntimeV1:
    """One pure OvercookedV2 transition from a detached adapter snapshot."""

    adapter: Any
    ego_agent_id: str = R015_EGO_AGENT_ID
    partner_agent_id: str = R015_PARTNER_AGENT_ID

    def __post_init__(self) -> None:
        from experiments.overcooked_v2.env_adapter import OCV2Adapter

        if not isinstance(self.adapter, OCV2Adapter):
            raise TypeError("R015 OCV2 runtime requires an OCV2Adapter.")
        if self.ego_agent_id != R015_EGO_AGENT_ID or (
            self.partner_agent_id != R015_PARTNER_AGENT_ID
        ):
            raise ValueError("R015 fixes the ego cook at agent_1 and partner at agent_0.")
        if int(self.adapter.max_steps) != 400:
            raise ValueError("R015 OCV2 runtime requires a 400-step environment.")

    def step(
        self,
        snapshot: Any,
        *,
        ego_action: str,
        partner_action: str,
    ) -> OCV2PrimitiveTransitionV1:
        if ego_action not in OCV2_ACTION_INDEX or partner_action not in OCV2_ACTION_INDEX:
            raise ValueError("R015 OCV2 runtime received an unknown primitive action.")
        result = self.adapter.step_joint_from_state(
            snapshot,
            agent_0_action=OCV2_ACTION_INDEX[partner_action],
            agent_1_action=OCV2_ACTION_INDEX[ego_action],
        )
        reward_0 = _finite(result.step.rewards.get("agent_0"), field="agent_0 reward")
        reward_1 = _finite(result.step.rewards.get("agent_1"), field="agent_1 reward")
        if reward_0 != reward_1:
            raise ValueError("OCV2 did not broadcast one shared team reward.")
        done = result.step.dones.get("__all__")
        if not isinstance(done, (bool, np.bool_)):
            raise ValueError("OCV2 transition lacks the shared done boundary.")
        observation_key = self.ego_agent_id
        if observation_key not in result.step.obs:
            raise ValueError("OCV2 transition lacks the ego official observation.")
        return OCV2PrimitiveTransitionV1(
            next_snapshot=result.snapshot,
            official_observation_after=copy.deepcopy(result.step.obs[observation_key]),
            raw_team_reward=reward_0,
            done=bool(done),
            snapshot_before_sha256=_snapshot_sha256(snapshot),
            snapshot_after_sha256=_snapshot_sha256(result.snapshot),
        )


@runtime_checkable
class R015ReplayBackend(Protocol):
    """Frozen backend that re-executes checkpoints instead of trusting records."""

    def replay_trace(
        self,
        trace: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...

    def replay_probe_decision(
        self,
        evidence: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...

    def replay_safety_comparison(
        self,
        comparison: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class CallableR015ReplayBackendV1:
    """Explicit adapter used by a frozen manifest to bind three replay kernels."""

    trace_replayer: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]
    decision_replayer: Callable[
        [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
    ]
    safety_replayer: Callable[
        [Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]
    ]

    def __post_init__(self) -> None:
        if not all(
            callable(item)
            for item in (
                self.trace_replayer,
                self.decision_replayer,
                self.safety_replayer,
            )
        ):
            raise TypeError("Every R015 replay kernel must be callable.")

    def replay_trace(self, trace, context):
        return self.trace_replayer(trace, context)

    def replay_probe_decision(self, evidence, context):
        return self.decision_replayer(evidence, context)

    def replay_safety_comparison(self, comparison, context):
        return self.safety_replayer(comparison, context)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> Mapping[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        import yaml

        payload = yaml.safe_load(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("R015 controller manifest must be a mapping.")
    return payload


def _resolve_replay_backend(context: Mapping[str, Any]) -> R015ReplayBackend:
    injected = context.get("_r015_replay_backend")
    if injected is not None:
        if not isinstance(injected, R015ReplayBackend):
            raise TypeError("Injected R015 replay backend has the wrong interface.")
        return injected
    manifest_value = context.get("continuation_planner_path")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ValueError("R015 replay requires a bound continuation-planner manifest.")
    manifest_path = Path(manifest_value).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("The R015 continuation-planner manifest is missing.")
    expected_manifest_sha = context.get("continuation_planner_sha256")
    if not _is_sha256(expected_manifest_sha) or _file_sha256(
        manifest_path
    ) != expected_manifest_sha:
        raise ValueError("The R015 continuation-planner manifest hash changed.")
    manifest = _load_manifest(manifest_path)
    raw_backend = manifest.get("replay_backend")
    if not isinstance(raw_backend, Mapping) or set(raw_backend) != {
        "implementation_path",
        "implementation_sha256",
        "factory_name",
    }:
        raise ValueError("The R015 controller manifest lacks a replay backend binding.")
    if raw_backend.get("factory_name") != "build_r015_replay_backend" or not _is_sha256(
        raw_backend.get("implementation_sha256")
    ):
        raise ValueError("The R015 replay-backend factory binding changed.")
    implementation_path = Path(str(raw_backend["implementation_path"]))
    if not implementation_path.is_absolute():
        implementation_path = (manifest_path.parent / implementation_path).resolve()
    if not implementation_path.is_file() or _file_sha256(
        implementation_path
    ) != raw_backend["implementation_sha256"]:
        raise ValueError("The R015 replay-backend implementation is missing or changed.")
    module_name = f"path_c_r015_runtime_{raw_backend['implementation_sha256']}"
    module_spec = importlib.util.spec_from_file_location(module_name, implementation_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("The R015 replay-backend implementation cannot be loaded.")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    factory = getattr(module, str(raw_backend["factory_name"]), None)
    if not callable(factory):
        raise ValueError("The R015 replay-backend factory is missing.")
    backend = factory(copy.deepcopy(dict(context)))
    if not isinstance(backend, R015ReplayBackend):
        raise TypeError("The registered R015 replay factory returned the wrong interface.")
    return backend


def _verified_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or value.get("verified") is not True:
        raise ValueError(f"The R015 {label} replay did not verify its input.")
    return copy.deepcopy(dict(value))


def verify_r015_trace_manifest(
    trace: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Re-execute every recorded action, state write, reward, and boundary."""

    if not isinstance(trace, Mapping) or not isinstance(context, Mapping):
        raise TypeError("R015 trace replay requires mapping inputs.")
    backend = _resolve_replay_backend(context)
    return _verified_mapping(
        backend.replay_trace(copy.deepcopy(dict(trace)), copy.deepcopy(dict(context))),
        label="trace",
    )


def verify_r015_probe_decision(
    evidence: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Recompute six-script values and the first-positive decision from branches."""

    if not isinstance(evidence, Mapping) or not isinstance(context, Mapping):
        raise TypeError("R015 decision replay requires mapping inputs.")
    backend = _resolve_replay_backend(context)
    return _verified_mapping(
        backend.replay_probe_decision(
            copy.deepcopy(dict(evidence)),
            copy.deepcopy(dict(context)),
        ),
        label="probe-decision",
    )


def verify_r015_safety_comparison(
    comparison: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Reconstruct and replay every wrong-delivery safety branch."""

    if not isinstance(comparison, Mapping) or not isinstance(context, Mapping):
        raise TypeError("R015 safety replay requires mapping inputs.")
    backend = _resolve_replay_backend(context)
    return _verified_mapping(
        backend.replay_safety_comparison(
            copy.deepcopy(dict(comparison)),
            copy.deepcopy(dict(context)),
        ),
        label="safety",
    )


@dataclass(frozen=True)
class R015FormalBlockRequestV1:
    """一个冻结正式抽样尝试；编号和 seed 必须都来自 v4 日程。"""

    round_index: int
    audit_unit_id: str
    prototype_id: str
    episode_seed: int
    ego_position: int
    mechanical_attempt_index: int


class R015MechanicalBlockInvalidError(RuntimeError):
    """只表示真实环境回合或在线过滤确已失效，可按整块规则处理。"""


@dataclass(frozen=True)
class R015FormalBlockProductionResultV1:
    """正式块生产结果；机械无效时不得携带部分回报证据。"""

    mechanically_valid: bool
    paired_block: Mapping[str, Any] | None
    invalid_reason: str | None = None


@runtime_checkable
class R015FormalBlockProducer(Protocol):
    """从冻结尝试生成一个可由正式裁决器完整重放的 v2 配对块。"""

    def produce_formal_paired_block(
        self,
        request: R015FormalBlockRequestV1,
    ) -> R015FormalBlockProductionResultV1:
        ...


def build_r015_formal_sampling_schedule(
    *,
    prototype_ids: Sequence[str],
    root_seed: int,
    n_rounds: int = 2500,
) -> Mapping[str, Any]:
    """在正式数据前生成四原型等权轮向量及每块 32 个替换 seed。"""

    ordered_prototypes = tuple(str(item) for item in prototype_ids)
    if len(ordered_prototypes) != 4 or len(set(ordered_prototypes)) != 4:
        raise ValueError("R015 formal sampling requires exactly four prototypes.")
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise ValueError("R015 formal root seed must be a non-negative integer.")
    if isinstance(n_rounds, bool) or int(n_rounds) != 2500:
        raise ValueError("R015 formal sampling fixes N_rounds_max at 2500.")
    entries: list[Mapping[str, Any]] = []
    for round_index in range(1, n_rounds + 1):
        for prototype_index, prototype_id in enumerate(ordered_prototypes):
            primary_coordinate = [
                "r015_formal_sampling_schedule_v4",
                root_seed,
                round_index,
                prototype_index,
                prototype_id,
            ]
            episode_seeds_by_attempt_index: list[int] = []
            for mechanical_attempt_index in range(
                R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT
            ):
                digest = canonical_sha256(
                    [
                        "r015_formal_prefrozen_replacement_seed_v4",
                        root_seed,
                        round_index,
                        prototype_index,
                        prototype_id,
                        mechanical_attempt_index,
                    ]
                )
                episode_seed = int(digest[-8:], 16)
                episode_seeds_by_attempt_index.append(episode_seed)
            entries.append(
                {
                    "round_index": round_index,
                    "audit_unit_id": canonical_sha256(
                        ["r015_formal_audit_unit_v4", *primary_coordinate]
                    ),
                    "prototype_id": prototype_id,
                    "episode_seeds_by_attempt_index": (
                        episode_seeds_by_attempt_index
                    ),
                    "ego_position": 1,
                }
            )
    return {
        "schema_version": R015_FORMAL_SAMPLING_SCHEDULE_SCHEMA,
        "experiment_id": "R015",
        "frozen_before_formal_data": True,
        "selection_uses_outcomes": False,
        "round_sampling_contract": R015_FORMAL_ROUND_SAMPLING_CONTRACT,
        "mechanical_replacement_attempts_per_coordinate": (
            R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT
        ),
        "mechanical_replacement_attempt_indices": "0_through_31",
        "replacement_seed_selection_uses_outcomes": False,
        "replacement_episode_seed_derivation_id": (
            R015_FORMAL_REPLACEMENT_SEED_DERIVATION_ID
        ),
        "replacement_episode_seed_collision_policy": (
            "allow_value_collisions_without_redraw"
        ),
        "audit_unit_changes_across_replacement_attempts": False,
        "mechanical_attempt_exhaustion_rule": "terminate_entire_formal_audit",
        "round_vectors_generated_independently": True,
        "ego_agent_id": R015_EGO_AGENT_ID,
        "partner_agent_id": R015_PARTNER_AGENT_ID,
        "zero_count_independence_contract": (
            "independent_firing_prefixes_across_prototypes_v1"
        ),
        "firing_prefixes_independent_across_prototypes": True,
        "cross_prototype_shared_randomness_before_firing": False,
        "n_rounds": n_rounds,
        "formal_root_seed": root_seed,
        "entries": entries,
    }


def _jsonable_formal(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            str(field): _jsonable_formal(getattr(value, field))
            for field in value.__dataclass_fields__
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable_formal(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_formal(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"R015 formal evidence contains unsupported value {type(value)!r}.")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(
        _jsonable_formal(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append_jsonl_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonable_formal(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _ensure_empty_jsonl_atomic(path: Path) -> None:
    """Create the empty invalid-attempt ledger even when no block was replaced."""

    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("R015 formal JSON artifact must contain one mapping.")
    return payload


def _load_formal_ledger(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"R015 formal ledger has an incomplete line {line_number}."
            ) from error
        if not isinstance(row, Mapping):
            raise ValueError("R015 formal ledger rows must be mappings.")
        rows.append(row)
    return rows


FormalRuntimeCoordinate = tuple[int, str, str, tuple[int, ...], int]


def _formal_ledger_coordinate(
    row: Mapping[str, Any],
) -> tuple[int, str, str, int, int, int]:
    integer_values = (
        row.get("round_index"),
        row.get("episode_seed"),
        row.get("ego_position"),
        row.get("mechanical_attempt_index"),
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integer_values
    ) or integer_values[0] == 0 or not isinstance(
        row.get("audit_unit_id"), str
    ) or not isinstance(row.get("prototype_id"), str):
        raise ValueError("R015 formal ledger has an invalid coordinate type.")
    return (
        integer_values[0],
        row["audit_unit_id"],
        row["prototype_id"],
        integer_values[1],
        integer_values[2],
        integer_values[3],
    )


def _validate_formal_block_identity_without_effect_read(
    payload: Mapping[str, Any],
    *,
    expected: tuple[int, str, str, int, int, int],
    runtime_binding_sha256: str,
) -> bool:
    """只读正式坐标和触发标记；这里不访问任何实验组回报或配对差。"""

    (
        round_index,
        audit_unit_id,
        prototype_id,
        episode_seed,
        ego_position,
        mechanical_attempt_index,
    ) = expected
    if payload.get("schema_version") != "path_c_r015_paired_block_v2" or (
        payload.get("phase") != "formal"
    ):
        raise ValueError("R015 formal producer returned a non-formal block schema.")
    recorded_integers = (
        payload.get("round_index"),
        payload.get("episode_seed"),
        payload.get("ego_position"),
        payload.get("mechanical_attempt_index"),
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in recorded_integers
    ) or not isinstance(payload.get("audit_unit_id"), str) or not isinstance(
        payload.get("prototype_id"), str
    ):
        raise ValueError("R015 formal block has an invalid sampling-coordinate type.")
    identity = (
        recorded_integers[0],
        payload["audit_unit_id"],
        payload["prototype_id"],
        recorded_integers[1],
        recorded_integers[2],
        recorded_integers[3],
    )
    if identity != (
        round_index,
        audit_unit_id,
        prototype_id,
        episode_seed,
        ego_position,
        mechanical_attempt_index,
    ):
        raise ValueError("R015 formal block differs from its frozen sampling coordinate.")
    if payload.get("formal_effect_look_number") != 1 or (
        payload.get("pilot_data") is not False
    ):
        raise ValueError("R015 formal producer mixed pilot or another effect-look index.")
    if not _is_sha256(runtime_binding_sha256) or payload.get(
        "runtime_binding_sha256"
    ) != runtime_binding_sha256:
        raise ValueError("R015 formal block changed its frozen runtime binding.")
    firing_indicator = payload.get("firing_indicator")
    if not isinstance(firing_indicator, bool):
        raise ValueError("R015 formal block lacks one shared firing indicator.")
    return firing_indicator


def _bind_formal_runtime_to_block(
    payload: Mapping[str, Any],
    *,
    runtime_binding_sha256: str,
) -> Mapping[str, Any]:
    """Add the runner-owned runtime digest after producer replay has succeeded."""

    if not _is_sha256(runtime_binding_sha256):
        raise ValueError("R015 formal block lacks a valid runtime binding.")
    block = _jsonable_formal(payload)
    if not isinstance(block, Mapping):
        raise TypeError("R015 formal producer did not return one block mapping.")
    producer_runtime_binding = block.get("runtime_binding_sha256")
    if producer_runtime_binding not in (None, runtime_binding_sha256):
        raise ValueError("R015 producer supplied a different frozen runtime binding.")
    return {**block, "runtime_binding_sha256": runtime_binding_sha256}


_FORMAL_NONDETERMINISTIC_THROUGHPUT_FIELDS = frozenset(
    {
        "wall_seconds",
        "true_transitions_per_second",
        "computed_transitions_per_second",
        "jit_compilations",
        "microbatch_wall_seconds",
        "microbatch_true_transitions_per_second",
    }
)
_FORMAL_NONDETERMINISTIC_TOP_LEVEL_TELEMETRY_FIELDS = frozenset(
    {
        "actual_execution_wall_seconds",
        "independent_replay_wall_seconds",
    }
)


def _validate_nonnegative_finite_telemetry(
    value: Any,
    *,
    field: str,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_nonnegative_finite_telemetry(
                child,
                field=f"{field}.{key}",
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_nonnegative_finite_telemetry(
                child,
                field=f"{field}[{index}]",
            )
        return
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(
                f"R015 formal block has invalid nonnegative telemetry field {field}."
            )
        return
    raise TypeError(f"R015 formal block has unsupported telemetry field {field}.")


def _deterministic_throughput_projection(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: copy.deepcopy(child)
        for key, child in value.items()
        if key not in _FORMAL_NONDETERMINISTIC_THROUGHPUT_FIELDS
    }


def _require_nonnegative_finite_telemetry_number(value: Any, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (
        not math.isfinite(float(value))
    ) or float(value) < 0.0:
        raise ValueError(
            f"R015 formal block has invalid nonnegative telemetry field {field}."
        )


def _validate_nondeterministic_execution_telemetry(
    execution: Mapping[str, Any],
    *,
    field: str,
) -> None:
    _require_nonnegative_finite_telemetry_number(
        execution.get("wall_seconds"),
        field=f"{field}.wall_seconds",
    )
    jit_compilations = execution.get("jit_compilations")
    if isinstance(jit_compilations, bool) or not isinstance(
        jit_compilations, int
    ) or jit_compilations < 0:
        raise ValueError(
            f"R015 formal block has invalid nonnegative telemetry field "
            f"{field}.jit_compilations."
        )
    for name in (
        "true_transitions_per_second",
        "computed_transitions_per_second",
    ):
        if name in execution and execution[name] is not None:
            _require_nonnegative_finite_telemetry_number(
                execution[name],
                field=f"{field}.{name}",
            )
    for name, allow_none in (
        ("microbatch_wall_seconds", False),
        ("microbatch_true_transitions_per_second", True),
    ):
        if name not in execution:
            continue
        values = execution[name]
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                f"R015 formal block has malformed telemetry field {field}.{name}."
            )
        for index, value in enumerate(values):
            if value is None and allow_none:
                continue
            _require_nonnegative_finite_telemetry_number(
                value,
                field=f"{field}.{name}[{index}]",
            )


def _formal_orphan_replay_projection(
    block: Mapping[str, Any],
) -> Mapping[str, Any]:
    """孤立块重算比较时剔除已核验的非确定机械吞吐字段。"""

    normalized = _jsonable_formal(block)
    if not isinstance(normalized, Mapping):
        raise TypeError("R015 formal orphan recovery requires one block mapping.")
    telemetry = normalized.get("mechanical_execution_telemetry")
    if not isinstance(telemetry, Mapping) or telemetry.get("schema_version") != (
        "path_c_r015_formal_production_telemetry_v1"
    ):
        raise ValueError("R015 formal block lacks registered mechanical telemetry.")
    if telemetry.get("effect_values_in_telemetry") is not False or telemetry.get(
        "independent_replay_reused_compiled_device_programs"
    ) is not True:
        raise ValueError("R015 formal telemetry leaked an effect or changed replay use.")
    _validate_nonnegative_finite_telemetry(
        telemetry,
        field="mechanical_execution_telemetry",
    )
    for field in (
        "actual_execution_wall_seconds",
        "independent_replay_wall_seconds",
    ):
        _require_nonnegative_finite_telemetry_number(
            telemetry.get(field),
            field=f"mechanical_execution_telemetry.{field}",
        )
    actual_execution = telemetry.get("actual_execution")
    replay_execution = telemetry.get("independent_replay_execution")
    if not isinstance(actual_execution, Mapping) or not isinstance(
        replay_execution, Mapping
    ):
        raise ValueError("R015 formal telemetry lacks both execution summaries.")
    for label, execution in (
        ("actual_execution", actual_execution),
        ("independent_replay_execution", replay_execution),
    ):
        _validate_nondeterministic_execution_telemetry(
            execution,
            field=f"mechanical_execution_telemetry.{label}",
        )
    if actual_execution.get("host_sync_inside_environment_loop") is not False or (
        replay_execution.get("host_sync_inside_environment_loop") is not False
    ) or _deterministic_throughput_projection(
        actual_execution
    ) != _deterministic_throughput_projection(replay_execution):
        raise ValueError(
            "R015 formal actual and replay deterministic execution counts differ."
        )
    actual_planning = telemetry.get("actual_planning_real_environment_transitions")
    replay_planning = telemetry.get(
        "independent_replay_planning_real_environment_transitions"
    )
    if isinstance(actual_planning, bool) or not isinstance(actual_planning, int) or (
        actual_planning < 0
    ) or replay_planning != actual_planning:
        raise ValueError("R015 formal actual and replay planning counts differ.")
    try:
        consultations = normalized["probe_decision"]["consultations"]
        planning_costs = [
            row["cost_accounting"]["new_real_environment_transitions"]
            for row in consultations
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "R015 formal block lacks consultation planning-cost evidence."
        ) from error
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in planning_costs
    ):
        raise ValueError("R015 formal block has an invalid consultation planning cost.")
    evidence_planning = sum(planning_costs)
    if evidence_planning != actual_planning:
        raise ValueError(
            "R015 formal telemetry planning count differs from block evidence."
        )
    projected = copy.deepcopy(dict(normalized))
    projected_telemetry = copy.deepcopy(dict(telemetry))
    for field in _FORMAL_NONDETERMINISTIC_TOP_LEVEL_TELEMETRY_FIELDS:
        projected_telemetry.pop(field, None)
    projected_telemetry["actual_execution"] = (
        _deterministic_throughput_projection(actual_execution)
    )
    projected_telemetry["independent_replay_execution"] = (
        _deterministic_throughput_projection(replay_execution)
    )
    projected["mechanical_execution_telemetry"] = projected_telemetry
    return projected


def _reconcile_formal_ledger(
    *,
    ledger_path: Path,
    expected_coordinates: Sequence[FormalRuntimeCoordinate],
    block_directory: Path,
    runtime_binding_sha256: str,
) -> tuple[list[Mapping[str, Any]], Mapping[str, int]]:
    rows = _load_formal_ledger(ledger_path)
    if len(rows) > len(expected_coordinates):
        raise ValueError("R015 formal ledger exceeds the frozen schedule.")
    firing_counts: dict[str, int] = {
        coordinate[2]: 0 for coordinate in expected_coordinates[:4]
    }
    for index, row in enumerate(rows):
        if set(row) != {
            "schema_version",
            "sequence_index",
            "round_index",
            "audit_unit_id",
            "prototype_id",
            "episode_seed",
            "ego_position",
            "mechanical_attempt_index",
            "firing_indicator",
            "block_path",
            "block_sha256",
            "runtime_binding_sha256",
        } or row.get("schema_version") != "path_c_r015_formal_block_ledger_v2":
            raise ValueError("R015 formal ledger has the wrong schema.")
        expected_coordinate = expected_coordinates[index]
        row_coordinate = _formal_ledger_coordinate(row)
        attempt_index = row_coordinate[-1]
        attempt_seeds = expected_coordinate[3]
        if not 0 <= attempt_index < len(attempt_seeds):
            raise ValueError("R015 formal ledger has an invalid replacement attempt.")
        expected_ledger_coordinate = (
            expected_coordinate[0],
            expected_coordinate[1],
            expected_coordinate[2],
            attempt_seeds[attempt_index],
            expected_coordinate[4],
            attempt_index,
        )
        if row.get("sequence_index") != index or row_coordinate != (
            expected_ledger_coordinate
        ):
            raise ValueError("R015 formal resume order differs from the frozen schedule.")
        if row.get("runtime_binding_sha256") != runtime_binding_sha256:
            raise ValueError("R015 formal resume changed its runtime binding.")
        block_path = Path(str(row.get("block_path", ""))).resolve()
        try:
            block_path.relative_to(block_directory.resolve())
        except ValueError as error:
            raise ValueError("R015 formal ledger points outside its block directory.") from error
        if not block_path.is_file():
            raise ValueError("R015 formal block file is missing or changed.")
        block_bytes = block_path.read_bytes()
        if hashlib.sha256(block_bytes).hexdigest() != row.get("block_sha256"):
            raise ValueError("R015 formal block file is missing or changed.")
        if not isinstance(row.get("firing_indicator"), bool):
            raise ValueError("R015 formal ledger has a non-boolean firing indicator.")
        recovered_block = json.loads(block_bytes)
        if not isinstance(recovered_block, Mapping):
            raise ValueError("R015 formal block file must contain one mapping.")
        recovered_firing = _validate_formal_block_identity_without_effect_read(
            recovered_block,
            expected=expected_ledger_coordinate,
            runtime_binding_sha256=runtime_binding_sha256,
        )
        if recovered_firing is not row["firing_indicator"]:
            raise ValueError("R015 formal ledger changed its block firing indicator.")
        firing_counts[str(row["prototype_id"])] += int(row["firing_indicator"])
    return rows, firing_counts


def _reconcile_formal_invalid_attempts(
    *,
    invalid_path: Path,
    expected_coordinates: Sequence[FormalRuntimeCoordinate],
    ledger_rows: Sequence[Mapping[str, Any]],
    runtime_binding_sha256: str,
) -> Mapping[int, int]:
    """核对失效尝试严格使用日程前缀，并返回每个坐标的下一个尝试编号。"""

    rows = _load_formal_ledger(invalid_path)
    invalid_count_by_sequence: dict[int, int] = {}
    latest_sequence = -1
    for row in rows:
        if set(row) != {
            "schema_version",
            "sequence_index",
            "round_index",
            "audit_unit_id",
            "prototype_id",
            "episode_seed",
            "ego_position",
            "mechanical_attempt_index",
            "reason",
            "schedule_bound_seed",
            "outcome_field_read",
            "runtime_binding_sha256",
        } or row.get("schema_version") != (
            "path_c_r015_mechanical_replacement_attempt_v2"
        ):
            raise ValueError("R015 formal invalid-attempt ledger has the wrong schema.")
        raw_sequence_index = row.get("sequence_index")
        if isinstance(raw_sequence_index, bool) or not isinstance(
            raw_sequence_index, int
        ):
            raise ValueError("R015 formal invalid attempt has a non-integer sequence.")
        sequence_index = raw_sequence_index
        if not 0 <= sequence_index < len(expected_coordinates):
            raise ValueError("R015 formal invalid attempt is outside the schedule.")
        if sequence_index < latest_sequence or sequence_index > len(ledger_rows):
            raise ValueError("R015 formal invalid attempts are out of resume order.")
        latest_sequence = sequence_index
        expected_coordinate = expected_coordinates[sequence_index]
        expected_attempt_index = invalid_count_by_sequence.get(sequence_index, 0)
        attempt_seeds = expected_coordinate[3]
        if expected_attempt_index >= len(attempt_seeds):
            raise ValueError("R015 formal invalid attempts exceed the frozen schedule.")
        expected_identity = (
            expected_coordinate[0],
            expected_coordinate[1],
            expected_coordinate[2],
            attempt_seeds[expected_attempt_index],
            expected_coordinate[4],
            expected_attempt_index,
        )
        if _formal_ledger_coordinate(row) != expected_identity or (
            row.get("schedule_bound_seed") is not True
        ) or row.get("outcome_field_read") is not False or (
            row.get("runtime_binding_sha256") != runtime_binding_sha256
        ) or not str(row.get("reason", "")):
            raise ValueError(
                "R015 formal invalid attempt differs from its frozen replacement seed."
            )
        invalid_count_by_sequence[sequence_index] = expected_attempt_index + 1

    for sequence_index, ledger_row in enumerate(ledger_rows):
        invalid_count = invalid_count_by_sequence.get(sequence_index, 0)
        if int(ledger_row["mechanical_attempt_index"]) != invalid_count:
            raise ValueError(
                "R015 formal success does not follow its complete invalid-attempt prefix."
            )
    return invalid_count_by_sequence


def _write_count_checkpoint(
    *,
    path: Path,
    checkpoint_round: int,
    firing_counts: Mapping[str, int],
) -> Mapping[str, Any]:
    payload = {
        "schema_version": "path_c_r015_firing_count_checkpoint_v1",
        "experiment_id": "R015",
        "checkpoint_round_count": int(checkpoint_round),
        "firing_counts_by_prototype": {
            str(key): int(value) for key, value in firing_counts.items()
        },
    }
    if path.exists():
        if _load_json_mapping(path) != payload:
            raise ValueError("An existing R015 firing-count checkpoint changed.")
    else:
        _write_json_atomic(path, payload)
    return payload


def _assemble_formal_dataset_without_effect_read(
    *,
    path: Path,
    preregistration_sha256: str,
    support_report_sha256: str,
    checkpoint_paths: Sequence[Path],
    ledger_rows: Sequence[Mapping[str, Any]],
    ledger_path: Path,
    invalid_attempts_path: Path,
    runtime_binding_sha256: str,
) -> None:
    """按字节组装最终数据集；配对差第一次计算仍只发生在裁决器中。"""

    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not ledger_path.is_file() or not invalid_attempts_path.is_file():
        raise ValueError("R015 final dataset requires both complete execution ledgers.")
    if not _is_sha256(runtime_binding_sha256):
        raise ValueError("R015 final dataset lacks its frozen runtime binding.")
    prefix = {
        "schema_version": "path_c_r015_formal_dataset_v3",
        "experiment_id": "R015",
        "analysis_kind": "effect_final",
        "formal_effect_look_number": 1,
        "prior_formal_effect_look_count": 0,
        "formal_effect_look_round": 2500,
        "pilot_data_included": False,
        "preregistration_sha256": preregistration_sha256,
        "support_report_sha256": support_report_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
        "formal_block_ledger": {
            "path": str(ledger_path.resolve()),
            "sha256": _file_sha256(ledger_path),
        },
        "mechanically_invalid_attempts": {
            "path": str(invalid_attempts_path.resolve()),
            "sha256": _file_sha256(invalid_attempts_path),
        },
        "firing_count_checkpoint_sha256": [
            _file_sha256(checkpoint_path) for checkpoint_path in checkpoint_paths
        ],
    }
    prefix_bytes = json.dumps(
        prefix,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not prefix_bytes.endswith(b"}"):
        raise RuntimeError("R015 dataset prefix serialization failed.")
    with temporary.open("wb") as handle:
        handle.write(prefix_bytes[:-1])
        handle.write(b',"paired_blocks":[')
        for index, row in enumerate(ledger_rows):
            if index:
                handle.write(b",")
            block_path = Path(str(row["block_path"]))
            block_bytes = block_path.read_bytes().strip()
            if not block_bytes.startswith(b"{") or not block_bytes.endswith(b"}"):
                raise ValueError("R015 formal block file is not one JSON mapping.")
            if hashlib.sha256(block_bytes + b"\n").hexdigest() == row["block_sha256"]:
                # _write_json_atomic terminates files with one newline.
                pass
            elif hashlib.sha256(block_bytes).hexdigest() != row["block_sha256"]:
                raise ValueError("R015 formal block changed during dataset assembly.")
            handle.write(block_bytes)
        handle.write(b"]}\n")
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        candidate_bytes = temporary.read_bytes()
        existing_bytes = path.read_bytes()
        temporary.unlink()
        if existing_bytes != candidate_bytes:
            raise ValueError(
                "An existing R015 final dataset differs from the current ledgers "
                "or runtime binding."
            )
    else:
        temporary.replace(path)


def _decision_payload(decision: Any) -> Mapping[str, Any]:
    if hasattr(decision, "__dataclass_fields__"):
        return {
            field: _jsonable_formal(getattr(decision, field))
            for field in decision.__dataclass_fields__
        }
    if isinstance(decision, Mapping):
        return _jsonable_formal(decision)
    raise TypeError("R015 formal adjudicator returned an unsupported result.")


def _decision_sha256(decision_payload: Mapping[str, Any]) -> str:
    return canonical_sha256(_jsonable_formal(decision_payload))


def _publish_or_verify_terminal(
    path: Path,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """原子发布一次终局；若已存在，则要求重算内容完全相同。"""

    normalized = _jsonable_formal(payload)
    if path.exists():
        if _load_json_mapping(path) != normalized:
            raise ValueError("Existing R015 terminal differs from recomputed evidence.")
    else:
        _write_json_atomic(path, normalized)
    return normalized


def _kill_terminal_payload(
    *,
    preregistration_sha256: str,
    runtime_binding_sha256: str,
    completed_round_count: int,
    completed_block_count: int,
    ledger_path: Path,
    invalid_path: Path,
    checkpoint_paths: Sequence[Path],
    decision: Any,
) -> Mapping[str, Any]:
    decision_payload = _decision_payload(decision)
    return {
        "schema_version": "path_c_r015_formal_terminal_v2",
        "experiment_id": "R015",
        "termination_kind": "firing_count_registered_negative",
        "preregistration_sha256": preregistration_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
        "completed_round_count": completed_round_count,
        "completed_block_count": completed_block_count,
        "effect_value_read": False,
        "formal_block_ledger_sha256": _file_sha256(ledger_path),
        "mechanically_invalid_attempts_sha256": _file_sha256(invalid_path),
        "firing_count_checkpoint_sha256": [
            _file_sha256(path) for path in checkpoint_paths
        ],
        "decision_sha256": _decision_sha256(decision_payload),
        "decision": decision_payload,
    }


def _effect_receipt_base(
    *,
    preregistration: Any,
    support_report_sha256: str,
) -> Mapping[str, Any]:
    assert preregistration.formal_dataset_path is not None
    return {
        "schema_version": "path_c_r015_formal_effect_view_receipt_v3",
        "experiment_id": "R015",
        "formal_effect_look_number": 1,
        "formal_effect_look_round": preregistration.statistics.n_rounds_max,
        "preregistration_sha256": preregistration.source_sha256,
        "formal_dataset_path": str(preregistration.formal_dataset_path),
        "formal_dataset_sha256": _file_sha256(preregistration.formal_dataset_path),
        "support_report_sha256": support_report_sha256,
    }


def _effect_terminal_payload(
    *,
    preregistration: Any,
    runtime_binding_sha256: str,
    completed_block_count: int,
    ledger_path: Path,
    invalid_path: Path,
    checkpoint_paths: Sequence[Path],
    started_receipt_sha256: str,
    decision_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not _is_sha256(started_receipt_sha256):
        raise ValueError("R015 effect terminal lacks its started-view receipt digest.")
    return {
        "schema_version": "path_c_r015_formal_terminal_v2",
        "experiment_id": "R015",
        "termination_kind": "single_effect_view_completed",
        "preregistration_sha256": preregistration.source_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
        "completed_round_count": preregistration.statistics.n_rounds_max,
        "completed_block_count": completed_block_count,
        "effect_value_read": True,
        "formal_dataset_sha256": _file_sha256(preregistration.formal_dataset_path),
        "formal_block_ledger_sha256": _file_sha256(ledger_path),
        "mechanically_invalid_attempts_sha256": _file_sha256(invalid_path),
        "firing_count_checkpoint_sha256": [
            _file_sha256(path) for path in checkpoint_paths
        ],
        "formal_effect_view_started_sha256": started_receipt_sha256,
        "decision_sha256": _decision_sha256(decision_payload),
        "decision": _jsonable_formal(decision_payload),
    }


def _complete_or_validate_effect_receipt(
    *,
    receipt_path: Path,
    receipt_base: Mapping[str, Any],
    terminal_path: Path,
    terminal_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """终局发布后完成无摘要循环的收据，恢复时重验同一链。"""

    if not receipt_path.is_file() or not terminal_path.is_file():
        raise ValueError("R015 effect completion requires both receipt and terminal files.")
    started_sha256 = terminal_payload.get("formal_effect_view_started_sha256")
    decision_sha256 = terminal_payload.get("decision_sha256")
    if not _is_sha256(started_sha256) or not _is_sha256(decision_sha256):
        raise ValueError("R015 effect completion has invalid receipt or decision digests.")
    if _file_sha256(terminal_path) != hashlib.sha256(
        json.dumps(
            _jsonable_formal(terminal_payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    ).hexdigest():
        raise ValueError("R015 effect terminal bytes changed before receipt completion.")
    initial_receipt = {**receipt_base, "status": "consumed_before_adjudication"}
    terminal_sha256 = _file_sha256(terminal_path)
    completed_receipt = {
        **receipt_base,
        "status": "completed",
        "started_receipt_sha256": started_sha256,
        "terminal_path": str(terminal_path),
        "terminal_sha256": terminal_sha256,
        "decision_sha256": decision_sha256,
    }
    recorded_receipt = _load_json_mapping(receipt_path)
    if recorded_receipt == initial_receipt:
        if _file_sha256(receipt_path) != started_sha256:
            raise ValueError("R015 started-view receipt digest changed.")
        _write_json_atomic(receipt_path, completed_receipt)
        recorded_receipt = _load_json_mapping(receipt_path)
    if recorded_receipt != completed_receipt:
        raise ValueError("R015 completed effect receipt differs from terminal evidence.")
    return completed_receipt


def run_r015_formal(
    preregistration_path: str | Path,
    *,
    producer: R015FormalBlockProducer,
    output_directory: str | Path,
    runtime_binding_sha256: str,
    maximum_mechanical_attempts: int = 32,
) -> Mapping[str, Any]:
    """运行冻结正式轮次、次数提前停止和第 2500 轮唯一效应查看。"""

    from experiments.overcooked_v2.path_c_r015 import (
        _load_formal_sampling_schedule,
        evaluate_r015_firing_count_checkpoint,
        load_and_adjudicate_r015,
        load_r015_preregistration,
    )

    if not isinstance(producer, R015FormalBlockProducer):
        raise TypeError("R015 formal execution requires a v2 formal block producer.")
    if not _is_sha256(runtime_binding_sha256):
        raise ValueError("R015 formal execution requires a frozen runtime digest.")
    if isinstance(maximum_mechanical_attempts, bool) or int(
        maximum_mechanical_attempts
    ) != R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT:
        raise ValueError(
            "R015 formal execution requires all 32 pre-frozen replacement seeds."
        )
    preregistration = load_r015_preregistration(preregistration_path)
    support_members = preregistration._require_frozen_support()
    schedule_set = _load_formal_sampling_schedule(preregistration, support_members)
    prototype_order = tuple(
        candidate.candidate_id for candidate in preregistration.support_spec.candidates
    )
    prototype_rank = {value: index for index, value in enumerate(prototype_order)}
    expected_coordinates = tuple(
        sorted(schedule_set, key=lambda item: (item[0], prototype_rank[item[2]]))
    )
    if preregistration.formal_dataset_path is None or (
        preregistration.firing_count_checkpoint_paths is None
    ) or preregistration.formal_view_record_path is None:
        raise ValueError("R015 frozen formal output paths are incomplete.")
    if len(preregistration.firing_count_checkpoint_paths) != 5:
        raise ValueError("R015 requires five frozen firing-count checkpoint paths.")
    output_dir = Path(output_directory).resolve()
    block_directory = output_dir / "paired_blocks"
    ledger_path = output_dir / "formal_block_ledger.jsonl"
    invalid_path = output_dir / "mechanically_invalid_attempts.jsonl"
    state_path = output_dir / "formal_run_state.json"
    terminal_path = output_dir / "formal_terminal_result.json"
    terminal_exists = terminal_path.exists()
    if preregistration.formal_view_record_path.exists() and not terminal_exists:
        raise ValueError(
            "The single R015 formal effect view was consumed without a terminal; "
            "the run is fail-closed."
        )
    _ensure_empty_jsonl_atomic(invalid_path)
    ledger_rows, firing_counts = _reconcile_formal_ledger(
        ledger_path=ledger_path,
        expected_coordinates=expected_coordinates,
        block_directory=block_directory,
        runtime_binding_sha256=runtime_binding_sha256,
    )
    invalid_counts = _reconcile_formal_invalid_attempts(
        invalid_path=invalid_path,
        expected_coordinates=expected_coordinates,
        ledger_rows=ledger_rows,
        runtime_binding_sha256=runtime_binding_sha256,
    )
    completed = len(ledger_rows)
    completed_rounds = completed // len(prototype_order)
    for checkpoint_index, checkpoint_round in enumerate(
        preregistration.statistics.kill_checkpoint_rounds
    ):
        checkpoint_path = preregistration.firing_count_checkpoint_paths[
            checkpoint_index
        ]
        if checkpoint_round > completed_rounds:
            if checkpoint_path.exists():
                raise ValueError(
                    "A firing-count checkpoint exists before its round is complete."
                )
            continue
        checkpoint_counts = {prototype_id: 0 for prototype_id in prototype_order}
        for row in ledger_rows[: checkpoint_round * len(prototype_order)]:
            checkpoint_counts[str(row["prototype_id"])] += int(
                row["firing_indicator"]
            )
        checkpoint_payload = _write_count_checkpoint(
            path=checkpoint_path,
            checkpoint_round=checkpoint_round,
            firing_counts=checkpoint_counts,
        )
        recovered_decision = evaluate_r015_firing_count_checkpoint(
            checkpoint_payload,
            statistics=preregistration.statistics,
            expected_prototype_ids=prototype_order,
        )
        if recovered_decision.verdict == "REGISTERED_NEGATIVE":
            expected_block_count = checkpoint_round * len(prototype_order)
            if completed != expected_block_count:
                raise ValueError(
                    "Formal blocks exist after a recovered firing-count stop."
                )
            if preregistration.formal_view_record_path.exists():
                raise ValueError("A count-only stop cannot have an effect-view receipt.")
            terminal = _kill_terminal_payload(
                preregistration_sha256=preregistration.source_sha256,
                runtime_binding_sha256=runtime_binding_sha256,
                completed_round_count=checkpoint_round,
                completed_block_count=completed,
                ledger_path=ledger_path,
                invalid_path=invalid_path,
                checkpoint_paths=preregistration.firing_count_checkpoint_paths[
                    : checkpoint_index + 1
                ],
                decision=recovered_decision,
            )
            return _publish_or_verify_terminal(terminal_path, terminal)
    if terminal_exists and completed < len(expected_coordinates):
        raise ValueError(
            "Existing R015 terminal is not supported by a recovered count-only stop."
        )
    for sequence_index in range(completed, len(expected_coordinates)):
        coordinate = expected_coordinates[sequence_index]
        (
            round_index,
            audit_unit_id,
            prototype_id,
            attempt_episode_seeds,
            ego_position,
        ) = coordinate
        safe_prototype = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in prototype_id
        )
        block_path = block_directory / (
            f"round_{round_index:04d}__{safe_prototype}.json"
        )
        if block_path.exists():
            block = _load_json_mapping(block_path)
            selected_attempt_index = int(
                block.get("mechanical_attempt_index", -1)
            )
            if selected_attempt_index != invalid_counts.get(sequence_index, 0) or not (
                0 <= selected_attempt_index < len(attempt_episode_seeds)
            ):
                raise ValueError(
                    "R015 recovered block does not follow its invalid-attempt prefix."
                )
            episode_seed = attempt_episode_seeds[selected_attempt_index]
            recovery_request = R015FormalBlockRequestV1(
                round_index=round_index,
                audit_unit_id=audit_unit_id,
                prototype_id=prototype_id,
                episode_seed=episode_seed,
                ego_position=ego_position,
                mechanical_attempt_index=selected_attempt_index,
            )
            regenerated = producer.produce_formal_paired_block(recovery_request)
            if not isinstance(regenerated, R015FormalBlockProductionResultV1) or (
                not regenerated.mechanically_valid
            ) or regenerated.paired_block is None or regenerated.invalid_reason is not None:
                raise ValueError(
                    "R015 orphan block could not be regenerated as the same valid block."
                )
            regenerated_block = _bind_formal_runtime_to_block(
                regenerated.paired_block,
                runtime_binding_sha256=runtime_binding_sha256,
            )
            if _formal_orphan_replay_projection(
                regenerated_block
            ) != _formal_orphan_replay_projection(block):
                raise ValueError(
                    "R015 orphan block differs from producer replay outside the two "
                    "non-deterministic elapsed-time measurements."
                )
            firing_indicator = _validate_formal_block_identity_without_effect_read(
                block,
                expected=(
                    round_index,
                    audit_unit_id,
                    prototype_id,
                    episode_seed,
                    ego_position,
                    selected_attempt_index,
                ),
                runtime_binding_sha256=runtime_binding_sha256,
            )
        else:
            result: R015FormalBlockProductionResultV1 | None = None
            selected_attempt_index = invalid_counts.get(sequence_index, 0)
            for attempt_index in range(
                selected_attempt_index,
                R015_FORMAL_REPLACEMENT_ATTEMPT_COUNT,
            ):
                episode_seed = attempt_episode_seeds[attempt_index]
                request = R015FormalBlockRequestV1(
                    round_index=round_index,
                    audit_unit_id=audit_unit_id,
                    prototype_id=prototype_id,
                    episode_seed=episode_seed,
                    ego_position=ego_position,
                    mechanical_attempt_index=attempt_index,
                )
                result = producer.produce_formal_paired_block(request)
                if not isinstance(result, R015FormalBlockProductionResultV1):
                    raise TypeError("R015 formal producer returned the wrong result type.")
                if result.mechanically_valid:
                    if result.paired_block is None or result.invalid_reason is not None:
                        raise ValueError(
                            "A valid R015 production result lacks its paired block."
                        )
                    break
                if result.paired_block is not None or not str(
                    result.invalid_reason or ""
                ):
                    raise ValueError(
                        "An invalid R015 result leaked partial outcome evidence."
                    )
                _append_jsonl_fsync(
                    invalid_path,
                    {
                        "schema_version": (
                            "path_c_r015_mechanical_replacement_attempt_v2"
                        ),
                        "sequence_index": sequence_index,
                        "round_index": round_index,
                        "audit_unit_id": audit_unit_id,
                        "prototype_id": prototype_id,
                        "episode_seed": episode_seed,
                        "ego_position": ego_position,
                        "mechanical_attempt_index": attempt_index,
                        "reason": str(result.invalid_reason),
                        "schedule_bound_seed": True,
                        "outcome_field_read": False,
                        "runtime_binding_sha256": runtime_binding_sha256,
                    },
                )
            if (
                result is None
                or not result.mechanically_valid
                or result.paired_block is None
            ):
                raise RuntimeError("R015 exhausted 32 whole-block mechanical reruns.")
            block = _bind_formal_runtime_to_block(
                result.paired_block,
                runtime_binding_sha256=runtime_binding_sha256,
            )
            selected_attempt_index = attempt_index
            if block.get("mechanical_attempt_index") != selected_attempt_index:
                raise ValueError(
                    "R015 producer changed its schedule-bound replacement attempt."
                )
            episode_seed = attempt_episode_seeds[selected_attempt_index]
            firing_indicator = _validate_formal_block_identity_without_effect_read(
                block,
                expected=(
                    round_index,
                    audit_unit_id,
                    prototype_id,
                    episode_seed,
                    ego_position,
                    selected_attempt_index,
                ),
                runtime_binding_sha256=runtime_binding_sha256,
            )
            _write_json_atomic(block_path, block)
        ledger_row = {
            "schema_version": "path_c_r015_formal_block_ledger_v2",
            "sequence_index": sequence_index,
            "round_index": round_index,
            "audit_unit_id": audit_unit_id,
            "prototype_id": prototype_id,
            "episode_seed": episode_seed,
            "ego_position": ego_position,
            "mechanical_attempt_index": selected_attempt_index,
            "firing_indicator": firing_indicator,
            "block_path": str(block_path),
            "block_sha256": _file_sha256(block_path),
            "runtime_binding_sha256": runtime_binding_sha256,
        }
        _append_jsonl_fsync(ledger_path, ledger_row)
        ledger_rows.append(ledger_row)
        firing_counts[prototype_id] += int(firing_indicator)
        round_completed_now = (
            sequence_index % len(prototype_order) == len(prototype_order) - 1
        )
        completed_round = round_index if round_completed_now else round_index - 1
        _write_json_atomic(
            state_path,
            {
                "schema_version": "path_c_r015_formal_run_state_v1",
                "preregistration_sha256": preregistration.source_sha256,
                "runtime_binding_sha256": runtime_binding_sha256,
                "completed_block_count": len(ledger_rows),
                "completed_round_count": completed_round,
                "firing_counts_by_prototype": firing_counts,
                "effect_value_read": False,
            },
        )
        if round_completed_now and completed_round in (
            preregistration.statistics.kill_checkpoint_rounds
        ):
            checkpoint_index = preregistration.statistics.kill_checkpoint_rounds.index(
                completed_round
            )
            checkpoint_payload = _write_count_checkpoint(
                path=preregistration.firing_count_checkpoint_paths[checkpoint_index],
                checkpoint_round=completed_round,
                firing_counts=firing_counts,
            )
            kill_decision = evaluate_r015_firing_count_checkpoint(
                checkpoint_payload,
                statistics=preregistration.statistics,
                expected_prototype_ids=prototype_order,
            )
            if kill_decision.verdict == "REGISTERED_NEGATIVE":
                if preregistration.formal_view_record_path.exists():
                    raise ValueError(
                        "A count-only stop cannot have an effect-view receipt."
                    )
                terminal = _kill_terminal_payload(
                    preregistration_sha256=preregistration.source_sha256,
                    runtime_binding_sha256=runtime_binding_sha256,
                    completed_round_count=completed_round,
                    completed_block_count=len(ledger_rows),
                    ledger_path=ledger_path,
                    invalid_path=invalid_path,
                    checkpoint_paths=preregistration.firing_count_checkpoint_paths[
                        : checkpoint_index + 1
                    ],
                    decision=kill_decision,
                )
                return _publish_or_verify_terminal(terminal_path, terminal)

    if len(ledger_rows) != preregistration.statistics.formal_paired_block_budget:
        raise ValueError("R015 formal run ended before all 2500 rounds were complete.")
    if any(
        not path.is_file() for path in preregistration.firing_count_checkpoint_paths
    ):
        raise ValueError("R015 final dataset lacks its five count-only checkpoints.")
    _ensure_empty_jsonl_atomic(invalid_path)
    assert preregistration.support_report_path is not None
    _assemble_formal_dataset_without_effect_read(
        path=preregistration.formal_dataset_path,
        preregistration_sha256=preregistration.source_sha256,
        support_report_sha256=_file_sha256(preregistration.support_report_path),
        checkpoint_paths=preregistration.firing_count_checkpoint_paths,
        ledger_rows=ledger_rows,
        ledger_path=ledger_path,
        invalid_attempts_path=invalid_path,
        runtime_binding_sha256=runtime_binding_sha256,
    )
    support_report_sha256 = _file_sha256(preregistration.support_report_path)
    receipt_base = _effect_receipt_base(
        preregistration=preregistration,
        support_report_sha256=support_report_sha256,
    )
    if terminal_exists:
        recorded_terminal = _load_json_mapping(terminal_path)
        decision_payload = _decision_payload(recorded_terminal.get("decision"))
        terminal = _effect_terminal_payload(
            preregistration=preregistration,
            runtime_binding_sha256=runtime_binding_sha256,
            completed_block_count=len(ledger_rows),
            ledger_path=ledger_path,
            invalid_path=invalid_path,
            checkpoint_paths=preregistration.firing_count_checkpoint_paths,
            started_receipt_sha256=str(
                recorded_terminal.get("formal_effect_view_started_sha256", "")
            ),
            decision_payload=decision_payload,
        )
        terminal = _publish_or_verify_terminal(terminal_path, terminal)
        _complete_or_validate_effect_receipt(
            receipt_path=preregistration.formal_view_record_path,
            receipt_base=receipt_base,
            terminal_path=terminal_path,
            terminal_payload=terminal,
        )
        return terminal

    decision = load_and_adjudicate_r015(
        preregistration.source_path,
        expected_runtime_binding_sha256=runtime_binding_sha256,
    )
    started_receipt_sha256 = _file_sha256(
        preregistration.formal_view_record_path
    )
    decision_payload = _decision_payload(decision)
    terminal = _effect_terminal_payload(
        preregistration=preregistration,
        runtime_binding_sha256=runtime_binding_sha256,
        completed_block_count=len(ledger_rows),
        ledger_path=ledger_path,
        invalid_path=invalid_path,
        checkpoint_paths=preregistration.firing_count_checkpoint_paths,
        started_receipt_sha256=started_receipt_sha256,
        decision_payload=decision_payload,
    )
    terminal = _publish_or_verify_terminal(terminal_path, terminal)
    _complete_or_validate_effect_receipt(
        receipt_path=preregistration.formal_view_record_path,
        receipt_base=receipt_base,
        terminal_path=terminal_path,
        terminal_payload=terminal,
    )
    return terminal
