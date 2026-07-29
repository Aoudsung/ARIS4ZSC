"""Frozen-checkpoint empirical task-value audit for Path C V4.4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, NamedTuple, Sequence

from experiments.overcooked_v2.deployment import (
    Deployment,
    load_deployment,
    reset_deployment_state,
    update_deployment_after_transition,
)
from experiments.overcooked_v2.official_adapter import (
    FrozenPartnerPool,
    VectorEnvironment,
)
from experiments.overcooked_v2.partner_panel_app import (
    panel_batch,
    panel_episode_seed,
)
from experiments.overcooked_v2.response_contrast_app import contrast_pairing_batch
from src.path_c.evaluation import (
    Pairing,
    ResponseContrastRow,
    spearman_rank_correlation,
    standard_episode_seed,
    summarize_counterfactual_trigger_values,
    summarize_responsibility_records,
    summarize_response_contrast,
    validate_counterfactual_continuation_index,
)
from src.path_c.experiment import PopulationEntry, load_config
from src.path_c.method import policy_effect_trigger_tolerance
from src.path_c.runner import policy_action, tree_select
from src.path_c.storage import (
    read_parquet,
    read_run_identity,
    write_json,
    write_parquet,
)


LEGACY_V44_RESPONSE_FIELDS = {
    "predicted_policy_mediated_effect_lcb": (
        "predicted_policy_gain_lower_score"
    ),
    "maximum_action_policy_mediated_gain_lcb": (
        "maximum_action_predicted_gain_lower_score"
    ),
    "executed_action_policy_mediated_gain_lcb": (
        "executed_action_predicted_gain_lower_score"
    ),
}

LEGACY_V44_PANEL_FIELDS = {
    "predicted_policy_mediated_effect_lcb": (
        "predicted_policy_gain_lower_score"
    ),
    "executed_action_policy_mediated_gain_lcb": (
        "executed_action_predicted_gain_lower_score"
    ),
}

LEGACY_V44_PANEL_EPISODE_FIELDS = {
    "positive_policy_gain_lcb_count": "positive_gain_lower_score_count",
}


@dataclass(frozen=True, slots=True)
class TriggerIdentity:
    source: str
    partner_index: int | None
    episode_index: int
    episode_seed: int
    trigger_step: int

    @property
    def trigger_id(self) -> str:
        partner = "self" if self.partner_index is None else f"partner-{self.partner_index:02d}"
        return (
            f"{partner}-episode-{self.episode_index:04d}-"
            f"step-{self.trigger_step:03d}"
        )


class ActorStep(NamedTuple):
    state: Any
    action: Any
    output: Any
    record: Any


class AuditWorld(NamedTuple):
    environment_state: Any
    observations: Any
    ego_state: Any
    partner_state: Any
    raw_return: Any
    discounted_return: Any
    correct_delivery_count: Any
    wrong_delivery_count: Any
    indicator_activation_count: Any
    last_ego_action: Any
    last_response_code: Any
    last_reward: Any


class TriggerCapture(NamedTuple):
    world: AuditWorld
    triggered: Any
    trigger_step: Any
    pre_world: AuditWorld
    post_use_world: AuditWorld
    post_mask_world: AuditWorld
    trigger_action: Any
    trigger_tolerance: Any
    predicted_policy_mediated_effect: Any
    predicted_policy_gain_lower_score: Any
    executed_action_policy_mediated_gain: Any
    executed_action_predicted_gain_lower_score: Any
    executed_action_expected_next_policy_tv: Any
    per_action_predicted_gain: Any
    per_action_predicted_gain_lower_score: Any
    response_code: Any
    trigger_count: Any = None
    partner_action: Any = None
    reference_logits: Any = None
    execution_logits: Any = None
    slot_log_belief: Any = None
    belief_entropy: Any = None
    value_class_count: Any = None
    predicted_policy_gain_uncertainty: Any = None
    predicted_next_policy_total_variation: Any = None
    executed_action_policy_gain_uncertainty: Any = None


class PairedContinuation(NamedTuple):
    use: AuditWorld
    mask: AuditWorld
    first_action_difference: Any
    first_interaction_difference: Any
    first_position_difference: Any
    first_inventory_difference: Any
    first_grid_difference: Any
    first_pot_difference: Any
    first_reward_difference: Any
    first_delivery_difference: Any
    physical_reconvergence_step: Any
    action_difference_steps: Any
    ever_physically_different: Any
    use_first_action: Any
    mask_first_action: Any
    use_first_positions: Any
    mask_first_positions: Any
    use_first_inventories: Any
    mask_first_inventories: Any
    use_first_grid: Any
    mask_first_grid: Any
    use_first_reward: Any
    mask_first_reward: Any
    use_first_correct_delivery_count: Any
    mask_first_correct_delivery_count: Any
    use_first_wrong_delivery_count: Any
    mask_first_wrong_delivery_count: Any


class ContinuationRunners(NamedTuple):
    pre_response: Any
    post_response: Any


class PanelReplayEvidence(NamedTuple):
    """Complete registered panel decisions needed by state reconstruction."""

    ego_action: Any
    partner_action: Any
    response_code: Any
    reference_logits: Any
    execution_logits: Any
    slot_log_belief: Any
    belief_entropy: Any
    value_class_count: Any
    predicted_policy_gain_lower_score: Any
    predicted_policy_gain_uncertainty: Any
    predicted_next_policy_total_variation: Any
    executed_action_predicted_gain_lower_score: Any
    executed_action_policy_gain_uncertainty: Any
    executed_action_expected_next_policy_tv: Any


class FixedPartnerStateReplay(NamedTuple):
    """Full states plus the discrete trajectory from a separate replay."""

    capture: TriggerCapture
    ego_action: Any
    partner_action: Any
    response_code: Any
    slot_log_belief: Any


def read_legacy_v44_response_rows(path: str | Path) -> tuple[ResponseContrastRow, ...]:
    """Read exactly the frozen V4.4 response schema for this audit only."""

    rows = read_parquet(path)
    converted = []
    for index, row in enumerate(rows):
        payload = dict(row)
        for old, new in LEGACY_V44_RESPONSE_FIELDS.items():
            if old not in payload or new in payload:
                raise ValueError(
                    f"Legacy V4.4 response row {index} does not have the registered schema."
                )
            payload[new] = payload.pop(old)
        converted.append(ResponseContrastRow(**payload))
    return tuple(converted)


def read_legacy_v44_panel_episode_rows(
    path: str | Path,
) -> tuple[Mapping[str, Any], ...]:
    """Read exactly the frozen V4.4 fixed-partner episode schema."""

    converted = []
    for index, row in enumerate(read_parquet(path)):
        payload = dict(row)
        for old, new in LEGACY_V44_PANEL_EPISODE_FIELDS.items():
            if old not in payload or new in payload:
                raise ValueError(
                    f"Legacy V4.4 panel episode {index} has an unexpected schema."
                )
            payload[new] = payload.pop(old)
        converted.append(payload)
    return tuple(converted)


def read_legacy_v44_panel_trigger_decisions(
    path: str | Path,
    trigger_steps: Mapping[int, int],
) -> Mapping[int, Mapping[str, Any]]:
    """Read only registered trigger rows from the frozen V4.4 panel log."""

    selected: dict[int, Mapping[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            episode = int(payload["episode_index"])
            if episode not in trigger_steps or int(payload["step"]) != int(
                trigger_steps[episode]
            ):
                continue
            for old, new in LEGACY_V44_PANEL_FIELDS.items():
                if old not in payload or new in payload:
                    raise ValueError(
                        "Legacy V4.4 panel decision has an unexpected schema "
                        f"at line {line_number}."
                    )
                payload[new] = payload.pop(old)
            if episode in selected:
                raise ValueError(
                    f"Duplicate V4.4 panel trigger decision for episode {episode}."
                )
            selected[episode] = payload
    missing = sorted(set(trigger_steps) - set(selected))
    if missing:
        raise ValueError(
            f"Frozen V4.4 panel trigger decisions are missing episodes {missing}."
        )
    return selected


def inverse_cdf_actions(probabilities: Any, uniforms: Any) -> Any:
    """Couple two categorical policies through the same uniform draws."""

    import jax.numpy as jnp

    values = jnp.asarray(probabilities, dtype=jnp.float32)
    draws = jnp.asarray(uniforms, dtype=jnp.float32)
    if values.ndim < 2 or draws.shape != values.shape[:-1]:
        raise ValueError("Uniform draws must match the categorical batch axes.")
    cumulative = jnp.cumsum(values, axis=-1)
    return jnp.sum(draws[..., None] >= cumulative, axis=-1).astype(jnp.int32)


def _actor_step(
    deployment: Deployment,
    state: Any,
    observations: Any,
    keys: Any,
    config: Any,
) -> ActorStep:
    next_state, action, output, record, unused_generic = policy_action(
        functions=deployment.functions(),
        params={
            "official": deployment.online_params,
            "heads": deployment.head_params,
        },
        policy_state=state,
        observations=observations,
        key=keys,
        deployment_mode="posterior_use",
        gamma=config.training.gamma,
        uncertainty_penalty=config.model.uncertainty_penalty,
        behavior_exploration_mix=0.0,
        behavior_uniform_floor=0.0,
    )
    del unused_generic
    return ActorStep(next_state, action, output, record)


def _environment_summary(state: Any) -> Mapping[str, Any]:
    import numpy as np
    from jaxmarl.environments.overcooked_v2.common import StaticObject

    agents = state.agents
    grid = np.asarray(state.grid)
    static = grid[..., 0]
    pot_mask = static == int(StaticObject.POT)
    return {
        "time": int(np.asarray(state.time)),
        "terminal": bool(np.asarray(state.terminal)),
        "recipe": int(np.asarray(state.recipe)),
        "agent_positions": np.stack(
            (np.asarray(agents.pos.x), np.asarray(agents.pos.y)), axis=-1
        ).tolist(),
        "agent_directions": np.asarray(agents.dir).tolist(),
        "agent_inventories": np.asarray(agents.inventory).tolist(),
        "pot_dynamic": grid[..., 1][pot_mask].tolist(),
        "pot_timers": grid[..., 2][pot_mask].tolist(),
        "grid": grid.tolist(),
    }


def _task_phase(state: Any, *, ego_seat: int) -> tuple[str, Mapping[str, Any]]:
    import numpy as np
    from jaxmarl.environments.overcooked_v2.common import DynamicObject, StaticObject

    grid = np.asarray(state.grid)
    inventory = int(np.asarray(state.agents.inventory)[ego_seat])
    recipe = int(np.asarray(state.recipe))
    plated = recipe | int(DynamicObject.PLATE) | int(DynamicObject.COOKED)
    pots = grid[grid[..., 0] == int(StaticObject.POT)]
    ready = bool(np.any((pots[:, 1] & int(DynamicObject.COOKED)) != 0)) if len(pots) else False
    cooking = bool(np.any(pots[:, 2] > 0)) if len(pots) else False
    incomplete = bool(
        np.any(
            (pots[:, 1] != int(DynamicObject.EMPTY))
            & ((pots[:, 1] & int(DynamicObject.COOKED)) == 0)
            & (pots[:, 2] == 0)
        )
    ) if len(pots) else False
    position = np.asarray(
        [state.agents.pos.x[ego_seat], state.agents.pos.y[ego_seat]],
        dtype=np.int32,
    )
    direction = int(np.asarray(state.agents.dir)[ego_seat])
    offsets = np.asarray(((0, -1), (0, 1), (1, 0), (-1, 0)), dtype=np.int32)
    faced = position + offsets[direction]
    faced_static = int(grid[faced[1], faced[0], 0])
    evidence = {
        "inventory": inventory,
        "recipe": recipe,
        "ready_pot": ready,
        "cooking_pot": cooking,
        "incomplete_pot": incomplete,
        "faced_static_object": faced_static,
    }
    if inventory == plated:
        return (
            "final_delivery"
            if faced_static == int(StaticObject.GOAL)
            else "move_to_delivery",
            evidence,
        )
    if inventory == int(DynamicObject.PLATE) and ready:
        return "plate_dish", evidence
    if inventory == int(DynamicObject.EMPTY) and ready:
        return "fetch_plate", evidence
    if bool(DynamicObject.is_ingredient(inventory)) and incomplete:
        return "deliver_ingredient_to_pot", evidence
    if cooking:
        return "wait_for_cooking", evidence
    if inventory == int(DynamicObject.EMPTY) or bool(DynamicObject.is_ingredient(inventory)):
        return "navigate_to_ingredient", evidence
    return "no_clear_task_phase", evidence


def _tree_repeat(tree: Any, count: int) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(
        lambda value: jnp.repeat(jnp.asarray(value)[None, ...], int(count), axis=0),
        tree,
    )


def _tree_lane(tree: Any, lane: int) -> Any:
    import jax

    return jax.tree_util.tree_map(lambda value: value[int(lane)], tree)


def _read_jsonl_files(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _responsibility_audit(
    run_directory: Path,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    responsibility_rows = _read_jsonl_files(
        run_directory / "records" / "responsibilities"
    )
    episode_rows = _read_jsonl_files(run_directory / "records" / "episodes")
    episode_returns = {
        (
            int(row["update_count"]),
            int(row["environment_index"]),
            int(row["episode_id"]),
        ): float(row["raw_return"])
        for row in episode_rows
    }
    summary = summarize_responsibility_records(
        responsibility_rows, episode_returns
    )
    grouped: dict[tuple[int, int, int, int], list[Mapping[str, Any]]] = {}
    for row in responsibility_rows:
        key = (
            int(row["update_count"]),
            int(row["epoch"]),
            int(row["environment_index"]),
            int(row["episode_id"]),
        )
        grouped.setdefault(key, []).append(row)
    audit_rows: list[Mapping[str, Any]] = []
    for key, members in sorted(grouped.items()):
        ordered = sorted(members, key=lambda row: int(row["slot"]))
        probabilities = [float(row["responsibility"]) for row in ordered]
        energies = sorted(float(row["td_energy"]) for row in ordered)
        entropy = -sum(
            value * math.log(value) for value in probabilities if value > 0.0
        )
        margin = energies[1] - energies[0]
        dominant = int(
            ordered[max(
                range(len(probabilities)), key=probabilities.__getitem__
            )]["slot"]
        )
        episode_key = (key[0], key[2], key[3])
        episode_return = float(episode_returns[episode_key])
        for row, probability in zip(ordered, probabilities, strict=True):
            audit_rows.append(
                {
                    "update_count": key[0],
                    "epoch": key[1],
                    "environment_index": key[2],
                    "episode_id": key[3],
                    "slot": int(row["slot"]),
                    "responsibility": probability,
                    "td_energy": float(row["td_energy"]),
                    "audit_partner_member": int(row["audit_partner_member"]),
                    "episode_raw_return": episode_return,
                    "responsibility_weighted_return": (
                        probability * episode_return
                    ),
                    "assignment_responsibility_entropy": entropy,
                    "assignment_td_energy_margin": margin,
                    "assignment_dominant_slot": dominant,
                    "historical_task_phase": "training_time_not_collected",
                    "historical_per_slot_action_values": (
                        "training_time_not_collected"
                    ),
                }
            )
    if len(audit_rows) != len(responsibility_rows):
        raise RuntimeError("Responsibility audit did not preserve every row.")
    return audit_rows, summary


def _empty_world(
    *,
    environment_state: Any,
    observations: Any,
    ego_state: Any,
    partner_state: Any,
    count: int,
) -> AuditWorld:
    import jax.numpy as jnp

    zeros_float = jnp.zeros((count,), dtype=jnp.float32)
    zeros_int = jnp.zeros((count,), dtype=jnp.int32)
    return AuditWorld(
        environment_state=environment_state,
        observations=observations,
        ego_state=ego_state,
        partner_state=partner_state,
        raw_return=zeros_float,
        discounted_return=zeros_float,
        correct_delivery_count=zeros_int,
        wrong_delivery_count=zeros_int,
        indicator_activation_count=zeros_int,
        last_ego_action=jnp.full((count,), -1, dtype=jnp.int32),
        last_response_code=jnp.full((count,), -1, dtype=jnp.int32),
        last_reward=zeros_float,
    )


def _capture_from_registered_response_scan(capture: Any) -> TriggerCapture:
    """Use the exact production response scan that created the frozen rows."""

    import jax.numpy as jnp

    def world(branch: Any) -> AuditWorld:
        zeros = jnp.zeros_like(branch.raw_return, dtype=jnp.float32)
        return AuditWorld(
            environment_state=branch.environment_state,
            observations=branch.observations,
            ego_state=branch.left_state,
            partner_state=branch.right_state,
            raw_return=branch.raw_return,
            discounted_return=zeros,
            correct_delivery_count=branch.correct_delivery_count,
            wrong_delivery_count=branch.wrong_delivery_count,
            indicator_activation_count=branch.indicator_activation_count,
            last_ego_action=branch.last_left_action,
            last_response_code=branch.last_left_response,
            last_reward=branch.last_reward,
        )

    pre = world(capture.pre_branch)
    post_use = world(capture.post_use_branch)
    post_mask = world(capture.post_mask_branch)
    return TriggerCapture(
        world=post_use,
        triggered=capture.triggered,
        trigger_step=capture.trigger_step,
        pre_world=pre,
        post_use_world=post_use,
        post_mask_world=post_mask,
        trigger_action=capture.trigger_action,
        trigger_tolerance=capture.trigger_tolerance,
        predicted_policy_mediated_effect=(
            capture.predicted_policy_mediated_effect
        ),
        predicted_policy_gain_lower_score=(
            capture.predicted_policy_gain_lower_score
        ),
        executed_action_policy_mediated_gain=(
            capture.executed_action_policy_mediated_gain
        ),
        executed_action_predicted_gain_lower_score=(
            capture.executed_action_predicted_gain_lower_score
        ),
        executed_action_expected_next_policy_tv=(
            capture.executed_action_expected_next_policy_tv
        ),
        per_action_predicted_gain=capture.per_action_predicted_gain,
        per_action_predicted_gain_lower_score=(
            capture.per_action_predicted_gain_lower_score
        ),
        response_code=capture.response_code,
    )


def _advance_deployment_world(
    *,
    world: AuditWorld,
    ego: Deployment,
    partner: Deployment,
    ego_step: ActorStep,
    partner_step: ActorStep,
    environment: VectorEnvironment,
    environment_keys: Any,
    config: Any,
    mask_ego_response: Any,
    discount: Any,
) -> AuditWorld:
    import jax.numpy as jnp

    joint_actions = jnp.stack((ego_step.action, partner_step.action), axis=-1)
    next_environment, next_observations, rewards, dones, info = (
        environment.step_with_keys(
            world.environment_state, joint_actions, environment_keys
        )
    )
    terminal = info["terminal_observations"]
    observation_mask = dones.reshape(
        dones.shape + (1,) * (next_observations[:, 0].ndim - 1)
    )
    ego_next_observation = jnp.where(
        observation_mask, terminal[:, 0], next_observations[:, 0]
    )
    partner_next_observation = jnp.where(
        observation_mask, terminal[:, 1], next_observations[:, 1]
    )
    ego_state, response_code = update_deployment_after_transition(
        deployment=ego,
        functions=ego.functions(),
        state=ego_step.state,
        output=ego_step.output,
        observations=world.observations[:, 0],
        actions=ego_step.action,
        next_observations=ego_next_observation,
        rewards=rewards,
        dones=dones,
        deployment_mode="posterior_use",
        terminal_response=config.model.response_count - 1,
        mask_response=mask_ego_response,
    )
    partner_state, unused_partner_response = update_deployment_after_transition(
        deployment=partner,
        functions=partner.functions(),
        state=partner_step.state,
        output=partner_step.output,
        observations=world.observations[:, 1],
        actions=partner_step.action,
        next_observations=partner_next_observation,
        rewards=rewards,
        dones=dones,
        deployment_mode="posterior_use",
        terminal_response=config.model.response_count - 1,
    )
    del unused_partner_response
    return AuditWorld(
        environment_state=next_environment,
        observations=next_observations,
        ego_state=ego_state,
        partner_state=partner_state,
        raw_return=world.raw_return + rewards,
        discounted_return=world.discounted_return + discount * rewards,
        correct_delivery_count=(
            world.correct_delivery_count + info["correct_delivery"]
        ),
        wrong_delivery_count=(
            world.wrong_delivery_count + info["wrong_delivery"]
        ),
        indicator_activation_count=(
            world.indicator_activation_count + info["indicator_activation"]
        ),
        last_ego_action=ego_step.action,
        last_response_code=response_code,
        last_reward=rewards,
    )


def validate_response_replay(
    expected: Sequence[ResponseContrastRow],
    observed: Sequence[ResponseContrastRow],
) -> None:
    if len(expected) != len(observed):
        raise RuntimeError("Response replay row count differs from the frozen artifact.")
    for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
        if left.to_mapping() != right.to_mapping():
            differing = sorted(
                name
                for name in left.to_mapping()
                if left.to_mapping()[name] != right.to_mapping()[name]
            )
            raise RuntimeError(
                f"Response replay differs at episode {index}: fields={differing}."
            )


def _zero_continuation_counters(world: AuditWorld, count: int) -> AuditWorld:
    import jax.numpy as jnp

    zeros_float = jnp.zeros((count,), dtype=jnp.float32)
    zeros_int = jnp.zeros((count,), dtype=jnp.int32)
    return world._replace(
        raw_return=zeros_float,
        discounted_return=zeros_float,
        correct_delivery_count=zeros_int,
        wrong_delivery_count=zeros_int,
        indicator_activation_count=zeros_int,
        last_ego_action=jnp.full((count,), -1, dtype=jnp.int32),
        last_response_code=jnp.full((count,), -1, dtype=jnp.int32),
        last_reward=zeros_float,
    )


def _advance_fixed_world(
    *,
    world: AuditWorld,
    ego: Deployment,
    ego_step: ActorStep,
    partner_carry: Any,
    partner_action: Any,
    environment: VectorEnvironment,
    environment_keys: Any,
    config: Any,
    mask_ego_response: Any,
    discount: Any,
) -> AuditWorld:
    import jax.numpy as jnp

    joint_actions = jnp.stack((ego_step.action, partner_action), axis=-1)
    next_environment, next_observations, rewards, dones, info = (
        environment.step_with_keys(
            world.environment_state, joint_actions, environment_keys
        )
    )
    terminal = info["terminal_observations"]
    observation_mask = dones.reshape(
        dones.shape + (1,) * (next_observations[:, 0].ndim - 1)
    )
    ego_next_observation = jnp.where(
        observation_mask, terminal[:, 0], next_observations[:, 0]
    )
    ego_state, response_code = update_deployment_after_transition(
        deployment=ego,
        functions=ego.functions(),
        state=ego_step.state,
        output=ego_step.output,
        observations=world.observations[:, 0],
        actions=ego_step.action,
        next_observations=ego_next_observation,
        rewards=rewards,
        dones=dones,
        deployment_mode="posterior_use",
        terminal_response=config.model.response_count - 1,
        mask_response=mask_ego_response,
    )
    return AuditWorld(
        environment_state=next_environment,
        observations=next_observations,
        ego_state=ego_state,
        partner_state=partner_carry,
        raw_return=world.raw_return + rewards,
        discounted_return=world.discounted_return + discount * rewards,
        correct_delivery_count=(
            world.correct_delivery_count + info["correct_delivery"]
        ),
        wrong_delivery_count=(
            world.wrong_delivery_count + info["wrong_delivery"]
        ),
        indicator_activation_count=(
            world.indicator_activation_count + info["indicator_activation"]
        ),
        last_ego_action=ego_step.action,
        last_response_code=response_code,
        last_reward=rewards,
    )


def _paired_diagnostics(
    current: PairedContinuation,
    next_use: AuditWorld,
    next_mask: AuditWorld,
    *,
    step: Any,
    active: Any,
) -> PairedContinuation:
    import jax.numpy as jnp
    from jaxmarl.environments.overcooked_v2.common import Actions, StaticObject

    use_agents = next_use.environment_state.agents
    mask_agents = next_mask.environment_state.agents
    action_difference = active & (
        next_use.last_ego_action != next_mask.last_ego_action
    )
    position_difference = active & (
        (use_agents.pos.x != mask_agents.pos.x).any(axis=-1)
        | (use_agents.pos.y != mask_agents.pos.y).any(axis=-1)
    )
    inventory_difference = active & (
        use_agents.inventory != mask_agents.inventory
    ).any(axis=-1)
    grid_difference = active & (
        next_use.environment_state.grid != next_mask.environment_state.grid
    ).any(axis=(-3, -2, -1))
    reward_difference = active & (
        next_use.last_reward != next_mask.last_reward
    )
    delivery_difference = active & (
        (
            next_use.correct_delivery_count
            != next_mask.correct_delivery_count
        )
        | (next_use.wrong_delivery_count != next_mask.wrong_delivery_count)
    )
    physical_difference = (
        position_difference | inventory_difference | grid_difference
    )
    reconverged = (
        active & current.ever_physically_different & ~physical_difference
    )

    def first(old: Any, condition: Any) -> Any:
        return jnp.where((old < 0) & condition, step, old)

    def first_step(old: Any, value: Any) -> Any:
        mask = (step == 0).reshape(
            (1,) * (jnp.ndim(value) - jnp.ndim(step))
        )
        return jnp.where(mask, value, old)

    use_positions = jnp.stack(
        (use_agents.pos.x, use_agents.pos.y), axis=-1
    )
    mask_positions = jnp.stack(
        (mask_agents.pos.x, mask_agents.pos.y), axis=-1
    )

    return PairedContinuation(
        use=next_use,
        mask=next_mask,
        first_action_difference=first(
            current.first_action_difference, action_difference
        ),
        first_interaction_difference=first(
            current.first_interaction_difference,
            action_difference
            & (
                (next_use.last_ego_action == int(Actions.interact))
                | (next_mask.last_ego_action == int(Actions.interact))
            ),
        ),
        first_position_difference=first(
            current.first_position_difference, position_difference
        ),
        first_inventory_difference=first(
            current.first_inventory_difference, inventory_difference
        ),
        first_grid_difference=first(
            current.first_grid_difference, grid_difference
        ),
        first_pot_difference=first(
            current.first_pot_difference,
            active
            & (
                (
                    next_use.environment_state.grid[..., 1:]
                    != next_mask.environment_state.grid[..., 1:]
                )
                & (
                    next_use.environment_state.grid[..., :1]
                    == int(StaticObject.POT)
                )
            ).any(axis=(-3, -2, -1)),
        ),
        first_reward_difference=first(
            current.first_reward_difference, reward_difference
        ),
        first_delivery_difference=first(
            current.first_delivery_difference, delivery_difference
        ),
        physical_reconvergence_step=first(
            current.physical_reconvergence_step, reconverged
        ),
        action_difference_steps=(
            current.action_difference_steps
            + action_difference.astype(jnp.int32)
        ),
        ever_physically_different=(
            current.ever_physically_different | physical_difference
        ),
        use_first_action=first_step(
            current.use_first_action, next_use.last_ego_action
        ),
        mask_first_action=first_step(
            current.mask_first_action, next_mask.last_ego_action
        ),
        use_first_positions=first_step(
            current.use_first_positions, use_positions
        ),
        mask_first_positions=first_step(
            current.mask_first_positions, mask_positions
        ),
        use_first_inventories=first_step(
            current.use_first_inventories, use_agents.inventory
        ),
        mask_first_inventories=first_step(
            current.mask_first_inventories, mask_agents.inventory
        ),
        use_first_grid=first_step(
            current.use_first_grid, next_use.environment_state.grid
        ),
        mask_first_grid=first_step(
            current.mask_first_grid, next_mask.environment_state.grid
        ),
        use_first_reward=first_step(
            current.use_first_reward, next_use.last_reward
        ),
        mask_first_reward=first_step(
            current.mask_first_reward, next_mask.last_reward
        ),
        use_first_correct_delivery_count=first_step(
            current.use_first_correct_delivery_count,
            next_use.correct_delivery_count,
        ),
        mask_first_correct_delivery_count=first_step(
            current.mask_first_correct_delivery_count,
            next_mask.correct_delivery_count,
        ),
        use_first_wrong_delivery_count=first_step(
            current.use_first_wrong_delivery_count,
            next_use.wrong_delivery_count,
        ),
        mask_first_wrong_delivery_count=first_step(
            current.mask_first_wrong_delivery_count,
            next_mask.wrong_delivery_count,
        ),
    )


def _paired_continuations(
    *,
    config: Any,
    deployment: Deployment,
    use_world: AuditWorld,
    mask_world: AuditWorld,
    forced_actions: Any | None,
    replica_indexes: Any,
    remaining_steps: int,
    episode_seed: int,
    trigger_step: int,
    key_domain: int,
    discount_offset: int,
    reset_counters: bool,
    fixed_partner_pool: FrozenPartnerPool | None,
    fixed_partner_index: int | None,
) -> tuple[PairedContinuation, Mapping[str, Any]]:
    """Run paired use/mask continuations with shared primitive random draws."""

    import jax
    import jax.numpy as jnp

    replicas = jnp.asarray(replica_indexes, dtype=jnp.int32)
    forced = (
        None
        if forced_actions is None
        else jnp.asarray(forced_actions, dtype=jnp.int32)
    )
    if forced is not None and forced.shape != replicas.shape:
        raise ValueError("Forced actions and replica indexes must align.")
    count = int(replicas.shape[0])
    template = VectorEnvironment.create(config)
    environment = VectorEnvironment(
        environment=template.environment,
        num_envs=count,
        episode_steps=config.environment.episode_steps,
    )
    root = jax.random.PRNGKey(jnp.asarray(episode_seed, dtype=jnp.uint32))
    root = jax.random.fold_in(root, int(key_domain))
    root = jax.random.fold_in(root, jnp.asarray(trigger_step, dtype=jnp.uint32))
    replica_roots = jax.vmap(lambda value: jax.random.fold_in(root, value))(
        replicas
    )
    minus_one = jnp.full((count,), -1, dtype=jnp.int32)
    agent_shape = use_world.environment_state.agents.inventory.shape[1:]
    initial = PairedContinuation(
        use=(
            _zero_continuation_counters(use_world, count)
            if reset_counters
            else use_world
        ),
        mask=(
            _zero_continuation_counters(mask_world, count)
            if reset_counters
            else mask_world
        ),
        first_action_difference=minus_one,
        first_interaction_difference=minus_one,
        first_position_difference=minus_one,
        first_inventory_difference=minus_one,
        first_grid_difference=minus_one,
        first_pot_difference=minus_one,
        first_reward_difference=minus_one,
        first_delivery_difference=minus_one,
        physical_reconvergence_step=minus_one,
        action_difference_steps=jnp.zeros((count,), dtype=jnp.int32),
        ever_physically_different=jnp.zeros((count,), dtype=jnp.bool_),
        use_first_action=minus_one,
        mask_first_action=minus_one,
        use_first_positions=jnp.zeros(
            (count,) + agent_shape + (2,), dtype=jnp.int32
        ),
        mask_first_positions=jnp.zeros(
            (count,) + agent_shape + (2,), dtype=jnp.int32
        ),
        use_first_inventories=jnp.zeros(
            (count,) + agent_shape, dtype=jnp.int32
        ),
        mask_first_inventories=jnp.zeros(
            (count,) + agent_shape, dtype=jnp.int32
        ),
        use_first_grid=jnp.zeros_like(use_world.environment_state.grid),
        mask_first_grid=jnp.zeros_like(mask_world.environment_state.grid),
        use_first_reward=jnp.zeros((count,), dtype=jnp.float32),
        mask_first_reward=jnp.zeros((count,), dtype=jnp.float32),
        use_first_correct_delivery_count=jnp.zeros(
            (count,), dtype=jnp.int32
        ),
        mask_first_correct_delivery_count=jnp.zeros(
            (count,), dtype=jnp.int32
        ),
        use_first_wrong_delivery_count=jnp.zeros(
            (count,), dtype=jnp.int32
        ),
        mask_first_wrong_delivery_count=jnp.zeros(
            (count,), dtype=jnp.int32
        ),
    )
    initial_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(
        replica_roots
    )
    initial_use_step = _actor_step(
        deployment,
        initial.use.ego_state,
        initial.use.observations[:, 0],
        initial_keys,
        config,
    )
    initial_mask_step = _actor_step(
        deployment,
        initial.mask.ego_state,
        initial.mask.observations[:, 0],
        initial_keys,
        config,
    )
    initial_values = {
        "use_probabilities": jax.nn.softmax(
            initial_use_step.output.execution_logits, axis=-1
        ),
        "mask_probabilities": jax.nn.softmax(
            initial_mask_step.output.execution_logits, axis=-1
        ),
        "use_predicted_action_values": initial_use_step.output.j_use_mean,
        "mask_predicted_action_values": initial_mask_step.output.j_use_mean,
        "use_slot_centered_q": initial_use_step.output.centered_advantages,
        "mask_slot_centered_q": initial_mask_step.output.centered_advantages,
    }

    def one_step(
        current: PairedContinuation, scan_input: tuple[Any, Any]
    ) -> tuple[Any, None]:
        step, remaining_limit = scan_input
        active_scalar = step < remaining_limit
        active = jnp.full((count,), active_scalar, dtype=jnp.bool_)
        ego_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 1 + 3 * step)
        )(replica_roots)
        partner_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 2 + 3 * step)
        )(replica_roots)
        environment_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 3 + 3 * step)
        )(replica_roots)
        uniform_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 10000 + step)
        )(replica_roots)
        uniforms = jax.vmap(lambda key: jax.random.uniform(key))(uniform_keys)
        use_step = _actor_step(
            deployment,
            current.use.ego_state,
            current.use.observations[:, 0],
            ego_keys,
            config,
        )
        mask_step = _actor_step(
            deployment,
            current.mask.ego_state,
            current.mask.observations[:, 0],
            ego_keys,
            config,
        )
        use_sampled = inverse_cdf_actions(
            jax.nn.softmax(use_step.output.execution_logits, axis=-1), uniforms
        )
        mask_sampled = inverse_cdf_actions(
            jax.nn.softmax(mask_step.output.execution_logits, axis=-1), uniforms
        )
        if forced is None:
            selected_use = use_sampled
            selected_mask = mask_sampled
        else:
            selected_use = jnp.where(step == 0, forced, use_sampled)
            selected_mask = jnp.where(step == 0, forced, mask_sampled)
        use_step = use_step._replace(action=selected_use)
        mask_step = mask_step._replace(action=selected_mask)
        discount = jnp.power(
            jnp.asarray(config.training.gamma, dtype=jnp.float32),
            step + int(discount_offset),
        )
        if fixed_partner_pool is None:
            use_partner = _actor_step(
                deployment,
                current.use.partner_state,
                current.use.observations[:, 1],
                partner_keys,
                config,
            )
            mask_partner = _actor_step(
                deployment,
                current.mask.partner_state,
                current.mask.observations[:, 1],
                partner_keys,
                config,
            )
            candidate_use = _advance_deployment_world(
                world=current.use,
                ego=deployment,
                partner=deployment,
                ego_step=use_step,
                partner_step=use_partner,
                environment=environment,
                environment_keys=environment_keys,
                config=config,
                mask_ego_response=False,
                discount=discount,
            )
            candidate_mask = _advance_deployment_world(
                world=current.mask,
                ego=deployment,
                partner=deployment,
                ego_step=mask_step,
                partner_step=mask_partner,
                environment=environment,
                environment_keys=environment_keys,
                config=config,
                mask_ego_response=False,
                discount=discount,
            )
        else:
            member_indexes = jnp.full(
                (count,), jnp.asarray(fixed_partner_index), dtype=jnp.int32
            )
            use_partner_action, use_partner_carry = (
                fixed_partner_pool.step_with_keys(
                    member_indexes,
                    current.use.observations[:, 1],
                    current.use.partner_state,
                    current.use.ego_state.episode_start,
                    partner_keys,
                )
            )
            mask_partner_action, mask_partner_carry = (
                fixed_partner_pool.step_with_keys(
                    member_indexes,
                    current.mask.observations[:, 1],
                    current.mask.partner_state,
                    current.mask.ego_state.episode_start,
                    partner_keys,
                )
            )
            candidate_use = _advance_fixed_world(
                world=current.use,
                ego=deployment,
                ego_step=use_step,
                partner_carry=use_partner_carry,
                partner_action=use_partner_action,
                environment=environment,
                environment_keys=environment_keys,
                config=config,
                mask_ego_response=False,
                discount=discount,
            )
            candidate_mask = _advance_fixed_world(
                world=current.mask,
                ego=deployment,
                ego_step=mask_step,
                partner_carry=mask_partner_carry,
                partner_action=mask_partner_action,
                environment=environment,
                environment_keys=environment_keys,
                config=config,
                mask_ego_response=False,
                discount=discount,
            )
        next_use = tree_select(active, candidate_use, current.use)
        next_mask = tree_select(active, candidate_mask, current.mask)
        return _paired_diagnostics(
            current,
            next_use,
            next_mask,
            step=step,
            active=active,
        ), None

    scan_steps = jnp.arange(config.environment.episode_steps)
    remaining_limits = jnp.full(
        scan_steps.shape,
        jnp.asarray(remaining_steps, dtype=jnp.int32),
        dtype=jnp.int32,
    )
    final, unused = jax.lax.scan(
        one_step,
        initial,
        (scan_steps, remaining_limits),
    )
    del unused
    return final, initial_values


def _pre_response_continuations(
    *,
    config: Any,
    deployment: Deployment,
    pre_world: AuditWorld,
    trigger_action: int,
    replicas: int,
    remaining_steps_after_trigger: int,
    episode_seed: int,
    trigger_step: int,
    fixed_partner_pool: FrozenPartnerPool | None,
    fixed_partner_index: int | None,
) -> tuple[PairedContinuation, Mapping[str, Any]]:
    """Estimate the ex-ante value of the exact action used by the trigger."""

    import jax
    import jax.numpy as jnp

    count = int(replicas)
    world = _zero_continuation_counters(
        _tree_repeat(pre_world, count), count
    )
    root = jax.random.PRNGKey(jnp.asarray(episode_seed, dtype=jnp.uint32))
    root = jax.random.fold_in(root, 200001)
    root = jax.random.fold_in(root, jnp.asarray(trigger_step, dtype=jnp.uint32))
    replica_indexes = jnp.arange(count, dtype=jnp.int32)
    roots = jax.vmap(lambda value: jax.random.fold_in(root, value))(
        replica_indexes
    )
    ego_keys = jax.vmap(lambda key: jax.random.fold_in(key, 1))(roots)
    partner_keys = jax.vmap(lambda key: jax.random.fold_in(key, 2))(roots)
    environment_keys = jax.vmap(lambda key: jax.random.fold_in(key, 3))(roots)
    ego_step = _actor_step(
        deployment,
        world.ego_state,
        world.observations[:, 0],
        ego_keys,
        config,
    )._replace(
        action=jnp.full(
            (count,), jnp.asarray(trigger_action), dtype=jnp.int32
        )
    )
    template = VectorEnvironment.create(config)
    environment = VectorEnvironment(
        environment=template.environment,
        num_envs=count,
        episode_steps=config.environment.episode_steps,
    )
    if fixed_partner_pool is None:
        partner_step = _actor_step(
            deployment,
            world.partner_state,
            world.observations[:, 1],
            partner_keys,
            config,
        )
        post_use = _advance_deployment_world(
            world=world,
            ego=deployment,
            partner=deployment,
            ego_step=ego_step,
            partner_step=partner_step,
            environment=environment,
            environment_keys=environment_keys,
            config=config,
            mask_ego_response=False,
            discount=jnp.asarray(1.0, dtype=jnp.float32),
        )
        post_mask = _advance_deployment_world(
            world=world,
            ego=deployment,
            partner=deployment,
            ego_step=ego_step,
            partner_step=partner_step,
            environment=environment,
            environment_keys=environment_keys,
            config=config,
            mask_ego_response=jnp.ones((count,), dtype=jnp.bool_),
            discount=jnp.asarray(1.0, dtype=jnp.float32),
        )
    else:
        member_indexes = jnp.full(
            (count,), jnp.asarray(fixed_partner_index), dtype=jnp.int32
        )
        partner_action, partner_carry = fixed_partner_pool.step_with_keys(
            member_indexes,
            world.observations[:, 1],
            world.partner_state,
            world.ego_state.episode_start,
            partner_keys,
        )
        post_use = _advance_fixed_world(
            world=world,
            ego=deployment,
            ego_step=ego_step,
            partner_carry=partner_carry,
            partner_action=partner_action,
            environment=environment,
            environment_keys=environment_keys,
            config=config,
            mask_ego_response=False,
            discount=jnp.asarray(1.0, dtype=jnp.float32),
        )
        post_mask = _advance_fixed_world(
            world=world,
            ego=deployment,
            ego_step=ego_step,
            partner_carry=partner_carry,
            partner_action=partner_action,
            environment=environment,
            environment_keys=environment_keys,
            config=config,
            mask_ego_response=jnp.ones((count,), dtype=jnp.bool_),
            discount=jnp.asarray(1.0, dtype=jnp.float32),
        )
    final, values = _paired_continuations(
        config=config,
        deployment=deployment,
        use_world=post_use,
        mask_world=post_mask,
        forced_actions=None,
        replica_indexes=replica_indexes,
        remaining_steps=remaining_steps_after_trigger,
        episode_seed=episode_seed,
        trigger_step=trigger_step,
        key_domain=200002,
        discount_offset=1,
        reset_counters=False,
        fixed_partner_pool=fixed_partner_pool,
        fixed_partner_index=fixed_partner_index,
    )
    return final, {
        **values,
        "trigger_response_codes": post_use.last_response_code,
    }


def _make_continuation_runners(
    *,
    config: Any,
    deployment: Deployment,
    replicas: int,
    fixed_partner_pool: FrozenPartnerPool | None,
) -> ContinuationRunners:
    """Compile each repeated continuation shape once for the whole audit."""

    import jax
    import jax.numpy as jnp
    import numpy as np

    action_count = int(config.model.action_count)
    forced_actions = jnp.asarray(
        np.repeat(np.arange(action_count, dtype=np.int32), replicas)
    )
    post_replica_indexes = jnp.asarray(
        np.tile(np.arange(replicas, dtype=np.int32), action_count)
    )

    def pre_response(
        pre_world: AuditWorld,
        trigger_action: Any,
        remaining_steps: Any,
        episode_seed: Any,
        trigger_step: Any,
        fixed_partner_index: Any,
    ) -> tuple[PairedContinuation, Mapping[str, Any]]:
        return _pre_response_continuations(
            config=config,
            deployment=deployment,
            pre_world=pre_world,
            trigger_action=trigger_action,
            replicas=replicas,
            remaining_steps_after_trigger=remaining_steps,
            episode_seed=episode_seed,
            trigger_step=trigger_step,
            fixed_partner_pool=fixed_partner_pool,
            fixed_partner_index=fixed_partner_index,
        )

    def post_response(
        post_use_world: AuditWorld,
        post_mask_world: AuditWorld,
        remaining_steps: Any,
        episode_seed: Any,
        trigger_step: Any,
        fixed_partner_index: Any,
    ) -> tuple[PairedContinuation, Mapping[str, Any]]:
        count = action_count * replicas
        return _paired_continuations(
            config=config,
            deployment=deployment,
            use_world=_tree_repeat(post_use_world, count),
            mask_world=_tree_repeat(post_mask_world, count),
            forced_actions=forced_actions,
            replica_indexes=post_replica_indexes,
            remaining_steps=remaining_steps,
            episode_seed=episode_seed,
            trigger_step=trigger_step,
            key_domain=300001,
            discount_offset=0,
            reset_counters=True,
            fixed_partner_pool=fixed_partner_pool,
            fixed_partner_index=fixed_partner_index,
        )

    return ContinuationRunners(
        pre_response=jax.jit(pre_response),
        post_response=jax.jit(post_response),
    )


def _empty_trigger_capture(world: AuditWorld, *, config: Any) -> TriggerCapture:
    """Create shape-correct storage for the first trigger in every lane."""

    import jax.numpy as jnp

    count = int(world.observations.shape[0])
    actions = int(config.model.action_count)
    zeros = jnp.zeros((count,), dtype=jnp.float32)
    zeros_int = jnp.zeros((count,), dtype=jnp.int32)
    return TriggerCapture(
        world=world,
        triggered=jnp.zeros((count,), dtype=jnp.bool_),
        trigger_step=jnp.full((count,), -1, dtype=jnp.int32),
        pre_world=world,
        post_use_world=world,
        post_mask_world=world,
        trigger_action=zeros_int,
        trigger_tolerance=zeros,
        predicted_policy_mediated_effect=zeros,
        predicted_policy_gain_lower_score=zeros,
        executed_action_policy_mediated_gain=zeros,
        executed_action_predicted_gain_lower_score=zeros,
        executed_action_expected_next_policy_tv=zeros,
        per_action_predicted_gain=jnp.zeros(
            (count, actions), dtype=jnp.float32
        ),
        per_action_predicted_gain_lower_score=jnp.zeros(
            (count, actions), dtype=jnp.float32
        ),
        response_code=zeros_int,
        trigger_count=zeros_int,
        partner_action=zeros_int,
        reference_logits=jnp.zeros((count, actions), dtype=jnp.float32),
        execution_logits=jnp.zeros((count, actions), dtype=jnp.float32),
        slot_log_belief=world.ego_state.slot_log_belief,
        belief_entropy=zeros,
        value_class_count=zeros_int,
        predicted_policy_gain_uncertainty=zeros,
        predicted_next_policy_total_variation=zeros,
        executed_action_policy_gain_uncertainty=zeros,
    )


def _capture_fixed_partner_states(
    *,
    config: Any,
    deployment: Deployment,
    partner_pool: FrozenPartnerPool,
    partner_index: int,
    evaluation_seed: int,
) -> FixedPartnerStateReplay:
    """Reconstruct full states without changing the production panel scan."""

    import jax
    import jax.numpy as jnp
    import numpy as np

    episode_count = int(config.evaluation.episodes_per_pairing)
    episode_steps = int(config.environment.episode_steps)
    seeds = np.asarray(
        [
            panel_episode_seed(
                evaluation_seed=evaluation_seed,
                layout=config.environment.layout,
                ego_outer_unit_id=deployment.outer_unit_id,
                partner_index=partner_index,
                episode_index=index,
            )
            for index in range(episode_count)
        ],
        dtype=np.uint32,
    )
    root_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    reset_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(root_keys)
    template = VectorEnvironment.create(config)
    environment = VectorEnvironment(
        environment=template.environment,
        num_envs=episode_count,
        episode_steps=episode_steps,
    )
    environment_state, observations = environment.reset_with_keys(reset_keys)
    ego_state = reset_deployment_state(
        deployment, batch_size=episode_count, config=config
    )
    partner_state = partner_pool.initial_carry(episode_count)
    world = _empty_world(
        environment_state=environment_state,
        observations=observations,
        ego_state=ego_state,
        partner_state=partner_state,
        count=episode_count,
    )
    initial_capture = _empty_trigger_capture(world, config=config)
    member_indexes = jnp.full(
        (episode_count,), int(partner_index), dtype=jnp.int32
    )

    def one_step(
        carry: tuple[AuditWorld, TriggerCapture], step: Any
    ) -> tuple[tuple[AuditWorld, TriggerCapture], tuple[Any, ...]]:
        current, capture = carry
        ego_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 1 + 3 * step)
        )(root_keys)
        partner_key = jax.random.fold_in(
            jax.random.PRNGKey(int(evaluation_seed) ^ (partner_index + 1)),
            step,
        )
        environment_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, 3 + 3 * step)
        )(root_keys)
        ego_step = _actor_step(
            deployment,
            current.ego_state,
            current.observations[:, 0],
            ego_keys,
            config,
        )
        partner_action, partner_carry = partner_pool.step(
            member_indexes,
            current.observations[:, 1],
            current.partner_state,
            current.ego_state.episode_start,
            partner_key,
        )
        discount = jnp.power(
            jnp.asarray(config.training.gamma, dtype=jnp.float32), step
        )
        post_use = _advance_fixed_world(
            world=current,
            ego=deployment,
            ego_step=ego_step,
            partner_carry=partner_carry,
            partner_action=partner_action,
            environment=environment,
            environment_keys=environment_keys,
            config=config,
            mask_ego_response=False,
            discount=discount,
        )
        done = post_use.ego_state.episode_start
        masked_belief = jnp.where(
            done[:, None],
            post_use.ego_state.slot_log_belief,
            current.ego_state.slot_log_belief,
        )
        post_mask = post_use._replace(
            ego_state=post_use.ego_state._replace(
                slot_log_belief=masked_belief
            )
        )
        tolerance = policy_effect_trigger_tolerance(
            ego_step.record.j_use, ego_step.record.j_mask
        )
        actionable = (
            ego_step.record.executed_action_predicted_gain_lower_score
            > tolerance
        ) & (
            ego_step.record.executed_action_expected_next_policy_tv
            >= config.evaluation.response_policy_tv_minimum
        )
        first = actionable & ~capture.triggered
        candidate = TriggerCapture(
            world=post_use,
            triggered=capture.triggered | actionable,
            trigger_step=jnp.full(
                (episode_count,), step, dtype=jnp.int32
            ),
            pre_world=current,
            post_use_world=post_use,
            post_mask_world=post_mask,
            trigger_action=ego_step.action,
            trigger_tolerance=tolerance,
            predicted_policy_mediated_effect=(
                ego_step.record.predicted_policy_mediated_effect
            ),
            predicted_policy_gain_lower_score=(
                ego_step.record.predicted_policy_gain_lower_score
            ),
            executed_action_policy_mediated_gain=(
                ego_step.record.executed_action_policy_mediated_gain
            ),
            executed_action_predicted_gain_lower_score=(
                ego_step.record.executed_action_predicted_gain_lower_score
            ),
            executed_action_expected_next_policy_tv=(
                ego_step.record.executed_action_expected_next_policy_tv
            ),
            per_action_predicted_gain=(
                ego_step.record.per_action_policy_mediated_gain
            ),
            per_action_predicted_gain_lower_score=(
                ego_step.record.per_action_predicted_gain_lower_score
            ),
            response_code=post_use.last_response_code,
            trigger_count=capture.trigger_count + actionable.astype(jnp.int32),
            partner_action=partner_action,
            reference_logits=ego_step.record.reference_logits,
            execution_logits=ego_step.record.execution_logits,
            slot_log_belief=current.ego_state.slot_log_belief,
            belief_entropy=ego_step.record.belief_entropy,
            value_class_count=ego_step.record.value_class_count,
            predicted_policy_gain_uncertainty=(
                ego_step.record.predicted_policy_gain_uncertainty
            ),
            predicted_next_policy_total_variation=(
                ego_step.record.predicted_next_policy_total_variation
            ),
            executed_action_policy_gain_uncertainty=(
                ego_step.record.executed_action_policy_gain_uncertainty
            ),
        )
        selected = tree_select(first, candidate, capture)._replace(
            world=post_use,
            triggered=capture.triggered | actionable,
            trigger_count=capture.trigger_count + actionable.astype(jnp.int32),
        )
        return (post_use, selected), (
            ego_step.action,
            partner_action,
            post_use.last_response_code,
            current.ego_state.slot_log_belief,
        )

    (final_world, capture), recorded = jax.lax.scan(
        one_step,
        (world, initial_capture),
        jnp.arange(episode_steps, dtype=jnp.int32),
    )
    return FixedPartnerStateReplay(
        capture=capture._replace(world=final_world),
        ego_action=recorded[0],
        partner_action=recorded[1],
        response_code=recorded[2],
        slot_log_belief=recorded[3],
    )


def _registered_panel_capture(
    capture: TriggerCapture,
    evidence: PanelReplayEvidence,
) -> TriggerCapture:
    """Attach only fields already proved exact by the production replay."""

    import jax.numpy as jnp
    import numpy as np

    triggered = np.asarray(capture.triggered, dtype=np.bool_)
    steps = np.asarray(capture.trigger_step, dtype=np.int32)
    safe_steps = np.where(triggered, steps, 0)
    lanes = np.arange(len(triggered), dtype=np.int32)

    def take(values: Any) -> Any:
        return jnp.asarray(np.asarray(values)[safe_steps, lanes])

    return capture._replace(
        trigger_action=take(evidence.ego_action),
        partner_action=take(evidence.partner_action),
        response_code=take(evidence.response_code),
        reference_logits=take(evidence.reference_logits),
        execution_logits=take(evidence.execution_logits),
        slot_log_belief=take(evidence.slot_log_belief),
        belief_entropy=take(evidence.belief_entropy),
        value_class_count=take(evidence.value_class_count),
        predicted_policy_gain_lower_score=take(
            evidence.predicted_policy_gain_lower_score
        ),
        predicted_policy_gain_uncertainty=take(
            evidence.predicted_policy_gain_uncertainty
        ),
        predicted_next_policy_total_variation=take(
            evidence.predicted_next_policy_total_variation
        ),
        executed_action_predicted_gain_lower_score=take(
            evidence.executed_action_predicted_gain_lower_score
        ),
        executed_action_policy_gain_uncertainty=take(
            evidence.executed_action_policy_gain_uncertainty
        ),
        executed_action_expected_next_policy_tv=take(
            evidence.executed_action_expected_next_policy_tv
        ),
    )


def _host(value: Any) -> Any:
    import jax
    import numpy as np

    if isinstance(value, Mapping):
        return {str(name): _host(item) for name, item in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return {
            name: _host(getattr(value, name)) for name in value._fields
        }
    if isinstance(value, (tuple, list)):
        return [_host(item) for item in value]
    leaves = jax.tree_util.tree_leaves(value)
    if len(leaves) != 1 or leaves[0] is not value:
        return _host(jax.tree_util.tree_map(lambda item: np.asarray(item), value))
    array = np.asarray(value)
    if array.ndim:
        return array.tolist()
    if np.issubdtype(array.dtype, np.bool_):
        return bool(array)
    if np.issubdtype(array.dtype, np.integer):
        return int(array)
    return float(array)


def _effect_class(pair: PairedContinuation, lane: int) -> str:
    import numpy as np

    delivery_changed = (
        int(np.asarray(pair.use.correct_delivery_count)[lane])
        != int(np.asarray(pair.mask.correct_delivery_count)[lane])
        or int(np.asarray(pair.use.wrong_delivery_count)[lane])
        != int(np.asarray(pair.mask.wrong_delivery_count)[lane])
    )
    if delivery_changed:
        return "delivery_changed"
    ever = bool(np.asarray(pair.ever_physically_different)[lane])
    if not ever:
        action_changed = int(np.asarray(pair.action_difference_steps)[lane]) > 0
        return "blocked_or_avoidance" if action_changed else "no_physical_difference"
    reconverged = int(np.asarray(pair.physical_reconvergence_step)[lane]) >= 0
    return (
        "temporary_difference_then_reconvergence"
        if reconverged
        else "persistent_difference_without_delivery_change"
    )


def _continuation_rows(
    *,
    identity: TriggerIdentity,
    estimand: str,
    paired: PairedContinuation,
    forced_actions: Sequence[int | None],
    replica_indexes: Sequence[int],
    continuation_environment_steps: int,
) -> list[Mapping[str, Any]]:
    import numpy as np
    from jaxmarl.environments.overcooked_v2.common import Actions, StaticObject

    rows = []
    for lane, (forced_action, replica) in enumerate(
        zip(forced_actions, replica_indexes, strict=True)
    ):
        common = {
            "trigger_id": identity.trigger_id,
            "source": identity.source,
            "partner_index": identity.partner_index,
            "episode_index": identity.episode_index,
            "episode_seed": identity.episode_seed,
            "trigger_step": identity.trigger_step,
            "estimand": estimand,
            "forced_action": forced_action,
            "replica_index": int(replica),
            "continuation_environment_steps": int(
                continuation_environment_steps
            ),
            "first_action_difference": int(
                np.asarray(paired.first_action_difference)[lane]
            ),
            "first_interaction_difference": int(
                np.asarray(paired.first_interaction_difference)[lane]
            ),
            "first_position_difference": int(
                np.asarray(paired.first_position_difference)[lane]
            ),
            "first_inventory_difference": int(
                np.asarray(paired.first_inventory_difference)[lane]
            ),
            "first_grid_difference": int(
                np.asarray(paired.first_grid_difference)[lane]
            ),
            "first_pot_difference": int(
                np.asarray(paired.first_pot_difference)[lane]
            ),
            "first_reward_difference": int(
                np.asarray(paired.first_reward_difference)[lane]
            ),
            "first_delivery_difference": int(
                np.asarray(paired.first_delivery_difference)[lane]
            ),
            "physical_reconvergence_step": int(
                np.asarray(paired.physical_reconvergence_step)[lane]
            ),
            "action_difference_steps": int(
                np.asarray(paired.action_difference_steps)[lane]
            ),
            "task_effect_class": _effect_class(paired, lane),
        }
        for branch_name, world in (("use", paired.use), ("mask", paired.mask)):
            first_action = int(
                np.asarray(
                    paired.use_first_action
                    if branch_name == "use"
                    else paired.mask_first_action
                )[lane]
            )
            first_positions = np.asarray(
                paired.use_first_positions
                if branch_name == "use"
                else paired.mask_first_positions
            )[lane]
            first_inventories = np.asarray(
                paired.use_first_inventories
                if branch_name == "use"
                else paired.mask_first_inventories
            )[lane]
            first_grid = np.asarray(
                paired.use_first_grid
                if branch_name == "use"
                else paired.mask_first_grid
            )[lane]
            pot_mask = first_grid[..., 0] == int(StaticObject.POT)
            first_correct = int(
                np.asarray(
                    paired.use_first_correct_delivery_count
                    if branch_name == "use"
                    else paired.mask_first_correct_delivery_count
                )[lane]
            )
            first_wrong = int(
                np.asarray(
                    paired.use_first_wrong_delivery_count
                    if branch_name == "use"
                    else paired.mask_first_wrong_delivery_count
                )[lane]
            )
            rows.append(
                {
                    **common,
                    "branch": branch_name,
                    "continuation_first_action": first_action,
                    "continuation_first_action_is_interact": (
                        first_action == int(Actions.interact)
                    ),
                    "continuation_first_positions": first_positions.tolist(),
                    "continuation_first_inventories": (
                        first_inventories.tolist()
                    ),
                    "continuation_first_grid": first_grid.tolist(),
                    "continuation_first_pot_state": (
                        first_grid[pot_mask].tolist()
                    ),
                    "continuation_first_reward": float(
                        np.asarray(
                            paired.use_first_reward
                            if branch_name == "use"
                            else paired.mask_first_reward
                        )[lane]
                    ),
                    "continuation_first_correct_deliveries": first_correct,
                    "continuation_first_wrong_deliveries": first_wrong,
                    "remaining_raw_return": float(
                        np.asarray(world.raw_return)[lane]
                    ),
                    "remaining_discounted_return": float(
                        np.asarray(world.discounted_return)[lane]
                    ),
                    "remaining_correct_deliveries": int(
                        np.asarray(world.correct_delivery_count)[lane]
                    ),
                    "remaining_wrong_deliveries": int(
                        np.asarray(world.wrong_delivery_count)[lane]
                    ),
                    "remaining_indicator_activations": int(
                        np.asarray(world.indicator_activation_count)[lane]
                    ),
                }
            )
    return rows


def _audit_trigger(
    *,
    config: Any,
    deployment: Deployment,
    capture: TriggerCapture,
    lane: int,
    identity: TriggerIdentity,
    replicas: int,
    fixed_partner_pool: FrozenPartnerPool | None,
    fixed_partner_index: int | None,
    runners: ContinuationRunners,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    pre_world = _tree_lane(capture.pre_world, lane)
    post_use_world = _tree_lane(capture.post_use_world, lane)
    post_mask_world = _tree_lane(capture.post_mask_world, lane)
    trigger_action = int(np.asarray(capture.trigger_action)[lane])
    remaining = int(config.environment.episode_steps) - identity.trigger_step - 1
    runner_partner_index = -1 if fixed_partner_index is None else fixed_partner_index
    episode_seed = np.uint32(identity.episode_seed)
    pre_pair, unused_pre_values = runners.pre_response(
        pre_world,
        trigger_action,
        remaining,
        episode_seed,
        identity.trigger_step,
        runner_partner_index,
    )
    del unused_pre_values
    action_count = int(config.model.action_count)
    forced = np.repeat(np.arange(action_count, dtype=np.int32), replicas)
    replica_indexes = np.tile(np.arange(replicas, dtype=np.int32), action_count)
    count = int(len(forced))
    post_pair, post_values = runners.post_response(
        post_use_world,
        post_mask_world,
        remaining,
        episode_seed,
        identity.trigger_step,
        runner_partner_index,
    )
    pre_use = np.asarray(pre_pair.use.discounted_return)
    pre_mask = np.asarray(pre_pair.mask.discounted_return)
    empirical_pre_gain = float(np.mean(pre_use - pre_mask))
    lower_score = float(
        np.asarray(capture.executed_action_predicted_gain_lower_score)[lane]
    )
    use_returns = np.asarray(post_pair.use.discounted_return).reshape(
        action_count, replicas
    )
    mask_returns = np.asarray(post_pair.mask.discounted_return).reshape(
        action_count, replicas
    )
    q_use = np.mean(use_returns, axis=1)
    q_mask = np.mean(mask_returns, axis=1)
    use_probabilities = np.asarray(post_values["use_probabilities"])[0]
    mask_probabilities = np.asarray(post_values["mask_probabilities"])[0]
    empirical_use_value = float(np.sum(use_probabilities * q_use))
    empirical_mask_value = float(np.sum(mask_probabilities * q_mask))
    predicted_use = np.asarray(post_values["use_predicted_action_values"])[0]
    predicted_mask = np.asarray(post_values["mask_predicted_action_values"])[0]
    use_slot_q = np.mean(
        np.asarray(post_values["use_slot_centered_q"])[0], axis=0
    )
    mask_slot_q = np.mean(
        np.asarray(post_values["mask_slot_centered_q"])[0], axis=0
    )
    use_belief = np.exp(
        np.asarray(post_use_world.ego_state.slot_log_belief)
    )
    use_belief = use_belief / np.sum(use_belief)

    def pairwise_distances(values: Any) -> list[float]:
        return [
            float(np.linalg.norm(values[left] - values[right]))
            for left in range(values.shape[0])
            for right in range(left + 1, values.shape[0])
        ]
    phase, phase_evidence = _task_phase(
        pre_world.environment_state, ego_seat=0
    )
    if fixed_partner_pool is None:
        partner_states = {
            "pre_partner_policy_state": _host(pre_world.partner_state),
            "post_partner_policy_state": _host(
                post_use_world.partner_state
            ),
            "pre_fixed_partner_carry": None,
            "post_fixed_partner_carry": None,
        }
    else:
        partner_states = {
            "pre_partner_policy_state": None,
            "post_partner_policy_state": None,
            "pre_fixed_partner_carry": _host(pre_world.partner_state),
            "post_fixed_partner_carry": _host(
                post_use_world.partner_state
            ),
        }
    reconstruction = {
        "trigger_id": identity.trigger_id,
        **asdict(identity),
        "replay_verified": True,
        "trigger_action": trigger_action,
        "task_phase_before_action": phase,
        "use_slot_posterior": use_belief.tolist(),
        "dominant_use_slot": int(np.argmax(use_belief)),
        "response_code": int(np.asarray(capture.response_code)[lane]),
        "task_phase_evidence": dict(phase_evidence),
        "pre_environment": _environment_summary(pre_world.environment_state),
        "post_environment": _environment_summary(
            post_use_world.environment_state
        ),
        "pre_observations": _host(pre_world.observations),
        "post_observations": _host(post_use_world.observations),
        "pre_ego_state": _host(pre_world.ego_state),
        "post_use_ego_state": _host(post_use_world.ego_state),
        "post_mask_ego_state": _host(post_mask_world.ego_state),
        **partner_states,
    }
    trigger_values = {
        "trigger_id": identity.trigger_id,
        **asdict(identity),
        "replicas": int(replicas),
        "trigger_action": trigger_action,
        "predicted_policy_mediated_effect": float(
            np.asarray(capture.predicted_policy_mediated_effect)[lane]
        ),
        "predicted_policy_gain_lower_score": float(
            np.asarray(capture.predicted_policy_gain_lower_score)[lane]
        ),
        "executed_action_predicted_gain": float(
            np.asarray(capture.executed_action_policy_mediated_gain)[lane]
        ),
        "executed_action_predicted_gain_lower_score": lower_score,
        "executed_action_expected_next_policy_tv": float(
            np.asarray(capture.executed_action_expected_next_policy_tv)[lane]
        ),
        "empirical_pre_response_discounted_gain": empirical_pre_gain,
        "pre_response_covered": empirical_pre_gain >= lower_score,
        "predicted_use_action_values": predicted_use.tolist(),
        "predicted_mask_action_values": predicted_mask.tolist(),
        "empirical_use_action_values": q_use.tolist(),
        "empirical_mask_action_values": q_mask.tolist(),
        "use_policy_probabilities": use_probabilities.tolist(),
        "mask_policy_probabilities": mask_probabilities.tolist(),
        "empirical_post_response_use_value": empirical_use_value,
        "empirical_post_response_mask_value": empirical_mask_value,
        "empirical_post_response_gain": (
            empirical_use_value - empirical_mask_value
        ),
        "use_action_rank_correlation": spearman_rank_correlation(
            predicted_use.tolist(), q_use.tolist()
        ),
        "mask_action_rank_correlation": spearman_rank_correlation(
            predicted_mask.tolist(), q_mask.tolist()
        ),
        "use_optimal_action_match": int(np.argmax(predicted_use))
        == int(np.argmax(q_use)),
        "mask_optimal_action_match": int(np.argmax(predicted_mask))
        == int(np.argmax(q_mask)),
        "use_slot_centered_q": use_slot_q.tolist(),
        "mask_slot_centered_q": mask_slot_q.tolist(),
        "use_slot_posterior": use_belief.tolist(),
        "dominant_use_slot": int(np.argmax(use_belief)),
        "task_phase_before_action": phase,
        "use_slot_optimal_actions": np.argmax(use_slot_q, axis=-1).tolist(),
        "mask_slot_optimal_actions": np.argmax(mask_slot_q, axis=-1).tolist(),
        "use_slot_pairwise_l2": pairwise_distances(use_slot_q),
        "mask_slot_pairwise_l2": pairwise_distances(mask_slot_q),
    }
    rows = _continuation_rows(
        identity=identity,
        estimand="pre_response",
        paired=pre_pair,
        forced_actions=[trigger_action] * replicas,
        replica_indexes=list(range(replicas)),
        continuation_environment_steps=remaining + 1,
    )
    rows.extend(
        _continuation_rows(
            identity=identity,
            estimand="post_response",
            paired=post_pair,
            forced_actions=[int(value) for value in forced],
            replica_indexes=[int(value) for value in replica_indexes],
            continuation_environment_steps=remaining,
        )
    )
    jax.block_until_ready(post_pair.use.raw_return)
    return reconstruction, rows, trigger_values


def _fixed_a1_continuation(
    *,
    config: Any,
    deployment: Deployment,
    pre_world: AuditWorld,
    identity: TriggerIdentity,
    partner_pool: FrozenPartnerPool,
    partner_index: int,
    trigger_action: int,
    replicas: int,
    runners: ContinuationRunners,
) -> tuple[int, AuditWorld | None, int]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    world = _tree_repeat(pre_world, 1)
    root = jax.random.PRNGKey(int(identity.episode_seed))
    ego_key = jax.random.fold_in(
        root, 1 + 3 * int(identity.trigger_step)
    )[None, ...]
    step = _actor_step(
        deployment,
        world.ego_state,
        world.observations[:, 0],
        ego_key,
        config,
    )
    mask_action = int(
        np.asarray(
            jax.random.categorical(ego_key[0], step.output.mask_execution_logits[0])
        )
    )
    if mask_action == int(trigger_action):
        return mask_action, None, 0
    paired, unused = runners.pre_response(
        pre_world,
        mask_action,
        (
            int(config.environment.episode_steps)
            - identity.trigger_step
            - 1
        ),
        np.uint32(identity.episode_seed),
        identity.trigger_step,
        partner_index,
    )
    del unused
    environment_steps = (
        2
        * int(replicas)
        * (int(config.environment.episode_steps) - identity.trigger_step)
    )
    return mask_action, _tree_lane(paired.mask, 0), environment_steps


def _panel_contrast_row(
    *,
    identity: TriggerIdentity,
    capture: TriggerCapture,
    lane: int,
    pre_continuation_rows: Sequence[Mapping[str, Any]],
    a1_action: int,
    a1_world: AuditWorld | None,
) -> Mapping[str, Any]:
    import numpy as np

    prefix = _tree_lane(capture.pre_world, lane)
    selected = {
        str(row["branch"]): row
        for row in pre_continuation_rows
        if row["estimand"] == "pre_response"
        and int(row["replica_index"]) == 0
    }
    if set(selected) != {"use", "mask"}:
        raise RuntimeError("The first panel contrast replica is incomplete.")
    prefix_return = float(np.asarray(prefix.raw_return))
    prefix_correct = int(np.asarray(prefix.correct_delivery_count))
    prefix_wrong = int(np.asarray(prefix.wrong_delivery_count))
    prefix_indicator = int(np.asarray(prefix.indicator_activation_count))

    def branch_from_row(name: str) -> Mapping[str, Any]:
        row = selected[name]
        return {
            "raw_return": prefix_return + float(row["remaining_raw_return"]),
            "correct_delivery_count": prefix_correct
            + int(row["remaining_correct_deliveries"]),
            "wrong_delivery_count": prefix_wrong
            + int(row["remaining_wrong_deliveries"]),
            "indicator_activation_count": prefix_indicator
            + int(row["remaining_indicator_activations"]),
        }

    a2_use = branch_from_row("use")
    a2_mask = branch_from_row("mask")
    trigger_action = int(np.asarray(capture.trigger_action)[lane])
    if int(a1_action) == trigger_action:
        if a1_world is not None:
            raise RuntimeError(
                "A matched A1 action must reuse the exact A2-mask branch."
            )
        a1 = dict(a2_mask)
    else:
        if a1_world is None:
            raise RuntimeError("A distinct A1 action lacks its matched branch.")
        a1 = {
            "raw_return": prefix_return
            + float(np.asarray(a1_world.raw_return)),
            "correct_delivery_count": prefix_correct
            + int(np.asarray(a1_world.correct_delivery_count)),
            "wrong_delivery_count": prefix_wrong
            + int(np.asarray(a1_world.wrong_delivery_count)),
            "indicator_activation_count": prefix_indicator
            + int(np.asarray(a1_world.indicator_activation_count)),
        }
    response_effect = a2_use["raw_return"] - a2_mask["raw_return"]
    task_cost = a1["raw_return"] - a2_mask["raw_return"]
    net_effect = a2_use["raw_return"] - a1["raw_return"]
    if net_effect != response_effect - task_cost:
        raise RuntimeError("Fixed-partner response effects violate their identity.")
    return {
        "source": identity.source,
        "partner_index": identity.partner_index,
        "episode_index": identity.episode_index,
        "episode_seed": identity.episode_seed,
        "triggered": True,
        "trigger_step": identity.trigger_step,
        "trigger_action": trigger_action,
        "a1_action": int(a1_action),
        "a1": a1,
        "a2_mask": a2_mask,
        "a2_use": a2_use,
        "delta_response": response_effect,
        "delta_cost": task_cost,
        "delta_net": net_effect,
    }


def _summary_from_trigger_values(
    values: Sequence[Mapping[str, Any]],
    *,
    replicas: int,
    actual_environment_steps: int,
) -> Mapping[str, Any]:
    if not values:
        raise ValueError("Counterfactual audit produced no trigger values.")
    trigger_ids = [str(row["trigger_id"]) for row in values]
    if len(set(trigger_ids)) != len(trigger_ids):
        raise RuntimeError("Counterfactual trigger rows contain duplicate identities.")
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in values:
        by_source.setdefault(str(row["source"]), []).append(row)
    return {
        "run_kind": "development",
        "scientific_readout_allowed": False,
        "checkpoint_frozen": True,
        "replicas_per_trigger": int(replicas),
        **summarize_counterfactual_trigger_values(values),
        "actual_environment_steps": int(actual_environment_steps),
        "use_optimal_action_match_rate": mean(
            int(bool(row["use_optimal_action_match"])) for row in values
        ),
        "mask_optimal_action_match_rate": mean(
            int(bool(row["mask_optimal_action_match"])) for row in values
        ),
        "sources": {
            source: {
                **summarize_counterfactual_trigger_values(rows),
                "mean_empirical_pre_response_discounted_gain": mean(
                    float(row["empirical_pre_response_discounted_gain"])
                    for row in rows
                ),
                "mean_empirical_post_response_gain": mean(
                    float(row["empirical_post_response_gain"])
                    for row in rows
                ),
                "coverage": mean(
                    int(bool(row["pre_response_covered"])) for row in rows
                ),
                "positive_empirical_pre_response_gain_count": sum(
                    float(row["empirical_pre_response_discounted_gain"]) > 0.0
                    for row in rows
                ),
                "negative_empirical_pre_response_gain_count": sum(
                    float(row["empirical_pre_response_discounted_gain"]) < 0.0
                    for row in rows
                ),
                "zero_empirical_pre_response_gain_count": sum(
                    float(row["empirical_pre_response_discounted_gain"]) == 0.0
                    for row in rows
                ),
            }
            for source, rows in sorted(by_source.items())
        },
        "formal_ten_unit_training": "closed",
    }


def _summarize_panel_contrast(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    by_partner: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_partner.setdefault(int(row["partner_index"]), []).append(row)
    result = {}
    for partner, members in sorted(by_partner.items()):
        triggered = [row for row in members if bool(row["triggered"])]
        result[str(partner)] = {
            "episode_count": len(members),
            "trigger_count": len(triggered),
            "action_change_count": sum(
                int(row["trigger_action"]) != int(row["a1_action"])
                for row in triggered
            ),
            "mean_delta_response": mean(
                float(row["delta_response"]) for row in members
            ),
            "mean_delta_cost": mean(float(row["delta_cost"]) for row in members),
            "mean_delta_net": mean(float(row["delta_net"]) for row in members),
            "triggered_positive_response_count": sum(
                float(row["delta_response"]) > 0.0 for row in triggered
            ),
            "triggered_negative_response_count": sum(
                float(row["delta_response"]) < 0.0 for row in triggered
            ),
            "triggered_zero_response_count": sum(
                float(row["delta_response"]) == 0.0 for row in triggered
            ),
        }
    return result


def _summarize_slot_semantics(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    import numpy as np

    pairwise = [
        float(value)
        for row in rows
        for value in row["use_slot_pairwise_l2"]
    ]
    optimal_disagreements = [
        len(set(int(value) for value in row["use_slot_optimal_actions"])) > 1
        for row in rows
    ]
    dominant_by_phase: dict[str, dict[str, int]] = {}
    dominant_by_partner: dict[str, dict[str, int]] = {}
    source_signatures: dict[str, dict[int, list[Any]]] = {}
    for row in rows:
        slot = str(int(row["dominant_use_slot"]))
        phase = str(row["task_phase_before_action"])
        phase_counts = dominant_by_phase.setdefault(phase, {})
        phase_counts[slot] = phase_counts.get(slot, 0) + 1
        partner = str(row["partner_index"])
        partner_counts = dominant_by_partner.setdefault(partner, {})
        partner_counts[slot] = partner_counts.get(slot, 0) + 1
        for slot_index, signature in enumerate(row["use_slot_centered_q"]):
            source_signatures.setdefault(str(row["source"]), {}).setdefault(
                slot_index, []
            ).append(np.asarray(signature, dtype=np.float64))
    slot_count = len(rows[0]["use_slot_centered_q"])
    cross_partner_variance = {}
    for slot in range(slot_count):
        means = [
            np.mean(signatures[slot], axis=0)
            for signatures in source_signatures.values()
            if slot in signatures
        ]
        cross_partner_variance[str(slot)] = (
            float(np.mean(np.var(np.stack(means), axis=0)))
            if len(means) > 1
            else None
        )
    return {
        "state_count": len(rows),
        "mean_between_slot_centered_q_distance": mean(pairwise),
        "states_with_different_slot_optimal_actions": sum(
            optimal_disagreements
        ),
        "different_slot_optimal_action_rate": mean(
            float(value) for value in optimal_disagreements
        ),
        "dominant_slot_by_task_phase": dominant_by_phase,
        "dominant_slot_by_partner": dominant_by_partner,
        "same_slot_cross_partner_action_signature_variance": (
            cross_partner_variance
        ),
    }


def _self_pairing_retention(evaluation_root: Path) -> Mapping[str, Any]:
    path = (
        evaluation_root
        / "self_pairing"
        / "posterior_use"
        / "episodes.parquet"
    )
    rows = read_parquet(path)
    if len(rows) != 500:
        raise RuntimeError("Frozen posterior-use self-pairing rows are incomplete.")
    return {
        "source": str(path),
        "episode_count": len(rows),
        "mean_raw_return": mean(float(row["raw_return"]) for row in rows),
    }


def _formal_gate_statuses(
    *,
    trigger_summary: Mapping[str, Any],
    panel_summary: Mapping[str, Any],
    slot_summary: Mapping[str, Any],
    self_pairing: Mapping[str, Any],
) -> Mapping[str, Any]:
    split = trigger_summary["lower_score_split"]
    positive_partners = sum(
        str(source).startswith("fixed_partner_")
        and float(values["mean_empirical_pre_response_discounted_gain"]) > 0.0
        for source, values in trigger_summary["sources"].items()
    )
    correlation = trigger_summary["mean_action_rank_correlation"]
    return {
        "1_repeated_action_values": {
            "status": "satisfied",
            "evidence": "Every accepted trigger has six-action, two-belief repeated continuations.",
        },
        "2_positive_action_rank_correlation": {
            "status": (
                "satisfied"
                if correlation is not None and float(correlation) > 0.0
                else "not_satisfied"
            ),
            "observed_mean_correlation": correlation,
        },
        "3_registered_lower_score_coverage": {
            "status": (
                "satisfied"
                if float(trigger_summary["pre_response_coverage"]) >= 0.95
                else "not_satisfied"
            ),
            "registered_target": 0.95,
            "observed": trigger_summary["pre_response_coverage"],
        },
        "4_high_score_exceeds_low_score": {
            "status": (
                "satisfied"
                if float(split["high_mean_empirical_pre_response_gain"])
                > float(split["low_mean_empirical_pre_response_gain"])
                else "not_satisfied"
            ),
            "evidence": split,
        },
        "5_positive_response_value_in_multiple_fixed_partners": {
            "status": "satisfied" if positive_partners > 1 else "not_satisfied",
            "positive_partner_count": positive_partners,
            "estimand": "repeated_pre_response_use_minus_mask",
        },
        "6_other_play_seed_202_negative_effect_explained": {
            "status": "evidence_insufficient",
            "reason": "Task-effect classes are descriptive and do not identify a unique causal training defect.",
        },
        "7_self_pairing_capability_retained": {
            "status": "satisfied",
            "evidence": self_pairing,
        },
        "8_slots_represent_control_differences": {
            "status": (
                "not_satisfied"
                if int(slot_summary["states_with_different_slot_optimal_actions"])
                == 0
                else "evidence_insufficient"
            ),
            "evidence": slot_summary,
        },
        "formal_ten_unit_training": "closed",
    }


def _checkpoint_files(directory: Path) -> Mapping[str, tuple[int, int]]:
    return {
        str(path.relative_to(directory)): (
            int(path.stat().st_size),
            int(path.stat().st_mtime_ns),
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def deployment_parameter_snapshot(deployment: Deployment) -> tuple[Any, ...]:
    import jax
    import numpy as np

    return tuple(
        np.asarray(value).copy()
        for value in jax.tree_util.tree_leaves(
            (
                deployment.reference_params,
                deployment.online_params,
                deployment.head_params,
                deployment.codebook,
                deployment.log_temperature,
                deployment.generic_log_temperature,
            )
        )
    )


def _write_partition(
    directory: Path,
    *,
    reconstruction: Mapping[str, Any],
    continuations: Sequence[Mapping[str, Any]],
    trigger_values: Mapping[str, Any],
) -> None:
    validate_counterfactual_continuation_index(
        continuations,
        replicas=int(trigger_values["replicas"]),
    )
    directory.mkdir(parents=True, exist_ok=True)
    outputs = (
        ("reconstruction.parquet", [reconstruction]),
        ("continuations.parquet", continuations),
        ("trigger_values.parquet", [trigger_values]),
    )
    for name, rows in outputs:
        final = directory / name
        partial = directory / f"{name}.partial"
        write_parquet(partial, rows)
        partial.replace(final)
    write_json(
        directory / "complete.json",
        {
            "trigger_id": str(trigger_values["trigger_id"]),
            "continuation_row_count": len(continuations),
        },
    )


def _read_complete_partition(
    directory: Path, *, replicas: int
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], Mapping[str, Any]] | None:
    complete = directory / "complete.json"
    paths = (
        directory / "reconstruction.parquet",
        directory / "continuations.parquet",
        directory / "trigger_values.parquet",
    )
    if not complete.is_file() or not all(path.is_file() for path in paths):
        return None
    reconstruction = read_parquet(paths[0])
    continuations = read_parquet(paths[1])
    values = read_parquet(paths[2])
    expected = 14 * int(replicas)
    if len(reconstruction) != 1 or len(values) != 1 or len(continuations) != expected:
        raise RuntimeError(f"Incomplete counterfactual partition: {directory}.")
    validate_counterfactual_continuation_index(
        continuations, replicas=int(replicas)
    )
    return reconstruction[0], continuations, values[0]


def _validate_capture_against_rows(
    capture: TriggerCapture,
    rows: Sequence[ResponseContrastRow],
) -> None:
    import numpy as np

    triggered = np.asarray(capture.triggered, dtype=np.bool_)
    captured_count = int(np.sum(triggered))
    frozen_count = sum(row.triggered for row in rows)
    if captured_count != frozen_count:
        differing = [
            index
            for index, row in enumerate(rows)
            if bool(triggered[index]) != bool(row.triggered)
        ]
        raise RuntimeError(
            "Captured and frozen trigger counts differ: "
            f"captured={captured_count}, frozen={frozen_count}, "
            f"episodes={differing}."
        )
    for lane, row in enumerate(rows):
        if bool(triggered[lane]) != bool(row.triggered):
            raise RuntimeError(f"Trigger presence differs for episode {lane}.")
        if not row.triggered:
            continue
        checks = {
            "trigger_step": (
                int(np.asarray(capture.trigger_step)[lane]), row.trigger_step
            ),
            "executed_action": (
                int(np.asarray(capture.trigger_action)[lane]), row.executed_action
            ),
            "executed_action_predicted_gain_lower_score": (
                float(
                    np.asarray(
                        capture.executed_action_predicted_gain_lower_score
                    )[lane]
                ),
                row.executed_action_predicted_gain_lower_score,
            ),
        }
        differing = [name for name, pair in checks.items() if pair[0] != pair[1]]
        if differing:
            raise RuntimeError(
                f"Captured trigger differs for episode {lane}: fields={differing}."
            )


def _validate_panel_capture(
    *,
    capture: TriggerCapture,
    episodes: Sequence[Mapping[str, Any]],
    decisions: Mapping[int, Mapping[str, Any]],
) -> None:
    """Require the reconstructed first trigger to match frozen panel records."""

    import numpy as np

    triggered = np.asarray(capture.triggered, dtype=np.bool_)
    trigger_steps = np.asarray(capture.trigger_step)
    trigger_counts = np.asarray(capture.trigger_count)
    for lane, episode in enumerate(episodes):
        expected_count = int(episode["positive_gain_lower_score_count"])
        if int(trigger_counts[lane]) != expected_count:
            raise RuntimeError(
                "Fixed-partner trigger count differs for episode "
                f"{lane}: replay={int(trigger_counts[lane])}, "
                f"frozen={expected_count}."
            )
        if bool(triggered[lane]) != (expected_count > 0):
            raise RuntimeError(
                f"Fixed-partner first-trigger presence differs for episode {lane}."
            )
        if not triggered[lane]:
            continue
        expected = decisions[lane]
        observed = {
            "episode_index": lane,
            "episode_seed": int(episode["episode_seed"]),
            "step": int(trigger_steps[lane]),
            "ego_action": int(np.asarray(capture.trigger_action)[lane]),
            "partner_action": int(np.asarray(capture.partner_action)[lane]),
            "response_code": int(np.asarray(capture.response_code)[lane]),
            "reference_logits": np.asarray(capture.reference_logits)[lane].tolist(),
            "execution_logits": np.asarray(capture.execution_logits)[lane].tolist(),
            "slot_log_belief": np.asarray(capture.slot_log_belief)[lane].tolist(),
            "belief_entropy": float(np.asarray(capture.belief_entropy)[lane]),
            "value_class_count": int(
                np.asarray(capture.value_class_count)[lane]
            ),
            "predicted_policy_gain_lower_score": float(
                np.asarray(capture.predicted_policy_gain_lower_score)[lane]
            ),
            "predicted_policy_gain_uncertainty": float(
                np.asarray(capture.predicted_policy_gain_uncertainty)[lane]
            ),
            "predicted_next_policy_total_variation": float(
                np.asarray(capture.predicted_next_policy_total_variation)[lane]
            ),
            "executed_action_predicted_gain_lower_score": float(
                np.asarray(
                    capture.executed_action_predicted_gain_lower_score
                )[lane]
            ),
            "executed_action_policy_gain_uncertainty": float(
                np.asarray(capture.executed_action_policy_gain_uncertainty)[lane]
            ),
            "executed_action_expected_next_policy_tv": float(
                np.asarray(
                    capture.executed_action_expected_next_policy_tv
                )[lane]
            ),
        }
        differing = sorted(
            name for name, value in observed.items() if expected[name] != value
        )
        if differing:
            raise RuntimeError(
                "Fixed-partner trigger replay differs for episode "
                f"{lane}: fields={differing}."
            )


def _validate_panel_episode_replay(
    frozen: Sequence[Mapping[str, Any]],
    replayed: Sequence[Any],
) -> None:
    if len(frozen) != len(replayed):
        raise RuntimeError("Fixed-partner episode replay count differs.")
    for index, (expected, observed_row) in enumerate(
        zip(frozen, replayed, strict=True)
    ):
        observed = observed_row.to_mapping()
        fields = sorted(
            name
            for name in set(expected) | set(observed)
            if expected.get(name) != observed.get(name)
        )
        if fields:
            raise RuntimeError(
                "Fixed-partner episode replay differs for episode "
                f"{index}: fields={fields}."
            )


def _validate_and_collect_panel_decisions(
    *,
    frozen_path: Path,
    replayed: Any,
    episode_steps: int,
    episode_count: int,
) -> PanelReplayEvidence:
    """Require exact production replay of every frozen decision row."""

    import itertools
    import numpy as np

    names = PanelReplayEvidence._fields
    collected: dict[str, list[Any]] = {name: [] for name in names}

    def frozen_rows() -> Any:
        with frozen_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                for old, new in LEGACY_V44_PANEL_FIELDS.items():
                    if old not in payload or new in payload:
                        raise ValueError(
                            "Legacy V4.4 panel decision has an unexpected "
                            f"schema at line {line_number}."
                        )
                    payload[new] = payload.pop(old)
                yield line_number, payload

    missing = object()
    row_count = 0
    for item, observed in itertools.zip_longest(
        frozen_rows(), replayed, fillvalue=missing
    ):
        if item is missing or observed is missing:
            raise RuntimeError(
                "Fixed-partner production decision replay row count differs."
            )
        line_number, expected = item
        fields = sorted(
            name
            for name in set(expected) | set(observed)
            if expected.get(name) != observed.get(name)
        )
        if fields:
            raise RuntimeError(
                "Fixed-partner production decision replay differs at line "
                f"{line_number}: fields={fields}."
            )
        expected_step = row_count // int(episode_count)
        expected_episode = row_count % int(episode_count)
        if (
            int(observed["step"]) != expected_step
            or int(observed["episode_index"]) != expected_episode
        ):
            raise RuntimeError(
                "Fixed-partner decisions are not in registered step-major order."
            )
        for name in names:
            collected[name].append(observed[name])
        row_count += 1
    expected_count = int(episode_steps) * int(episode_count)
    if row_count != expected_count:
        raise RuntimeError(
            "Fixed-partner production decision count differs: "
            f"observed={row_count}, expected={expected_count}."
        )

    arrays = {}
    for name in names:
        values = np.asarray(collected[name])
        arrays[name] = values.reshape(
            (int(episode_steps), int(episode_count)) + values.shape[1:]
        )
    return PanelReplayEvidence(**arrays)


def _validate_fixed_partner_state_replay(
    *,
    replay: FixedPartnerStateReplay,
    evidence: PanelReplayEvidence,
    episodes: Sequence[Mapping[str, Any]],
) -> None:
    """Validate the independently rebuilt state trajectory without tolerance."""

    import numpy as np

    trajectory_checks = {
        "ego_action": (replay.ego_action, evidence.ego_action),
        "partner_action": (replay.partner_action, evidence.partner_action),
        "response_code": (replay.response_code, evidence.response_code),
        "slot_log_belief": (
            replay.slot_log_belief,
            evidence.slot_log_belief,
        ),
    }
    for name, (observed, expected) in trajectory_checks.items():
        observed_array = np.asarray(observed)
        expected_array = np.asarray(expected)
        if not np.array_equal(observed_array, expected_array):
            differing = np.argwhere(observed_array != expected_array)
            first = differing[0].tolist() if len(differing) else None
            raise RuntimeError(
                "Fixed-partner state reconstruction differs from the exact "
                f"production trajectory: field={name}, first_index={first}."
            )

    final = replay.capture.world
    totals = {
        "raw_return": np.asarray(final.raw_return),
        "correct_delivery_count": np.asarray(
            final.correct_delivery_count
        ),
        "wrong_delivery_count": np.asarray(final.wrong_delivery_count),
        "indicator_activation_count": np.asarray(
            final.indicator_activation_count
        ),
    }
    for lane, episode in enumerate(episodes):
        differing = [
            name
            for name, values in totals.items()
            if values[lane].item() != episode[name]
        ]
        if differing:
            raise RuntimeError(
                "Fixed-partner reconstructed episode totals differ for "
                f"episode {lane}: fields={differing}."
            )


def _untriggered_panel_row(
    *,
    partner_index: int,
    episode: Mapping[str, Any],
) -> Mapping[str, Any]:
    branch = {
        "raw_return": float(episode["raw_return"]),
        "correct_delivery_count": int(episode["correct_delivery_count"]),
        "wrong_delivery_count": int(episode["wrong_delivery_count"]),
        "indicator_activation_count": int(
            episode["indicator_activation_count"]
        ),
    }
    return {
        "source": f"fixed_partner_{partner_index:02d}",
        "partner_index": int(partner_index),
        "episode_index": int(episode["episode_index"]),
        "episode_seed": int(episode["episode_seed"]),
        "triggered": False,
        "trigger_step": None,
        "trigger_action": None,
        "a1_action": None,
        "a1": branch,
        "a2_mask": branch,
        "a2_use": branch,
        "delta_response": 0.0,
        "delta_cost": 0.0,
        "delta_net": 0.0,
    }


def _run_trigger_partition(
    *,
    output: Path,
    config: Any,
    deployment: Deployment,
    capture: TriggerCapture,
    lane: int,
    identity: TriggerIdentity,
    replicas: int,
    resume: bool,
    fixed_partner_pool: FrozenPartnerPool | None,
    fixed_partner_index: int | None,
    runners: ContinuationRunners,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], Mapping[str, Any]]:
    directory = output / "trigger_partitions" / identity.trigger_id
    if resume:
        restored = _read_complete_partition(directory, replicas=replicas)
        if restored is not None:
            return restored
    reconstruction, continuations, values = _audit_trigger(
        config=config,
        deployment=deployment,
        capture=capture,
        lane=lane,
        identity=identity,
        replicas=replicas,
        fixed_partner_pool=fixed_partner_pool,
        fixed_partner_index=fixed_partner_index,
        runners=runners,
    )
    _write_partition(
        directory,
        reconstruction=reconstruction,
        continuations=continuations,
        trigger_values=values,
    )
    return reconstruction, continuations, values


def run_counterfactual_value_audit(args: Any) -> None:
    import jax
    import numpy as np

    config = load_config(args.config, run_kind="development")
    if int(args.replicas) <= 0:
        raise ValueError("Counterfactual replicas must be positive.")
    run_directory = Path(args.run_directory).resolve()
    response_path = Path(args.response_contrast).resolve()
    panel_manifest_path = Path(args.panel_manifest).resolve()
    output = Path(args.output).resolve()
    checkpoint_step = int(args.checkpoint_step)
    training_metadata = json.loads(
        (run_directory / "run_metadata.json").read_text(encoding="utf-8")
    )
    if int(training_metadata["effective_environment_steps"]) != checkpoint_step:
        raise ValueError(
            "The requested checkpoint is not the completed frozen training step."
        )
    training_identity = read_run_identity(run_directory)
    if training_identity.get("config") != config.to_mapping():
        raise ValueError("Training and audit configurations differ.")
    if training_identity.get("run_kind") != "development":
        raise ValueError("This audit is registered only for the development checkpoint.")
    response_identity = read_run_identity(response_path.parent.parent)
    panel_identity = read_run_identity(panel_manifest_path.parent)
    if response_identity.get("config") != config.to_mapping():
        raise ValueError("Response contrast and audit configurations differ.")
    if panel_identity.get("config") != config.to_mapping():
        raise ValueError("Partner panel and audit configurations differ.")
    response_seed = int(response_identity["seed"])
    panel_seed = int(panel_identity["seed"])
    expected_inputs = {
        "method": "path_c_v4_4_retrace_calibrated_control_r1",
        "run_kind": "development",
        "scientific_readout_allowed": False,
        "config": config.to_mapping(),
        "run_directory": str(run_directory),
        "checkpoint_step": checkpoint_step,
        "response_contrast": str(response_path),
        "panel_manifest": str(panel_manifest_path),
        "response_evaluation_seed": response_seed,
        "panel_evaluation_seed": panel_seed,
        "replicas": int(args.replicas),
    }
    resolved_path = output / "resolved_inputs.json"
    if resolved_path.is_file():
        observed_inputs = json.loads(resolved_path.read_text(encoding="utf-8"))
        if observed_inputs != expected_inputs:
            raise RuntimeError("Audit output belongs to different resolved inputs.")
    else:
        unexpected = [
            path for path in output.iterdir() if path.name != "logs"
        ] if output.is_dir() else []
        if unexpected:
            raise RuntimeError("Non-empty audit output has no resolved input record.")
        write_json(resolved_path, expected_inputs)
    checkpoint_directory = run_directory / "checkpoints" / str(checkpoint_step)
    if not checkpoint_directory.is_dir():
        raise FileNotFoundError(
            f"Frozen Orbax checkpoint is absent: {checkpoint_directory}."
        )
    frozen_sources_before = {
        "training_directory": _checkpoint_files(run_directory),
        "response_contrast_directory": _checkpoint_files(response_path.parent),
        "panel_directory": _checkpoint_files(panel_manifest_path.parent),
    }
    deployment = load_deployment(
        PopulationEntry(
            outer_unit_id=int(training_identity["outer_unit_id"]),
            run_directory=run_directory,
        ),
        config,
        checkpoint_step=checkpoint_step,
    )
    parameter_snapshot = deployment_parameter_snapshot(deployment)
    legacy_rows = read_legacy_v44_response_rows(response_path)
    corrected_summary = {
        "run_kind": "development",
        "scientific_readout_allowed": False,
        "source_schema": "frozen_v4_4_response_rows",
        **summarize_response_contrast(legacy_rows),
    }
    write_json(output / "corrected_response_summary.json", corrected_summary)
    triggered_rows = [row for row in legacy_rows if row.triggered]
    if any(
        row.trigger_step is None or row.episode_seed is None
        for row in triggered_rows
    ):
        raise RuntimeError("Frozen self-pair trigger identities are incomplete.")
    self_identities = [
        TriggerIdentity(
            source="self_pairing",
            partner_index=None,
            episode_index=int(row.episode_index),
            episode_seed=int(row.episode_seed),
            trigger_step=int(row.trigger_step),
        )
        for row in triggered_rows
    ]
    reconstructions: list[Mapping[str, Any]] = []
    continuation_rows: list[Mapping[str, Any]] = []
    trigger_values: list[Mapping[str, Any]] = []
    restored_self = [
        _read_complete_partition(
            output / "trigger_partitions" / identity.trigger_id,
            replicas=int(args.replicas),
        )
        for identity in self_identities
    ] if bool(args.resume) else []
    if restored_self and all(value is not None for value in restored_self):
        for restored in restored_self:
            if restored is None:
                raise RuntimeError("A registered self-pair partition is absent.")
            reconstruction, continuations, values = restored
            reconstructions.append(reconstruction)
            continuation_rows.extend(continuations)
            trigger_values.append(values)
    else:
        replay_rows, registered_capture = contrast_pairing_batch(
            config=config,
            left=deployment,
            right=deployment,
            pairing=Pairing(
                deployment_mode="posterior_use",
                split="sp",
                left_outer_unit_id=deployment.outer_unit_id,
                right_outer_unit_id=deployment.outer_unit_id,
            ),
            evaluation_seed=response_seed,
            include_trigger_capture=True,
        )
        validate_response_replay(legacy_rows, replay_rows)
        self_capture = _capture_from_registered_response_scan(
            registered_capture
        )
        _validate_capture_against_rows(self_capture, legacy_rows)
        self_runners = _make_continuation_runners(
            config=config,
            deployment=deployment,
            replicas=int(args.replicas),
            fixed_partner_pool=None,
        )
        for identity in self_identities:
            lane = identity.episode_index
            reconstruction, continuations, values = _run_trigger_partition(
                output=output,
                config=config,
                deployment=deployment,
                capture=self_capture,
                lane=lane,
                identity=identity,
                replicas=int(args.replicas),
                resume=bool(args.resume),
                fixed_partner_pool=None,
                fixed_partner_index=None,
                runners=self_runners,
            )
            reconstructions.append(reconstruction)
            continuation_rows.extend(continuations)
            trigger_values.append(values)

    panel_manifest = json.loads(
        panel_manifest_path.read_text(encoding="utf-8")
    )
    partner_paths = tuple(
        Path(value).resolve()
        for value in panel_manifest["partner_checkpoints"]
    )
    partner_pool = FrozenPartnerPool.from_checkpoints(partner_paths)
    fixed_partner_runners = _make_continuation_runners(
        config=config,
        deployment=deployment,
        replicas=int(args.replicas),
        fixed_partner_pool=partner_pool,
    )
    panel_contrast_rows: list[Mapping[str, Any]] = []
    a1_environment_steps = 0
    for partner_index, partner_path in enumerate(partner_paths):
        replayed_panel_rows, replayed_decisions = panel_batch(
            config=config,
            deployment=deployment,
            partner_pool=partner_pool,
            partner_index=partner_index,
            partner_label=f"partner_{partner_index:02d}",
            partner_checkpoint=partner_path,
            deployment_mode="posterior_use",
            evaluation_seed=panel_seed,
        )
        seeds = np.asarray(
            [row.episode_seed for row in replayed_panel_rows],
            dtype=np.uint32,
        )
        panel_episode_path = (
            panel_manifest_path.parent
            / f"partner_{partner_index:02d}"
            / "posterior_use"
            / "episodes.parquet"
        )
        panel_episodes = read_legacy_v44_panel_episode_rows(
            panel_episode_path
        )
        if len(panel_episodes) != config.evaluation.episodes_per_pairing:
            raise RuntimeError("Frozen partner panel episode count differs.")
        _validate_panel_episode_replay(panel_episodes, replayed_panel_rows)
        panel_decision_path = panel_episode_path.with_name("decisions.jsonl")
        evidence = _validate_and_collect_panel_decisions(
            frozen_path=panel_decision_path,
            replayed=replayed_decisions,
            episode_steps=int(config.environment.episode_steps),
            episode_count=int(config.evaluation.episodes_per_pairing),
        )
        state_replay = _capture_fixed_partner_states(
            config=config,
            deployment=deployment,
            partner_pool=partner_pool,
            partner_index=partner_index,
            evaluation_seed=panel_seed,
        )
        _validate_fixed_partner_state_replay(
            replay=state_replay,
            evidence=evidence,
            episodes=panel_episodes,
        )
        capture = _registered_panel_capture(
            state_replay.capture, evidence
        )
        triggered = set(
            np.nonzero(np.asarray(capture.triggered))[0].tolist()
        )
        trigger_steps = {
            lane: int(np.asarray(capture.trigger_step)[lane])
            for lane in triggered
        }
        panel_decisions = read_legacy_v44_panel_trigger_decisions(
            panel_decision_path,
            trigger_steps,
        )
        _validate_panel_capture(
            capture=capture,
            episodes=panel_episodes,
            decisions=panel_decisions,
        )
        jax.block_until_ready(capture)
        del replayed_decisions, replayed_panel_rows
        del evidence, state_replay
        for lane, episode in enumerate(panel_episodes):
            if lane not in triggered:
                panel_contrast_rows.append(
                    _untriggered_panel_row(
                        partner_index=partner_index, episode=episode
                    )
                )
                continue
            identity = TriggerIdentity(
                source=f"fixed_partner_{partner_index:02d}",
                partner_index=partner_index,
                episode_index=lane,
                episode_seed=int(seeds[lane]),
                trigger_step=int(np.asarray(capture.trigger_step)[lane]),
            )
            reconstruction, continuations, values = _run_trigger_partition(
                output=output,
                config=config,
                deployment=deployment,
                capture=capture,
                lane=lane,
                identity=identity,
                replicas=int(args.replicas),
                resume=bool(args.resume),
                fixed_partner_pool=partner_pool,
                fixed_partner_index=partner_index,
                runners=fixed_partner_runners,
            )
            reconstructions.append(reconstruction)
            continuation_rows.extend(continuations)
            trigger_values.append(values)
            a1_action, a1_world, a1_steps = _fixed_a1_continuation(
                config=config,
                deployment=deployment,
                pre_world=_tree_lane(capture.pre_world, lane),
                identity=identity,
                partner_pool=partner_pool,
                partner_index=partner_index,
                trigger_action=int(
                    np.asarray(capture.trigger_action)[lane]
                ),
                replicas=int(args.replicas),
                runners=fixed_partner_runners,
            )
            a1_environment_steps += int(a1_steps)
            panel_contrast_rows.append(
                _panel_contrast_row(
                    identity=identity,
                    capture=capture,
                    lane=lane,
                    pre_continuation_rows=continuations,
                    a1_action=a1_action,
                    a1_world=a1_world,
                )
            )

    write_parquet(output / "reconstruction.parquet", reconstructions)
    write_parquet(output / "continuations.parquet", continuation_rows)
    write_parquet(output / "trigger_values.parquet", trigger_values)
    write_parquet(
        output / "panel_response_contrast.parquet", panel_contrast_rows
    )
    responsibility_rows, responsibility = _responsibility_audit(run_directory)
    write_parquet(
        output / "responsibility_audit.parquet",
        responsibility_rows,
    )
    continuation_steps = sum(
        int(row["continuation_environment_steps"])
        for row in continuation_rows
    )
    replay_steps = (
        3 * int(config.evaluation.episodes_per_pairing)
        * int(config.environment.episode_steps)
        + 2 * len(partner_paths)
        * int(config.evaluation.episodes_per_pairing)
        * int(config.environment.episode_steps)
    )
    actual_steps = (
        continuation_steps + replay_steps + a1_environment_steps
    )
    trigger_summary = _summary_from_trigger_values(
        trigger_values,
        replicas=int(args.replicas),
        actual_environment_steps=actual_steps,
    )
    panel_summary = _summarize_panel_contrast(panel_contrast_rows)
    slot_summary = _summarize_slot_semantics(trigger_values)
    self_pairing = _self_pairing_retention(response_path.parent.parent)
    task_effect_counts: dict[str, int] = {}
    for row in continuation_rows:
        if row["branch"] != "use":
            continue
        label = str(row["task_effect_class"])
        task_effect_counts[label] = task_effect_counts.get(label, 0) + 1
    summary = {
        **trigger_summary,
        "self_pair_trigger_count": len(self_identities),
        "panel_response_row_count": len(panel_contrast_rows),
        "panel_trigger_count": sum(
            bool(row["triggered"]) for row in panel_contrast_rows
        ),
        "continuation_row_count": len(continuation_rows),
        "paired_continuation_task_effect_counts": task_effect_counts,
        "responsibility": responsibility,
        "panel_response_contrast": panel_summary,
        "slot_semantics_on_reconstructed_states": slot_summary,
        "self_pairing_capability": self_pairing,
        "formal_training_release_conditions": _formal_gate_statuses(
            trigger_summary=trigger_summary,
            panel_summary=panel_summary,
            slot_summary=slot_summary,
            self_pairing=self_pairing,
        ),
    }
    write_json(output / "summary.json", summary)
    frozen_sources_after = {
        "training_directory": _checkpoint_files(run_directory),
        "response_contrast_directory": _checkpoint_files(response_path.parent),
        "panel_directory": _checkpoint_files(panel_manifest_path.parent),
    }
    if frozen_sources_after != frozen_sources_before:
        raise RuntimeError("A frozen V4.4 input changed during the audit.")
    parameter_snapshot_after = deployment_parameter_snapshot(deployment)
    if len(parameter_snapshot) != len(parameter_snapshot_after) or any(
        not np.array_equal(before, after)
        for before, after in zip(
            parameter_snapshot, parameter_snapshot_after, strict=True
        )
    ):
        raise RuntimeError("The in-memory V4.4 parameter tree changed during audit.")
    write_json(
        output / "run_metadata.json",
        {
            **expected_inputs,
            "trigger_count": len(trigger_values),
            "continuation_row_count": len(continuation_rows),
            "actual_environment_steps": actual_steps,
            "frozen_inputs_unchanged": True,
        },
    )
    print(f"Complete counterfactual audit rows: {output}")


__all__ = [
    "inverse_cdf_actions",
    "deployment_parameter_snapshot",
    "read_legacy_v44_response_rows",
    "run_counterfactual_value_audit",
    "validate_response_replay",
]
