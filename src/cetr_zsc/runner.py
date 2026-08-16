"""Whole-episode vectorized collection for self-play and external lanes."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

from .losses import return_to_go
from .types import EpisodeBatch


class PartnerFunctions(NamedTuple):
    initial_state: Callable[..., Any]
    step: Callable[..., tuple[Any, Any, Any, Any]]
    observe: Callable[..., Any]
    run_id: Callable[..., Any]
    diagnostics: Callable[..., Mapping[str, Any]]


class PartnerState(NamedTuple):
    carry: Any
    member: Any
    parent: Any
    fold: Any
    role: Any
    stage_slot: Any


class PartnerContext(NamedTuple):
    reset_key: Any


def external_lane_assignment(
    ext_lanes: int,
    parent_count: int,
    update_index: Any,
    parent_members: Any,
) -> tuple[Any, Any, Any, Any]:
    """Assign every external lane without sampling.

    For E=64,P=16 and E=16,P=4, each parent receives exactly one A0, A1,
    B0, and B1 lane on every update.  The stage slot rotates with the update
    and lane block, so all three checkpoints are balanced over time.  The
    mechanical E=2<P path is intentionally a partial smoke allocation.
    """

    import jax.numpy as jnp

    count = int(ext_lanes)
    parents = int(parent_count)
    lanes = jnp.arange(count, dtype=jnp.int32)
    parent = jnp.mod(lanes, parents)
    fold = jnp.mod(lanes // parents, 2)
    role = jnp.mod(lanes // (2 * parents), 2)
    slot = lanes // parents
    stage_slot = jnp.mod(jnp.asarray(update_index, dtype=jnp.int32) + slot, 3)
    members = jnp.asarray(parent_members, dtype=jnp.int32)
    member = members[parent, stage_slot]
    return member, parent, fold, role


def make_static_partner_functions(*, pool_metadata: Any, official_pool: Any) -> PartnerFunctions:
    """Create the deterministic parent assignment used by external lanes."""

    import jax
    import jax.numpy as jnp

    nominal = jnp.asarray(pool_metadata.parent_nominal_weights, dtype=jnp.float32)
    parent_members = jnp.asarray(pool_metadata.parent_members, dtype=jnp.int32)
    parent_count = int(len(pool_metadata.parent_ids))
    if nominal.shape != (parent_count,):
        raise ValueError("Parent nominal weights do not match parent IDs.")
    if parent_members.shape != (parent_count, 3):
        raise ValueError("Every parent must expose three checkpoint slots.")

    def initial_state(
        ext_lane_count: int, key: Any, update_index: Any = 0
    ) -> PartnerState:
        del key
        count = int(ext_lane_count)
        if count <= 0 or count % 2:
            raise ValueError("External lane count must be a positive even number.")
        member, parent, fold, role = external_lane_assignment(
            count, parent_count, update_index, parent_members
        )
        stage_slot = jnp.mod(
            jnp.asarray(update_index, dtype=jnp.int32) + jnp.arange(count) // parent_count,
            3,
        )
        return PartnerState(
            carry=official_pool.initial_carry(count),
            member=member,
            parent=parent,
            fold=fold,
            role=role,
            stage_slot=stage_slot,
        )

    def step(
        parameters: Any,
        state: PartnerState,
        observations: Any,
        episode_start: Any,
        keys: Any,
    ) -> tuple[Any, PartnerState, PartnerContext, Any]:
        del parameters
        import jax
        import jax.numpy as jnp

        action, next_carry = official_pool.step_with_keys(
            state.member, observations, state.carry, episode_start, keys
        )
        reset_key = jax.random.fold_in(keys[0], 911)
        next_state = PartnerState(
            carry=next_carry,
            member=state.member,
            parent=state.parent,
            fold=state.fold,
            role=state.role,
            stage_slot=state.stage_slot,
        )
        return (
            action,
            next_state,
            PartnerContext(reset_key=reset_key),
            jnp.zeros_like(action, dtype=jnp.float32),
        )

    def observe(
        parameters: Any,
        state: PartnerState,
        context: PartnerContext,
        observations: Any,
        actions: Any,
        rewards: Any,
        dones: Any,
        next_observations: Any,
    ) -> PartnerState:
        del parameters, context, observations, actions, rewards, next_observations
        import jax.numpy as jnp

        done = jnp.asarray(dones, dtype=jnp.bool_)
        return PartnerState(
            carry=jnp.where(done[..., None], jnp.zeros_like(state.carry), state.carry),
            member=state.member,
            parent=state.parent,
            fold=state.fold,
            role=state.role,
            stage_slot=state.stage_slot,
        )

    def run_id(parameters: Any, state: PartnerState, context: Any) -> Any:
        del parameters, context
        return state.member

    def diagnostics(
        parameters: Any, state: PartnerState, context: Any
    ) -> Mapping[str, Any]:
        del parameters, context
        import jax.numpy as jnp

        return {
            "source": jnp.full(state.member.shape, 1, dtype=jnp.int32),
            "member": state.member,
            "run_id": state.member,
            "parent": state.parent,
            "fold": state.fold,
            "role": state.role,
            "stage_slot": state.stage_slot,
            "parent_count": jnp.asarray(parent_count, dtype=jnp.int32),
        }

    return PartnerFunctions(initial_state, step, observe, run_id, diagnostics)


def _categorical_log_probability(logits: Any, actions: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    logp = jnn.log_softmax(logits, axis=-1)
    indexes = jnp.asarray(actions, dtype=jnp.int32)[..., None]
    return jnp.take_along_axis(logp, indexes, axis=-1)[..., 0]


def _sample_actions(keys: Any, logits: Any) -> Any:
    import jax

    return jax.vmap(lambda key, value: jax.random.categorical(key, value))(
        keys, logits
    )


def collect_episodes(
    *,
    environment: Any,
    model: Any,
    params: Any,
    partner_functions: PartnerFunctions,
    random_key: Any,
    update_index: Any,
) -> tuple[Any, EpisodeBatch, Mapping[str, Any]]:
    """Collect one fresh, complete episode on every lane.

    The first half of the vector is self-play and the second half is external.
    Self-play roles alternate; external roles and parent/fold assignments come
    from ``external_lane_assignment`` for the supplied update index.
    """

    import jax
    import jax.numpy as jnp

    update_index = jnp.asarray(update_index, dtype=jnp.int32)

    lane_count = int(environment.num_envs)
    steps = int(environment.episode_steps)
    if lane_count <= 0 or lane_count % 4:
        raise ValueError("Episode collection requires lanes divisible by four.")
    if steps <= 0:
        raise ValueError("Episode length must be positive.")
    self_count = lane_count // 2
    external_count = lane_count - self_count
    lane_indexes = jnp.arange(lane_count, dtype=jnp.int32)
    self_roles = jnp.mod(jnp.arange(self_count, dtype=jnp.int32), 2)

    reset_root = jax.random.fold_in(random_key, 101)
    reset_keys = jax.vmap(
        lambda index: jax.random.fold_in(reset_root, index)
    )(lane_indexes)
    environment_state, joint_observations = environment.reset_with_keys(reset_keys)
    partner_root = jax.random.fold_in(random_key, 103)
    initial_partner_state = partner_functions.initial_state(
        external_count, partner_root, update_index
    )
    external_roles = jnp.asarray(initial_partner_state.role, dtype=jnp.int32)
    ego_roles = jnp.concatenate((self_roles, external_roles), axis=0)
    ego_carry = model.initial_carry(lane_count)
    self_carry = model.initial_carry(self_count)
    episode_start = jnp.ones((lane_count,), dtype=jnp.bool_)
    loop_key = jax.random.fold_in(random_key, 107)
    external_indexes = jnp.arange(external_count, dtype=jnp.int32) + self_count

    def one(carry: Any, unused_time: Any) -> tuple[Any, tuple[Any, ...]]:
        del unused_time
        (
            current_environment,
            current_observations,
            current_ego_carry,
            current_self_carry,
            current_partner_state,
            current_episode_start,
            current_key,
        ) = carry
        step_key, next_key = jax.random.split(current_key)
        ego_root, self_root, partner_root, environment_root = jax.random.split(
            step_key, 4
        )
        ego_keys = jax.random.split(ego_root, lane_count)
        self_keys = jax.random.split(self_root, self_count)
        partner_keys = jax.random.split(partner_root, external_count)
        environment_keys = jax.random.split(environment_root, lane_count)

        ego_observation = current_observations[lane_indexes, ego_roles]
        self_observation = current_observations[
            lane_indexes[:self_count], 1 - self_roles
        ]
        next_ego_carry, ego_logits, ego_value = model.step(
            params,
            current_ego_carry,
            ego_observation,
            current_episode_start,
        )
        next_self_carry, self_logits, self_value = model.step(
            params,
            current_self_carry,
            self_observation,
            current_episode_start[:self_count],
        )
        ego_action = _sample_actions(ego_keys, ego_logits)
        self_action = _sample_actions(self_keys, self_logits)
        ego_old_logp = _categorical_log_probability(ego_logits, ego_action)
        self_old_logp = _categorical_log_probability(self_logits, self_action)

        external_observation = current_observations[
            external_indexes, 1 - ego_roles[external_indexes]
        ]
        external_action, stepped_partner_state, partner_context, unused_value = (
            partner_functions.step(
                params,
                current_partner_state,
                external_observation,
                current_episode_start[external_indexes],
                partner_keys,
            )
        )
        del unused_value
        partner_action = jnp.concatenate((self_action, external_action), axis=0)
        ego_first = jnp.stack((ego_action, partner_action), axis=-1)
        partner_first = jnp.stack((partner_action, ego_action), axis=-1)
        joint_action = jnp.where(
            (ego_roles == 0)[:, None], ego_first, partner_first
        )
        (
            next_environment,
            next_observations,
            unused_reward,
            done,
            info,
        ) = environment.step_training_fast_with_keys(
            current_environment, joint_action, environment_keys
        )
        del unused_reward
        raw_by_agent = jnp.asarray(info["raw_rewards_by_agent"], dtype=jnp.float32)
        reward = raw_by_agent[lane_indexes, ego_roles]
        next_external_observation = next_observations[
            external_indexes, 1 - ego_roles[external_indexes]
        ]
        next_partner_state = partner_functions.observe(
            params,
            stepped_partner_state,
            partner_context,
            external_observation,
            external_action,
            reward[external_indexes],
            done[external_indexes],
            next_external_observation,
        )
        next_carry = (
            next_environment,
            next_observations,
            next_ego_carry,
            next_self_carry,
            next_partner_state,
            jnp.asarray(done, dtype=jnp.bool_),
            next_key,
        )
        row = (
            ego_observation,
            self_observation,
            current_episode_start,
            ego_action,
            ego_old_logp,
            ego_value,
            self_action,
            self_old_logp,
            self_value,
            reward,
            jnp.asarray(done, dtype=jnp.bool_),
        )
        return next_carry, row

    final_carry, rows = jax.lax.scan(
        one,
        (
            environment_state,
            joint_observations,
            ego_carry,
            self_carry,
            initial_partner_state,
            episode_start,
            loop_key,
        ),
        jnp.arange(steps, dtype=jnp.int32),
    )
    (
        unused_environment,
        unused_observations,
        unused_ego_carry,
        unused_self_carry,
        unused_partner_state,
        unused_episode_start,
        next_key,
    ) = final_carry
    del (
        unused_environment,
        unused_observations,
        unused_ego_carry,
        unused_self_carry,
        unused_partner_state,
        unused_episode_start,
    )

    (
        observations,
        self_observations,
        episode_starts,
        actions,
        old_log_probabilities,
        old_values,
        self_actions,
        self_old_log_probabilities,
        self_old_values,
        rewards,
        dones,
    ) = rows
    value_targets = return_to_go(rewards)
    sp_other_value_targets = value_targets[:, :self_count]
    advantages = jnp.asarray(value_targets, dtype=jnp.float32) - jnp.asarray(
        old_values, dtype=jnp.float32
    )
    sp_other_advantages = sp_other_value_targets - jnp.asarray(
        self_old_values, dtype=jnp.float32
    )
    episode_return = jnp.sum(jnp.asarray(rewards, dtype=jnp.float32), axis=0)
    parent = initial_partner_state.parent
    partner_diagnostics = partner_functions.diagnostics(
        params, initial_partner_state, None
    )
    parent_count_value = partner_diagnostics.get("parent_count")
    parent_count = (
        int(jnp.max(parent)) + 1
        if parent_count_value is None
        else int(parent_count_value)
    )
    lane_stream = jnp.concatenate(
        (
            jnp.zeros((self_count,), dtype=jnp.int32),
            jnp.ones((external_count,), dtype=jnp.int32),
        )
    )
    lane_parent = jnp.concatenate(
        (jnp.full((self_count,), -1, dtype=jnp.int32), initial_partner_state.parent)
    )
    lane_fold = jnp.concatenate(
        (jnp.full((self_count,), -1, dtype=jnp.int32), initial_partner_state.fold)
    )
    lane_weight = jnp.ones((lane_count,), dtype=jnp.float32)
    member_index = jnp.concatenate(
        (jnp.full((self_count,), -1, dtype=jnp.int32), initial_partner_state.member)
    )

    parent_counts = jnp.bincount(parent, length=parent_count)
    metrics = {
        "sp_return_mean": jnp.mean(episode_return[:self_count]),
        "external_return_mean": jnp.mean(episode_return[self_count:]),
        "premature_done_count": jnp.sum(dones[:-1].astype(jnp.float32)),
        "final_done_fraction": jnp.mean(dones[-1].astype(jnp.float32)),
        "external_parent_episode_counts": parent_counts,
        "parent_episode_counts": parent_counts,
    }
    batch = EpisodeBatch(
        observations=observations,
        actions=actions,
        old_log_probabilities=old_log_probabilities,
        old_values=old_values,
        rewards=rewards,
        dones=dones,
        value_targets=value_targets,
        sp_other_value_targets=sp_other_value_targets,
        advantages=advantages,
        sp_other_advantages=sp_other_advantages,
        episode_starts=episode_starts,
        sp_other_observations=self_observations,
        sp_other_actions=self_actions,
        sp_other_old_log_probabilities=self_old_log_probabilities,
        sp_other_old_values=self_old_values,
        lane_stream=lane_stream,
        lane_parent=lane_parent,
        lane_fold=lane_fold,
        lane_weight=lane_weight,
        episode_return=episode_return,
        ego_roles=ego_roles,
        member_index=member_index,
    )
    return next_key, batch, metrics


__all__ = [
    "PartnerContext",
    "PartnerFunctions",
    "PartnerState",
    "collect_episodes",
    "external_lane_assignment",
    "make_static_partner_functions",
]
