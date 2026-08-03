"""Owner-SP behavioral initialization for the V6 single actor.

Only the Official policy surface (observation, done/carry and action logits) is
recorded.  Environment state and partner identity never enter the DELTA model.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class OwnerBehaviorBatch(NamedTuple):
    observations: Any
    previous_actions: Any
    episode_starts: Any
    owner_logits: Any
    valid_mask: Any
    source_members: Any


def _carry_leaf_max_error(left: Any, right: Any) -> Any:
    """Compare recurrent leaves without subtracting bool/integer state."""

    import jax.numpy as jnp

    left_array = jnp.asarray(left)
    right_array = jnp.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        raise ValueError("Replayed carry leaves must have identical shape and dtype.")
    if jnp.issubdtype(left_array.dtype, jnp.inexact):
        return jnp.max(jnp.abs(left_array - right_array))
    return jnp.max(jnp.not_equal(left_array, right_array).astype(jnp.float32))


def collect_owner_behavior(
    *,
    environment: Any,
    owner_pool: Any,
    partner_pool: Any,
    length: int,
    key: Any,
    owner_members: Any | None = None,
) -> OwnerBehaviorBatch:
    """Collect exact stochastic owner-SP trajectories against qualified support."""

    import jax
    import jax.numpy as jnp

    count = int(environment.num_envs)
    reset_key, owner_member_key, partner_member_key, scan_key = jax.random.split(key, 4)
    environment_state, joint_observation = environment.reset(reset_key)
    owner_members = (
        owner_pool.sample_members(owner_member_key, count)
        if owner_members is None
        else jnp.asarray(owner_members, dtype=jnp.int32)
    )
    if owner_members.shape != (count,):
        raise ValueError("Owner member schedule must match the vector environment.")
    partner_members = partner_pool.sample_members(partner_member_key, count)
    owner_carry = owner_pool.initial_carry(count)
    partner_carry = partner_pool.initial_carry(count)
    starts = jnp.ones((count,), dtype=jnp.bool_)
    previous_actions = jnp.full((count,), 6, dtype=jnp.int32)
    # Alternate the ego role deterministically; this is a data-layout choice,
    # not a random scientific variable.
    owner_roles = (jnp.arange(count, dtype=jnp.int32) >= count // 2).astype(jnp.int32)

    def one(carry: Any, time_index: Any) -> tuple[Any, Any]:
        (
            current_environment,
            observations,
            owner_state,
            partner_state,
            episode_start,
            prior_action,
        ) = carry
        lane = jnp.arange(count, dtype=jnp.int32)
        owner_observation = observations[lane, owner_roles]
        partner_observation = observations[lane, 1 - owner_roles]
        step_root = jax.random.fold_in(scan_key, time_index)
        owner_action_key, partner_action_key, environment_key = jax.random.split(step_root, 3)
        owner_keys = jax.random.split(owner_action_key, count)
        partner_keys = jax.random.split(partner_action_key, count)
        next_owner_state, owner_logits, unused_owner_value = owner_pool.logits_and_value(
            owner_members,
            owner_observation,
            owner_state,
            episode_start,
        )
        del unused_owner_value
        owner_actions = jax.vmap(jax.random.categorical)(owner_keys, owner_logits)
        partner_actions, next_partner_state = partner_pool.step_with_keys(
            partner_members,
            partner_observation,
            partner_state,
            episode_start,
            partner_keys,
        )
        owner_first = jnp.stack((owner_actions, partner_actions), axis=-1)
        partner_first = jnp.stack((partner_actions, owner_actions), axis=-1)
        joint_actions = jnp.where(owner_roles[:, None] == 0, owner_first, partner_first)
        next_environment, next_observations, unused_reward, dones, unused_info = environment.step(
            current_environment, joint_actions, environment_key
        )
        del unused_reward, unused_info
        return (
            next_environment,
            next_observations,
            next_owner_state,
            next_partner_state,
            dones,
            owner_actions,
        ), (owner_observation, prior_action, episode_start, owner_logits)

    final, rows = jax.lax.scan(
        one,
        (
            environment_state,
            joint_observation,
            owner_carry,
            partner_carry,
            starts,
            previous_actions,
        ),
        jnp.arange(int(length), dtype=jnp.int32),
    )
    final_observation = final[1][jnp.arange(count), owner_roles]
    observations, actions, episode_starts, logits = rows
    return OwnerBehaviorBatch(
        observations=jnp.concatenate((observations, final_observation[None]), axis=0),
        previous_actions=jnp.concatenate((actions, final[5][None]), axis=0),
        episode_starts=jnp.concatenate((episode_starts, final[4][None]), axis=0),
        owner_logits=logits,
        valid_mask=jnp.ones(logits.shape[:-1], dtype=jnp.float32),
        source_members=jnp.broadcast_to(owner_members, logits.shape[:2]),
    )


def owner_behavior_loss(
    *,
    model: Any,
    params: Any,
    initial_state: Any,
    batch: OwnerBehaviorBatch,
) -> tuple[Any, dict[str, Any]]:
    """Forward KL from the owner-SP source into the one V6 actor."""

    import jax
    import jax.numpy as jnp

    dropout = jnp.zeros(batch.previous_actions.shape, dtype=jnp.bool_)
    unused_state, output = model.apply(
        {"params": params},
        initial_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        dropout,
        method=model.sequence,
    )
    del unused_state
    source_log = jax.nn.log_softmax(batch.owner_logits, axis=-1)
    source = jnp.exp(source_log)
    candidate_log = jax.nn.log_softmax(output.policy_logits[:-1], axis=-1)
    pointwise = jnp.sum(source * (source_log - candidate_log), axis=-1)
    denominator = jnp.maximum(jnp.sum(batch.valid_mask), 1.0)
    loss = jnp.sum(batch.valid_mask * pointwise) / denominator
    return loss, {
        "owner_distillation_kl": loss,
    }


def distill_owner_actor(
    *,
    model: Any,
    params: Any,
    batch: OwnerBehaviorBatch,
    initial_state_factory: Any,
    optimizer: Any,
    optimizer_state: Any,
    schedule: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Execute the registered four-by-64 behavior initialization scan."""

    import jax
    import jax.numpy as jnp
    import optax

    from .gradient_routing import keep_owned_gradients

    indexes = jnp.asarray(schedule, dtype=jnp.int32).reshape(
        (-1, int(schedule.shape[-1]))
    )

    def one(carry: Any, lane_indexes: Any):
        current, current_optimizer_state = carry
        sliced = OwnerBehaviorBatch(
            observations=batch.observations[:, lane_indexes],
            previous_actions=batch.previous_actions[:, lane_indexes],
            episode_starts=batch.episode_starts[:, lane_indexes],
            owner_logits=batch.owner_logits[:, lane_indexes],
            valid_mask=batch.valid_mask[:, lane_indexes],
            source_members=batch.source_members[:, lane_indexes],
        )
        initial_state = initial_state_factory(int(lane_indexes.shape[0]))

        def objective(candidate: Any):
            return owner_behavior_loss(
                model=model,
                params=candidate,
                initial_state=initial_state,
                batch=sliced,
            )

        (loss, metrics), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(current)
        gradients = keep_owned_gradients(
            gradients, loss_name="owner_distillation"
        )
        updates, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current
        )
        return (
            optax.apply_updates(current, updates),
            next_optimizer_state,
        ), {**metrics, "owner_distillation_loss": loss}

    (next_params, next_state), metrics = jax.lax.scan(
        one, (params, optimizer_state), indexes
    )
    return next_params, next_state, jax.tree_util.tree_map(
        lambda value: value[-1], metrics
    )


__all__ = [
    "OwnerBehaviorBatch",
    "collect_owner_behavior",
    "distill_owner_actor",
    "owner_behavior_loss",
]
