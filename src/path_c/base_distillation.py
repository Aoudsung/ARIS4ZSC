"""Owner-SP behavioral initialization for the r3 robust base.

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
    previous_actions = jnp.zeros((count,), dtype=jnp.int32)
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
    """Forward-KL owner distillation; only base/task parameters may own it."""

    import jax
    import jax.numpy as jnp

    zeros = jnp.zeros(batch.previous_actions.shape, dtype=jnp.float32)
    gates = jnp.zeros(batch.previous_actions.shape, dtype=jnp.float32)
    unused_state, output = model.apply(
        {"params": params},
        initial_state,
        batch.observations,
        batch.previous_actions,
        zeros,
        batch.episode_starts,
        gates,
        method=model.sequence,
    )
    del unused_state
    source_log = jax.nn.log_softmax(batch.owner_logits, axis=-1)
    source = jnp.exp(source_log)
    candidate_log = jax.nn.log_softmax(output.base_logits[:-1], axis=-1)
    pointwise = jnp.sum(source * (source_log - candidate_log), axis=-1)
    denominator = jnp.maximum(jnp.sum(batch.valid_mask), 1.0)
    loss = jnp.sum(batch.valid_mask * pointwise) / denominator
    return loss, {
        "base_distillation_kl": loss,
        "residual_rms_at_initialization": jnp.sqrt(
            jnp.mean(jnp.square(output.residual_logits[:-1]))
        ),
    }


def gate_zero_consistency(
    *,
    model: Any,
    params: Any,
    initial_state: Any,
    batch: OwnerBehaviorBatch,
    atol: float = 1.0e-7,
) -> dict[str, Any]:
    """Verify that ``gate=0`` is exactly the deterministic robust-base path.

    Official and DELTA recurrent carries have different registered structures,
    so source carries cannot be compared leaf-for-leaf.  The enforceable
    deployment contract is instead checked directly: identical legal history
    replay must produce identical DELTA carry/logits, and execution logits at
    gate zero must equal the base logits without a residual contribution.
    """

    import jax
    import jax.numpy as jnp

    zeros = jnp.zeros(batch.previous_actions.shape, dtype=jnp.float32)

    def replay() -> tuple[Any, Any]:
        return model.apply(
            {"params": params},
            initial_state,
            batch.observations,
            batch.previous_actions,
            zeros,
            batch.episode_starts,
            zeros,
            method=model.sequence,
        )

    first_state, first = replay()
    second_state, second = replay()
    carry_errors = tuple(
        _carry_leaf_max_error(left, right)
        for left, right in zip(
            jax.tree_util.tree_leaves(first_state),
            jax.tree_util.tree_leaves(second_state),
            strict=True,
        )
    )
    carry_error = (
        jnp.max(jnp.stack(carry_errors))
        if carry_errors
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    execution_error = jnp.max(
        jnp.abs(first.execution_logits - first.base_logits)
    )
    replay_logit_error = jnp.max(
        jnp.abs(first.execution_logits - second.execution_logits)
    )
    residual_rms = jnp.sqrt(jnp.mean(jnp.square(first.residual_logits)))
    passed = (
        (carry_error <= float(atol))
        & (execution_error <= float(atol))
        & (replay_logit_error <= float(atol))
    )
    return {
        "passed": passed,
        "carry_replay_max_error": carry_error,
        "gate_zero_execution_max_error": execution_error,
        "gate_zero_logit_replay_max_error": replay_logit_error,
        "conditional_residual_rms": residual_rms,
        "tolerance": jnp.asarray(float(atol), dtype=jnp.float32),
    }


__all__ = [
    "OwnerBehaviorBatch",
    "collect_owner_behavior",
    "gate_zero_consistency",
    "owner_behavior_loss",
]
