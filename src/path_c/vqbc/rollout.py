"""Device-side complete-episode rollout for the fourth Path C model."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

from .model import encode_response_codes, vqbc_forward
from .policy import (
    deployment_belief_after_response,
    regularized_policy,
    uniform_slot_log_belief,
)
from .quotient import slot_bayes_update
from .types import (
    VQBCDecisionRecord,
    VQBCPolicyState,
    VQBCRolloutBatch,
    VQBCRolloutState,
)


class VQBCRolloutCallbacks(NamedTuple):
    """Injected pure operations for official networks and training partners."""

    model_apply: Callable[..., Any]
    reference_apply: Callable[..., Any]
    partner_step: Callable[..., Any]
    partner_observe: Callable[..., Any]


def _seat_observations(observations: Any, seats: Any) -> tuple[Any, Any]:
    import jax.numpy as jnp

    values = jnp.asarray(observations)
    if values.ndim < 3 or values.shape[1] != 2:
        raise ValueError("Vector observations must have [environment, seat, ...] shape.")
    selector = jnp.asarray(seats, dtype=jnp.bool_).reshape(
        (values.shape[0],) + (1,) * (values.ndim - 2)
    )
    return (
        jnp.where(selector, values[:, 1], values[:, 0]),
        jnp.where(selector, values[:, 0], values[:, 1]),
    )


def initialize_policy_state(
    *,
    model_initial_carry: Callable[[int], Any],
    reference_initial_carry: Callable[[int], Any],
    batch_size: int,
    slot_count: int,
    action_count: int,
    initial_temperature: float,
) -> VQBCPolicyState:
    import jax.numpy as jnp

    return VQBCPolicyState(
        reference_carry=reference_initial_carry(batch_size),
        value_carry=model_initial_carry(batch_size),
        slot_log_belief=uniform_slot_log_belief((batch_size,), slot_count),
        previous_action=jnp.full(
            (batch_size,), action_count, dtype=jnp.int32
        ),
        previous_team_reward=jnp.zeros((batch_size,), dtype=jnp.float32),
        episode_start=jnp.ones((batch_size,), dtype=jnp.bool_),
        log_temperature=jnp.full(
            (batch_size,), jnp.log(initial_temperature), dtype=jnp.float32
        ),
        generic_log_temperature=jnp.full(
            (batch_size,), jnp.log(initial_temperature), dtype=jnp.float32
        ),
    )


def initialize_rollout(
    *,
    environment: Any,
    model_initial_carry: Callable[[int], Any],
    reference_initial_carry: Callable[[int], Any],
    partner_initial_carry: Callable[[int], Any],
    random_key: Any,
    slot_count: int = 8,
    action_count: int = 6,
    partner_member_count: int,
    initial_temperature: float = 1.0,
) -> VQBCRolloutState:
    import jax
    import jax.numpy as jnp

    if partner_member_count <= 0:
        raise ValueError("At least one immutable training partner is required.")
    next_key, environment_key, partner_key, seat_key = jax.random.split(random_key, 4)
    environment_state, observations = environment.reset(environment_key)
    count = int(environment.num_envs)
    return VQBCRolloutState(
        environment_state=environment_state,
        observations=observations,
        policy_state=initialize_policy_state(
            model_initial_carry=model_initial_carry,
            reference_initial_carry=reference_initial_carry,
            batch_size=count,
            slot_count=slot_count,
            action_count=action_count,
            initial_temperature=initial_temperature,
        ),
        partner_carry=partner_initial_carry(count),
        # Member zero is the rollout-frozen current policy.  Remaining indexes
        # address immutable historical checkpoints.
        partner_member_index=jax.random.randint(
            partner_key, (count,), 0, partner_member_count + 1
        ),
        ego_seat=jax.random.bernoulli(seat_key, shape=(count,)).astype(jnp.int32),
        episode_step=jnp.zeros((count,), dtype=jnp.int32),
        episode_id=jnp.arange(count, dtype=jnp.int64),
        episode_return=jnp.zeros((count,), dtype=jnp.float32),
        completed_episodes=jnp.asarray(0, dtype=jnp.int64),
        effective_environment_steps=jnp.asarray(0, dtype=jnp.int64),
        random_key=next_key,
    )


def policy_action(
    *,
    callbacks: VQBCRolloutCallbacks,
    params: Mapping[str, Any],
    policy_state: VQBCPolicyState,
    observations: Any,
    key: Any,
    deployment_mode: str,
    gamma: float,
) -> tuple[VQBCPolicyState, Any, Any, VQBCDecisionRecord, Any]:
    import jax
    import jax.numpy as jnp

    next_reference_carry, reference_logits = callbacks.reference_apply(
        policy_state.reference_carry,
        observations,
        policy_state.episode_start,
    )
    next_value_carry, raw_sequence = callbacks.model_apply(
        params,
        policy_state.value_carry,
        observations[None, ...],
        policy_state.previous_action[None, ...],
        policy_state.previous_team_reward[None, ...],
        policy_state.episode_start[None, ...],
    )
    raw = jax.tree_util.tree_map(lambda value: value[0], raw_sequence)
    belief = policy_state.slot_log_belief
    if deployment_mode == "prior_only":
        belief = uniform_slot_log_belief(
            belief.shape[:-1], int(belief.shape[-1])
        )
    output = vqbc_forward(
        raw_output=raw,
        reference_logits=reference_logits,
        slot_log_belief=belief,
        temperature=jnp.exp(policy_state.log_temperature),
        generic_temperature=jnp.exp(policy_state.generic_log_temperature),
        deployment_mode=deployment_mode,
        gamma=gamma,
    )
    generic = regularized_policy(
        reference_logits,
        output.information_gain,
        jnp.exp(policy_state.generic_log_temperature),
    )
    key_array = jnp.asarray(key)
    action = (
        jax.vmap(
            lambda lane_key, lane_logits: jax.random.categorical(
                lane_key, lane_logits, axis=-1
            )
        )(key_array, output.execution_logits)
        if key_array.ndim == 2
        else jax.random.categorical(key_array, output.execution_logits, axis=-1)
    )
    execution_log_probs = jax.nn.log_softmax(output.execution_logits, axis=-1)
    reference_log_probs = jax.nn.log_softmax(reference_logits, axis=-1)
    execution_probs = jnp.exp(execution_log_probs)
    kl = jnp.maximum(
        jnp.sum(
            execution_probs * (execution_log_probs - reference_log_probs),
            axis=-1,
        ),
        0.0,
    )
    class_count = jnp.max(output.quotient_ids, axis=-1) + 1
    slot_probs = jnp.exp(belief)
    entropy = -jnp.sum(slot_probs * belief, axis=-1)
    next_state = policy_state._replace(
        reference_carry=next_reference_carry,
        value_carry=next_value_carry,
        slot_log_belief=belief,
    )
    record = VQBCDecisionRecord(
        reference_logits=reference_logits,
        execution_logits=output.execution_logits,
        j_use=output.j_use,
        j_mask=output.j_mask,
        kl_divergence=kl,
        action=action,
        reference_greedy_action=jnp.argmax(reference_logits, axis=-1),
        quotient_count=class_count,
        belief_entropy=entropy,
        response_code=jnp.full(action.shape, -1, dtype=jnp.int32),
    )
    return next_state, action, output, record, generic.logits


def collect_rollout(
    *,
    state: VQBCRolloutState,
    length: int,
    environment: Any,
    callbacks: VQBCRolloutCallbacks,
    model: Any,
    params: Mapping[str, Any],
    codebook_embeddings: Any,
    partner_member_count: int,
    deployment_mode: str = "posterior_use",
    gamma: float = 0.99,
    terminal_response: int = 15,
) -> tuple[VQBCRolloutState, VQBCRolloutBatch]:
    """Collect synchronized complete episodes without carrying a replay pool."""

    import jax
    import jax.numpy as jnp

    if length != int(environment.episode_steps):
        raise ValueError("Fourth-model rollout length must equal one full episode.")
    count = int(environment.num_envs)
    initial_value_carry = state.policy_state.value_carry

    def one_step(
        current: VQBCRolloutState, unused: Any
    ) -> tuple[VQBCRolloutState, tuple[Any, ...]]:
        del unused
        (
            next_root_key,
            action_key,
            environment_key,
            partner_key,
            next_partner_key,
            next_seat_key,
        ) = jax.random.split(current.random_key, 6)
        ego_observation, partner_observation = _seat_observations(
            current.observations, current.ego_seat
        )
        (
            stepped_policy,
            ego_action,
            output,
            decision,
            generic_logits,
        ) = policy_action(
            callbacks=callbacks,
            params=params,
            policy_state=current.policy_state,
            observations=ego_observation,
            key=action_key,
            deployment_mode=deployment_mode,
            gamma=gamma,
        )
        (
            partner_action,
            tentative_partner_carry,
            partner_context,
        ) = callbacks.partner_step(
            params,
            current.partner_member_index,
            partner_observation,
            current.partner_carry,
            current.policy_state.episode_start,
            partner_key,
        )
        joint_actions = jnp.stack(
            (
                jnp.where(current.ego_seat == 0, ego_action, partner_action),
                jnp.where(current.ego_seat == 1, ego_action, partner_action),
            ),
            axis=-1,
        )
        (
            next_environment_state,
            next_observations,
            rewards,
            dones,
            info,
        ) = environment.step(
            current.environment_state, joint_actions, environment_key
        )
        dones = jnp.asarray(dones, dtype=jnp.bool_)
        rewards = jnp.asarray(rewards, dtype=jnp.float32)
        next_ego_observation, unused_partner_observation = _seat_observations(
            next_observations, current.ego_seat
        )
        del unused_partner_observation
        terminal_observations, unused_terminal_partner = _seat_observations(
            info["terminal_observations"], current.ego_seat
        )
        del unused_terminal_partner
        response_next_observation = jnp.where(
            dones.reshape(dones.shape + (1,) * (next_ego_observation.ndim - 1)),
            terminal_observations,
            next_ego_observation,
        )
        response_code, unused_logits, unused_signature = encode_response_codes(
            model=model,
            params=params,
            observations=ego_observation[None, ...],
            actions=ego_action[None, ...],
            next_observations=response_next_observation[None, ...],
            dones=dones[None, ...],
            codebook_embeddings=codebook_embeddings,
            terminal_response=terminal_response,
        )
        del unused_logits, unused_signature
        response_code = response_code[0]
        decision = decision._replace(response_code=response_code)
        unused_ego_observation, partner_next_observation = _seat_observations(
            next_observations, current.ego_seat
        )
        del unused_ego_observation
        unused_terminal_ego, terminal_partner_observation = _seat_observations(
            info["terminal_observations"], current.ego_seat
        )
        del unused_terminal_ego
        partner_response_next_observation = jnp.where(
            dones.reshape(
                dones.shape + (1,) * (partner_next_observation.ndim - 1)
            ),
            terminal_partner_observation,
            partner_next_observation,
        )
        next_partner_carry = callbacks.partner_observe(
            params,
            current.partner_member_index,
            tentative_partner_carry,
            partner_context,
            partner_observation,
            partner_response_next_observation,
            partner_action,
            rewards,
            dones,
            codebook_embeddings,
        )
        updated_belief = slot_bayes_update(
            slot_log_belief=stepped_policy.slot_log_belief,
            slot_response_probabilities=output.response_probabilities,
            action=ego_action,
            response_code=response_code,
        )
        uniform = uniform_slot_log_belief(
            (count,), int(updated_belief.shape[-1])
        )
        updated_belief = deployment_belief_after_response(
            mode=deployment_mode,
            current_log_belief=stepped_policy.slot_log_belief,
            updated_log_belief=updated_belief,
        )
        next_policy = stepped_policy._replace(
            slot_log_belief=jnp.where(
                dones[:, None], uniform, updated_belief
            ),
            previous_action=jnp.where(
                dones,
                jnp.full((count,), output.q_values.shape[-1], dtype=jnp.int32),
                ego_action,
            ),
            previous_team_reward=jnp.where(dones, 0.0, rewards),
            episode_start=dones,
        )
        completed_return = current.episode_return + rewards
        next_episode_id = current.episode_id + count
        sampled_partner = jax.random.randint(
            next_partner_key, (count,), 0, partner_member_count + 1
        )
        sampled_seat = jax.random.bernoulli(
            next_seat_key, shape=(count,)
        ).astype(jnp.int32)
        next_state = VQBCRolloutState(
            environment_state=next_environment_state,
            observations=next_observations,
            policy_state=next_policy,
            partner_carry=next_partner_carry,
            partner_member_index=jnp.where(
                dones, sampled_partner, current.partner_member_index
            ),
            ego_seat=jnp.where(dones, sampled_seat, current.ego_seat),
            episode_step=jnp.where(dones, 0, current.episode_step + 1),
            episode_id=jnp.where(dones, next_episode_id, current.episode_id),
            episode_return=jnp.where(dones, 0.0, completed_return),
            completed_episodes=(
                current.completed_episodes + jnp.sum(dones, dtype=jnp.int64)
            ),
            effective_environment_steps=(
                current.effective_environment_steps + count
            ),
            random_key=next_root_key,
        )
        return next_state, (
            ego_observation,
            response_next_observation,
            current.policy_state.episode_start,
            current.policy_state.previous_action,
            current.policy_state.previous_team_reward,
            decision.reference_logits,
            decision.execution_logits,
            generic_logits,
            current.policy_state.slot_log_belief,
            ego_action,
            rewards,
            dones,
            response_code,
            current.episode_id,
            current.episode_step,
            jnp.where(dones, completed_return, jnp.nan),
            decision.quotient_count,
            decision.j_use,
            decision.j_mask,
        )

    final_state, recorded = jax.lax.scan(
        one_step, state, xs=None, length=length
    )
    final_ego_observation, unused_final_partner = _seat_observations(
        final_state.observations, final_state.ego_seat
    )
    del unused_final_partner
    unused_carry, final_reference_logits = callbacks.reference_apply(
        final_state.policy_state.reference_carry,
        final_ego_observation,
        final_state.policy_state.episode_start,
    )
    del unused_carry
    batch = VQBCRolloutBatch(
        observations=jnp.concatenate(
            (recorded[0], final_ego_observation[None, ...]), axis=0
        ),
        response_next_observations=recorded[1],
        episode_start=jnp.concatenate(
            (
                recorded[2],
                final_state.policy_state.episode_start[None, ...],
            ),
            axis=0,
        ),
        previous_actions=jnp.concatenate(
            (
                recorded[3],
                final_state.policy_state.previous_action[None, ...],
            ),
            axis=0,
        ),
        previous_team_rewards=jnp.concatenate(
            (
                recorded[4],
                final_state.policy_state.previous_team_reward[None, ...],
            ),
            axis=0,
        ),
        initial_value_carry=initial_value_carry,
        reference_logits=jnp.concatenate(
            (recorded[5], final_reference_logits[None, ...]), axis=0
        ),
        execution_logits=recorded[6],
        generic_execution_logits=recorded[7],
        slot_log_beliefs=jnp.concatenate(
            (
                recorded[8],
                final_state.policy_state.slot_log_belief[None, ...],
            ),
            axis=0,
        ),
        actions=recorded[9],
        rewards=recorded[10],
        dones=recorded[11],
        response_codes=recorded[12],
        episode_ids=recorded[13],
        episode_steps=recorded[14],
        completed_episode_returns=recorded[15],
        quotient_counts=recorded[16],
        j_use=recorded[17],
        j_mask=recorded[18],
    )
    return final_state, batch


def make_compiled_collector(**static_arguments: Any) -> Callable[..., Any]:
    import functools

    import jax

    return jax.jit(functools.partial(collect_rollout, **static_arguments))


__all__ = [
    "VQBCRolloutCallbacks",
    "collect_rollout",
    "initialize_policy_state",
    "initialize_rollout",
    "make_compiled_collector",
    "policy_action",
]
