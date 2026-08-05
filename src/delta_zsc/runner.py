"""Vectorized legal-history rollout for unified DELTA-ZSC."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

from .model import observe_after_transition
from .types import RolloutBatch, RunnerState


class PartnerFunctions(NamedTuple):
    initial_state: Callable[..., Any]
    step: Callable[..., tuple[Any, Any, Any, Any]]
    observe: Callable[..., Any]
    run_id: Callable[..., Any]
    diagnostics: Callable[..., Mapping[str, Any]]


def official_ego_roles(environment_count: int) -> Any:
    import jax.numpy as jnp

    count = int(environment_count)
    if count <= 0 or count % 2:
        raise ValueError("Official role balancing requires a positive even lane count.")
    return (jnp.arange(count, dtype=jnp.int32) % 2).astype(jnp.int32)


def initialize_runner(
    *,
    environment: Any,
    model: Any,
    partner_functions: PartnerFunctions,
    random_key: Any,
) -> RunnerState:
    import jax
    import jax.numpy as jnp

    next_key, environment_key, partner_key = jax.random.split(random_key, 3)
    environment_state, observations = environment.reset(environment_key)
    count = int(environment.num_envs)
    return RunnerState(
        environment_state=environment_state,
        joint_observations=observations,
        ego_policy_state=model.initial_state(count),
        partner_state=partner_functions.initial_state(count, partner_key),
        partner_episode_start=jnp.ones((count,), dtype=jnp.bool_),
        ego_roles=official_ego_roles(count),
        random_key=next_key,
        completed_episodes=jnp.asarray(0, dtype=jnp.int32),
    )


def _categorical_log_probability(logits: Any, actions: Any) -> Any:
    import jax.nn as jnn
    import jax.numpy as jnp

    logp = jnn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(
        logp, jnp.asarray(actions, dtype=jnp.int32)[..., None], axis=-1
    )[..., 0]


def collect_rollout(
    *,
    state: RunnerState,
    length: int,
    environment: Any,
    model: Any,
    base_params: Any,
    latent_params: Any,
    partner_functions: PartnerFunctions,
    partner_parameters: Any,
    official_shaping_factor: float,
    record_anchors: bool,
    use_deployment_policy: bool = False,
) -> tuple[RunnerState, RolloutBatch, Mapping[str, Any]]:
    """Collect one time-major rollout.

    Training calls this with ``use_deployment_policy=False``.  Consequently the
    PPO sample and the continuation policy behind every CRN anchor are exactly
    the base policy.  Evaluation may explicitly request the analytic DELTA
    deployment policy without changing the estimator used during training.
    """

    import jax
    import jax.numpy as jnp

    steps = int(length)
    if steps <= 0:
        raise ValueError("Rollout length must be positive.")
    count = int(environment.num_envs)
    initial_policy_state = state.ego_policy_state
    lanes = jnp.arange(count, dtype=jnp.int32)

    def one(current: RunnerState, unused_time: Any):
        del unused_time
        next_root, ego_root, partner_root, environment_root = jax.random.split(
            current.random_key, 4
        )
        ego_keys = jax.random.split(ego_root, count)
        partner_keys = jax.random.split(partner_root, count)
        environment_keys = jax.random.split(environment_root, count)
        ego_observation = current.joint_observations[lanes, current.ego_roles]
        partner_observation = current.joint_observations[
            lanes, 1 - current.ego_roles
        ]
        compute_latent = bool(use_deployment_policy) or model.config.method_variant in {
            "response_only",
            "delta_passive",
            "delta_active",
        }
        stepped_ego, output = model.step(
            base_params,
            latent_params,
            current.ego_policy_state,
            ego_observation,
            compute_latent=compute_latent,
            execute_adaptation=bool(use_deployment_policy),
        )
        behavior_logits = (
            output.policy_logits if bool(use_deployment_policy) else output.base_policy_logits
        )
        ego_action = jax.vmap(
            lambda key, logits: jax.random.categorical(key, logits)
        )(ego_keys, behavior_logits)
        old_log_probability = _categorical_log_probability(
            behavior_logits, ego_action
        )
        partner_action, stepped_partner, partner_context, unused_partner_value = (
            partner_functions.step(
                partner_parameters,
                current.partner_state,
                partner_observation,
                current.partner_episode_start,
                partner_keys,
            )
        )
        del unused_partner_value
        ego_first = jnp.stack((ego_action, partner_action), axis=-1)
        partner_first = jnp.stack((partner_action, ego_action), axis=-1)
        joint_action = jnp.where(
            (current.ego_roles == 0)[:, None], ego_first, partner_first
        )
        (
            next_environment_state,
            next_joint_observations,
            reward,
            done,
            info,
        ) = environment.step_with_keys(
            current.environment_state, joint_action, environment_keys
        )
        terminal = info.get("terminal_observations", next_joint_observations)
        mask = jnp.asarray(done, dtype=jnp.bool_).reshape(
            jnp.asarray(done).shape
            + (1,) * (next_joint_observations[:, 0].ndim - 1)
        )
        terminal_ego = terminal[lanes, current.ego_roles]
        terminal_partner = terminal[lanes, 1 - current.ego_roles]
        next_ego_observation = next_joint_observations[lanes, current.ego_roles]
        next_partner_observation = next_joint_observations[
            lanes, 1 - current.ego_roles
        ]
        ego_response_next = jnp.where(mask, terminal_ego, next_ego_observation)
        partner_response_next = jnp.where(
            mask, terminal_partner, next_partner_observation
        )
        raw_by_agent = info.get("raw_rewards_by_agent")
        raw_reward = (
            jnp.asarray(reward, dtype=jnp.float32)
            if raw_by_agent is None
            else jnp.asarray(raw_by_agent, dtype=jnp.float32)[
                lanes, current.ego_roles
            ]
        )
        shaped_by_agent = info.get("official_shaped_rewards_by_agent")
        shaped_reward = (
            jnp.zeros_like(raw_reward)
            if shaped_by_agent is None
            else jnp.asarray(shaped_by_agent, dtype=jnp.float32)[
                lanes, current.ego_roles
            ]
            * float(official_shaping_factor)
        )
        next_ego_state = observe_after_transition(
            stepped_ego, action=ego_action, done=done
        )
        next_partner_state = partner_functions.observe(
            partner_parameters,
            stepped_partner,
            partner_context,
            partner_observation,
            partner_action,
            raw_reward,
            done,
            partner_response_next,
        )
        next_state = RunnerState(
            environment_state=next_environment_state,
            joint_observations=next_joint_observations,
            ego_policy_state=next_ego_state,
            partner_state=next_partner_state,
            partner_episode_start=jnp.asarray(done, dtype=jnp.bool_),
            ego_roles=current.ego_roles,
            random_key=next_root,
            completed_episodes=current.completed_episodes
            + jnp.sum(jnp.asarray(done, dtype=jnp.int32)),
        )
        row = {
            "environment_state": current.environment_state,
            "joint_observations": current.joint_observations,
            "ego_policy_state": current.ego_policy_state,
            "partner_state": current.partner_state,
            "episode_starts": current.partner_episode_start,
            "ego_roles": current.ego_roles,
            "ego_observation": ego_observation,
            "response_next_observation": ego_response_next,
            "previous_action": current.ego_policy_state.previous_action,
            "action": ego_action,
            "raw_reward": raw_reward,
            "shaped_reward": shaped_reward,
            "done": jnp.asarray(done, dtype=jnp.bool_),
            "old_log_probability": old_log_probability,
            "old_value": output.value,
            "base_policy_logits": output.base_policy_logits,
            "deployment_policy_logits": output.policy_logits,
            "belief": output.belief,
            "active_voi": output.active_voi,
            "active_voi_raw": output.active_voi_raw,
            "active_information_gain": output.active_information_gain,
            "active_voi_quadrature_error": output.active_voi_quadrature_error,
            "adaptation_kl": output.adaptation_kl,
        }
        return next_state, row

    final_state, rows = jax.lax.scan(
        one, state, jnp.arange(steps, dtype=jnp.int32)
    )
    final_lanes = jnp.arange(count, dtype=jnp.int32)
    final_observation = final_state.joint_observations[
        final_lanes, final_state.ego_roles
    ]
    compute_latent = bool(use_deployment_policy) or model.config.method_variant in {
        "response_only",
        "delta_passive",
        "delta_active",
    }
    _, final_output = model.step(
        base_params,
        latent_params,
        final_state.ego_policy_state,
        final_observation,
        compute_latent=compute_latent,
        execute_adaptation=bool(use_deployment_policy),
    )
    observations = jnp.concatenate(
        (rows["ego_observation"], final_observation[None]), axis=0
    )
    previous_actions = jnp.concatenate(
        (
            rows["previous_action"],
            final_state.ego_policy_state.previous_action[None],
        ),
        axis=0,
    )
    episode_starts = jnp.concatenate(
        (
            rows["episode_starts"],
            final_state.ego_policy_state.episode_start[None],
        ),
        axis=0,
    )
    values = jnp.concatenate((rows["old_value"], final_output.value[None]), axis=0)
    batch = RolloutBatch(
        observations=observations,
        response_next_observations=rows["response_next_observation"],
        previous_actions=previous_actions,
        episode_starts=episode_starts,
        actions=rows["action"],
        rewards=rows["raw_reward"],
        shaped_rewards=rows["shaped_reward"],
        dones=rows["done"],
        old_log_probabilities=rows["old_log_probability"],
        old_values=values[:-1],
        ppo_mask=jnp.ones((steps, count), dtype=jnp.float32),
        initial_policy_state=initial_policy_state,
    )
    records = (
        rows
        if bool(record_anchors)
        else {
            "belief": rows["belief"],
            "base_policy_logits": rows["base_policy_logits"],
            "deployment_policy_logits": rows["deployment_policy_logits"],
            "active_voi": rows["active_voi"],
            "active_voi_raw": rows["active_voi_raw"],
            "active_information_gain": rows["active_information_gain"],
            "active_voi_quadrature_error": rows["active_voi_quadrature_error"],
            "adaptation_kl": rows["adaptation_kl"],
        }
    )
    return final_state, batch, records


__all__ = [
    "PartnerFunctions",
    "collect_rollout",
    "initialize_runner",
    "official_ego_roles",
]
