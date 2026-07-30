"""Device-side rollout collection for the unified DELTA-ZSC controller."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

from .belief_set_encoder import stratified_mixture_samples
from .counterfactual_anchor import tree_select
from .model import initial_policy_state
from .regret_potential import decision_regret_from_action_values, potential_shaping
from .training import categorical_log_probability
from .types import PolicyState, RolloutBatch


class PartnerFunctions(NamedTuple):
    initial_state: Callable[[int], Any]
    step: Callable[..., tuple[Any, Any, Any, Any]]
    observe: Callable[..., Any]
    pre_teacher_latent: Callable[..., Any]
    teacher_latent: Callable[..., Any]
    run_id: Callable[..., Any]
    diagnostics: Callable[..., Mapping[str, Any]]


class RunnerState(NamedTuple):
    environment_state: Any
    observations: Any
    ego_policy: PolicyState
    partner_state: Any
    ego_roles: Any
    episode_return: Any
    completed_episodes: Any
    effective_environment_steps: Any
    random_key: Any


def official_ego_roles(environment_count: int) -> Any:
    """Exact Official actor-axis mask: first half agent 0, second half agent 1."""

    import jax.numpy as jnp

    count = int(environment_count)
    if count <= 0 or count % 2:
        raise ValueError("Official role balancing requires a positive even lane count.")
    return jnp.linspace(0, 2, count, endpoint=False, dtype=jnp.int32)



def initialize_runner(
    *,
    environment: Any,
    model_config: Any,
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
        observations=observations,
        ego_policy=initial_policy_state(
            batch_size=count,
            observation_shape=environment.observation_shape,
            action_count=6,
            task_hidden_dim=model_config.task_hidden_dim,
            belief_hidden_dim=model_config.belief_hidden_dim,
            latent_dim=model_config.latent_dim,
            mixture_components=model_config.mixture_components,
        ),
        partner_state=partner_functions.initial_state(count, partner_key),
        ego_roles=official_ego_roles(count),
        episode_return=jnp.zeros((count,), dtype=jnp.float32),
        completed_episodes=jnp.asarray(0, dtype=jnp.int64),
        effective_environment_steps=jnp.asarray(0, dtype=jnp.int64),
        random_key=next_key,
    )


def policy_action(
    *,
    model: Any,
    params: Mapping[str, Any],
    state: PolicyState,
    observation: Any,
    gate: Any,
    keys: Any,
) -> tuple[PolicyState, Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    next_state, output = model.apply(
        {"params": params}, state, observation, gate, method=model.step
    )
    key_array = jnp.asarray(keys)
    if key_array.ndim == 2:
        action = jax.vmap(
            lambda key, logits: jax.random.categorical(key, logits)
        )(key_array, output.execution_logits)
    else:
        action = jax.random.categorical(key_array, output.execution_logits)
    log_probability = categorical_log_probability(output.execution_logits, action)
    return next_state, action, output, log_probability


def observe_policy_after_transition(
    *,
    stepped_state: PolicyState,
    action: Any,
    reward: Any,
    done: Any,
    next_observation: Any,
    model_config: Any,
) -> PolicyState:
    import jax.numpy as jnp

    count = int(jnp.asarray(done).shape[0])
    fresh = initial_policy_state(
        batch_size=count,
        observation_shape=tuple(next_observation.shape[1:]),
        action_count=6,
        task_hidden_dim=model_config.task_hidden_dim,
        belief_hidden_dim=model_config.belief_hidden_dim,
        latent_dim=model_config.latent_dim,
        mixture_components=model_config.mixture_components,
    )
    candidate = stepped_state._replace(
        previous_action=jnp.asarray(action, dtype=jnp.int32),
        # Official AbstractPolicy does not receive reward.  The deployable
        # student therefore never consumes this privileged transition field;
        # raw reward remains available to PPO targets, the response objective,
        # and the training-only full-trajectory teacher.
        previous_reward=jnp.zeros_like(jnp.asarray(reward, dtype=jnp.float32)),
        episode_start=jnp.asarray(done, dtype=jnp.bool_),
    )
    return tree_select(jnp.asarray(done, dtype=jnp.bool_), fresh, candidate)


def collect_rollout(
    *,
    state: RunnerState,
    length: int,
    environment: Any,
    model: Any,
    params: Mapping[str, Any],
    model_config: Any,
    partner_functions: PartnerFunctions,
    partner_parameters: Any,
    gate_values: Any,
    teacher_lane_mask: Any,
    official_shaping_factor: float,
) -> tuple[RunnerState, RolloutBatch, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp

    if length <= 0:
        raise ValueError("Rollout length must be positive.")
    count = int(environment.num_envs)
    initial_policy = state.ego_policy

    def one_step(current: RunnerState, unused: Any) -> tuple[Any, Mapping[str, Any]]:
        del unused
        (
            next_root,
            ego_root,
            partner_root,
            environment_root,
        ) = jax.random.split(current.random_key, 4)
        ego_keys = jax.random.split(ego_root, count)
        partner_keys = jax.random.split(partner_root, count)
        environment_keys = jax.random.split(environment_root, count)
        lane_indexes = jnp.arange(count, dtype=jnp.int32)
        ego_observation = current.observations[lane_indexes, current.ego_roles]
        partner_observation = current.observations[
            lane_indexes, 1 - current.ego_roles
        ]
        gate = jnp.broadcast_to(
            jnp.asarray(gate_values, dtype=jnp.float32), (count,)
        )
        stepped_ego, output = model.apply(
            {"params": params},
            current.ego_policy,
            ego_observation,
            gate,
            method=model.step,
        )
        teacher_latent_pre = partner_functions.pre_teacher_latent(
            partner_parameters, current.partner_state, output.task_features
        )
        valid_teacher = jnp.all(jnp.isfinite(teacher_latent_pre), axis=-1)
        safe_teacher = jnp.where(valid_teacher[..., None], teacher_latent_pre, 0.0)
        teacher_output = model.apply(
            {"params": params},
            output.task_features,
            safe_teacher,
            1.0,
            method=model.from_features_and_latent,
        )
        teacher_mask = jnp.broadcast_to(
            jnp.asarray(teacher_lane_mask, dtype=jnp.bool_), (count,)
        ) & valid_teacher
        behavior_logits = jnp.where(
            teacher_mask[..., None], teacher_output.logits, output.execution_logits
        )
        ego_action = jax.vmap(
            lambda key, logits: jax.random.categorical(key, logits)
        )(ego_keys, behavior_logits)
        log_probability = categorical_log_probability(behavior_logits, ego_action)
        (
            partner_action,
            stepped_partner,
            partner_context,
            partner_log_probability,
        ) = partner_functions.step(
            partner_parameters,
            current.partner_state,
            partner_observation,
            current.ego_policy.episode_start,
            partner_keys,
        )
        ego_first = jnp.stack((ego_action, partner_action), axis=-1)
        partner_first = jnp.stack((partner_action, ego_action), axis=-1)
        joint = jnp.where(
            (current.ego_roles == 0)[:, None], ego_first, partner_first
        )
        (
            next_environment,
            next_observations,
            rewards,
            dones,
            info,
        ) = environment.step_with_keys(
            current.environment_state, joint, environment_keys
        )
        terminal = info["terminal_observations"]
        observation_mask = dones.reshape(
            dones.shape + (1,) * (next_observations[:, 0].ndim - 1)
        )
        terminal_ego = terminal[lane_indexes, current.ego_roles]
        terminal_partner = terminal[lane_indexes, 1 - current.ego_roles]
        next_ego_observation = next_observations[
            lane_indexes, current.ego_roles
        ]
        next_partner_observation = next_observations[
            lane_indexes, 1 - current.ego_roles
        ]
        ego_response_next = jnp.where(
            observation_mask, terminal_ego, next_ego_observation
        )
        partner_response_next = jnp.where(
            observation_mask, terminal_partner, next_partner_observation
        )
        raw_rewards_by_agent = info["raw_rewards_by_agent"]
        raw_rewards = raw_rewards_by_agent[lane_indexes, current.ego_roles]
        official_shaped_by_agent = info["official_shaped_rewards_by_agent"]
        official_shaped_rewards = official_shaped_by_agent[
            lane_indexes, current.ego_roles
        ]
        next_ego = observe_policy_after_transition(
            stepped_state=stepped_ego,
            action=ego_action,
            reward=raw_rewards,
            done=dones,
            next_observation=next_ego_observation,
            model_config=model_config,
        )
        partner_diagnostics = partner_functions.diagnostics(
            partner_parameters, current.partner_state, partner_context
        )
        next_partner = partner_functions.observe(
            partner_parameters,
            stepped_partner,
            partner_context,
            partner_observation,
            partner_action,
            raw_rewards,
            dones,
            partner_response_next,
        )
        completed_return = current.episode_return + raw_rewards
        next_state = RunnerState(
            environment_state=next_environment,
            observations=next_observations,
            ego_policy=next_ego,
            partner_state=next_partner,
            ego_roles=current.ego_roles,
            episode_return=jnp.where(dones, 0.0, completed_return),
            completed_episodes=current.completed_episodes
            + jnp.sum(dones.astype(jnp.int64)),
            effective_environment_steps=current.effective_environment_steps + count,
            random_key=next_root,
        )
        return next_state, {
            "observations": ego_observation,
            "response_next_observations": ego_response_next,
            "previous_actions": current.ego_policy.previous_action,
            "previous_rewards": current.ego_policy.previous_reward,
            "episode_starts": current.ego_policy.episode_start,
            "gate": gate,
            "action_key": ego_keys,
            "actions": ego_action,
            "rewards": raw_rewards,
            "official_shaped_rewards": official_shaped_rewards,
            "official_shaping_factor": jnp.full(
                raw_rewards.shape,
                float(official_shaping_factor),
                dtype=jnp.float32,
            ),
            "dones": dones,
            "old_log_probabilities": log_probability,
            "old_values": output.state_value,
            "ppo_mask": (~teacher_mask).astype(jnp.float32),
            "partner_run_ids": partner_functions.run_id(
                partner_parameters, current.partner_state, partner_context
            ),
            "partner_actions": partner_action,
            "partner_log_probabilities": partner_log_probability,
            "partner_source": partner_diagnostics["source"],
            "partner_code": partner_diagnostics["code"],
            "partner_generator_logits": partner_diagnostics["generator_logits"],
            "partner_generator_value": partner_diagnostics["generator_value"],
            "completed_returns": jnp.where(dones, completed_return, 0.0),
            "completed_mask": dones,
            "correct_deliveries": info["correct_delivery"],
            "wrong_deliveries": info["wrong_delivery"],
            "support_score": output.support_score,
            "base_logits": output.base_logits,
            "conditional_logits": output.base_logits + output.residual_logits,
            "action_values": output.action_values,
            "mixture_logits": output.mixture_logits,
            "mixture_means": output.mixture_means,
            "mixture_log_variances": output.mixture_log_variances,
            "task_features": output.task_features,
            "environment_state": current.environment_state,
            "ego_policy_state": current.ego_policy,
            "partner_state": current.partner_state,
            "joint_observations": current.observations,
            "partner_observations": partner_observation,
            "ego_roles": current.ego_roles,
        }

    final_state, recorded = jax.lax.scan(one_step, state, xs=None, length=int(length))
    lane_indexes = jnp.arange(count, dtype=jnp.int32)
    final_observation = final_state.observations[
        lane_indexes, final_state.ego_roles
    ]
    final_gate = jnp.broadcast_to(jnp.asarray(gate_values, dtype=jnp.float32), (count,))
    unused_state, final_output = model.apply(
        {"params": params},
        final_state.ego_policy,
        final_observation,
        final_gate,
        method=model.step,
    )
    del unused_state
    batch = RolloutBatch(
        observations=jnp.concatenate(
            (recorded["observations"], final_observation[None, ...]), axis=0
        ),
        response_next_observations=recorded["response_next_observations"],
        previous_actions=jnp.concatenate(
            (
                recorded["previous_actions"],
                final_state.ego_policy.previous_action[None, ...],
            ),
            axis=0,
        ),
        previous_rewards=jnp.concatenate(
            (
                recorded["previous_rewards"],
                final_state.ego_policy.previous_reward[None, ...],
            ),
            axis=0,
        ),
        episode_starts=jnp.concatenate(
            (
                recorded["episode_starts"],
                final_state.ego_policy.episode_start[None, ...],
            ),
            axis=0,
        ),
        action_keys=jnp.concatenate(
            (recorded["action_key"], recorded["action_key"][-1:]), axis=0
        ),
        gate_overrides=jnp.concatenate(
            (recorded["gate"], final_gate[None, ...]), axis=0
        ),
        actions=recorded["actions"],
        rewards=recorded["rewards"],
        official_shaped_rewards=recorded["official_shaped_rewards"],
        official_shaping_factors=recorded["official_shaping_factor"],
        decision_regret_shaping=jnp.zeros_like(recorded["rewards"]),
        shaped_rewards=(
            recorded["rewards"]
            + recorded["official_shaping_factor"]
            * recorded["official_shaped_rewards"]
        ),
        dones=recorded["dones"],
        old_log_probabilities=recorded["old_log_probabilities"],
        old_values=jnp.concatenate(
            (recorded["old_values"], final_output.state_value[None, ...]), axis=0
        ),
        ppo_mask=recorded["ppo_mask"],
        partner_codes=recorded["partner_code"],
        partner_sources=recorded["partner_source"],
        partner_run_ids=recorded["partner_run_ids"],
        initial_policy_state=initial_policy,
    )
    return final_state, batch, recorded


def attach_decision_regret_shaping(
    *,
    batch: RolloutBatch,
    model: Any,
    target_params: Mapping[str, Any],
    config: Any,
    key: Any,
) -> tuple[RolloutBatch, Mapping[str, Any]]:
    """Compute a frozen-potential shaped reward on the collected batch."""

    import jax
    import jax.numpy as jnp

    unused_final, output = model.apply(
        {"params": target_params},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.previous_rewards,
        batch.episode_starts,
        jnp.ones_like(batch.gate_overrides),
        method=model.sequence,
    )
    del unused_final
    time_count, batch_count = output.task_features.shape[:2]
    keys = jax.random.split(key, time_count * batch_count).reshape(
        (time_count, batch_count, 2)
    )

    def one_time(
        task_features: Any,
        mixture_logits: Any,
        means: Any,
        log_variances: Any,
        sample_key: Any,
    ) -> Any:
        def one_lane(feature: Any, logits: Any, mu: Any, log_var: Any, lane_key: Any) -> Any:
            samples, weights = stratified_mixture_samples(
                lane_key,
                mixture_logits=logits,
                means=mu,
                log_variances=log_var,
                sample_count=config.model.posterior_particles,
            )
            features = jnp.broadcast_to(
                feature[None, :],
                (config.model.posterior_particles, feature.shape[-1]),
            )
            teacher = model.apply(
                {"params": target_params},
                features,
                samples,
                1.0,
                method=model.from_features_and_latent,
            )
            return decision_regret_from_action_values(
                teacher.action_values, weights
            )

        return jax.vmap(one_lane)(
            task_features, mixture_logits, means, log_variances, sample_key
        )

    regrets = jax.vmap(one_time)(
        output.task_features,
        output.mixture_logits,
        output.mixture_means,
        output.mixture_log_variances,
        keys,
    )
    shaping = potential_shaping(
        regrets[:-1],
        regrets[1:],
        batch.dones,
        gamma=config.ppo.gamma,
        weight=config.loss.decision_regret_weight,
    )
    shaped = batch.shaped_rewards + shaping
    return batch._replace(
        decision_regret_shaping=shaping,
        shaped_rewards=shaped,
    ), {
        "decision_regret_values": regrets[:-1],
        "mean_decision_regret": jnp.mean(regrets[:-1]),
        "mean_decision_regret_shaping": jnp.mean(shaping),
        "mean_official_shaped_reward": jnp.mean(batch.official_shaped_rewards),
        "official_shaping_factor": jnp.mean(batch.official_shaping_factors),
        "mean_combined_training_reward": jnp.mean(shaped),
        "maximum_decision_regret": jnp.max(regrets[:-1]),
    }


__all__ = [
    "PartnerFunctions",
    "RunnerState",
    "attach_decision_regret_shaping",
    "collect_rollout",
    "initialize_runner",
    "official_ego_roles",
    "observe_policy_after_transition",
    "policy_action",
]
