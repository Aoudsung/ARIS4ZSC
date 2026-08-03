"""CUDA-friendly rollout and detached V6 decision-regret shaping."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

from .belief_set_encoder import gaussian_samples
from .counterfactual_anchor import tree_select
from .model import initial_policy_state
from .regret_potential import decision_regret_from_action_values
from .training import categorical_log_probability
from .types import PolicyState, RolloutBatch


DECISION_REGRET_STATE_CHUNK_SIZE = 4_096
DECISION_REGRET_NUMERICAL_TILE_SIZE = 32


class PartnerFunctions(NamedTuple):
    initial_state: Callable[..., Any]
    step: Callable[..., tuple[Any, Any, Any, Any]]
    observe: Callable[..., Any]
    run_id: Callable[..., Any]
    diagnostics: Callable[..., Mapping[str, Any]]


class RunnerState(NamedTuple):
    environment_state: Any
    observations: Any
    ego_policy: PolicyState
    target_ego_policy: PolicyState
    partner_state: Any
    ego_roles: Any
    episode_return: Any
    completed_episodes: Any
    effective_environment_steps: Any
    random_key: Any


def official_ego_roles(environment_count: int) -> Any:
    import jax.numpy as jnp

    count = int(environment_count)
    if count <= 0 or count % 2:
        raise ValueError("Official role balancing requires a positive even lane count.")
    return jnp.linspace(0, 2, count, endpoint=False, dtype=jnp.int32)


def initialize_runner(
    *, environment: Any, model_config: Any, partner_functions: PartnerFunctions, random_key: Any
) -> RunnerState:
    import jax
    import jax.numpy as jnp

    next_key, environment_key, partner_key = jax.random.split(random_key, 3)
    environment_state, observations = environment.reset(environment_key)
    count = int(environment.num_envs)
    policy = initial_policy_state(
        batch_size=count,
        observation_shape=environment.observation_shape,
        action_count=6,
        task_hidden_dim=model_config.task_hidden_dim,
        belief_hidden_dim=model_config.belief_hidden_dim,
        latent_dim=model_config.latent_dim,
    )
    return RunnerState(
        environment_state,
        observations,
        policy,
        policy,
        partner_functions.initial_state(count, partner_key),
        official_ego_roles(count),
        jnp.zeros((count,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int64),
        jnp.asarray(0, dtype=jnp.int64),
        next_key,
    )


def policy_action(
    *,
    model: Any,
    params: Mapping[str, Any],
    target_params: Mapping[str, Any] | None,
    state: PolicyState,
    observation: Any,
    context_dropout_mask: Any,
    keys: Any,
) -> tuple[PolicyState, Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    del target_params
    next_state, output = model.apply(
        {"params": params},
        state,
        observation,
        context_dropout_mask,
        method=model.step,
    )
    key_array = jnp.asarray(keys)
    action = (
        jax.vmap(lambda key, logits: jax.random.categorical(key, logits))(
            key_array, output.policy_logits
        )
        if key_array.ndim == 2
        else jax.random.categorical(key_array, output.policy_logits)
    )
    log_probability = categorical_log_probability(output.policy_logits, action)
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

    del reward
    count = int(jnp.asarray(done).shape[0])
    fresh = initial_policy_state(
        batch_size=count,
        observation_shape=tuple(next_observation.shape[1:]),
        action_count=6,
        task_hidden_dim=model_config.task_hidden_dim,
        belief_hidden_dim=model_config.belief_hidden_dim,
        latent_dim=model_config.latent_dim,
    )
    candidate = stepped_state._replace(
        previous_action=jnp.asarray(action, dtype=jnp.int32),
        episode_start=jnp.asarray(done, dtype=jnp.bool_),
    )
    return tree_select(jnp.asarray(done, dtype=jnp.bool_), fresh, candidate)


def context_dropout_probability(
    effective_steps: Any, *, total_steps: int, initial: float, final: float
) -> Any:
    import jax.numpy as jnp

    progress = jnp.clip(
        jnp.asarray(effective_steps, dtype=jnp.float32) / float(max(total_steps, 1)),
        0.0,
        1.0,
    )
    return float(initial) + progress * (float(final) - float(initial))


def collect_rollout(
    *,
    state: RunnerState,
    length: int,
    environment: Any,
    model: Any,
    params: Mapping[str, Any],
    target_params: Mapping[str, Any],
    model_config: Any,
    partner_functions: PartnerFunctions,
    partner_parameters: Any,
    context_dropout_probability_value: Any,
    context_dropout_root: Any,
    official_shaping_factor: float,
    record_mode: str = "anchor_full",
) -> tuple[RunnerState, RolloutBatch | None, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp

    if length <= 0:
        raise ValueError("Rollout length must be positive.")
    if record_mode not in {"minimal", "anchor_full", "support"}:
        raise ValueError(f"Unknown rollout record mode: {record_mode!r}.")
    count = int(environment.num_envs)
    initial_policy = state.ego_policy
    initial_target_policy = state.target_ego_policy

    def one_step(current: RunnerState, time_index: Any):
        next_root, ego_root, partner_root, environment_root = jax.random.split(
            current.random_key, 4
        )
        ego_keys = jax.random.split(ego_root, count)
        partner_keys = jax.random.split(partner_root, count)
        environment_keys = jax.random.split(environment_root, count)
        dropout_step_key = jax.random.fold_in(
            jnp.asarray(context_dropout_root, dtype=jnp.uint32), time_index
        )
        dropout_keys = jax.random.split(dropout_step_key, count)
        dropout_mask = jax.vmap(
            lambda key: jax.random.bernoulli(
                key, jnp.asarray(context_dropout_probability_value, dtype=jnp.float32)
            )
        )(dropout_keys)
        lane_indexes = jnp.arange(count, dtype=jnp.int32)
        ego_observation = current.observations[lane_indexes, current.ego_roles]
        partner_observation = current.observations[lane_indexes, 1 - current.ego_roles]
        stepped_ego, output = model.apply(
            {"params": params},
            current.ego_policy,
            ego_observation,
            dropout_mask,
            method=model.step,
        )
        stepped_target, _ = model.apply(
            {"params": target_params},
            current.target_ego_policy,
            ego_observation,
            jnp.zeros((count,), dtype=jnp.bool_),
            method=model.step,
        )
        ego_action = jax.vmap(
            lambda key, logits: jax.random.categorical(key, logits)
        )(ego_keys, output.policy_logits)
        log_probability = categorical_log_probability(output.policy_logits, ego_action)
        behavior_probability = jnp.exp(log_probability)
        partner_action, stepped_partner, partner_context, partner_log_probability = (
            partner_functions.step(
                partner_parameters,
                current.partner_state,
                partner_observation,
                current.ego_policy.episode_start,
                partner_keys,
            )
        )
        ego_first = jnp.stack((ego_action, partner_action), axis=-1)
        partner_first = jnp.stack((partner_action, ego_action), axis=-1)
        joint = jnp.where((current.ego_roles == 0)[:, None], ego_first, partner_first)
        next_environment, next_observations, rewards, dones, info = (
            environment.step_with_keys(current.environment_state, joint, environment_keys)
        )
        terminal = info["terminal_observations"]
        observation_mask = dones.reshape(
            dones.shape + (1,) * (next_observations[:, 0].ndim - 1)
        )
        terminal_ego = terminal[lane_indexes, current.ego_roles]
        terminal_partner = terminal[lane_indexes, 1 - current.ego_roles]
        next_ego_observation = next_observations[lane_indexes, current.ego_roles]
        next_partner_observation = next_observations[lane_indexes, 1 - current.ego_roles]
        ego_response_next = jnp.where(observation_mask, terminal_ego, next_ego_observation)
        partner_response_next = jnp.where(
            observation_mask, terminal_partner, next_partner_observation
        )
        raw_rewards_by_agent = info["raw_rewards_by_agent"]
        raw_rewards = raw_rewards_by_agent[lane_indexes, current.ego_roles]
        official_shaped = info["official_shaped_rewards_by_agent"][
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
        next_target = observe_policy_after_transition(
            stepped_state=stepped_target,
            action=ego_action,
            reward=raw_rewards,
            done=dones,
            next_observation=next_ego_observation,
            model_config=model_config,
        )
        diagnostics = partner_functions.diagnostics(
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
            next_environment,
            next_observations,
            next_ego,
            next_target,
            next_partner,
            current.ego_roles,
            jnp.where(dones, 0.0, completed_return),
            current.completed_episodes + jnp.sum(dones.astype(jnp.int64)),
            current.effective_environment_steps + count,
            next_root,
        )
        common = {
            "observations": ego_observation,
            "response_next_observations": ego_response_next,
            "previous_actions": current.ego_policy.previous_action,
            "episode_starts": current.ego_policy.episode_start,
            "context_dropout_mask": dropout_mask,
            "action_key": ego_keys,
            "actions": ego_action,
            "rewards": raw_rewards,
            "official_shaped_rewards": official_shaped,
            "official_shaping_factor": jnp.full_like(
                raw_rewards, jnp.asarray(official_shaping_factor, dtype=jnp.float32)
            ),
            "dones": dones,
            "old_log_probabilities": log_probability,
            "behavior_probabilities": behavior_probability,
            "old_values": output.state_value,
            "ppo_mask": jnp.ones_like(raw_rewards, dtype=jnp.float32),
            "partner_run_ids": partner_functions.run_id(
                partner_parameters, current.partner_state, partner_context
            ),
            "partner_code": diagnostics["code"],
            "partner_source": diagnostics["source"],
            "completed_returns": jnp.where(dones, completed_return, 0.0),
            "completed_mask": dones,
        }
        if record_mode == "minimal":
            return next_state, common
        if record_mode == "support":
            return next_state, {
                "partner_run_ids": common["partner_run_ids"],
                "partner_source": common["partner_source"],
                "belief_mean": output.belief_mean,
                "belief_log_standard_deviation": output.belief_log_standard_deviation,
                "normalized_uncertainty": output.normalized_uncertainty,
            }
        return next_state, {
            **common,
            "partner_actions": partner_action,
            "partner_log_probabilities": partner_log_probability,
            "partner_generator_logits": diagnostics["generator_logits"],
            "partner_generator_value": diagnostics["generator_value"],
            "correct_deliveries": info["correct_delivery"],
            "wrong_deliveries": info["wrong_delivery"],
            "policy_logits": output.policy_logits,
            "action_values": output.action_values,
            "raw_q1": output.raw_q1,
            "raw_q2": output.raw_q2,
            "belief_mean": output.belief_mean,
            "belief_log_standard_deviation": output.belief_log_standard_deviation,
            "normalized_uncertainty": output.normalized_uncertainty,
            "task_features": output.task_features,
            "environment_state": current.environment_state,
            "ego_policy_state": current.ego_policy,
            "target_ego_policy_state": current.target_ego_policy,
            "partner_state": current.partner_state,
            "joint_observations": current.observations,
            "partner_observations": partner_observation,
            "ego_roles": current.ego_roles,
        }

    final_state, recorded = jax.lax.scan(
        one_step, state, jnp.arange(int(length), dtype=jnp.uint32)
    )
    if record_mode == "support":
        return final_state, None, recorded
    lane_indexes = jnp.arange(count, dtype=jnp.int32)
    final_observation = final_state.observations[lane_indexes, final_state.ego_roles]
    final_drop = jnp.zeros((count,), dtype=jnp.bool_)
    _, final_output = model.apply(
        {"params": params},
        final_state.ego_policy,
        final_observation,
        final_drop,
        method=model.step,
    )
    batch = RolloutBatch(
        observations=jnp.concatenate((recorded["observations"], final_observation[None]), axis=0),
        response_next_observations=recorded["response_next_observations"],
        previous_actions=jnp.concatenate(
            (recorded["previous_actions"], final_state.ego_policy.previous_action[None]), axis=0
        ),
        episode_starts=jnp.concatenate(
            (recorded["episode_starts"], final_state.ego_policy.episode_start[None]), axis=0
        ),
        action_keys=jnp.concatenate((recorded["action_key"], recorded["action_key"][-1:]), axis=0),
        context_dropout_masks=jnp.concatenate(
            (recorded["context_dropout_mask"], final_drop[None]), axis=0
        ),
        actions=recorded["actions"],
        rewards=recorded["rewards"],
        official_shaped_rewards=recorded["official_shaped_rewards"],
        official_shaping_factors=recorded["official_shaping_factor"],
        decision_regret_shaping=jnp.zeros_like(recorded["rewards"]),
        shaped_rewards=recorded["rewards"] + recorded["official_shaping_factor"] * recorded["official_shaped_rewards"],
        dones=recorded["dones"],
        old_log_probabilities=recorded["old_log_probabilities"],
        old_values=jnp.concatenate((recorded["old_values"], final_output.state_value[None]), axis=0),
        behavior_probabilities=recorded["behavior_probabilities"],
        ppo_mask=recorded["ppo_mask"],
        partner_codes=recorded["partner_code"],
        partner_sources=recorded["partner_source"],
        partner_run_ids=recorded["partner_run_ids"],
        initial_policy_state=initial_policy,
        initial_target_policy_state=initial_target_policy,
    )
    if record_mode == "anchor_full":
        retained = recorded
    else:
        retained = {
            "completed_returns": recorded["completed_returns"],
            "completed_mask": recorded["completed_mask"],
            "partner_source": recorded["partner_source"],
        }
    return final_state, batch, retained


def collect_rollout_minimal(**kwargs: Any):
    return collect_rollout(**kwargs, record_mode="minimal")


def collect_support_rollout(**kwargs: Any):
    return collect_rollout(**kwargs, record_mode="support")


def decision_regret_chunk(
    *,
    model: Any,
    target_params: Mapping[str, Any],
    task_features: Any,
    belief_mean: Any,
    belief_log_standard_deviation: Any,
    sample_keys: Any,
    posterior_particles: int,
) -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    features = jax.lax.stop_gradient(jnp.asarray(task_features))
    means = jax.lax.stop_gradient(jnp.asarray(belief_mean))
    log_std = jax.lax.stop_gradient(jnp.asarray(belief_log_standard_deviation))
    keys = jnp.asarray(sample_keys)
    samples, weights = jax.vmap(
        lambda key, mean, std: gaussian_samples(
            key,
            mean=mean,
            log_standard_deviation=std,
            sample_count=int(posterior_particles),
        )
    )(keys, means, log_std)
    particle_features = jnp.broadcast_to(
        features[:, None, :],
        (features.shape[0], int(posterior_particles), features.shape[-1]),
    )
    action_values = model.apply(
        {"params": target_params},
        particle_features,
        samples,
        method=model.action_values_from_features_and_latent,
    )
    values = jax.lax.stop_gradient(action_values)
    regret = decision_regret_from_action_values(values, jax.lax.stop_gradient(weights))
    action_range = jnp.mean(jnp.max(values, axis=-1) - jnp.min(values, axis=-1), axis=-1)
    return regret, action_range


def chunked_decision_regret(
    *,
    model: Any,
    target_params: Mapping[str, Any],
    context: Any,
    key: Any,
    posterior_particles: int,
    state_chunk_size: int = DECISION_REGRET_STATE_CHUNK_SIZE,
) -> tuple[Any, Any]:
    import math
    import jax
    import jax.numpy as jnp

    features = jnp.asarray(context.task_features)
    means = jnp.asarray(context.belief_mean)
    log_std = jnp.asarray(context.belief_log_standard_deviation)
    prefix = features.shape[:-1]
    state_count = int(math.prod(prefix))
    chunk_size = int(state_chunk_size)
    padded_count = ((state_count + chunk_size - 1) // chunk_size) * chunk_size
    padding = padded_count - state_count
    keys = jnp.pad(jax.random.split(key, state_count), ((0, padding), (0, 0)))

    def chunks(value: Any):
        flat = value.reshape((state_count,) + value.shape[len(prefix):])
        widths = ((0, padding),) + ((0, 0),) * (flat.ndim - 1)
        return jnp.pad(flat, widths).reshape(
            (padded_count // chunk_size, chunk_size) + flat.shape[1:]
        )

    values = (chunks(features), chunks(means), chunks(log_std), keys.reshape((-1, chunk_size, 2)))

    def one(items: tuple[Any, Any, Any, Any]):
        f, m, s, k = items
        return decision_regret_chunk(
            model=model,
            target_params=target_params,
            task_features=f,
            belief_mean=m,
            belief_log_standard_deviation=s,
            sample_keys=k,
            posterior_particles=posterior_particles,
        )

    regrets, ranges = jax.lax.map(one, values)
    return (
        regrets.reshape((padded_count,))[:state_count].reshape(prefix),
        ranges.reshape((padded_count,))[:state_count].reshape(prefix),
    )


def target_context_sequence(*, model: Any, target_params: Mapping[str, Any], batch: RolloutBatch) -> Any:
    _, context = model.apply(
        {"params": target_params},
        batch.initial_target_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        method=model.context_sequence,
    )
    return context


def decision_regret_weight(*, effective_steps: Any, total_steps: int, maximum: float, midpoint: float, temperature: float) -> Any:
    import jax
    import jax.numpy as jnp

    progress = jnp.asarray(effective_steps, dtype=jnp.float32) / float(max(total_steps, 1))
    return float(maximum) * jax.nn.sigmoid((progress - float(midpoint)) / float(temperature))


def finalize_decision_regret_shaping(
    *,
    batch: RolloutBatch,
    regrets: Any,
    action_ranges: Any,
    action_range_ema: Any,
    gamma: float,
    weight: Any,
    ema_decay: float = 0.99,
) -> tuple[RolloutBatch, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp

    values = jax.lax.stop_gradient(jnp.asarray(regrets, dtype=jnp.float32))
    ranges = jax.lax.stop_gradient(jnp.asarray(action_ranges, dtype=jnp.float32))
    next_scale = float(ema_decay) * jnp.asarray(action_range_ema) + (1.0 - float(ema_decay)) * jnp.mean(ranges)
    normalized = values / (jnp.maximum(next_scale, 0.0) + 1.0e-3)
    following = jnp.where(batch.dones, 0.0, normalized[1:])
    unweighted = jnp.clip(normalized[:-1] - float(gamma) * following, -1.0, 1.0)
    shaping = jnp.asarray(weight, dtype=jnp.float32) * unweighted
    shaped = batch.shaped_rewards + shaping
    return batch._replace(decision_regret_shaping=shaping, shaped_rewards=shaped), {
        "decision_regret_values": values[:-1],
        "normalized_decision_regret": normalized[:-1],
        "mean_decision_regret": jnp.mean(values[:-1]),
        "maximum_decision_regret": jnp.max(values[:-1]),
        "mean_decision_regret_shaping": jnp.mean(shaping),
        "decision_regret_weight": jnp.asarray(weight),
        "action_range_ema": next_scale,
        "mean_action_range": jnp.mean(ranges),
        "mean_combined_training_reward": jnp.mean(shaped),
    }


def attach_decision_regret_shaping(
    *,
    batch: RolloutBatch,
    model: Any,
    target_params: Mapping[str, Any],
    config: Any,
    key: Any,
    action_range_ema: Any = 1.0,
    effective_steps: Any = 0,
) -> tuple[RolloutBatch, Mapping[str, Any]]:
    context = target_context_sequence(model=model, target_params=target_params, batch=batch)
    regrets, ranges = chunked_decision_regret(
        model=model,
        target_params=target_params,
        context=context,
        key=key,
        posterior_particles=config.model.posterior_particles,
    )
    weight = decision_regret_weight(
        effective_steps=effective_steps,
        total_steps=config.training.environment_steps,
        maximum=config.loss.decision_regret_weight_maximum,
        midpoint=config.loss.decision_regret_schedule_midpoint,
        temperature=config.loss.decision_regret_schedule_temperature,
    )
    return finalize_decision_regret_shaping(
        batch=batch,
        regrets=regrets,
        action_ranges=ranges,
        action_range_ema=action_range_ema,
        gamma=config.ppo.gamma,
        weight=weight,
    )


__all__ = [
    "PartnerFunctions",
    "RunnerState",
    "DECISION_REGRET_STATE_CHUNK_SIZE",
    "DECISION_REGRET_NUMERICAL_TILE_SIZE",
    "attach_decision_regret_shaping",
    "chunked_decision_regret",
    "collect_rollout",
    "collect_rollout_minimal",
    "collect_support_rollout",
    "context_dropout_probability",
    "decision_regret_chunk",
    "decision_regret_weight",
    "finalize_decision_regret_shaping",
    "initialize_runner",
    "official_ego_roles",
    "observe_policy_after_transition",
    "policy_action",
    "target_context_sequence",
]
