"""Device-side rollout collection for the unified DELTA-ZSC controller."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

from .belief_set_encoder import stratified_mixture_samples
from .counterfactual_anchor import tree_select
from .model import initial_policy_state
from .regret_potential import decision_regret_from_action_values, potential_shaping
from .training import categorical_log_probability
from .types import PolicyState, RolloutBatch


DECISION_REGRET_STATE_CHUNK_SIZE = 4_096
DECISION_REGRET_NUMERICAL_TILE_SIZE = 32


class PartnerFunctions(NamedTuple):
    initial_state: Callable[[int], Any]
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
    ego_policy = initial_policy_state(
        batch_size=count,
        observation_shape=environment.observation_shape,
        action_count=6,
        task_hidden_dim=model_config.task_hidden_dim,
        belief_hidden_dim=model_config.belief_hidden_dim,
        latent_dim=model_config.latent_dim,
        mixture_components=model_config.mixture_components,
    )
    return RunnerState(
        environment_state=environment_state,
        observations=observations,
        ego_policy=ego_policy,
        target_ego_policy=ego_policy,
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
    target_params: Mapping[str, Any],
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
        # policy therefore never consumes this privileged transition field;
        # raw reward remains available only to training targets.
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
    target_params: Mapping[str, Any],
    model_config: Any,
    partner_functions: PartnerFunctions,
    partner_parameters: Any,
    gate_values: Any,
    official_shaping_factor: float,
    record_mode: str = "anchor_full",
) -> tuple[RunnerState, RolloutBatch, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp

    if length <= 0:
        raise ValueError("Rollout length must be positive.")
    if record_mode not in {"minimal", "anchor_full", "support"}:
        raise ValueError(f"Unknown rollout record mode: {record_mode!r}.")
    count = int(environment.num_envs)
    initial_policy = state.ego_policy
    initial_target_policy = state.target_ego_policy

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
        stepped_target_ego, unused_target_output = model.apply(
            {"params": target_params},
            current.target_ego_policy,
            ego_observation,
            gate,
            method=model.step,
        )
        del unused_target_output
        behavior_logits = output.execution_logits
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
        next_target_ego = observe_policy_after_transition(
            stepped_state=stepped_target_ego,
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
            target_ego_policy=next_target_ego,
            partner_state=next_partner,
            ego_roles=current.ego_roles,
            episode_return=jnp.where(dones, 0.0, completed_return),
            completed_episodes=current.completed_episodes
            + jnp.sum(dones.astype(jnp.int64)),
            effective_environment_steps=current.effective_environment_steps + count,
            random_key=next_root,
        )
        batch_record = {
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
                jnp.asarray(official_shaping_factor, dtype=jnp.float32),
                dtype=jnp.float32,
            ),
            "dones": dones,
            "old_log_probabilities": log_probability,
            "old_values": output.state_value,
            "ppo_mask": jnp.ones_like(raw_rewards, dtype=jnp.float32),
            "partner_run_ids": partner_functions.run_id(
                partner_parameters, current.partner_state, partner_context
            ),
            # Compact partner context is consumed by registered auxiliary
            # losses in every PPO update, so it belongs to the minimal batch.
            "partner_code": partner_diagnostics["code"],
            "partner_source": partner_diagnostics["source"],
        }
        if record_mode == "minimal":
            return next_state, batch_record
        if record_mode == "support":
            return next_state, {
                "partner_run_ids": batch_record["partner_run_ids"],
                "partner_source": partner_diagnostics["source"],
                "mixture_logits": output.mixture_logits,
                "mixture_means": output.mixture_means,
                "mixture_log_variances": output.mixture_log_variances,
            }
        return next_state, {
            **batch_record,
            "partner_actions": partner_action,
            "partner_log_probabilities": partner_log_probability,
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
            "target_ego_policy_state": current.target_ego_policy,
            "partner_state": current.partner_state,
            "joint_observations": current.observations,
            "partner_observations": partner_observation,
            "ego_roles": current.ego_roles,
        }

    final_state, recorded = jax.lax.scan(one_step, state, xs=None, length=int(length))
    if record_mode == "support":
        return final_state, None, recorded
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
        initial_target_policy_state=initial_target_policy,
    )
    return final_state, batch, (recorded if record_mode == "anchor_full" else {})


def collect_rollout_minimal(**kwargs: Any) -> tuple[RunnerState, RolloutBatch, Mapping[str, Any]]:
    """Collect a PPO rollout without materializing anchor-only scan outputs."""

    return collect_rollout(**kwargs, record_mode="minimal")


def collect_support_rollout(**kwargs: Any) -> tuple[RunnerState, None, Mapping[str, Any]]:
    """Collect only final-model posterior/support records."""

    return collect_rollout(**kwargs, record_mode="support")


def chunked_decision_regret(
    *,
    model: Any,
    target_params: Mapping[str, Any],
    context: Any,
    key: Any,
    posterior_particles: int,
    state_chunk_size: int = DECISION_REGRET_STATE_CHUNK_SIZE,
) -> Any:
    """Compute target-critic regret without actor/decoder materialization.

    Real state keys are generated before padding, so changing the operational
    chunking cannot change any scientific posterior sample.
    """

    import math

    import jax
    import jax.numpy as jnp

    particle_count = int(posterior_particles)
    chunk_size = int(state_chunk_size)
    if particle_count <= 0 or chunk_size <= 0:
        raise ValueError("Decision-regret particle and chunk counts must be positive.")

    task_features = jax.lax.stop_gradient(jnp.asarray(context.task_features))
    mixture_logits = jax.lax.stop_gradient(jnp.asarray(context.mixture_logits))
    means = jax.lax.stop_gradient(jnp.asarray(context.mixture_means))
    log_variances = jax.lax.stop_gradient(
        jnp.asarray(context.mixture_log_variances)
    )
    prefix = task_features.shape[:-1]
    state_count = int(math.prod(prefix))
    if state_count <= 0:
        raise ValueError("Decision-regret context must contain at least one state.")

    real_keys = jax.random.split(key, state_count)
    padded_count = ((state_count + chunk_size - 1) // chunk_size) * chunk_size
    padding = padded_count - state_count

    def flatten_and_pad(value: Any) -> Any:
        array = jnp.asarray(value)
        flat = array.reshape((state_count,) + array.shape[len(prefix) :])
        pad_width = ((0, padding),) + ((0, 0),) * (flat.ndim - 1)
        return jnp.pad(flat, pad_width).reshape(
            (padded_count // chunk_size, chunk_size) + flat.shape[1:]
        )

    key_padding = ((0, padding), (0, 0))
    chunk_keys = jnp.pad(real_keys, key_padding).reshape(
        (padded_count // chunk_size, chunk_size, 2)
    )
    chunks = (
        flatten_and_pad(task_features),
        flatten_and_pad(mixture_logits),
        flatten_and_pad(means),
        flatten_and_pad(log_variances),
        chunk_keys,
    )

    def one_chunk(values: tuple[Any, Any, Any, Any, Any]) -> Any:
        features, logits, mu, log_var, sample_keys = values
        return decision_regret_chunk(
            model=model,
            target_params=target_params,
            task_features=features,
            mixture_logits=logits,
            mixture_means=mu,
            mixture_log_variances=log_var,
            sample_keys=sample_keys,
            posterior_particles=particle_count,
        )

    chunked = jax.lax.map(one_chunk, chunks)
    return chunked.reshape((padded_count,))[:state_count].reshape(prefix)


def decision_regret_chunk(
    *,
    model: Any,
    target_params: Mapping[str, Any],
    task_features: Any,
    mixture_logits: Any,
    mixture_means: Any,
    mixture_log_variances: Any,
    sample_keys: Any,
    posterior_particles: int,
) -> Any:
    """Evaluate one fixed-size target-critic chunk and no other model heads."""

    import jax
    import jax.numpy as jnp

    features = jax.lax.stop_gradient(jnp.asarray(task_features))
    logits = jax.lax.stop_gradient(jnp.asarray(mixture_logits))
    means = jax.lax.stop_gradient(jnp.asarray(mixture_means))
    log_variances = jax.lax.stop_gradient(jnp.asarray(mixture_log_variances))
    keys = jnp.asarray(sample_keys)
    particle_count = int(posterior_particles)
    real_count = int(features.shape[0])
    tile_size = DECISION_REGRET_NUMERICAL_TILE_SIZE
    padded_count = ((real_count + tile_size - 1) // tile_size) * tile_size
    padding = padded_count - real_count

    def tile(value: Any) -> Any:
        array = jnp.asarray(value)
        widths = ((0, padding),) + ((0, 0),) * (array.ndim - 1)
        return jnp.pad(array, widths).reshape(
            (padded_count // tile_size, tile_size) + array.shape[1:]
        )

    tiled = (
        tile(features),
        tile(logits),
        tile(means),
        tile(log_variances),
        tile(keys),
    )

    def one_tile(values: tuple[Any, Any, Any, Any, Any]) -> Any:
        tile_features, tile_logits, tile_means, tile_log_variances, tile_keys = values
        samples, weights = jax.vmap(
            lambda lane_key, lane_logits, lane_mu, lane_log_var: (
                stratified_mixture_samples(
                    lane_key,
                    mixture_logits=lane_logits,
                    means=lane_mu,
                    log_variances=lane_log_var,
                    sample_count=particle_count,
                )
            )
        )(tile_keys, tile_logits, tile_means, tile_log_variances)
        particle_features = jnp.broadcast_to(
            tile_features[:, None, :],
            (tile_size, particle_count, tile_features.shape[-1]),
        )
        action_values = model.apply(
            {"params": target_params},
            particle_features,
            samples,
            method=model.action_values_from_features_and_latent,
        )
        return decision_regret_from_action_values(
            jax.lax.stop_gradient(action_values),
            jax.lax.stop_gradient(weights),
        )

    regrets = jax.lax.map(one_tile, tiled).reshape((padded_count,))
    return regrets[:real_count]


def target_context_sequence(
    *,
    model: Any,
    target_params: Mapping[str, Any],
    batch: RolloutBatch,
) -> Any:
    """Replay target task/belief recurrence without actor, critic, or decoder."""

    unused_final, context = model.apply(
        {"params": target_params},
        batch.initial_target_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.previous_rewards,
        batch.episode_starts,
        method=model.context_sequence,
    )
    del unused_final
    return context


def finalize_decision_regret_shaping(
    *,
    batch: RolloutBatch,
    regrets: Any,
    gamma: float,
    weight: float,
) -> tuple[RolloutBatch, Mapping[str, Any]]:
    """Apply the registered detached potential after chunk evaluation."""

    import jax.numpy as jnp

    values = jnp.asarray(regrets)
    shaping = potential_shaping(
        values[:-1],
        values[1:],
        batch.dones,
        gamma=gamma,
        weight=weight,
    )
    shaped = batch.shaped_rewards + shaping
    return batch._replace(
        decision_regret_shaping=shaping,
        shaped_rewards=shaped,
    ), {
        "decision_regret_values": values[:-1],
        "decision_regret_values_with_next": values,
        "mean_decision_regret": jnp.mean(values[:-1]),
        "mean_decision_regret_shaping": jnp.mean(shaping),
        "mean_official_shaped_reward": jnp.mean(batch.official_shaped_rewards),
        "official_shaping_factor": jnp.mean(batch.official_shaping_factors),
        "mean_combined_training_reward": jnp.mean(shaped),
        "maximum_decision_regret": jnp.max(values[:-1]),
        "decision_regret_state_chunk_size": jnp.asarray(
            DECISION_REGRET_STATE_CHUNK_SIZE, dtype=jnp.int32
        ),
    }


def attach_decision_regret_shaping(
    *,
    batch: RolloutBatch,
    model: Any,
    target_params: Mapping[str, Any],
    config: Any,
    key: Any,
) -> tuple[RolloutBatch, Mapping[str, Any]]:
    """Compute a frozen-potential shaped reward on the collected batch."""

    context = target_context_sequence(
        model=model,
        target_params=target_params,
        batch=batch,
    )
    regrets = chunked_decision_regret(
        model=model,
        target_params=target_params,
        context=context,
        key=key,
        posterior_particles=config.model.posterior_particles,
    )
    return finalize_decision_regret_shaping(
        batch=batch,
        regrets=regrets,
        gamma=config.ppo.gamma,
        weight=config.loss.decision_regret_weight,
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
    "decision_regret_chunk",
    "finalize_decision_regret_shaping",
    "initialize_runner",
    "official_ego_roles",
    "observe_policy_after_transition",
    "policy_action",
    "target_context_sequence",
]
