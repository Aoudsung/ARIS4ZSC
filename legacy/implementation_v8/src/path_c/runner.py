"""CUDA-friendly DEPI rollout over the legal ego information boundary."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple

from .counterfactual_anchor import tree_select
from .model import initial_policy_state
from .training import categorical_log_probability
from .types import PolicyState, RolloutBatch


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
        capability_hidden_dim=model_config.capability_hidden_dim,
        capability_dim=model_config.capability_dim,
        component_embedding_dim=model_config.component_embedding_dim,
        protocol_components=model_config.protocol_components,
    )
    return RunnerState(
        environment_state,
        observations,
        policy,
        policy,
        partner_functions.initial_state(count, partner_key),
        official_ego_roles(count),
        jnp.zeros((count,), dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
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
        capability_hidden_dim=model_config.capability_hidden_dim,
        capability_dim=model_config.capability_dim,
        component_embedding_dim=model_config.component_embedding_dim,
        protocol_components=model_config.protocol_components,
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
        # Retain the complete behaviour distribution so the post-update
        # trust-region diagnostic is the exact categorical KL, not a sampled
        # action estimator.
        behavior_probability = jax.nn.softmax(output.policy_logits, axis=-1)
        partner_action, stepped_partner, partner_context, _ = (
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
            current.completed_episodes + jnp.sum(dones.astype(jnp.int32)),
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
            "ppo_mask": diagnostics.get(
                "ppo_mask", jnp.ones_like(raw_rewards, dtype=jnp.float32)
            ),
            "partner_run_ids": partner_functions.run_id(
                partner_parameters, current.partner_state, partner_context
            ),
            "partner_source": diagnostics["source"],
            "partner_member": diagnostics["member"],
            "partner_family_id": diagnostics["family_id"],
            "partner_checkpoint_stage": diagnostics["checkpoint_stage"],
            "partner_partition": diagnostics.get(
                "partition", jnp.zeros_like(ego_action, dtype=jnp.int32)
            ),
            "completed_returns": jnp.where(dones, completed_return, 0.0),
            "completed_mask": dones,
        }
        if record_mode == "minimal":
            return next_state, common
        if record_mode == "support":
            return next_state, {
                "partner_run_ids": common["partner_run_ids"],
                "partner_source": common["partner_source"],
                "capability": output.capability,
                "protocol_probabilities": output.protocol_probabilities,
                "posterior_entropy": output.posterior_entropy,
            }
        return next_state, {
            **common,
            "correct_deliveries": info["correct_delivery"],
            "wrong_deliveries": info["wrong_delivery"],
            "policy_logits": output.policy_logits,
            "action_values": output.action_values,
            "raw_q1": output.raw_q1,
            "raw_q2": output.raw_q2,
            "capability": output.capability,
            "protocol_probabilities": output.protocol_probabilities,
            "protocol_embedding": output.protocol_embedding,
            "posterior_entropy": output.posterior_entropy,
            "task_features": output.task_features,
            "environment_state": current.environment_state,
            "ego_policy_state": current.ego_policy,
            "target_ego_policy_state": current.target_ego_policy,
            "partner_state": current.partner_state,
            "joint_observations": current.observations,
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
        shaped_rewards=recorded["rewards"] + recorded["official_shaping_factor"] * recorded["official_shaped_rewards"],
        dones=recorded["dones"],
        old_log_probabilities=recorded["old_log_probabilities"],
        old_values=jnp.concatenate((recorded["old_values"], final_output.state_value[None]), axis=0),
        behavior_probabilities=recorded["behavior_probabilities"],
        ppo_mask=recorded["ppo_mask"],
        partner_sources=recorded["partner_source"],
        partner_members=recorded["partner_member"],
        partner_family_ids=recorded["partner_family_id"],
        partner_checkpoint_stages=recorded["partner_checkpoint_stage"],
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


__all__ = [
    "PartnerFunctions",
    "RunnerState",
    "collect_rollout",
    "collect_rollout_minimal",
    "collect_support_rollout",
    "context_dropout_probability",
    "initialize_runner",
    "official_ego_roles",
    "observe_policy_after_transition",
    "policy_action",
    "target_context_sequence",
]
