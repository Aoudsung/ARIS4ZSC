"""Official-environment rollout for unified DELTA-ZSC."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

from .model import initial_agent_state
from .training import action_log_probability
from .types import AgentState, RolloutBatch


class RunnerState(NamedTuple):
    environment_state: Any
    observations: Any
    agent_state: AgentState
    partner_state: Any
    ego_roles: Any
    episode_return: Any
    completed_episodes: Any
    environment_steps: Any
    random_key: Any


def balanced_ego_roles(environment_count: int) -> Any:
    import jax.numpy as jnp

    count = int(environment_count)
    if count <= 0 or count % 2:
        raise ValueError("Role-balanced rollout requires a positive even lane count.")
    return jnp.arange(count, dtype=jnp.int32) % 2


def initialize_runner(
    *,
    environment: Any,
    task_hidden_dim: int,
    component_count: int,
    partner_functions: Any,
    random_key: Any,
) -> RunnerState:
    import jax
    import jax.numpy as jnp

    next_key, reset_key, partner_key = jax.random.split(random_key, 3)
    environment_state, observations = environment.reset(reset_key)
    count = int(environment.num_envs)
    return RunnerState(
        environment_state=environment_state,
        observations=observations,
        agent_state=initial_agent_state(
            batch_size=count,
            observation_shape=environment.observation_shape,
            task_hidden_dim=int(task_hidden_dim),
            component_count=int(component_count),
        ),
        partner_state=partner_functions.initial_state(count, partner_key),
        ego_roles=balanced_ego_roles(count),
        episode_return=jnp.zeros((count,), dtype=jnp.float32),
        completed_episodes=jnp.asarray(0, dtype=jnp.int32),
        environment_steps=jnp.asarray(0, dtype=jnp.int32),
        random_key=next_key,
    )


def _tree_select(mask: Any, selected: Any, alternative: Any) -> Any:
    import jax
    import jax.numpy as jnp

    values = jnp.asarray(mask, dtype=jnp.bool_)

    def choose(left: Any, right: Any) -> Any:
        expanded = values.reshape(
            values.shape + (1,) * (jnp.ndim(left) - values.ndim)
        )
        return jnp.where(expanded, left, right)

    return jax.tree_util.tree_map(choose, selected, alternative)


def update_agent_after_transition(
    *,
    state: AgentState,
    action: Any,
    done: Any,
    next_observation: Any,
    task_hidden_dim: int,
    component_count: int,
) -> AgentState:
    import jax.numpy as jnp

    done_array = jnp.asarray(done, dtype=jnp.bool_)
    candidate = state._replace(
        previous_action=jnp.asarray(action, dtype=jnp.int32),
        episode_start=done_array,
    )
    fresh = initial_agent_state(
        batch_size=int(done_array.shape[0]),
        observation_shape=tuple(int(value) for value in next_observation.shape[1:]),
        task_hidden_dim=int(task_hidden_dim),
        component_count=int(component_count),
    )
    # New episodes use an exactly fresh prior/carry; non-terminal lanes retain
    # the legal pre-action observation stored by ``UnifiedAgent.step``.
    return _tree_select(done_array, fresh, candidate)


def collect_rollout(
    *,
    state: RunnerState,
    length: int,
    environment: Any,
    agent: Any,
    base_params: Any,
    latent_params: Any,
    partner_functions: Any,
    partner_parameters: Any,
    variant: str,
    task_hidden_dim: int,
    component_count: int,
    shaping_factor: float,
    record_anchor_state: bool = False,
) -> tuple[RunnerState, RolloutBatch, Mapping[str, Any]]:
    """Collect one recurrent on-policy rollout and optional anchor snapshots."""

    import jax
    import jax.numpy as jnp

    if int(length) <= 0:
        raise ValueError("Rollout length must be positive.")
    count = int(environment.num_envs)
    initial_state = state.agent_state

    def one(current: RunnerState, unused_time: Any):
        del unused_time
        next_root, ego_root, partner_root, environment_root = jax.random.split(
            current.random_key, 4
        )
        ego_keys = jax.random.split(ego_root, count)
        partner_keys = jax.random.split(partner_root, count)
        environment_keys = jax.random.split(environment_root, count)
        lanes = jnp.arange(count, dtype=jnp.int32)
        ego_observation = current.observations[lanes, current.ego_roles]
        partner_observation = current.observations[lanes, 1 - current.ego_roles]
        stepped_agent, output = agent.step(
            base_params=base_params,
            latent_params=latent_params,
            state=current.agent_state,
            observation=ego_observation,
            variant=variant,
        )
        ego_action = jax.vmap(
            lambda key, logits: jax.random.categorical(key, logits)
        )(ego_keys, output.latent.adapted_logits)
        log_probability = action_log_probability(
            output.latent.adapted_logits, ego_action
        )
        behavior_probability = jax.nn.softmax(
            output.latent.adapted_logits, axis=-1
        )
        partner_action, stepped_partner, partner_context, _ = partner_functions.step(
            partner_parameters,
            current.partner_state,
            partner_observation,
            current.agent_state.episode_start,
            partner_keys,
        )
        ego_first = jnp.stack((ego_action, partner_action), axis=-1)
        partner_first = jnp.stack((partner_action, ego_action), axis=-1)
        joint_action = jnp.where(
            (current.ego_roles == 0)[:, None], ego_first, partner_first
        )
        (
            next_environment_state,
            next_observations,
            rewards,
            dones,
            info,
        ) = environment.step_with_keys(
            current.environment_state, joint_action, environment_keys
        )
        raw_rewards_by_agent = info["raw_rewards_by_agent"]
        raw_reward = raw_rewards_by_agent[lanes, current.ego_roles]
        official_shaped = info["official_shaped_rewards_by_agent"][
            lanes, current.ego_roles
        ]
        shaped_reward = raw_reward + float(shaping_factor) * official_shaped
        terminal = info["terminal_observations"]
        observation_mask = dones.reshape(
            dones.shape + (1,) * (next_observations[:, 0].ndim - 1)
        )
        response_next = jnp.where(
            observation_mask,
            terminal[lanes, current.ego_roles],
            next_observations[lanes, current.ego_roles],
        )
        partner_response_next = jnp.where(
            observation_mask,
            terminal[lanes, 1 - current.ego_roles],
            next_observations[lanes, 1 - current.ego_roles],
        )
        next_agent = update_agent_after_transition(
            state=stepped_agent,
            action=ego_action,
            done=dones,
            next_observation=next_observations[lanes, current.ego_roles],
            task_hidden_dim=task_hidden_dim,
            component_count=component_count,
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
            raw_reward,
            dones,
            partner_response_next,
        )
        completed_return = current.episode_return + raw_reward
        next_state = RunnerState(
            environment_state=next_environment_state,
            observations=next_observations,
            agent_state=next_agent,
            partner_state=next_partner,
            ego_roles=current.ego_roles,
            episode_return=jnp.where(dones, 0.0, completed_return),
            completed_episodes=current.completed_episodes
            + jnp.sum(dones.astype(jnp.int32)),
            environment_steps=current.environment_steps + count,
            random_key=next_root,
        )
        common = {
            "observations": ego_observation,
            "response_next_observations": response_next,
            "previous_actions": current.agent_state.previous_action,
            "episode_starts": current.agent_state.episode_start,
            "actions": ego_action,
            "rewards": raw_reward,
            "shaped_rewards": shaped_reward,
            "dones": dones,
            "behavior_probabilities": behavior_probability,
            "old_log_probabilities": log_probability,
            "old_values": output.base.value,
            "ppo_mask": diagnostics.get(
                "ppo_mask", jnp.ones_like(raw_reward, dtype=jnp.float32)
            ),
            "completed_returns": jnp.where(dones, completed_return, 0.0),
            "completed_mask": dones,
            "adaptation_kl": output.latent.adaptation_kl,
            "posterior": output.latent.belief,
            "voi": output.latent.value_of_information,
            "partner_run_ids": partner_functions.run_id(
                partner_parameters, current.partner_state, partner_context
            ),
        }
        if record_anchor_state:
            common = {
                **common,
                "environment_state": current.environment_state,
                "joint_observations": current.observations,
                "agent_state": current.agent_state,
                "partner_state": current.partner_state,
                "partner_episode_start": current.agent_state.episode_start,
                "ego_roles": current.ego_roles,
            }
        return next_state, common

    final_state, recorded = jax.lax.scan(
        one, state, jnp.arange(int(length), dtype=jnp.int32)
    )
    lanes = jnp.arange(count, dtype=jnp.int32)
    final_observation = final_state.observations[lanes, final_state.ego_roles]
    _, final_output = agent.step(
        base_params=base_params,
        latent_params=latent_params,
        state=final_state.agent_state,
        observation=final_observation,
        variant=variant,
    )
    batch = RolloutBatch(
        observations=jnp.concatenate(
            (recorded["observations"], final_observation[None]), axis=0
        ),
        response_next_observations=recorded["response_next_observations"],
        previous_actions=jnp.concatenate(
            (
                recorded["previous_actions"],
                final_state.agent_state.previous_action[None],
            ),
            axis=0,
        ),
        episode_starts=jnp.concatenate(
            (
                recorded["episode_starts"],
                final_state.agent_state.episode_start[None],
            ),
            axis=0,
        ),
        actions=recorded["actions"],
        rewards=recorded["rewards"],
        shaped_rewards=recorded["shaped_rewards"],
        dones=recorded["dones"],
        behavior_probabilities=recorded["behavior_probabilities"],
        old_log_probabilities=recorded["old_log_probabilities"],
        old_values=jnp.concatenate(
            (recorded["old_values"], final_output.base.value[None]), axis=0
        ),
        ppo_mask=recorded["ppo_mask"],
        initial_state=initial_state,
    )
    return final_state, batch, recorded


def deployment_action(
    *,
    agent: Any,
    base_params: Any,
    latent_params: Any,
    state: AgentState,
    observation: Any,
    keys: Any,
    variant: str,
) -> tuple[AgentState, Any, AgentOutput, Any]:
    import jax
    import jax.numpy as jnp

    stepped, output = agent.step(
        base_params=base_params,
        latent_params=latent_params,
        state=state,
        observation=observation,
        variant=variant,
    )
    key_array = jnp.asarray(keys)
    action = (
        jax.vmap(lambda key, logits: jax.random.categorical(key, logits))(
            key_array, output.latent.adapted_logits
        )
        if key_array.ndim == 2
        else jax.random.categorical(key_array, output.latent.adapted_logits)
    )
    return (
        stepped,
        action,
        output,
        action_log_probability(output.latent.adapted_logits, action),
    )


__all__ = [
    "RunnerState",
    "balanced_ego_roles",
    "collect_rollout",
    "deployment_action",
    "initialize_runner",
    "update_agent_after_transition",
]
