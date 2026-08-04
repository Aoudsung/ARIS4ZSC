"""Direct CRN counterfactual decision observations.

Anchors are sampled from the current rollout and consumed only with that same
rollout.  They are not replayed across policy updates and require no pair
comparator, matched-history classifier, separation margin, pseudo-label, or
stored posterior.  The latent forward algorithm inserts each decision emission
at its exact rollout time/lane coordinate.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .runner import update_agent_after_transition
from .types import DecisionAnchorBatch


def _gather_time_lanes(tree: Any, flat_indexes: Any, environment_count: int) -> Any:
    import jax
    import jax.numpy as jnp

    indexes = jnp.asarray(flat_indexes, dtype=jnp.int32)
    time = indexes // int(environment_count)
    lane = indexes % int(environment_count)
    return jax.tree_util.tree_map(lambda value: value[time, lane], tree)


def _world_from_records(
    records: Mapping[str, Any], flat_indexes: Any, *, environment_count: int
) -> Any:
    import jax.numpy as jnp

    from src.path_c.counterfactual_anchor import AnchorWorld

    ego_roles = _gather_time_lanes(
        records["ego_roles"], flat_indexes, environment_count
    )
    count = int(jnp.asarray(flat_indexes).shape[0])
    return AnchorWorld(
        environment_state=_gather_time_lanes(
            records["environment_state"], flat_indexes, environment_count
        ),
        observations=_gather_time_lanes(
            records["joint_observations"], flat_indexes, environment_count
        ),
        ego_state=_gather_time_lanes(
            records["agent_state"], flat_indexes, environment_count
        ),
        partner_state=_gather_time_lanes(
            records["partner_state"], flat_indexes, environment_count
        ),
        partner_episode_start=_gather_time_lanes(
            records["partner_episode_start"], flat_indexes, environment_count
        ),
        ego_roles=ego_roles,
        done=jnp.zeros((count,), dtype=jnp.bool_),
        raw_return=jnp.zeros((count,), dtype=jnp.float32),
    )


def make_anchor_functions(
    *,
    agent: Any,
    base_params: Any,
    latent_params: Any,
    variant: str,
    partner_functions: Any,
    partner_parameters: Any,
    environment: Any,
    task_hidden_dim: int,
    component_count: int,
) -> Any:
    import jax

    from src.path_c.counterfactual_anchor import AnchorFunctions

    def ego_policy_step(state: Any, observation: Any, keys: Any):
        del keys
        next_state, output = agent.step(
            base_params=base_params,
            latent_params=latent_params,
            state=state,
            observation=observation,
            variant=variant,
        )
        return next_state, jax.nn.softmax(output.latent.adapted_logits, axis=-1)

    def ego_observe(
        stepped_state: Any,
        previous_observation: Any,
        action: Any,
        reward: Any,
        done: Any,
        next_observation: Any,
    ):
        del previous_observation, reward
        return update_agent_after_transition(
            state=stepped_state,
            action=action,
            done=done,
            next_observation=next_observation,
            task_hidden_dim=task_hidden_dim,
            component_count=component_count,
        )

    def partner_policy_step(
        state: Any, observation: Any, episode_start: Any, keys: Any
    ):
        action, next_state, context, _ = partner_functions.step(
            partner_parameters, state, observation, episode_start, keys
        )
        return action, next_state, context

    def partner_observe(
        state: Any,
        context: Any,
        observation: Any,
        action: Any,
        reward: Any,
        done: Any,
        next_observation: Any,
    ):
        return partner_functions.observe(
            partner_parameters,
            state,
            context,
            observation,
            action,
            reward,
            done,
            next_observation,
        )

    def environment_step(state: Any, joint_action: Any, keys: Any):
        return environment.step_with_keys(state, joint_action, keys)

    return AnchorFunctions(
        ego_policy_step=ego_policy_step,
        ego_observe=ego_observe,
        partner_policy_step=partner_policy_step,
        partner_observe=partner_observe,
        environment_step=environment_step,
    )


def select_anchor_indexes(
    *,
    key: Any,
    ppo_mask: Any,
    anchor_count: int,
) -> Any:
    """Select a fixed number of valid current-rollout states without replacement."""

    import jax
    import jax.numpy as jnp

    mask = np.asarray(ppo_mask, dtype=np.float32).reshape((-1,)) > 0.5
    eligible = np.flatnonzero(mask)
    count = int(anchor_count)
    if count <= 0:
        raise ValueError("Anchor count must be positive.")
    if eligible.size < count:
        raise RuntimeError(
            f"Only {eligible.size} PPO-valid states are available for {count} anchors."
        )
    chosen = jax.random.choice(
        key,
        jnp.asarray(eligible, dtype=jnp.int32),
        shape=(count,),
        replace=False,
    )
    return jnp.sort(chosen)


def collect_decision_anchors(
    *,
    key: Any,
    records: Mapping[str, Any],
    environment: Any,
    agent: Any,
    base_params: Any,
    latent_params: Any,
    variant: str,
    partner_functions: Any,
    partner_parameters: Any,
    task_hidden_dim: int,
    component_count: int,
    anchor_count: int,
    fit_replicas: int,
    evaluation_replicas: int,
    continuation_horizon: int,
    gamma: float,
) -> tuple[DecisionAnchorBatch, Mapping[str, int]]:
    """Collect real all-action continuation observations from one rollout."""

    import jax
    import jax.numpy as jnp

    from src.path_c.counterfactual_anchor import collect_counterfactual_anchors

    time_count, environment_count = records["actions"].shape
    select_key, root_key = jax.random.split(key)
    flat_indexes = select_anchor_indexes(
        key=select_key,
        ppo_mask=records["ppo_mask"],
        anchor_count=anchor_count,
    )
    world = _world_from_records(
        records, flat_indexes, environment_count=environment_count
    )
    roots = jax.random.split(root_key, int(anchor_count))
    functions = make_anchor_functions(
        agent=agent,
        base_params=base_params,
        latent_params=latent_params,
        variant=variant,
        partner_functions=partner_functions,
        partner_parameters=partner_parameters,
        environment=environment,
        task_hidden_dim=task_hidden_dim,
        component_count=component_count,
    )
    zeros_i = jnp.zeros((int(anchor_count),), dtype=jnp.int32)
    zeros_f = jnp.zeros((int(anchor_count),), dtype=jnp.float32)
    batch = collect_counterfactual_anchors(
        anchor_ids=jnp.arange(int(anchor_count), dtype=jnp.int32),
        root_keys=roots,
        world=world,
        rollout_flat_indexes=flat_indexes,
        policy_states=world.ego_state,
        observations=_gather_time_lanes(
            records["observations"], flat_indexes, environment_count
        ),
        partner_sources=zeros_i,
        partner_members=zeros_i,
        partner_family_ids=zeros_i,
        partner_checkpoint_stages=zeros_f,
        partner_run_ids=_gather_time_lanes(
            records["partner_run_ids"], flat_indexes, environment_count
        ),
        functions=functions,
        action_count=6,
        fit_replicas=int(fit_replicas),
        evaluation_replicas=int(evaluation_replicas),
        continuation_horizon=int(continuation_horizon),
        discount=float(gamma),
    )
    replica_count = jnp.maximum(
        jnp.asarray(batch.replica_count, dtype=jnp.float32), 1.0
    )
    mean = jnp.asarray(batch.return_sum_by_action, dtype=jnp.float32) / replica_count
    second = (
        jnp.asarray(batch.return_squared_sum_by_action, dtype=jnp.float32)
        / replica_count
    )
    standard_error = jnp.sqrt(
        jnp.maximum(second - jnp.square(mean), 0.0) / replica_count
    )
    centered = mean - jnp.mean(mean, axis=-1, keepdims=True)
    time_indexes = flat_indexes // int(environment_count)
    lane_indexes = flat_indexes % int(environment_count)
    result = DecisionAnchorBatch(
        time_indexes=time_indexes,
        lane_indexes=lane_indexes,
        centered_returns=centered,
        standard_errors=standard_error,
        action_mask=jnp.asarray(batch.action_mask, dtype=jnp.bool_),
        fit_replica_returns=jnp.asarray(
            batch.fit_replica_returns_by_action, dtype=jnp.float32
        ),
        evaluation_returns=jnp.asarray(
            batch.evaluation_returns_by_action, dtype=jnp.float32
        ),
    )
    transition_steps = (
        int(anchor_count)
        * 6
        * (int(fit_replicas) + int(evaluation_replicas))
        * int(continuation_horizon)
    )
    return result, {
        "anchor_count": int(anchor_count),
        "counterfactual_continuation_steps": transition_steps,
        "rollout_time_count": int(time_count),
        "rollout_environment_count": int(environment_count),
    }


__all__ = [
    "collect_decision_anchors",
    "make_anchor_functions",
    "select_anchor_indexes",
]
