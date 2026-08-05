"""Sparse CRN all-action decision observations.

Anchors are used once in the same outer update that collected them.  No replay,
comparator, pair matching, separation geometry, or stale target policy exists.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from .types import AnchorBatch


class AnchorWorld(NamedTuple):
    environment_state: Any
    observations: Any
    ego_state: Any
    partner_state: Any
    partner_episode_start: Any
    ego_roles: Any
    done: Any


class AnchorFunctions(NamedTuple):
    ego_step: Callable[..., tuple[Any, Any]]
    ego_observe: Callable[..., Any]
    partner_step: Callable[..., tuple[Any, Any, Any]]
    partner_observe: Callable[..., Any]
    environment_step: Callable[..., tuple[Any, Any, Any, Any, Any]]


def _repeat_tree(tree: Any, repeats: int) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(
        lambda value: jnp.repeat(jnp.asarray(value), int(repeats), axis=0), tree
    )


def _select_active(active: Any, candidate: Any, current: Any) -> Any:
    import jax
    import jax.numpy as jnp

    mask = jnp.asarray(active, dtype=jnp.bool_)

    def one(new: Any, old: Any) -> Any:
        expanded = mask.reshape(mask.shape + (1,) * (jnp.ndim(new) - mask.ndim))
        return jnp.where(expanded, new, old)

    return jax.tree_util.tree_map(one, candidate, current)


def _branch_roots(root_keys: Any, action_count: int, replica_count: int) -> Any:
    import jax
    import jax.numpy as jnp

    roots = jnp.asarray(root_keys, dtype=jnp.uint32)
    replica_ids = jnp.arange(int(replica_count), dtype=jnp.uint32)

    def one(root: Any) -> Any:
        replica_roots = jax.vmap(lambda r: jax.random.fold_in(root, r))(replica_ids)
        return jnp.broadcast_to(
            replica_roots[None],
            (int(action_count), int(replica_count), 2),
        ).reshape((-1, 2))

    return jax.vmap(one)(roots).reshape((-1, 2))


def collect_all_action_continuations(
    *,
    world: AnchorWorld,
    root_keys: Any,
    functions: AnchorFunctions,
    base_params: Any,
    latent_params: Any,
    action_count: int,
    fit_replicas: int,
    evaluation_replicas: int,
    horizon: int,
    gamma: float,
) -> tuple[Any, Any, Any, Any]:
    """Return fit/evaluation means and raw replica tensors."""

    import jax
    import jax.numpy as jnp

    anchor_count = int(jnp.asarray(world.done).shape[0])
    replicas = int(fit_replicas + evaluation_replicas)
    if min(anchor_count, action_count, fit_replicas, evaluation_replicas, horizon) <= 0:
        raise ValueError("Anchor continuation dimensions must be positive.")
    repeats = int(action_count) * replicas
    forced_actions = jnp.tile(
        jnp.repeat(jnp.arange(int(action_count), dtype=jnp.int32), replicas),
        anchor_count,
    )
    roots = _branch_roots(root_keys, action_count, replicas)
    current = AnchorWorld(
        environment_state=_repeat_tree(world.environment_state, repeats),
        observations=_repeat_tree(world.observations, repeats),
        ego_state=_repeat_tree(world.ego_state, repeats),
        partner_state=_repeat_tree(world.partner_state, repeats),
        partner_episode_start=_repeat_tree(world.partner_episode_start, repeats),
        ego_roles=_repeat_tree(world.ego_roles, repeats),
        done=_repeat_tree(world.done, repeats),
    )
    raw_return = jnp.zeros((anchor_count * repeats,), dtype=jnp.float32)

    def advance(step: int, carry: tuple[AnchorWorld, Any]) -> tuple[AnchorWorld, Any]:
        branch, returns = carry
        active = ~jnp.asarray(branch.done, dtype=jnp.bool_)
        step_roots = jax.vmap(lambda key: jax.random.fold_in(key, step))(roots)
        split = jax.vmap(lambda key: jax.random.split(key, 3))(step_roots)
        ego_keys, partner_keys, environment_keys = split[:, 0], split[:, 1], split[:, 2]
        lanes = jnp.arange(branch.ego_roles.shape[0], dtype=jnp.int32)
        ego_observation = branch.observations[lanes, branch.ego_roles]
        partner_observation = branch.observations[lanes, 1 - branch.ego_roles]
        next_ego_pre, sampled_actions = functions.ego_step(
            base_params, latent_params, branch.ego_state, ego_observation, ego_keys
        )
        ego_actions = jnp.where(step == 0, forced_actions, sampled_actions)
        partner_actions, next_partner_pre, partner_context = functions.partner_step(
            branch.partner_state,
            partner_observation,
            branch.partner_episode_start,
            partner_keys,
        )
        ego_first = jnp.stack((ego_actions, partner_actions), axis=-1)
        partner_first = jnp.stack((partner_actions, ego_actions), axis=-1)
        joint = jnp.where((branch.ego_roles == 0)[:, None], ego_first, partner_first)
        next_environment, next_observations, reward, dones, info = functions.environment_step(
            branch.environment_state, joint, environment_keys
        )
        raw_by_agent = info.get("raw_rewards_by_agent")
        ego_reward = (
            reward
            if raw_by_agent is None
            else raw_by_agent[lanes, branch.ego_roles]
        )
        next_ego = functions.ego_observe(next_ego_pre, ego_actions, dones)
        next_partner = functions.partner_observe(
            next_partner_pre,
            partner_context,
            partner_observation,
            partner_actions,
            ego_reward,
            dones,
            next_observations[lanes, 1 - branch.ego_roles],
        )
        candidate = AnchorWorld(
            environment_state=next_environment,
            observations=next_observations,
            ego_state=next_ego,
            partner_state=next_partner,
            partner_episode_start=jnp.asarray(dones, dtype=jnp.bool_),
            ego_roles=branch.ego_roles,
            done=jnp.asarray(dones, dtype=jnp.bool_),
        )
        increment = jnp.where(
            active,
            (float(gamma) ** jnp.asarray(step, dtype=jnp.float32)) * ego_reward,
            0.0,
        )
        return _select_active(active, candidate, branch), returns + increment

    _, values = jax.lax.fori_loop(0, int(horizon), advance, (current, raw_return))
    values = values.reshape((anchor_count, int(action_count), replicas))
    fit_replica = values[..., : int(fit_replicas)]
    evaluation_replica = values[..., int(fit_replicas) :]
    return (
        jnp.mean(fit_replica, axis=-1),
        jnp.mean(evaluation_replica, axis=-1),
        fit_replica,
        evaluation_replica,
    )


def select_anchor_indexes(
    key: Any,
    *,
    time_count: int,
    environment_count: int,
    requested: int,
) -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    total = int(time_count) * int(environment_count)
    if not 0 < int(requested) <= total:
        raise ValueError("Requested anchor count is outside the rollout.")
    flat = jax.random.choice(
        key, total, shape=(int(requested),), replace=False
    )
    return flat // int(environment_count), flat % int(environment_count)


def gather_time_lane(tree: Any, time_indexes: Any, lane_indexes: Any) -> Any:
    import jax

    return jax.tree_util.tree_map(lambda value: value[time_indexes, lane_indexes], tree)


def collect_anchor_batch(
    *,
    key: Any,
    records: Any,
    functions: AnchorFunctions,
    base_params: Any,
    latent_params: Any,
    states_per_trigger: int,
    action_count: int,
    fit_replicas: int,
    evaluation_replicas: int,
    horizon: int,
    gamma: float,
) -> AnchorBatch:
    import jax
    import jax.numpy as jnp

    time_count, environment_count = records["ego_roles"].shape
    index_key, root_key = jax.random.split(key)
    time, lane = select_anchor_indexes(
        index_key,
        time_count=time_count,
        environment_count=environment_count,
        requested=states_per_trigger,
    )
    world = AnchorWorld(
        environment_state=gather_time_lane(records["environment_state"], time, lane),
        observations=gather_time_lane(records["joint_observations"], time, lane),
        ego_state=gather_time_lane(records["ego_policy_state"], time, lane),
        partner_state=gather_time_lane(records["partner_state"], time, lane),
        partner_episode_start=records["episode_starts"][time, lane],
        ego_roles=records["ego_roles"][time, lane],
        done=jnp.zeros((int(states_per_trigger),), dtype=jnp.bool_),
    )
    roots = jax.random.split(root_key, int(states_per_trigger))
    fit, evaluation, fit_replica, evaluation_replica = collect_all_action_continuations(
        world=world,
        root_keys=roots,
        functions=functions,
        base_params=base_params,
        latent_params=latent_params,
        action_count=action_count,
        fit_replicas=fit_replicas,
        evaluation_replicas=evaluation_replicas,
        horizon=horizon,
        gamma=gamma,
    )
    from .decision_model import action_contrast_matrix

    basis = action_contrast_matrix(action_count)
    # [anchor, action, replica] -> [anchor, replica, contrast].  The covariance
    # of the sample mean retains the CRN-induced cross-action correlations.
    contrast_samples = jnp.einsum("nar,ad->nrd", fit_replica, basis)
    centered_samples = contrast_samples - jnp.mean(
        contrast_samples, axis=1, keepdims=True
    )
    sample_covariance = jnp.einsum(
        "nrd,nre->nde", centered_samples, centered_samples
    ) / float(fit_replicas - 1)
    measurement_covariance = sample_covariance / float(fit_replicas)
    return AnchorBatch(
        time_indexes=time,
        lane_indexes=lane,
        fit_returns_by_action=fit,
        evaluation_returns_by_action=evaluation,
        measurement_covariances=(
            measurement_covariance
            + 1.0e-6 * jnp.eye(action_count - 1, dtype=jnp.float32)
        ),
        action_mask=jnp.ones_like(fit, dtype=jnp.bool_),
        fit_replica_returns_by_action=fit_replica,
        evaluation_replica_returns_by_action=evaluation_replica,
    )


__all__ = [
    "AnchorFunctions",
    "AnchorWorld",
    "collect_all_action_continuations",
    "collect_anchor_batch",
    "gather_time_lane",
    "select_anchor_indexes",
]
