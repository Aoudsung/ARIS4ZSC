"""Sparse CRN current and probe-conditioned successor decision observations.

Anchors are consumed once in the same outer update that collected them.  The
v4 active target forces a probe at time t, advances one collection-time-base
bridge step while the teammate can react, then forces every candidate decision
action at time t+2 and continues with the collection-time base policy. Rewards
at the probe and bridge transitions are excluded from the decision matrix.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from .types import AnchorBatch, AnchorSnapshots


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


def _replica_roots(root_keys: Any, replica_count: int) -> Any:
    import jax
    import jax.numpy as jnp

    roots = jnp.asarray(root_keys, dtype=jnp.uint32)
    replica_ids = jnp.arange(int(replica_count), dtype=jnp.uint32)
    return jax.vmap(
        lambda root: jax.vmap(lambda replica: jax.random.fold_in(root, replica))(
            replica_ids
        )
    )(roots)


def _expanded_roots(root_keys: Any, action_count: int, replica_count: int) -> Any:
    import jax.numpy as jnp

    replica = _replica_roots(root_keys, replica_count)
    return jnp.broadcast_to(
        replica[:, None, :, :],
        (replica.shape[0], int(action_count), int(replica_count), 2),
    ).reshape((-1, 2))


def _step_world(
    *,
    branch: AnchorWorld,
    roots: Any,
    step_id: int,
    functions: AnchorFunctions,
    base_params: Any,
    latent_params: Any,
    forced_actions: Any | None,
    force: Any = True,
) -> tuple[AnchorWorld, Any]:
    """Advance one branch step and return ego raw reward."""

    import jax
    import jax.numpy as jnp

    active = ~jnp.asarray(branch.done, dtype=jnp.bool_)
    step_roots = jax.vmap(
        lambda key: jax.random.fold_in(
            key, jnp.asarray(step_id, dtype=jnp.uint32)
        )
    )(roots)
    split = jax.vmap(lambda key: jax.random.split(key, 3))(step_roots)
    ego_keys, partner_keys, environment_keys = split[:, 0], split[:, 1], split[:, 2]
    lanes = jnp.arange(branch.ego_roles.shape[0], dtype=jnp.int32)
    ego_observation = branch.observations[lanes, branch.ego_roles]
    partner_observation = branch.observations[lanes, 1 - branch.ego_roles]
    next_ego_pre, sampled_actions = functions.ego_step(
        base_params, latent_params, branch.ego_state, ego_observation, ego_keys
    )
    if forced_actions is None:
        ego_actions = sampled_actions
    else:
        ego_actions = jnp.where(
            jnp.asarray(force, dtype=jnp.bool_),
            jnp.asarray(forced_actions, dtype=jnp.int32),
            sampled_actions,
        )
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
    ego_reward = reward if raw_by_agent is None else raw_by_agent[lanes, branch.ego_roles]
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
    return _select_active(active, candidate, branch), jnp.where(active, ego_reward, 0.0)


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
    """Return current-state all-action fit/evaluation means and replicas."""

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
    roots = _expanded_roots(root_keys, action_count, replicas)
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
        branch, reward = _step_world(
            branch=branch,
            roots=roots,
            step_id=step,
            functions=functions,
            base_params=base_params,
            latent_params=latent_params,
            forced_actions=forced_actions,
            force=(step == 0),
        )
        return branch, returns + (
            jnp.asarray(gamma, dtype=jnp.float32) ** jnp.asarray(step, dtype=jnp.float32)
        ) * reward

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


def collect_probe_successor_continuations(
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
) -> tuple[Any, Any, Any, Any, Any]:
    """Return bounded-memory ``Q(s[t+2]^probe, a_decision)`` matrices.

    Probe and post-response decision actions are traversed with ``lax.map``. At most
    ``anchor_count * replica_count`` worlds are live simultaneously, rather
    than materializing ``anchor * probe * next_action * replica`` worlds.  The
    root key and step IDs are shared across action alternatives, preserving the
    exact common-random-number estimand. Probe and bridge rewards are excluded:
    the bridge ego action is sampled from the collection-time base policy before
    the delayed response becomes observable, and the forced action is chosen
    only from the resulting time-t+2 world.
    """

    import jax
    import jax.numpy as jnp

    anchor_count = int(jnp.asarray(world.done).shape[0])
    actions = int(action_count)
    replicas = int(fit_replicas + evaluation_replicas)
    if min(anchor_count, actions, fit_replicas, evaluation_replicas, horizon) <= 0:
        raise ValueError("Probe continuation dimensions must be positive.")

    roots = _replica_roots(root_keys, replicas).reshape((-1, 2))
    base_world = AnchorWorld(
        environment_state=_repeat_tree(world.environment_state, replicas),
        observations=_repeat_tree(world.observations, replicas),
        ego_state=_repeat_tree(world.ego_state, replicas),
        partner_state=_repeat_tree(world.partner_state, replicas),
        partner_episode_start=_repeat_tree(world.partner_episode_start, replicas),
        ego_roles=_repeat_tree(world.ego_roles, replicas),
        done=_repeat_tree(world.done, replicas),
    )
    branch_count = anchor_count * replicas

    def one_probe(probe_action: Any) -> tuple[Any, Any]:
        forced_probe = jnp.full((branch_count,), probe_action, dtype=jnp.int32)
        post_probe, unused_probe_reward = _step_world(
            branch=base_world,
            roots=roots,
            step_id=10_000,
            functions=functions,
            base_params=base_params,
            latent_params=latent_params,
            forced_actions=forced_probe,
        )
        del unused_probe_reward

        # The teammate can react to the probe only on this bridge transition.
        # The ego has not observed that reaction yet, so its bridge action is
        # the fixed collection-time base policy rather than a forced/adapted
        # action. The bridge reward is deliberately outside Q(s[t+2], a').
        decision_world, unused_bridge_reward = _step_world(
            branch=post_probe,
            roots=roots,
            step_id=15_000,
            functions=functions,
            base_params=base_params,
            latent_params=latent_params,
            forced_actions=None,
        )
        del unused_bridge_reward
        survived = (~jnp.asarray(decision_world.done, dtype=jnp.bool_)).reshape(
            (anchor_count, replicas)
        )

        def one_followup(next_action: Any) -> Any:
            forced_next = jnp.full((branch_count,), next_action, dtype=jnp.int32)
            initial_return = jnp.zeros((branch_count,), dtype=jnp.float32)

            def advance(step: int, carry: tuple[AnchorWorld, Any]):
                branch, returns = carry
                branch, reward = _step_world(
                    branch=branch,
                    roots=roots,
                    step_id=20_000 + step,
                    functions=functions,
                    base_params=base_params,
                    latent_params=latent_params,
                    forced_actions=forced_next,
                    force=(step == 0),
                )
                discount = jnp.asarray(gamma, dtype=jnp.float32) ** jnp.asarray(
                    step, dtype=jnp.float32
                )
                return branch, returns + discount * reward

            _, returns = jax.lax.fori_loop(
                0, int(horizon), advance, (decision_world, initial_return)
            )
            return returns.reshape((anchor_count, replicas))

        # [next_action, anchor, replica] -> [anchor, next_action, replica]
        by_action = jax.lax.map(
            one_followup, jnp.arange(actions, dtype=jnp.int32)
        )
        return jnp.transpose(by_action, (1, 0, 2)), survived

    # [probe, anchor, next_action, replica] -> [anchor, probe, next_action, replica]
    values, survived = jax.lax.map(
        one_probe, jnp.arange(actions, dtype=jnp.int32)
    )
    values = jnp.transpose(values, (1, 0, 2, 3))
    survived = jnp.transpose(survived, (1, 0, 2))
    fit_replica = values[..., : int(fit_replicas)]
    evaluation_replica = values[..., int(fit_replicas) :]
    valid_probe = jnp.all(survived, axis=-1)
    action_mask = jnp.broadcast_to(
        valid_probe[..., None], (anchor_count, actions, actions)
    )
    return (
        jnp.mean(fit_replica, axis=-1),
        jnp.mean(evaluation_replica, axis=-1),
        fit_replica,
        evaluation_replica,
        action_mask,
    )


def select_anchor_indexes(
    key: Any,
    *,
    time_count: int,
    environment_count: int,
    requested: int,
) -> tuple[Any, Any]:
    import jax

    total = int(time_count) * int(environment_count)
    if not 0 < int(requested) <= total:
        raise ValueError("Requested anchor count is outside the rollout.")
    flat = jax.random.choice(key, total, shape=(int(requested),), replace=False)
    return flat // int(environment_count), flat % int(environment_count)


def gather_time_lane(tree: Any, time_indexes: Any, lane_indexes: Any) -> Any:
    import jax

    return jax.tree_util.tree_map(
        lambda value: value[time_indexes, lane_indexes], tree
    )


def _measurement_covariance(replica_returns: Any, action_count: int, fit_replicas: int) -> Any:
    import jax.numpy as jnp

    from .decision_model import action_contrast_matrix

    basis = action_contrast_matrix(action_count)
    contrast_samples = jnp.einsum("...ar,ad->...rd", replica_returns, basis)
    centered = contrast_samples - jnp.mean(contrast_samples, axis=-2, keepdims=True)
    sample_covariance = jnp.einsum("...rd,...re->...de", centered, centered) / float(
        fit_replicas - 1
    )
    return sample_covariance / float(fit_replicas) + 1.0e-6 * jnp.eye(
        action_count - 1, dtype=jnp.float32
    )


def _collect_from_world(
    *,
    world: AnchorWorld,
    time_indexes: Any,
    lane_indexes: Any,
    root_key: Any,
    functions: AnchorFunctions,
    base_params: Any,
    latent_params: Any,
    action_count: int,
    fit_replicas: int,
    evaluation_replicas: int,
    horizon: int,
    gamma: Any,
    collect_successor: bool,
) -> AnchorBatch:
    import jax
    import jax.numpy as jnp

    state_count = int(jnp.asarray(world.done).shape[0])
    immediate_key, probe_key = jax.random.split(root_key)
    immediate_roots = jax.random.split(immediate_key, state_count)
    probe_roots = jax.random.split(probe_key, state_count)
    fit, evaluation, fit_replica, evaluation_replica = collect_all_action_continuations(
        world=world,
        root_keys=immediate_roots,
        functions=functions,
        base_params=base_params,
        latent_params=latent_params,
        action_count=action_count,
        fit_replicas=fit_replicas,
        evaluation_replicas=evaluation_replicas,
        horizon=horizon,
        gamma=gamma,
    )
    if bool(collect_successor):
        (
            probe_fit,
            probe_evaluation,
            probe_fit_replica,
            probe_evaluation_replica,
            probe_action_mask,
        ) = collect_probe_successor_continuations(
            world=world,
            root_keys=probe_roots,
            functions=functions,
            base_params=base_params,
            latent_params=latent_params,
            action_count=action_count,
            fit_replicas=fit_replicas,
            evaluation_replicas=evaluation_replicas,
            horizon=horizon,
            gamma=gamma,
        )
        probe_covariance = _measurement_covariance(
            probe_fit_replica, action_count, fit_replicas
        )
    else:
        # Preserve one static AnchorBatch signature while avoiding every
        # privileged successor simulation in DELTA-passive.  False masks make
        # the placeholder arrays semantically absent, not zero-valued labels.
        action_shape = (state_count, int(action_count), int(action_count))
        probe_fit = jnp.zeros(action_shape, dtype=jnp.float32)
        probe_evaluation = jnp.zeros(action_shape, dtype=jnp.float32)
        probe_fit_replica = jnp.zeros(
            action_shape + (int(fit_replicas),), dtype=jnp.float32
        )
        probe_evaluation_replica = jnp.zeros(
            action_shape + (int(evaluation_replicas),), dtype=jnp.float32
        )
        probe_covariance = jnp.zeros(
            (state_count, int(action_count), int(action_count) - 1, int(action_count) - 1),
            dtype=jnp.float32,
        )
        probe_action_mask = jnp.zeros(action_shape, dtype=jnp.bool_)
    return AnchorBatch(
        time_indexes=time_indexes,
        lane_indexes=lane_indexes,
        fit_returns_by_action=fit,
        evaluation_returns_by_action=evaluation,
        measurement_covariances=_measurement_covariance(
            fit_replica, action_count, fit_replicas
        ),
        action_mask=jnp.ones_like(fit, dtype=jnp.bool_),
        fit_replica_returns_by_action=fit_replica,
        evaluation_replica_returns_by_action=evaluation_replica,
        probe_fit_returns_by_action=probe_fit,
        probe_evaluation_returns_by_action=probe_evaluation,
        probe_measurement_covariances=probe_covariance,
        probe_action_mask=probe_action_mask,
        probe_fit_replica_returns_by_action=probe_fit_replica,
        probe_evaluation_replica_returns_by_action=probe_evaluation_replica,
    )


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
    collect_successor: bool = True,
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
    return _collect_from_world(
        world=world,
        time_indexes=time,
        lane_indexes=lane,
        root_key=root_key,
        functions=functions,
        base_params=base_params,
        latent_params=latent_params,
        action_count=action_count,
        fit_replicas=fit_replicas,
        evaluation_replicas=evaluation_replicas,
        horizon=horizon,
        gamma=gamma,
        collect_successor=collect_successor,
    )


def collect_anchor_batch_from_snapshots(
    *,
    root_key: Any,
    snapshots: AnchorSnapshots,
    functions: AnchorFunctions,
    base_params: Any,
    latent_params: Any,
    action_count: int,
    fit_replicas: int,
    evaluation_replicas: int,
    horizon: int,
    gamma: Any,
    collect_successor: bool = True,
) -> AnchorBatch:
    """Collect both v4 CRN target families from sparse rollout worlds."""

    import jax.numpy as jnp

    state_count = int(snapshots.time_indexes.shape[0])
    world = AnchorWorld(
        environment_state=snapshots.environment_state,
        observations=snapshots.observations,
        ego_state=snapshots.ego_state,
        partner_state=snapshots.partner_state,
        partner_episode_start=snapshots.partner_episode_start,
        ego_roles=snapshots.ego_roles,
        done=jnp.zeros((state_count,), dtype=jnp.bool_),
    )
    return _collect_from_world(
        world=world,
        time_indexes=snapshots.time_indexes,
        lane_indexes=snapshots.lane_indexes,
        root_key=root_key,
        functions=functions,
        base_params=base_params,
        latent_params=latent_params,
        action_count=action_count,
        fit_replicas=fit_replicas,
        evaluation_replicas=evaluation_replicas,
        horizon=horizon,
        gamma=gamma,
        collect_successor=collect_successor,
    )


def make_anchor_batch_kernel(
    *,
    functions: AnchorFunctions,
    action_count: int,
    fit_replicas: int,
    evaluation_replicas: int,
    horizon: int,
    collect_successor: bool,
) -> Callable[..., AnchorBatch]:
    """Create a fixed current-only or current-plus-successor executable."""

    import jax

    @jax.jit
    def kernel(
        root_key: Any,
        snapshots: AnchorSnapshots,
        base_params: Any,
        latent_params: Any,
        gamma: Any,
    ) -> AnchorBatch:
        return collect_anchor_batch_from_snapshots(
            root_key=root_key,
            snapshots=snapshots,
            functions=functions,
            base_params=base_params,
            latent_params=latent_params,
            action_count=action_count,
            fit_replicas=fit_replicas,
            evaluation_replicas=evaluation_replicas,
            horizon=horizon,
            gamma=gamma,
            collect_successor=collect_successor,
        )

    return kernel


__all__ = [
    "AnchorFunctions",
    "AnchorWorld",
    "collect_all_action_continuations",
    "collect_anchor_batch",
    "collect_anchor_batch_from_snapshots",
    "collect_probe_successor_continuations",
    "gather_time_lane",
    "make_anchor_batch_kernel",
    "select_anchor_indexes",
]
