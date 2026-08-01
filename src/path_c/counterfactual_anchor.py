"""Common-random-number counterfactual action anchors.

The collector is environment-agnostic.  All callbacks operate on a flattened
batch of ``anchor × action × replica`` lanes.  Ground-truth targets are simulator
returns; no model Q value is used to create labels.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

from .types import CounterfactualAnchorBatch


class AnchorWorld(NamedTuple):
    environment_state: Any
    observations: Any
    ego_state: Any
    partner_state: Any
    partner_episode_start: Any
    ego_roles: Any
    done: Any
    raw_return: Any


class AnchorFunctions(NamedTuple):
    """Batched callbacks required by ``collect_counterfactual_anchors``."""

    ego_policy_step: Callable[..., tuple[Any, Any]]
    ego_observe: Callable[..., Any]
    partner_policy_step: Callable[..., tuple[Any, Any, Any]]
    partner_observe: Callable[..., Any]
    environment_step: Callable[..., tuple[Any, Any, Any, Any, Any]]


def tree_select(mask: Any, selected: Any, alternative: Any) -> Any:
    import jax
    import jax.numpy as jnp

    def choose(left: Any, right: Any) -> Any:
        values = jnp.asarray(mask, dtype=jnp.bool_)
        expanded = values.reshape(values.shape + (1,) * (jnp.ndim(left) - values.ndim))
        return jnp.where(expanded, left, right)

    return jax.tree_util.tree_map(choose, selected, alternative)


def tree_repeat_interleave(tree: Any, repeats: int) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(
        lambda value: jnp.repeat(jnp.asarray(value), int(repeats), axis=0), tree
    )


def inverse_cdf_actions(probabilities: Any, uniforms: Any) -> Any:
    import jax.numpy as jnp

    probs = jnp.asarray(probabilities, dtype=jnp.float32)
    draws = jnp.asarray(uniforms, dtype=jnp.float32)
    cumulative = jnp.cumsum(probs, axis=-1)
    return jnp.sum(draws[..., None] >= cumulative, axis=-1).astype(jnp.int32)


def _shared_lane_keys(
    root_keys: Any,
    *,
    action_count: int,
    replicas: int,
    domain: int,
) -> Any:
    """Keys repeat across action branches for each anchor/replica."""

    import jax
    import jax.numpy as jnp

    roots = jnp.asarray(root_keys)
    replica_ids = jnp.arange(int(replicas), dtype=jnp.uint32)

    def one(root: Any) -> Any:
        replica_keys = jax.vmap(
            lambda replica: jax.random.fold_in(
                jax.random.fold_in(root, int(domain)), replica
            )
        )(replica_ids)
        repeated = jnp.broadcast_to(
            replica_keys[None, ...],
            (int(action_count),) + replica_keys.shape,
        )
        return repeated.reshape((int(action_count) * int(replicas), 2))

    return jax.vmap(one)(roots).reshape((-1, 2))


def anchor_microbatch_candidates(
    *,
    maximum_anchor_worlds: int,
    action_count: int,
    replicas: int,
) -> tuple[int, ...]:
    """Return a fixed descending sequence aligned to complete anchor worlds."""

    branches_per_world = int(action_count) * int(replicas)
    if maximum_anchor_worlds <= 0 or branches_per_world <= 0:
        raise ValueError("Anchor candidate dimensions must be positive.")
    worlds = int(maximum_anchor_worlds)
    candidates = []
    while True:
        candidates.append(worlds * branches_per_world)
        if worlds == 1:
            break
        worlds = max(worlds // 2, 1)
    return tuple(candidates)


def collect_counterfactual_anchors(
    *,
    anchor_ids: Any,
    root_keys: Any,
    world: AnchorWorld,
    rollout_flat_indexes: Any,
    policy_states: Any,
    observations: Any,
    partner_codes: Any,
    partner_sources: Any,
    partner_run_ids: Any,
    functions: AnchorFunctions,
    action_count: int,
    fit_replicas: int,
    evaluation_replicas: int,
    continuation_horizon: int,
    gate: float = 1.0,
    microbatch_size: int | None = None,
    runtime: Any | None = None,
    chunk_kernel: Callable[..., Any] | None = None,
) -> CounterfactualAnchorBatch:
    """Estimate all-action continuation returns from identical anchor worlds."""

    import jax
    import jax.numpy as jnp

    if action_count <= 1:
        raise ValueError("Counterfactual anchors need at least two actions.")
    if fit_replicas <= 0 or evaluation_replicas <= 0:
        raise ValueError("Both replica splits must be non-empty.")
    if continuation_horizon <= 0:
        raise ValueError("Continuation horizon must be positive.")
    anchor_count = int(jnp.asarray(world.done).shape[0])
    replicas = int(fit_replicas + evaluation_replicas)
    repeats = int(action_count * replicas)
    if microbatch_size is not None:
        limit = int(microbatch_size)
        if limit < repeats:
            raise ValueError(
                "Anchor microbatch must fit every action/replica branch of one world."
            )
        anchors_per_batch = max(limit // repeats, 1)
        padded_count = (
            (anchor_count + anchors_per_batch - 1) // anchors_per_batch
        ) * anchors_per_batch
        padding = padded_count - anchor_count

        def pad_tree(tree: Any) -> Any:
            def pad(value: Any) -> Any:
                array = jnp.asarray(value)
                if not padding:
                    return array
                repeated = jnp.repeat(array[-1:], padding, axis=0)
                return jnp.concatenate((array, repeated), axis=0)

            return jax.tree_util.tree_map(pad, tree)

        padded_ids = jnp.concatenate(
            (
                jnp.asarray(anchor_ids),
                jnp.full((padding,), -1, dtype=jnp.asarray(anchor_ids).dtype),
            ),
            axis=0,
        )
        # All scientific root keys already exist.  Operational padding receives
        # zero keys and is cropped before any return enters a loss or artifact.
        padded_roots = jnp.concatenate(
            (
                jnp.asarray(root_keys),
                jnp.zeros((padding, 2), dtype=jnp.asarray(root_keys).dtype),
            ),
            axis=0,
        )
        padded_world = pad_tree(world)
        padded_indexes = pad_tree(jnp.asarray(rollout_flat_indexes))
        padded_policy = pad_tree(policy_states)
        padded_observations = pad_tree(jnp.asarray(observations))
        padded_codes = pad_tree(jnp.asarray(partner_codes))
        padded_sources = pad_tree(jnp.asarray(partner_sources))
        padded_run_ids = pad_tree(jnp.asarray(partner_run_ids))
        chunks = []
        for start in range(0, padded_count, anchors_per_batch):
            stop = start + anchors_per_batch

            def sliced(tree: Any) -> Any:
                return jax.tree_util.tree_map(
                    lambda value: jnp.asarray(value)[start:stop], tree
                )

            arguments = (
                runtime,
                padded_ids[start:stop],
                padded_roots[start:stop],
                sliced(padded_world),
                padded_indexes[start:stop],
                sliced(padded_policy),
                padded_observations[start:stop],
                padded_codes[start:stop],
                padded_sources[start:stop],
                padded_run_ids[start:stop],
            )
            if chunk_kernel is not None:
                chunks.append(chunk_kernel(*arguments))
            else:
                chunks.append(
                    collect_counterfactual_anchors(
                        anchor_ids=arguments[1],
                        root_keys=arguments[2],
                        world=arguments[3],
                        rollout_flat_indexes=arguments[4],
                        policy_states=arguments[5],
                        observations=arguments[6],
                        partner_codes=arguments[7],
                        partner_sources=arguments[8],
                        partner_run_ids=arguments[9],
                        functions=functions,
                        action_count=action_count,
                        fit_replicas=fit_replicas,
                        evaluation_replicas=evaluation_replicas,
                        continuation_horizon=continuation_horizon,
                        gate=gate,
                        runtime=runtime,
                    )
                )
        combined = jax.tree_util.tree_map(
            lambda *values: jnp.concatenate(values, axis=0), *chunks
        )
        return jax.tree_util.tree_map(lambda value: value[:anchor_count], combined)
    expanded = jax.tree_util.tree_map(
        lambda value: jnp.repeat(jnp.asarray(value), repeats, axis=0), world
    )
    branch_actions = jnp.tile(
        jnp.repeat(jnp.arange(action_count, dtype=jnp.int32), replicas),
        anchor_count,
    )
    branch_replica = jnp.tile(
        jnp.tile(jnp.arange(replicas, dtype=jnp.int32), action_count),
        anchor_count,
    )
    lane_keys = _shared_lane_keys(
        root_keys,
        action_count=action_count,
        replicas=replicas,
        domain=100_003,
    )

    def step_world(
        current: AnchorWorld,
        *,
        forced_action: Any | None,
        time_index: int,
    ) -> AnchorWorld:
        active = ~jnp.asarray(current.done, dtype=jnp.bool_)
        ego_key = jax.vmap(
            lambda key: jax.random.fold_in(key, 10 + time_index)
        )(lane_keys)
        partner_key = jax.vmap(
            lambda key: jax.random.fold_in(key, 20 + time_index)
        )(lane_keys)
        environment_key = jax.vmap(
            lambda key: jax.random.fold_in(key, 30 + time_index)
        )(lane_keys)
        uniform_key = jax.vmap(
            lambda key: jax.random.fold_in(key, 40 + time_index)
        )(lane_keys)
        uniforms = jax.vmap(lambda key: jax.random.uniform(key))(uniform_key)

        lane_indexes = jnp.arange(active.shape[0], dtype=jnp.int32)
        ego_observation = current.observations[
            lane_indexes, current.ego_roles
        ]
        partner_observation = current.observations[
            lane_indexes, 1 - current.ego_roles
        ]
        ego_arguments = (
            current.ego_state,
            ego_observation,
            jnp.full(active.shape, float(gate), dtype=jnp.float32),
            ego_key,
        )
        next_ego_pre, ego_probabilities = (
            functions.ego_policy_step(*ego_arguments)
            if runtime is None
            else functions.ego_policy_step(runtime, *ego_arguments)
        )
        sampled_ego = inverse_cdf_actions(ego_probabilities, uniforms)
        ego_action = sampled_ego if forced_action is None else jnp.asarray(forced_action)
        partner_arguments = (
            current.partner_state,
            partner_observation,
            current.partner_episode_start,
            partner_key,
        )
        partner_action, next_partner_pre, partner_context = (
            functions.partner_policy_step(*partner_arguments)
            if runtime is None
            else functions.partner_policy_step(runtime, *partner_arguments)
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
        ) = functions.environment_step(
            current.environment_state,
            joint,
            environment_key,
        )
        terminal = info["terminal_observations"]
        obs_mask = jnp.asarray(dones).reshape(
            jnp.asarray(dones).shape
            + (1,) * (next_observations[:, 0].ndim - 1)
        )
        ego_terminal = terminal[lane_indexes, current.ego_roles]
        partner_terminal = terminal[lane_indexes, 1 - current.ego_roles]
        ego_next = next_observations[lane_indexes, current.ego_roles]
        partner_next = next_observations[lane_indexes, 1 - current.ego_roles]
        ego_next_for_history = jnp.where(obs_mask, ego_terminal, ego_next)
        partner_next_for_history = jnp.where(
            obs_mask, partner_terminal, partner_next
        )
        next_ego = functions.ego_observe(
            next_ego_pre,
            ego_observation,
            ego_action,
            rewards,
            dones,
            ego_next_for_history,
        )
        observe_arguments = (
            next_partner_pre,
            partner_context,
            partner_observation,
            partner_action,
            rewards,
            dones,
            partner_next_for_history,
        )
        next_partner = (
            functions.partner_observe(*observe_arguments)
            if runtime is None
            else functions.partner_observe(runtime, *observe_arguments)
        )
        candidate = AnchorWorld(
            environment_state=next_environment,
            observations=next_observations,
            ego_state=next_ego,
            partner_state=next_partner,
            partner_episode_start=dones,
            ego_roles=current.ego_roles,
            done=jnp.asarray(dones, dtype=jnp.bool_),
            raw_return=current.raw_return + jnp.where(active, rewards, 0.0),
        )
        return tree_select(active, candidate, current)

    first = step_world(expanded, forced_action=branch_actions, time_index=0)

    def body(index: int, current: AnchorWorld) -> AnchorWorld:
        return step_world(current, forced_action=None, time_index=index)

    final = jax.lax.fori_loop(1, int(continuation_horizon), body, first)
    returns = jnp.asarray(final.raw_return).reshape(
        (anchor_count, action_count, replicas)
    )
    fit = jnp.mean(returns[..., :fit_replicas], axis=-1)
    evaluation = jnp.mean(returns[..., fit_replicas:], axis=-1)
    return CounterfactualAnchorBatch(
        anchor_ids=anchor_ids,
        rollout_flat_indexes=rollout_flat_indexes,
        policy_states=policy_states,
        observations=observations,
        partner_codes=partner_codes,
        partner_sources=partner_sources,
        fit_returns_by_action=fit,
        evaluation_returns_by_action=evaluation,
        partner_run_ids=partner_run_ids,
        action_mask=jnp.ones((anchor_count, action_count), dtype=jnp.bool_),
    )


def centered_fit_signature(batch: CounterfactualAnchorBatch) -> Any:
    import jax.numpy as jnp

    values = jnp.asarray(batch.fit_returns_by_action, dtype=jnp.float32)
    return values - jnp.mean(values, axis=-1, keepdims=True)


def evaluate_selected_actions(
    batch: CounterfactualAnchorBatch,
    selected_actions: Any,
) -> Any:
    import jax.numpy as jnp

    actions = jnp.asarray(selected_actions, dtype=jnp.int32)
    return jnp.take_along_axis(
        jnp.asarray(batch.evaluation_returns_by_action),
        actions[..., None],
        axis=-1,
    )[..., 0]


__all__ = [
    "AnchorFunctions",
    "AnchorWorld",
    "anchor_microbatch_candidates",
    "centered_fit_signature",
    "collect_counterfactual_anchors",
    "evaluate_selected_actions",
    "inverse_cdf_actions",
    "tree_repeat_interleave",
    "tree_select",
]
