"""Common-random-number counterfactual action anchors.

The collector is environment-agnostic.  All callbacks operate on a flattened
batch of ``anchor × action × replica`` lanes.  A label is the detached,
fixed-horizon raw simulator return under a common-random-number continuation.
No learned critic, value function, or endpoint bootstrap can create a label.
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


def select_anchor_microbatch_size(
    *,
    chunk_kernel: Any,
    runtime: Any,
    anchor_ids: Any,
    root_keys: Any,
    world: AnchorWorld,
    rollout_flat_indexes: Any,
    policy_states: Any,
    observations: Any,
    partner_sources: Any,
    partner_members: Any,
    partner_family_ids: Any,
    partner_checkpoint_stages: Any,
    partner_run_ids: Any,
    action_count: int,
    replicas: int,
) -> int:
    """Compile-probe the largest complete-world anchor microbatch.

    Scientific keys are generated before this operational choice.  Lowering
    the returned branch count only changes the number of complete anchor
    worlds dispatched per executable call.  When XLA memory analysis is not
    exposed by the runtime, the selector conservatively accepts one world
    rather than executing an unaccounted simulator preflight.
    """

    import jax
    import jax.numpy as jnp

    count = int(jnp.asarray(anchor_ids).shape[0])
    repeats = int(action_count) * int(replicas)
    if count <= 0 or repeats <= 0:
        raise ValueError("Anchor microbatch preflight dimensions must be positive.")

    def take(tree: Any, size: int) -> Any:
        return jax.tree_util.tree_map(lambda value: jnp.asarray(value)[:size], tree)

    failures: list[str] = []
    for branch_count in anchor_microbatch_candidates(
        maximum_anchor_worlds=count,
        action_count=action_count,
        replicas=replicas,
    ):
        worlds = int(branch_count) // repeats
        try:
            chunk_kernel.executable(
                runtime,
                take(anchor_ids, worlds),
                take(root_keys, worlds),
                take(world, worlds),
                take(rollout_flat_indexes, worlds),
                take(policy_states, worlds),
                take(observations, worlds),
                take(partner_sources, worlds),
                take(partner_members, worlds),
                take(partner_family_ids, worlds),
                take(partner_checkpoint_stages, worlds),
                take(partner_run_ids, worlds),
            )
        except Exception as error:
            if error.__class__.__name__ != "CompiledMemoryLimitError":
                raise
            failures.append(f"{branch_count}:{error}")
            continue
        metadata = {
            item.get("argument_signature"): item
            for item in chunk_kernel.metadata()
        }
        last = metadata.get(chunk_kernel.last_signature, {})
        memory = last.get("memory_analysis", {})
        if memory.get("conservative_device_bytes") is not None or worlds == 1:
            chunk_kernel.selected_microbatch_size = int(branch_count)
            return int(branch_count)
    raise RuntimeError(
        "No registered anchor microbatch fits the device memory limit: "
        + "; ".join(failures)
    )


def collect_counterfactual_anchors(
    *,
    anchor_ids: Any,
    root_keys: Any,
    world: AnchorWorld,
    rollout_flat_indexes: Any,
    policy_states: Any,
    observations: Any,
    partner_sources: Any,
    partner_members: Any,
    partner_family_ids: Any,
    partner_checkpoint_stages: Any,
    partner_run_ids: Any,
    functions: AnchorFunctions,
    action_count: int,
    fit_replicas: int,
    evaluation_replicas: int,
    continuation_horizon: int,
    discount: float = 1.0,
    microbatch_size: int | None = None,
    runtime: Any | None = None,
    chunk_kernel: Callable[..., Any] | None = None,
    collection_policy_logits: Any | None = None,
    collection_update: Any | None = None,
    collection_target_fingerprint: Any | None = None,
    matched_pair_ids: Any | None = None,
) -> CounterfactualAnchorBatch:
    """Estimate all-action continuation returns from identical anchor worlds."""

    import jax
    import jax.numpy as jnp

    if action_count <= 1:
        raise ValueError("Counterfactual anchors need at least two actions.")
    if fit_replicas <= 0 or evaluation_replicas < 0:
        raise ValueError("Fit replicas must be positive and evaluation replicas non-negative.")
    if continuation_horizon <= 0:
        raise ValueError("Continuation horizon must be positive.")
    anchor_count = int(jnp.asarray(world.done).shape[0])
    replicas = int(fit_replicas + evaluation_replicas)
    repeats = int(action_count * replicas)
    if microbatch_size is None and chunk_kernel is not None:
        microbatch_size = select_anchor_microbatch_size(
            chunk_kernel=chunk_kernel,
            runtime=runtime,
            anchor_ids=anchor_ids,
            root_keys=root_keys,
            world=world,
            rollout_flat_indexes=rollout_flat_indexes,
            policy_states=policy_states,
            observations=observations,
            partner_sources=partner_sources,
            partner_members=partner_members,
            partner_family_ids=partner_family_ids,
            partner_checkpoint_stages=partner_checkpoint_stages,
            partner_run_ids=partner_run_ids,
            action_count=action_count,
            replicas=replicas,
        )
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
        if padding:
            # Padding is an operational fixed-shape device lane, not an
            # additional counterfactual world.  Mark it terminal before the
            # continuation kernel so every reward/state update is masked even
            # though XLA still executes the statically shaped simulator body.
            padded_world = padded_world._replace(
                done=jnp.concatenate(
                    (
                        jnp.asarray(world.done, dtype=jnp.bool_),
                        jnp.ones((padding,), dtype=jnp.bool_),
                    ),
                    axis=0,
                )
            )
        padded_indexes = pad_tree(jnp.asarray(rollout_flat_indexes))
        padded_policy = pad_tree(policy_states)
        padded_observations = pad_tree(jnp.asarray(observations))
        padded_sources = pad_tree(jnp.asarray(partner_sources))
        padded_members = pad_tree(jnp.asarray(partner_members))
        padded_family_ids = pad_tree(jnp.asarray(partner_family_ids))
        padded_checkpoint_stages = pad_tree(
            jnp.asarray(partner_checkpoint_stages)
        )
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
                padded_sources[start:stop],
                padded_members[start:stop],
                padded_family_ids[start:stop],
                padded_checkpoint_stages[start:stop],
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
                        partner_sources=arguments[7],
                        partner_members=arguments[8],
                        partner_family_ids=arguments[9],
                        partner_checkpoint_stages=arguments[10],
                        partner_run_ids=arguments[11],
                        functions=functions,
                        action_count=action_count,
                        fit_replicas=fit_replicas,
                        evaluation_replicas=evaluation_replicas,
                        continuation_horizon=continuation_horizon,
                        discount=discount,
                        runtime=runtime,
                        collection_policy_logits=(
                            None
                            if collection_policy_logits is None
                            else jnp.asarray(collection_policy_logits)[start:stop]
                        ),
                        collection_update=collection_update,
                        collection_target_fingerprint=collection_target_fingerprint,
                        matched_pair_ids=(
                            None
                            if matched_pair_ids is None
                            else jnp.asarray(matched_pair_ids)[start:stop]
                        ),
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
        ego_arguments = (current.ego_state, ego_observation, ego_key)
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
        raw_by_agent = info.get("raw_rewards_by_agent")
        ego_rewards = (
            rewards
            if raw_by_agent is None
            else raw_by_agent[lane_indexes, current.ego_roles]
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
            ego_rewards,
            dones,
            ego_next_for_history,
        )
        observe_arguments = (
            next_partner_pre,
            partner_context,
            partner_observation,
            partner_action,
            ego_rewards,
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
            raw_return=current.raw_return
                + jnp.power(
                    jnp.asarray(discount, dtype=jnp.float32),
                    jnp.asarray(time_index, dtype=jnp.float32),
                )
                * jnp.where(active, ego_rewards, 0.0),
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
    fit_returns = returns[..., :fit_replicas]
    fit_sum = jnp.sum(fit_returns, axis=-1)
    fit_squared_sum = jnp.sum(jnp.square(fit_returns), axis=-1)
    evaluation = (
        jnp.mean(returns[..., fit_replicas:], axis=-1)
        if evaluation_replicas > 0
        else jnp.full_like(fit, jnp.nan)
    )
    logits = (
        jnp.zeros((anchor_count, action_count), dtype=jnp.float32)
        if collection_policy_logits is None
        else jnp.asarray(collection_policy_logits, dtype=jnp.float32)
    )
    update = (
        jnp.zeros((anchor_count,), dtype=jnp.int32)
        if collection_update is None
        else jnp.broadcast_to(
            jnp.asarray(collection_update, dtype=jnp.int32), (anchor_count,)
        )
    )
    fingerprint = (
        jnp.zeros((anchor_count, 2), dtype=jnp.uint32)
        if collection_target_fingerprint is None
        else jnp.broadcast_to(
            jnp.asarray(collection_target_fingerprint, dtype=jnp.uint32),
            (anchor_count, 2),
        )
    )
    pair_ids = (
        jnp.full((anchor_count,), -1, dtype=jnp.int32)
        if matched_pair_ids is None
        else jnp.asarray(matched_pair_ids, dtype=jnp.int32)
    )
    return CounterfactualAnchorBatch(
        anchor_ids=anchor_ids,
        rollout_flat_indexes=rollout_flat_indexes,
        policy_states=policy_states,
        observations=observations,
        partner_sources=partner_sources,
        partner_members=partner_members,
        partner_family_ids=partner_family_ids,
        partner_checkpoint_stages=partner_checkpoint_stages,
        fit_returns_by_action=fit,
        return_sum_by_action=fit_sum,
        return_squared_sum_by_action=fit_squared_sum,
        replica_count=jnp.full(
            (anchor_count, action_count), int(fit_replicas), dtype=jnp.int32
        ),
        collection_policy_logits=logits,
        collection_update=update,
        collection_target_fingerprint=fingerprint,
        matched_pair_ids=pair_ids,
        partner_run_ids=partner_run_ids,
        action_mask=jnp.ones((anchor_count, action_count), dtype=jnp.bool_),
        evaluation_returns_by_action=evaluation,
        evaluation_replica_count=jnp.full(
            (anchor_count, action_count),
            int(evaluation_replicas),
            dtype=jnp.int32,
        ),
        fit_replica_returns_by_action=fit_returns,
    )


def centered_fit_signature(batch: CounterfactualAnchorBatch) -> Any:
    import jax.numpy as jnp

    values = jnp.asarray(batch.fit_returns_by_action, dtype=jnp.float32)
    return values - jnp.mean(values, axis=-1, keepdims=True)


__all__ = [
    "AnchorFunctions",
    "AnchorWorld",
    "anchor_microbatch_candidates",
    "select_anchor_microbatch_size",
    "centered_fit_signature",
    "collect_counterfactual_anchors",
    "inverse_cdf_actions",
    "tree_select",
]
