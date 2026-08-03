"""Fixed-budget V6 counterfactual collection.

Training anchors are selected only from rollout time and partner-source strata.
No learned score or post-hoc qualification decision changes the sample budget.
Matched-code worlds are compared only after sixteen legal, observable probe
steps; hidden codes never enter the ego policy.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

from .counterfactual_anchor import (
    AnchorFunctions,
    AnchorWorld,
    advance_anchor_world,
    collect_counterfactual_anchors,
    select_anchor_microbatch_size,
)
from .decision_geometry import centered_action_values, decision_distance
from .runner import observe_policy_after_transition
from .types import QuotientPairBatch


class AnchorRuntime(NamedTuple):
    target_params: Any
    partner_parameters: Any


def gather_time_lanes(tree: Any, indexes: Any) -> Any:
    import jax
    import jax.numpy as jnp

    selected = jnp.asarray(indexes, dtype=jnp.int32)

    def gather(value: Any) -> Any:
        array = jnp.asarray(value)
        flat = array.reshape((array.shape[0] * array.shape[1],) + array.shape[2:])
        return flat[selected]

    return jax.tree_util.tree_map(gather, tree)


def _interleave(left: Any, right: Any) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(
        lambda a, b: jnp.stack((jnp.asarray(a), jnp.asarray(b)), axis=1).reshape(
            (2 * int(jnp.asarray(a).shape[0]),) + jnp.asarray(a).shape[1:]
        ),
        left,
        right,
    )


def _concatenate(left: Any, right: Any) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate((jnp.asarray(a), jnp.asarray(b)), axis=0),
        left,
        right,
    )


def time_source_stratified_indexes(
    key: Any,
    *,
    time_count: int,
    environment_count: int,
    requested: int,
    source_values: Any,
    time_bins: int = 4,
) -> Any:
    """Draw a fixed number without replacement across time/source strata."""

    import jax
    import jax.numpy as jnp
    import numpy as np

    total = int(time_count) * int(environment_count)
    if not 0 < int(requested) <= total:
        raise ValueError("Anchor request is outside the rollout state count.")
    sources = np.asarray(source_values, dtype=np.int64).reshape((-1,))
    if sources.shape != (total,):
        raise ValueError("Partner-source labels do not align with rollout states.")
    times = np.repeat(np.arange(int(time_count)), int(environment_count))
    time_labels = np.minimum(
        (times * int(time_bins)) // max(int(time_count), 1), int(time_bins) - 1
    )
    priorities = np.asarray(
        jax.random.uniform(key, (total,), dtype=jnp.float32), dtype=np.float64
    )
    strata: dict[tuple[int, int], list[int]] = {}
    for index in range(total):
        strata.setdefault((int(time_labels[index]), int(sources[index])), []).append(index)
    quota = max(1, int(np.ceil(float(requested) / max(len(strata), 1))))
    selected: list[int] = []
    used: set[int] = set()
    for label in sorted(strata):
        ranked = sorted(strata[label], key=lambda item: priorities[item], reverse=True)
        for index in ranked[:quota]:
            if index not in used:
                selected.append(index)
                used.add(index)
    if len(selected) < int(requested):
        ranked = sorted(range(total), key=lambda item: priorities[item], reverse=True)
        selected.extend(
            index for index in ranked if index not in used
        )
    return jnp.asarray(selected[: int(requested)], dtype=jnp.int32)


def _generator_intervention_state(state: Any, code: Any) -> Any:
    import jax
    import jax.numpy as jnp

    required = {"source", "current_carry", "target_carry", "code"}
    if not required.issubset(set(getattr(state, "_fields", ()))):
        raise TypeError("Partner runtime does not support continuous-code intervention.")
    count = int(jnp.asarray(code).shape[0])
    return state._replace(
        source=jnp.zeros((count,), dtype=jnp.int32),
        current_carry=jax.tree_util.tree_map(jnp.zeros_like, state.current_carry),
        target_carry=jax.tree_util.tree_map(jnp.zeros_like, state.target_carry),
        code=jnp.asarray(code, dtype=jnp.float32),
    )


def make_anchor_functions(
    *, model: Any, model_config: Any, partner_functions: Any, environment: Any
) -> AnchorFunctions:
    """Create static callbacks with all checkpoint values supplied at runtime."""

    def ego_policy_step(
        runtime: AnchorRuntime, state: Any, observation: Any, keys: Any
    ) -> tuple[Any, Any]:
        import jax
        import jax.numpy as jnp

        del keys
        dropped = jnp.zeros((jnp.asarray(observation).shape[0],), dtype=jnp.bool_)
        next_state, output = model.apply(
            {"params": runtime.target_params},
            state,
            observation,
            dropped,
            method=model.step,
        )
        return next_state, jax.nn.softmax(output.policy_logits, axis=-1)

    def ego_observe(
        stepped: Any,
        observation: Any,
        action: Any,
        reward: Any,
        done: Any,
        next_observation: Any,
    ) -> Any:
        del observation
        return observe_policy_after_transition(
            stepped_state=stepped,
            action=action,
            reward=reward,
            done=done,
            next_observation=next_observation,
            model_config=model_config,
        )

    def ego_endpoint_value(runtime: AnchorRuntime, state: Any, observation: Any) -> Any:
        import jax
        import jax.numpy as jnp

        dropped = jnp.zeros((jnp.asarray(observation).shape[0],), dtype=jnp.bool_)
        _, output = model.apply(
            {"params": runtime.target_params},
            state,
            observation,
            dropped,
            method=model.step,
        )
        probabilities = jax.nn.softmax(output.policy_logits, axis=-1)
        conservative_q = jnp.minimum(output.raw_q1, output.raw_q2)
        return jnp.sum(probabilities * conservative_q, axis=-1)

    def partner_policy_step(
        runtime: AnchorRuntime,
        state: Any,
        observation: Any,
        episode_start: Any,
        keys: Any,
    ) -> tuple[Any, Any, Any]:
        action, next_state, context, unused_log_probability = partner_functions.step(
            runtime.partner_parameters, state, observation, episode_start, keys
        )
        del unused_log_probability
        return action, next_state, context

    def partner_observe(
        runtime: AnchorRuntime,
        state: Any,
        context: Any,
        observation: Any,
        action: Any,
        reward: Any,
        done: Any,
        next_observation: Any,
    ) -> Any:
        return partner_functions.observe(
            runtime.partner_parameters,
            state,
            context,
            observation,
            action,
            reward,
            done,
            next_observation,
        )

    return AnchorFunctions(
        ego_policy_step=ego_policy_step,
        ego_observe=ego_observe,
        partner_policy_step=partner_policy_step,
        partner_observe=partner_observe,
        environment_step=environment.step_with_keys,
        ego_endpoint_value=ego_endpoint_value,
    )


def world_from_records(records: Mapping[str, Any], indexes: Any) -> AnchorWorld:
    import jax.numpy as jnp

    policy_state = gather_time_lanes(records["target_ego_policy_state"], indexes)
    count = int(jnp.asarray(indexes).shape[0])
    return AnchorWorld(
        environment_state=gather_time_lanes(records["environment_state"], indexes),
        observations=gather_time_lanes(records["joint_observations"], indexes),
        ego_state=policy_state,
        partner_state=gather_time_lanes(records["partner_state"], indexes),
        partner_episode_start=policy_state.episode_start,
        ego_roles=gather_time_lanes(records["ego_roles"], indexes),
        done=jnp.zeros((count,), dtype=jnp.bool_),
        raw_return=jnp.zeros((count,), dtype=jnp.float32),
    )


def _attach_metadata(
    batch: Any,
    *,
    logits: Any,
    update: int,
    target_fingerprint: Any,
    matched_pair_ids: Any,
) -> Any:
    import jax.numpy as jnp

    count = int(jnp.asarray(batch.anchor_ids).shape[0])
    return batch._replace(
        collection_policy_logits=jnp.asarray(logits, dtype=jnp.float32),
        collection_update=jnp.full((count,), int(update), dtype=jnp.int32),
        collection_target_fingerprint=jnp.broadcast_to(
            jnp.asarray(target_fingerprint, dtype=jnp.uint32), (count, 2)
        ),
        matched_pair_ids=jnp.asarray(matched_pair_ids, dtype=jnp.int32),
    )


def collect_anchor_batch(
    *,
    anchor_domain: int,
    key: Any,
    records: Mapping[str, Any],
    environment: Any,
    model: Any,
    target_params: Any,
    config: Any,
    partner_functions: Any,
    partner_parameters: Any,
    collection_update: int,
    target_fingerprint: Any,
    microbatch_size: int | None = None,
    anchor_functions: AnchorFunctions | None = None,
    chunk_kernel: Any | None = None,
) -> tuple[Any, QuotientPairBatch, Mapping[str, int]]:
    """Collect exactly 32 ordinary and 16 post-evidence matched pairs."""

    import jax
    import jax.numpy as jnp
    import numpy as np

    time_count, environment_count = records["actions"].shape
    ordinary_key, matched_key, return_key, code_key, probe_key = jax.random.split(key, 5)
    ordinary_count = int(config.anchors.ordinary_states)
    pair_count = int(config.anchors.matched_code_pairs)
    ordinary_indexes = time_source_stratified_indexes(
        ordinary_key,
        time_count=time_count,
        environment_count=environment_count,
        requested=ordinary_count,
        source_values=records["partner_source"],
    )
    matched_indexes = time_source_stratified_indexes(
        matched_key,
        time_count=time_count,
        environment_count=environment_count,
        requested=pair_count,
        source_values=records["partner_source"],
    )
    functions = anchor_functions or make_anchor_functions(
        model=model,
        model_config=config.model,
        partner_functions=partner_functions,
        environment=environment,
    )
    runtime = AnchorRuntime(target_params, partner_parameters)

    ordinary_world = world_from_records(records, ordinary_indexes)
    ordinary_roots, matched_roots = jax.random.split(return_key)
    ordinary_root_keys = jax.random.split(ordinary_roots, ordinary_count)
    ordinary = collect_counterfactual_anchors(
        anchor_ids=(
            jnp.asarray(anchor_domain, dtype=jnp.int64) * 1_000_000
            + jnp.arange(ordinary_count, dtype=jnp.int64)
        ),
        root_keys=ordinary_root_keys,
        world=ordinary_world,
        rollout_flat_indexes=ordinary_indexes,
        policy_states=ordinary_world.ego_state,
        observations=gather_time_lanes(records["observations"], ordinary_indexes),
        partner_codes=gather_time_lanes(records["partner_code"], ordinary_indexes),
        partner_sources=gather_time_lanes(records["partner_source"], ordinary_indexes),
        partner_run_ids=gather_time_lanes(records["partner_run_ids"], ordinary_indexes),
        functions=functions,
        action_count=6,
        fit_replicas=int(config.anchors.fit_replicas),
        evaluation_replicas=0,
        continuation_horizon=int(config.anchors.continuation_horizon),
        discount=config.ppo.gamma,
        microbatch_size=microbatch_size,
        runtime=runtime,
        chunk_kernel=chunk_kernel,
    )
    ordinary_observations = gather_time_lanes(records["observations"], ordinary_indexes)
    _, ordinary_target_output = model.apply(
        {"params": target_params},
        ordinary_world.ego_state,
        ordinary_observations,
        jnp.zeros((ordinary_count,), dtype=jnp.bool_),
        method=model.step,
    )
    ordinary = _attach_metadata(
        ordinary,
        logits=ordinary_target_output.policy_logits,
        update=collection_update,
        target_fingerprint=target_fingerprint,
        matched_pair_ids=jnp.full((ordinary_count,), -1, dtype=jnp.int32),
    )

    base_world = world_from_records(records, matched_indexes)
    from .partner_generator import sample_partner_codes

    code_a_key, code_b_key = jax.random.split(code_key)
    code_a = sample_partner_codes(
        code_a_key, batch_size=pair_count, code_dim=config.partner_generator.code_dim
    )
    code_b = sample_partner_codes(
        code_b_key, batch_size=pair_count, code_dim=config.partner_generator.code_dim
    )
    code_b = jnp.where(
        (jnp.linalg.norm(code_a - code_b, axis=-1) < 1.0e-4)[:, None],
        -code_a,
        code_b,
    )
    state_a = _generator_intervention_state(base_world.partner_state, code_a)
    state_b = _generator_intervention_state(base_world.partner_state, code_b)
    paired_world = AnchorWorld(
        environment_state=_interleave(
            base_world.environment_state, base_world.environment_state
        ),
        observations=_interleave(base_world.observations, base_world.observations),
        ego_state=_interleave(base_world.ego_state, base_world.ego_state),
        partner_state=_interleave(state_a, state_b),
        partner_episode_start=jnp.ones((2 * pair_count,), dtype=jnp.bool_),
        ego_roles=_interleave(base_world.ego_roles, base_world.ego_roles),
        done=jnp.zeros((2 * pair_count,), dtype=jnp.bool_),
        raw_return=jnp.zeros((2 * pair_count,), dtype=jnp.float32),
    )
    pair_root_keys = jax.random.split(probe_key, pair_count)
    paired_roots = jnp.repeat(pair_root_keys, 2, axis=0)
    paired_world = advance_anchor_world(
        world=paired_world,
        root_keys=paired_roots,
        functions=functions,
        steps=int(config.anchors.probe_steps),
        runtime=runtime,
        domain=80_003,
    )
    probe_valid = ~jnp.asarray(paired_world.done, dtype=jnp.bool_)
    paired_world = paired_world._replace(
        raw_return=jnp.zeros((2 * pair_count,), dtype=jnp.float32)
    )
    lanes = jnp.arange(2 * pair_count, dtype=jnp.int32)
    paired_observations = paired_world.observations[lanes, paired_world.ego_roles]
    matched = collect_counterfactual_anchors(
        anchor_ids=(
            jnp.asarray(anchor_domain, dtype=jnp.int64) * 1_000_000
            + 100_000
            + jnp.arange(2 * pair_count, dtype=jnp.int64)
        ),
        root_keys=jnp.repeat(jax.random.split(matched_roots, pair_count), 2, axis=0),
        world=paired_world,
        rollout_flat_indexes=jnp.repeat(matched_indexes, 2),
        policy_states=paired_world.ego_state,
        observations=paired_observations,
        partner_codes=_interleave(code_a, code_b),
        partner_sources=jnp.zeros((2 * pair_count,), dtype=jnp.int32),
        partner_run_ids=jnp.full((2 * pair_count,), -2, dtype=jnp.int32),
        functions=functions,
        action_count=6,
        fit_replicas=int(config.anchors.fit_replicas),
        evaluation_replicas=0,
        continuation_horizon=int(config.anchors.continuation_horizon),
        discount=config.ppo.gamma,
        microbatch_size=microbatch_size,
        runtime=runtime,
        chunk_kernel=chunk_kernel,
    )
    dropped = jnp.zeros((2 * pair_count,), dtype=jnp.bool_)
    _, matched_output = model.apply(
        {"params": target_params},
        paired_world.ego_state,
        paired_observations,
        dropped,
        method=model.step,
    )
    pair_ids = jnp.repeat(jnp.arange(pair_count, dtype=jnp.int32), 2)
    matched = matched._replace(
        action_mask=matched.action_mask & probe_valid[:, None]
    )
    matched = _attach_metadata(
        matched,
        logits=matched_output.policy_logits,
        update=collection_update,
        target_fingerprint=target_fingerprint,
        matched_pair_ids=pair_ids,
    )
    combined = _concatenate(ordinary, matched)
    offset = ordinary_count
    left = offset + 2 * jnp.arange(pair_count, dtype=jnp.int32)
    right = left + 1
    signatures = centered_action_values(combined.fit_returns_by_action)
    quotient = QuotientPairBatch(
        anchor_index_a=left,
        anchor_index_b=right,
        decision_distance=decision_distance(signatures[left], signatures[right]),
        weights=jnp.ones((pair_count,), dtype=jnp.float32),
    )
    budget = {
        "counterfactual_continuation_steps": (
            (ordinary_count + 2 * pair_count)
            * 6
            * int(config.anchors.fit_replicas)
            * int(config.anchors.continuation_horizon)
        ),
        "matched_code_probe_steps": pair_count * 2 * int(config.anchors.probe_steps),
        "invalid_matched_probe_states": int(
            np.sum(~np.asarray(probe_valid, dtype=bool))
        ),
    }
    return combined, quotient, budget


def preflight_anchor_microbatch_from_records(
    *,
    records: Mapping[str, Any],
    target_params: Any,
    partner_parameters: Any,
    config: Any,
    chunk_kernel: Any,
    key: Any,
) -> int:
    import jax
    import jax.numpy as jnp

    world_count = int(config.anchors.ordinary_states) + 2 * int(
        config.anchors.matched_code_pairs
    )
    indexes = jnp.zeros((world_count,), dtype=jnp.int32)
    world = world_from_records(records, indexes)
    return select_anchor_microbatch_size(
        chunk_kernel=chunk_kernel,
        runtime=AnchorRuntime(target_params, partner_parameters),
        anchor_ids=jnp.arange(world_count, dtype=jnp.int64),
        root_keys=jax.random.split(key, world_count),
        world=world,
        rollout_flat_indexes=indexes,
        policy_states=world.ego_state,
        observations=gather_time_lanes(records["observations"], indexes),
        partner_codes=gather_time_lanes(records["partner_code"], indexes),
        partner_sources=gather_time_lanes(records["partner_source"], indexes),
        partner_run_ids=gather_time_lanes(records["partner_run_ids"], indexes),
        action_count=6,
        replicas=int(config.anchors.fit_replicas),
    )


def anchor_budget_for_updates(total_updates: int, interval: int = 16) -> Mapping[str, int]:
    """Return the exact fixed V6 training-anchor attempted transition ledger."""

    triggers = (int(total_updates) - 1) // int(interval) + 1
    return {
        "triggers": triggers,
        "counterfactual_continuation_steps": triggers * 196_608,
        "matched_code_probe_steps": triggers * 512,
    }


__all__ = [
    "AnchorRuntime",
    "anchor_budget_for_updates",
    "collect_anchor_batch",
    "gather_time_lanes",
    "make_anchor_functions",
    "preflight_anchor_microbatch_from_records",
    "time_source_stratified_indexes",
    "world_from_records",
]
