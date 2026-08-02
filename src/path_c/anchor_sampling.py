"""Selection and collection of real-return counterfactual action anchors."""

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


class AnchorEgoState(NamedTuple):
    """Legal online state for one frozen TargetPolicyEpoch continuation."""

    policy_state: Any
    partner_code: Any
    partner_source: Any


class AnchorRuntime(NamedTuple):
    """Dynamic parameters passed to a cached continuation executable."""

    target_params: Any
    partner_parameters: Any


def anchor_preflight_world_count(config: Any, *, mode: str) -> int:
    """Return the largest real world batch compiled by one r3 anchor mode."""

    if mode == "training":
        return int(config.anchors.selected_ordinary) + 2 * int(
            config.anchors.selected_matched_code
        )
    if mode == "audit":
        return max(
            int(config.anchors.audit_ordinary_states),
            2 * int(config.anchors.audit_matched_code_states),
        )
    raise ValueError("Anchor preflight mode must be training or audit.")


def preflight_anchor_microbatch_from_records(
    *,
    records: Mapping[str, Any],
    target_params: Any,
    partner_parameters: Any,
    config: Any,
    chunk_kernel: Any,
    mode: str,
    key: Any,
) -> int:
    """Compile-probe a formal anchor kernel without executing transitions.

    One legal rollout snapshot is repeated only to establish the registered
    static shapes. ``CompiledCallable.executable`` lowers and memory-analyzes
    each candidate but never runs it, so this operational preflight consumes
    no simulator sample and cannot enter any training or audit statistic.
    """

    import jax
    import jax.numpy as jnp

    world_count = anchor_preflight_world_count(config, mode=mode)
    indexes = jnp.zeros((world_count,), dtype=jnp.int32)
    environment_state = gather_time_lanes(records["environment_state"], indexes)
    observations = gather_time_lanes(records["joint_observations"], indexes)
    live_ego_state = gather_time_lanes(records["ego_policy_state"], indexes)
    target_ego_state = gather_time_lanes(
        records["target_ego_policy_state"], indexes
    )
    partner_state = gather_time_lanes(records["partner_state"], indexes)
    partner_codes = gather_time_lanes(records["partner_code"], indexes)
    partner_sources = gather_time_lanes(records["partner_source"], indexes)
    partner_run_ids = gather_time_lanes(records["partner_run_ids"], indexes)
    ego_roles = gather_time_lanes(records["ego_roles"], indexes)
    anchor_ego_state = AnchorEgoState(
        policy_state=target_ego_state,
        partner_code=partner_codes,
        partner_source=partner_sources,
    )
    world = AnchorWorld(
        environment_state=environment_state,
        observations=observations,
        ego_state=anchor_ego_state,
        partner_state=partner_state,
        partner_episode_start=target_ego_state.episode_start,
        ego_roles=ego_roles,
        done=jnp.zeros((world_count,), dtype=jnp.bool_),
        raw_return=jnp.zeros((world_count,), dtype=jnp.float32),
    )
    if mode == "training":
        replicas = int(config.anchors.fit_replicas)
    else:
        replicas = int(config.anchors.audit_fit_replicas) + int(
            config.anchors.audit_evaluation_replicas
        )
    return select_anchor_microbatch_size(
        chunk_kernel=chunk_kernel,
        runtime=AnchorRuntime(
            target_params=target_params,
            partner_parameters=partner_parameters,
        ),
        anchor_ids=jnp.arange(world_count, dtype=jnp.int64),
        root_keys=jax.random.split(key, world_count),
        world=world,
        rollout_flat_indexes=indexes,
        policy_states=live_ego_state,
        observations=gather_time_lanes(records["observations"], indexes),
        partner_codes=partner_codes,
        partner_sources=partner_sources,
        partner_run_ids=partner_run_ids,
        action_count=6,
        replicas=replicas,
    )


def make_anchor_functions(
    *,
    model: Any,
    model_config: Any,
    partner_functions: Any,
    environment: Any,
) -> AnchorFunctions:
    """Create callbacks that close only static code, never checkpoint values."""

    def ego_policy_step(
        runtime: AnchorRuntime,
        state: AnchorEgoState,
        observation: Any,
        gate: Any,
        keys: Any,
    ) -> tuple[AnchorEgoState, Any]:
        import jax
        import jax.numpy as jnp

        del keys
        next_policy_state, online = model.apply(
            {"params": runtime.target_params},
            state.policy_state,
            observation,
            jnp.asarray(gate, dtype=jnp.float32),
            method=model.step,
        )
        next_state = state._replace(
            policy_state=next_policy_state,
        )
        return next_state, jax.nn.softmax(online.execution_logits, axis=-1)

    def ego_observe(
        stepped: AnchorEgoState,
        observations_at_time: Any,
        actions: Any,
        rewards: Any,
        dones: Any,
        next_observations: Any,
    ) -> AnchorEgoState:
        del observations_at_time
        policy_state = observe_policy_after_transition(
            stepped_state=stepped.policy_state,
            action=actions,
            reward=rewards,
            done=dones,
            next_observation=next_observations,
            model_config=model_config,
        )
        return stepped._replace(policy_state=policy_state)

    def ego_endpoint_value(
        runtime: AnchorRuntime,
        state: AnchorEgoState,
        observation: Any,
        gate: Any,
    ) -> Any:
        import jax
        import jax.numpy as jnp

        unused_state, output = model.apply(
            {"params": runtime.target_params},
            state.policy_state,
            observation,
            jnp.asarray(gate, dtype=jnp.float32),
            method=model.step,
        )
        del unused_state
        probabilities = jax.nn.softmax(output.execution_logits, axis=-1)
        return jnp.sum(probabilities * output.action_values, axis=-1)

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
        observations_at_time: Any,
        actions: Any,
        rewards: Any,
        dones: Any,
        next_observations: Any,
    ) -> Any:
        return partner_functions.observe(
            runtime.partner_parameters,
            state,
            context,
            observations_at_time,
            actions,
            rewards,
            dones,
            next_observations,
        )

    return AnchorFunctions(
        ego_policy_step=ego_policy_step,
        ego_observe=ego_observe,
        partner_policy_step=partner_policy_step,
        partner_observe=partner_observe,
        environment_step=environment.step_with_keys,
        ego_endpoint_value=ego_endpoint_value,
    )


def gather_time_lanes(tree: Any, indexes: Any) -> Any:
    import jax
    import jax.numpy as jnp

    index = jnp.asarray(indexes, dtype=jnp.int32)

    def one(value: Any) -> Any:
        array = jnp.asarray(value)
        flat = array.reshape((array.shape[0] * array.shape[1],) + array.shape[2:])
        return flat[index]

    return jax.tree_util.tree_map(one, tree)


def stratified_anchor_indexes(
    key: Any,
    *,
    time_count: int,
    environment_count: int,
    requested: int,
    time_bins: int,
    regret_bins: int,
    valid_mask: Any,
    regret_values: Any,
    source_values: Any,
) -> Any:
    """Select anchors jointly stratified by time, regret, and source.

    Selection uses no future return.  Random priorities are generated from the
    registered JAX key, while the balancing logic runs on host because anchor
    collection is already a non-jitted simulator branching operation.
    """

    import jax
    import jax.numpy as jnp
    import numpy as np

    total = int(time_count) * int(environment_count)
    if requested <= 0 or requested > total:
        raise ValueError("Requested anchor count is outside rollout states.")
    if time_bins <= 0 or regret_bins <= 0:
        raise ValueError("Anchor stratum counts must be positive.")
    valid = np.asarray(valid_mask, dtype=np.bool_).reshape((-1,))
    regrets = np.asarray(regret_values, dtype=np.float64).reshape((-1,))
    sources = np.asarray(source_values, dtype=np.int64).reshape((-1,))
    if not (valid.shape == regrets.shape == sources.shape == (total,)):
        raise ValueError("Anchor stratification columns do not align.")
    valid_indexes = np.flatnonzero(valid & np.isfinite(regrets))
    if valid_indexes.size < int(requested):
        raise ValueError(
            "Not enough valid rollout states for the registered anchor count."
        )
    priorities = np.asarray(
        jax.random.uniform(key, (total,), dtype=jnp.float32), dtype=np.float64
    )
    time_index = np.repeat(np.arange(time_count), environment_count)
    time_label = np.minimum(
        (time_index * int(time_bins)) // max(int(time_count), 1),
        int(time_bins) - 1,
    )
    valid_regrets = regrets[valid_indexes]
    if int(regret_bins) == 1 or np.allclose(valid_regrets, valid_regrets[0]):
        regret_label = np.zeros((total,), dtype=np.int64)
    else:
        quantiles = np.quantile(
            valid_regrets,
            np.linspace(0.0, 1.0, int(regret_bins) + 1)[1:-1],
            method="linear",
        )
        regret_label = np.searchsorted(quantiles, regrets, side="right")
        regret_label = np.minimum(regret_label, int(regret_bins) - 1)

    strata: dict[tuple[int, int, int], list[int]] = {}
    for index in valid_indexes.tolist():
        label = (
            int(time_label[index]),
            int(regret_label[index]),
            int(sources[index]),
        )
        strata.setdefault(label, []).append(index)
    quota = max(1, int(np.ceil(float(requested) / max(len(strata), 1))))
    selected: list[int] = []
    selected_set: set[int] = set()
    for label in sorted(strata):
        candidates = sorted(
            strata[label], key=lambda index: priorities[index], reverse=True
        )
        for index in candidates[:quota]:
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)
    if len(selected) < int(requested):
        remaining = sorted(
            (index for index in valid_indexes.tolist() if index not in selected_set),
            key=lambda index: priorities[index],
            reverse=True,
        )
        selected.extend(remaining[: int(requested) - len(selected)])
    selected = selected[: int(requested)]
    if len(selected) != int(requested):
        raise AssertionError("Anchor stratification did not produce the fixed size.")
    return jnp.asarray(selected, dtype=jnp.int32)

def make_quotient_pairs(
    anchors: Any,
    *,
    left_indexes: Any,
    right_indexes: Any,
) -> QuotientPairBatch:
    """Build exact matched-state pairs from partner-code interventions.

    The two index sets must refer to anchors that share the same physical state,
    ego legal history, future random primitives, and forced ego-action branches;
    only the continuous partner-generator code differs.
    """

    import jax.numpy as jnp

    left = jnp.asarray(left_indexes, dtype=jnp.int32)
    right = jnp.asarray(right_indexes, dtype=jnp.int32)
    if left.shape != right.shape or left.ndim != 1 or left.shape[0] == 0:
        raise ValueError("Matched quotient pairs must be non-empty aligned vectors.")
    signatures = centered_action_values(anchors.fit_returns_by_action)
    return QuotientPairBatch(
        anchor_index_a=left,
        anchor_index_b=right,
        decision_distance=decision_distance(signatures[left], signatures[right]),
        weights=jnp.ones(left.shape, dtype=jnp.float32),
    )


def _concatenate_anchor_batches(left: Any, right: Any) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate((jnp.asarray(a), jnp.asarray(b)), axis=0),
        left,
        right,
    )


def _generator_intervention_state(state: Any, code: Any) -> Any:
    """Switch a mixed partner state to the current generator at a fixed world."""

    import jax
    import jax.numpy as jnp

    required = {
        "source",
        "generator_carry",
        "code",
    }
    fields = set(getattr(state, "_fields", ()))
    if not required.issubset(fields):
        raise TypeError("Partner state does not support generator-code intervention.")
    count = int(jnp.asarray(code).shape[0])
    zeros = jax.tree_util.tree_map(jnp.zeros_like, state.generator_carry)
    return state._replace(
        source=jnp.zeros((count,), dtype=jnp.int32),
        generator_carry=zeros,
        code=jnp.asarray(code, dtype=jnp.float32),
    )


def _external_intervention_state(state: Any, member_indexes: Any) -> Any:
    """Switch a mixed state to one frozen development partner at common world."""

    import jax
    import jax.numpy as jnp

    required = {"source", "external_carry", "external_member"}
    if not required.issubset(set(getattr(state, "_fields", ()))):
        raise TypeError("Partner state does not support external-run intervention.")
    members = jnp.asarray(member_indexes, dtype=jnp.int32)
    return state._replace(
        source=jnp.full(members.shape, 2, dtype=jnp.int32),
        external_carry=jax.tree_util.tree_map(jnp.zeros_like, state.external_carry),
        external_member=members,
    )



def _validate_anchor_return_bounds(anchors: Any, *, lower: float, upper: float) -> None:
    """Fail closed when preregistered bounded-return assumptions are violated."""

    import numpy as np

    fit = np.asarray(anchors.fit_returns_by_action, dtype=np.float64)
    evaluation = np.asarray(anchors.evaluation_returns_by_action, dtype=np.float64)
    if not np.isfinite(fit).all():
        raise FloatingPointError("Counterfactual anchor returns are non-finite.")
    finite_evaluation = evaluation[np.isfinite(evaluation)]
    observed_lower = float(
        min(fit.min(), finite_evaluation.min())
        if finite_evaluation.size
        else fit.min()
    )
    observed_upper = float(
        max(fit.max(), finite_evaluation.max())
        if finite_evaluation.size
        else fit.max()
    )
    if observed_lower < float(lower) or observed_upper > float(upper):
        raise ValueError(
            "Observed simulator continuation return violates the frozen bounds: "
            f"observed=[{observed_lower}, {observed_upper}], "
            f"registered=[{float(lower)}, {float(upper)}]."
        )


def _collect_anchor_batch_at_indexes(
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
    enable_quotient_interventions: bool = False,
    microbatch_size: int | None = None,
    anchor_functions: AnchorFunctions | None = None,
    chunk_kernel: Any | None = None,
    preselected_indexes: Any | None = None,
    ordinary_count: int | None = None,
    matched_pair_count: int | None = None,
    fit_replicas: int | None = None,
    evaluation_replicas: int | None = None,
    continuation_horizon: int | None = None,
    probe_steps: int | None = None,
    collect_ordinary: bool = True,
) -> tuple[Any, Any, Any, Any]:
    """Collect split-replica all-action targets from rollout snapshots.

    Returns anchors, quotient pairs, selected generator codes, and flattened
    rollout indexes.  Ground truth is simulator raw return only.
    """

    import jax
    import jax.numpy as jnp

    time_count, environment_count = records["actions"].shape
    valid_candidates = jnp.ones(records["actions"].shape, dtype=jnp.bool_)
    selection_key, continuation_key = jax.random.split(key)
    requested = int(
        ordinary_count
        if ordinary_count is not None
        else config.anchors.states_per_interval
    )
    indexes = (
        jnp.asarray(preselected_indexes, dtype=jnp.int32)
        if preselected_indexes is not None
        else stratified_anchor_indexes(
            selection_key,
            time_count=time_count,
            environment_count=environment_count,
            requested=requested,
            time_bins=config.anchors.sampling_time_bins,
            regret_bins=config.anchors.sampling_regret_bins,
            valid_mask=valid_candidates,
            regret_values=records.get(
                "decision_regret", jnp.zeros(records["actions"].shape, dtype=jnp.float32)
            ),
            source_values=records["partner_source"],
        )
    )
    if indexes.ndim != 1 or int(indexes.shape[0]) != requested:
        raise ValueError("Preselected anchor indexes do not match the registered count.")
    environment_state = gather_time_lanes(records["environment_state"], indexes)
    observations = gather_time_lanes(records["joint_observations"], indexes)
    live_ego_state = gather_time_lanes(records["ego_policy_state"], indexes)
    ego_state = gather_time_lanes(records["target_ego_policy_state"], indexes)
    partner_state = gather_time_lanes(records["partner_state"], indexes)
    partner_codes = gather_time_lanes(records["partner_code"], indexes)
    partner_sources = gather_time_lanes(records["partner_source"], indexes)
    partner_run_ids = gather_time_lanes(records["partner_run_ids"], indexes)
    ego_roles = gather_time_lanes(records["ego_roles"], indexes)
    anchor_count = int(indexes.shape[0])
    anchor_ego_state = AnchorEgoState(
        policy_state=ego_state,
        partner_code=partner_codes,
        partner_source=partner_sources,
    )
    world = AnchorWorld(
        environment_state=environment_state,
        observations=observations,
        ego_state=anchor_ego_state,
        partner_state=partner_state,
        partner_episode_start=ego_state.episode_start,
        ego_roles=ego_roles,
        done=jnp.zeros((anchor_count,), dtype=jnp.bool_),
        raw_return=jnp.zeros((anchor_count,), dtype=jnp.float32),
    )

    functions = anchor_functions or make_anchor_functions(
        model=model,
        model_config=config.model,
        partner_functions=partner_functions,
        environment=environment,
    )
    runtime = AnchorRuntime(
        target_params=target_params,
        partner_parameters=partner_parameters,
    )

    root_keys = jax.random.split(continuation_key, anchor_count)
    registered_fit = int(fit_replicas or config.anchors.fit_replicas)
    registered_evaluation = int(
        config.anchors.evaluation_replicas
        if evaluation_replicas is None
        else evaluation_replicas
    )
    registered_horizon = int(
        continuation_horizon or config.anchors.continuation_horizon
    )
    anchors = (
        collect_counterfactual_anchors(
            anchor_ids=(
                jnp.asarray(anchor_domain, dtype=jnp.int64) * 1_000_000
                + jnp.arange(anchor_count, dtype=jnp.int64)
            ),
            root_keys=root_keys,
            world=world,
            rollout_flat_indexes=indexes,
            policy_states=live_ego_state,
            observations=gather_time_lanes(records["observations"], indexes),
            partner_codes=partner_codes,
            partner_sources=partner_sources,
            partner_run_ids=partner_run_ids,
            functions=functions,
            action_count=6,
            fit_replicas=registered_fit,
            evaluation_replicas=registered_evaluation,
            continuation_horizon=registered_horizon,
            discount=config.ppo.gamma,
            microbatch_size=microbatch_size,
            runtime=runtime,
            chunk_kernel=chunk_kernel,
        )
        if collect_ordinary
        else None
    )
    base_codes = gather_time_lanes(records["partner_code"], indexes)
    base_sources = gather_time_lanes(records["partner_source"], indexes)
    codes = jnp.where(
        (base_sources == 2)[..., None],
        jnp.full_like(base_codes, jnp.nan),
        base_codes,
    )
    quotient = None

    # Exact matched-state quotient supervision.  For a bounded subset of anchor
    # worlds, intervene on the hidden continuous partner code while holding the
    # environment, ego legal history, forced action, and future random streams
    # fixed.  This avoids the confounding created by pairing unrelated task
    # states merely because their trajectories look similar.
    state_fields = set(getattr(partner_state, "_fields", ()))
    can_intervene = {"source", "generator_carry", "code"}.issubset(state_fields)
    pair_count = min(
        anchor_count,
        int(
            config.partner_generator.codes_per_update
            if matched_pair_count is None
            else matched_pair_count
        ),
    )
    if enable_quotient_interventions and can_intervene and pair_count > 0:
        code_a_key, code_b_key, pair_key = jax.random.split(
            jax.random.fold_in(continuation_key, 91_337), 3
        )
        from .partner_generator import sample_partner_codes

        code_a = sample_partner_codes(
            code_a_key,
            batch_size=pair_count,
            code_dim=config.partner_generator.code_dim,
        )
        code_b = sample_partner_codes(
            code_b_key,
            batch_size=pair_count,
            code_dim=config.partner_generator.code_dim,
        )
        # Avoid identical antithetic samples without introducing a discrete type.
        code_b = jnp.where(
            (jnp.linalg.norm(code_a - code_b, axis=-1) < 1.0e-4)[..., None],
            -code_a,
            code_b,
        )
        pair_environment = jax.tree_util.tree_map(
            lambda value: value[:pair_count], environment_state
        )
        pair_observations = observations[:pair_count]
        pair_ego_state = jax.tree_util.tree_map(
            lambda value: value[:pair_count], ego_state
        )
        pair_partner_state = jax.tree_util.tree_map(
            lambda value: value[:pair_count], partner_state
        )
        pair_ego_roles = ego_roles[:pair_count]
        state_a = _generator_intervention_state(pair_partner_state, code_a)
        state_b = _generator_intervention_state(pair_partner_state, code_b)
        variant_policy_state = jax.tree_util.tree_map(
            lambda value: jnp.concatenate((value, value), axis=0),
            pair_ego_state,
        )
        variant_ego_state = AnchorEgoState(
            policy_state=variant_policy_state,
            partner_code=jnp.concatenate((code_a, code_b), axis=0),
            partner_source=jnp.zeros((2 * pair_count,), dtype=jnp.int32),
        )
        variant_world = AnchorWorld(
            environment_state=jax.tree_util.tree_map(
                lambda value: jnp.concatenate((value, value), axis=0),
                pair_environment,
            ),
            observations=jnp.concatenate(
                (pair_observations, pair_observations), axis=0
            ),
            ego_state=variant_ego_state,
            partner_state=jax.tree_util.tree_map(
                lambda a, b: jnp.concatenate((a, b), axis=0), state_a, state_b
            ),
            partner_episode_start=jnp.ones((2 * pair_count,), dtype=jnp.bool_),
            ego_roles=jnp.concatenate((pair_ego_roles, pair_ego_roles), axis=0),
            done=jnp.zeros((2 * pair_count,), dtype=jnp.bool_),
            raw_return=jnp.zeros((2 * pair_count,), dtype=jnp.float32),
        )
        pair_roots = jax.random.split(pair_key, pair_count)
        variant_roots = jnp.concatenate((pair_roots, pair_roots), axis=0)
        variant_world = advance_anchor_world(
            world=variant_world,
            root_keys=variant_roots,
            functions=functions,
            steps=int(probe_steps or config.anchors.probe_steps),
            gate=0.0,
            runtime=runtime,
            domain=80_003,
        )
        # A post-evidence anchor is valid only if both matched branches survived
        # the probe.  Failing closed avoids treating reset histories as evidence.
        import numpy as np

        if bool(np.any(np.asarray(variant_world.done))):
            raise RuntimeError(
                "A matched-code probe terminated before its post-evidence anchor; "
                "resample the registered candidate without using hidden-code labels."
            )
        variant_world = variant_world._replace(
            raw_return=jnp.zeros((2 * pair_count,), dtype=jnp.float32)
        )
        base_pair_indexes = indexes[:pair_count]
        variant_indexes = jnp.concatenate(
            (base_pair_indexes, base_pair_indexes), axis=0
        )
        variant_policy_states = variant_world.ego_state.policy_state
        variant_lane = jnp.arange(2 * pair_count, dtype=jnp.int32)
        variant_observations = variant_world.observations[
            variant_lane, variant_world.ego_roles
        ]
        variant_codes = jnp.concatenate((code_a, code_b), axis=0)
        variant_sources = jnp.zeros((2 * pair_count,), dtype=jnp.int32)
        variant = collect_counterfactual_anchors(
            anchor_ids=(
                jnp.asarray(anchor_domain, dtype=jnp.int64) * 1_000_000
                + 100_000
                + jnp.arange(2 * pair_count, dtype=jnp.int64)
            ),
            root_keys=variant_roots,
            world=variant_world,
            rollout_flat_indexes=variant_indexes,
            policy_states=variant_policy_states,
            observations=variant_observations,
            partner_codes=variant_codes,
            partner_sources=variant_sources,
            partner_run_ids=jnp.full(
                (2 * pair_count,), -2, dtype=jnp.int32
            ),
            functions=functions,
            action_count=6,
            fit_replicas=registered_fit,
            evaluation_replicas=registered_evaluation,
            continuation_horizon=registered_horizon,
            discount=config.ppo.gamma,
            microbatch_size=microbatch_size,
            runtime=runtime,
            chunk_kernel=chunk_kernel,
        )
        offset = 0 if anchors is None else int(anchors.anchor_ids.shape[0])
        anchors = variant if anchors is None else _concatenate_anchor_batches(anchors, variant)
        codes = variant_codes if not collect_ordinary else jnp.concatenate((codes, variant_codes), axis=0)
        quotient = make_quotient_pairs(
            anchors,
            left_indexes=offset + jnp.arange(pair_count, dtype=jnp.int32),
            right_indexes=(
                offset + pair_count + jnp.arange(pair_count, dtype=jnp.int32)
            ),
        )
    if anchors is None:
        raise ValueError("Anchor request collected neither ordinary nor matched worlds.")
    _validate_anchor_return_bounds(
        anchors,
        lower=config.anchors.return_lower_bound,
        upper=config.anchors.return_upper_bound,
    )
    return anchors, quotient, codes, indexes


def _top_indexes(values: Any, count: int) -> Any:
    """Stable deterministic top-k on host; ties preserve candidate order."""

    import jax.numpy as jnp
    import numpy as np

    scores = np.asarray(values, dtype=np.float64).reshape((-1,))
    if not 0 < int(count) <= scores.size or not np.all(np.isfinite(scores)):
        raise ValueError("Opportunity selection has an invalid top-k request.")
    order = np.lexsort((np.arange(scores.size), -scores))
    return jnp.asarray(order[: int(count)], dtype=jnp.int32)


def _take_anchor_batch(batch: Any, indexes: Any) -> Any:
    import jax
    import jax.numpy as jnp

    selected = jnp.asarray(indexes, dtype=jnp.int32)
    return jax.tree_util.tree_map(lambda value: jnp.asarray(value)[selected], batch)


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
    mode: str = "training",
    enable_quotient_interventions: bool | None = None,
    microbatch_size: int | None = None,
    anchor_functions: AnchorFunctions | None = None,
    chunk_kernel: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Collect r3 training, audit, or calibration anchors.

    Training uses a disjoint pilot domain to select high-leverage states, then
    spends the registered eight fit replicas only on 24 ordinary and 24 legal
    post-evidence matched-code pairs.  Audit/calibration never enter a loss.
    """

    import jax
    import jax.numpy as jnp

    time_count, environment_count = records["actions"].shape
    total = int(time_count * environment_count)
    valid = jnp.ones((time_count, environment_count), dtype=jnp.bool_)
    regret = records.get(
        "decision_regret", jnp.zeros((time_count, environment_count), dtype=jnp.float32)
    )
    sources = records["partner_source"]
    pilot_ordinary_key, pilot_matched_key, fit_ordinary_key, fit_matched_key = jax.random.split(key, 4)

    if mode == "training":
        ordinary_candidates = int(config.anchors.pilot_ordinary_candidates)
        matched_candidates = int(config.anchors.pilot_matched_code_candidates)
        ordinary_selected = int(config.anchors.selected_ordinary)
        matched_selected = int(config.anchors.selected_matched_code)
        if max(ordinary_candidates, matched_candidates) > total:
            raise ValueError("Rollout does not contain the registered training-anchor candidates.")
        ordinary_indexes = stratified_anchor_indexes(
            jax.random.fold_in(pilot_ordinary_key, 1),
            time_count=time_count,
            environment_count=environment_count,
            requested=ordinary_candidates,
            time_bins=1,
            regret_bins=1,
            valid_mask=valid,
            regret_values=regret,
            source_values=sources,
        )
        matched_indexes = stratified_anchor_indexes(
            jax.random.fold_in(pilot_matched_key, 1),
            time_count=time_count,
            environment_count=environment_count,
            requested=matched_candidates,
            time_bins=1,
            regret_bins=1,
            valid_mask=valid,
            regret_values=regret,
            source_values=sources,
        )
        pilot_ordinary, _, _, _ = _collect_anchor_batch_at_indexes(
            anchor_domain=anchor_domain * 10 + 1,
            key=pilot_ordinary_key,
            records=records,
            environment=environment,
            model=model,
            target_params=target_params,
            config=config,
            partner_functions=partner_functions,
            partner_parameters=partner_parameters,
            enable_quotient_interventions=False,
            microbatch_size=microbatch_size,
            anchor_functions=anchor_functions,
            chunk_kernel=None,
            preselected_indexes=ordinary_indexes,
            ordinary_count=ordinary_candidates,
            matched_pair_count=0,
            fit_replicas=int(config.anchors.pilot_replicas),
            evaluation_replicas=0,
            continuation_horizon=int(config.anchors.continuation_horizon),
            collect_ordinary=True,
        )
        ordinary_range = jnp.max(pilot_ordinary.fit_returns_by_action, axis=-1) - jnp.min(
            pilot_ordinary.fit_returns_by_action, axis=-1
        )
        selected_ordinary_indexes = ordinary_indexes[
            _top_indexes(ordinary_range, ordinary_selected)
        ]

        pilot_matched, pilot_pairs, _, _ = _collect_anchor_batch_at_indexes(
            anchor_domain=anchor_domain * 10 + 2,
            key=pilot_matched_key,
            records=records,
            environment=environment,
            model=model,
            target_params=target_params,
            config=config,
            partner_functions=partner_functions,
            partner_parameters=partner_parameters,
            enable_quotient_interventions=True,
            microbatch_size=microbatch_size,
            anchor_functions=anchor_functions,
            chunk_kernel=None,
            preselected_indexes=matched_indexes,
            ordinary_count=matched_candidates,
            matched_pair_count=matched_candidates,
            fit_replicas=int(config.anchors.pilot_replicas),
            evaluation_replicas=0,
            continuation_horizon=int(config.anchors.continuation_horizon),
            probe_steps=int(config.anchors.probe_steps),
            collect_ordinary=False,
        )
        if pilot_pairs is None:
            raise RuntimeError("Registered post-evidence matched-code pilot was not collected.")
        matched_scores = jnp.asarray(pilot_pairs.decision_distance)
        selected_pair_rows = _top_indexes(matched_scores, matched_selected)
        selected_matched_indexes = matched_indexes[selected_pair_rows]

        ordinary, _, ordinary_codes, ordinary_used = _collect_anchor_batch_at_indexes(
            anchor_domain=anchor_domain * 10 + 3,
            key=fit_ordinary_key,
            records=records,
            environment=environment,
            model=model,
            target_params=target_params,
            config=config,
            partner_functions=partner_functions,
            partner_parameters=partner_parameters,
            enable_quotient_interventions=False,
            microbatch_size=microbatch_size,
            anchor_functions=anchor_functions,
            chunk_kernel=chunk_kernel,
            preselected_indexes=selected_ordinary_indexes,
            ordinary_count=ordinary_selected,
            matched_pair_count=0,
            fit_replicas=int(config.anchors.fit_replicas),
            evaluation_replicas=0,
            continuation_horizon=int(config.anchors.continuation_horizon),
            collect_ordinary=True,
        )
        matched, matched_pairs, matched_codes, matched_used = _collect_anchor_batch_at_indexes(
            anchor_domain=anchor_domain * 10 + 4,
            key=fit_matched_key,
            records=records,
            environment=environment,
            model=model,
            target_params=target_params,
            config=config,
            partner_functions=partner_functions,
            partner_parameters=partner_parameters,
            enable_quotient_interventions=True,
            microbatch_size=microbatch_size,
            anchor_functions=anchor_functions,
            chunk_kernel=chunk_kernel,
            preselected_indexes=selected_matched_indexes,
            ordinary_count=matched_selected,
            matched_pair_count=matched_selected,
            fit_replicas=int(config.anchors.fit_replicas),
            evaluation_replicas=0,
            continuation_horizon=int(config.anchors.continuation_horizon),
            probe_steps=int(config.anchors.probe_steps),
            collect_ordinary=False,
        )
        offset = int(ordinary.anchor_ids.shape[0])
        combined = _concatenate_anchor_batches(ordinary, matched)
        quotient = None
        if matched_pairs is not None:
            quotient = matched_pairs._replace(
                anchor_index_a=matched_pairs.anchor_index_a + offset,
                anchor_index_b=matched_pairs.anchor_index_b + offset,
            )
        return (
            combined,
            quotient,
            jnp.concatenate((ordinary_codes, matched_codes), axis=0),
            jnp.concatenate((ordinary_used, matched_used), axis=0),
        )

    if mode not in {"audit", "calibration"}:
        raise ValueError("Anchor mode must be training, audit, or calibration.")
    if mode == "calibration":
        ordinary_count = int(config.calibration.anchors_per_run)
        matched_count = 0
    else:
        ordinary_count = int(config.anchors.audit_ordinary_states)
        matched_count = int(config.anchors.audit_matched_code_states)
    if ordinary_count + matched_count > total:
        raise ValueError("Rollout does not contain enough registered audit states.")
    # Twenty-five percent of the audit states use random priorities only; the
    # remainder are deterministically chosen from the highest predicted-regret
    # states.  No empirical return is read during selection.
    random_count = int(round((ordinary_count + matched_count) * float(config.anchors.audit_uniform_fraction)))
    all_flat = stratified_anchor_indexes(
        jax.random.fold_in(key, 17),
        time_count=time_count,
        environment_count=environment_count,
        requested=ordinary_count + matched_count,
        time_bins=1,
        regret_bins=1,
        valid_mask=valid,
        regret_values=regret,
        source_values=sources,
    )
    flat_regret = jnp.asarray(regret).reshape((-1,))
    opportunity = _top_indexes(flat_regret, ordinary_count + matched_count)
    selected = jnp.concatenate((all_flat[:random_count], opportunity))
    # Stable unique filtering happens on host, followed by deterministic fill.
    import numpy as np
    selected_host = list(dict.fromkeys(np.asarray(selected, dtype=np.int64).tolist()))
    for value in np.asarray(all_flat, dtype=np.int64).tolist():
        if len(selected_host) >= ordinary_count + matched_count:
            break
        if value not in selected_host:
            selected_host.append(value)
    selected = jnp.asarray(selected_host[: ordinary_count + matched_count], dtype=jnp.int32)
    ordinary_indexes = selected[:ordinary_count]
    matched_indexes = selected[ordinary_count:]
    ordinary, _, ordinary_codes, ordinary_used = _collect_anchor_batch_at_indexes(
        anchor_domain=anchor_domain * 10 + 7,
        key=fit_ordinary_key,
        records=records,
        environment=environment,
        model=model,
        target_params=target_params,
        config=config,
        partner_functions=partner_functions,
        partner_parameters=partner_parameters,
        enable_quotient_interventions=False,
        microbatch_size=microbatch_size,
        anchor_functions=anchor_functions,
        chunk_kernel=chunk_kernel,
        preselected_indexes=ordinary_indexes,
        ordinary_count=ordinary_count,
        matched_pair_count=0,
        fit_replicas=int(config.anchors.audit_fit_replicas),
        evaluation_replicas=int(config.anchors.audit_evaluation_replicas),
        continuation_horizon=int(config.anchors.audit_continuation_horizon),
        collect_ordinary=True,
    )
    if matched_count == 0:
        return ordinary, None, ordinary_codes, ordinary_used
    matched, pairs, matched_codes, matched_used = _collect_anchor_batch_at_indexes(
        anchor_domain=anchor_domain * 10 + 8,
        key=fit_matched_key,
        records=records,
        environment=environment,
        model=model,
        target_params=target_params,
        config=config,
        partner_functions=partner_functions,
        partner_parameters=partner_parameters,
        enable_quotient_interventions=True,
        microbatch_size=microbatch_size,
        anchor_functions=anchor_functions,
        chunk_kernel=chunk_kernel,
        preselected_indexes=matched_indexes,
        ordinary_count=matched_count,
        matched_pair_count=matched_count,
        fit_replicas=int(config.anchors.audit_fit_replicas),
        evaluation_replicas=int(config.anchors.audit_evaluation_replicas),
        continuation_horizon=int(config.anchors.audit_continuation_horizon),
        probe_steps=int(config.anchors.probe_steps),
        collect_ordinary=False,
    )
    offset = int(ordinary.anchor_ids.shape[0])
    if pairs is not None:
        pairs = pairs._replace(
            anchor_index_a=pairs.anchor_index_a + offset,
            anchor_index_b=pairs.anchor_index_b + offset,
        )
    return (
        _concatenate_anchor_batches(ordinary, matched),
        pairs,
        jnp.concatenate((ordinary_codes, matched_codes), axis=0),
        jnp.concatenate((ordinary_used, matched_used), axis=0),
    )


def collect_external_partner_support_audit(
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
    external_member_count: int,
    microbatch_size: int | None = None,
    anchor_functions: AnchorFunctions | None = None,
    chunk_kernel: Any | None = None,
) -> tuple[Any, Any]:
    """C1 common-world audit over run-disjoint capable external partners.

    Two uniformly selected physical worlds are copied across all sixteen
    development runs.  Each run receives sixteen legal evidence steps before
    all-action evaluation, so hidden run labels never supervise the posterior.
    """

    import jax
    import jax.numpy as jnp

    members = int(external_member_count)
    if members < 2:
        raise ValueError("Decision-support audit needs independent partner runs.")
    registered = int(config.anchors.audit_ordinary_states)
    if registered % members:
        raise ValueError("Audit ordinary states must divide across development runs.")
    common_worlds = registered // members
    time_count, environment_count = records["actions"].shape
    selection_key, evidence_key, continuation_key = jax.random.split(key, 3)
    base_indexes = stratified_anchor_indexes(
        selection_key,
        time_count=time_count,
        environment_count=environment_count,
        requested=common_worlds,
        time_bins=1,
        regret_bins=1,
        valid_mask=jnp.ones_like(records["actions"], dtype=jnp.bool_),
        regret_values=jnp.zeros_like(records["actions"], dtype=jnp.float32),
        source_values=records["partner_source"],
    )

    def repeat_world(tree: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda value: jnp.repeat(value, members, axis=0), tree
        )

    environment_state = repeat_world(
        gather_time_lanes(records["environment_state"], base_indexes)
    )
    observations = repeat_world(
        gather_time_lanes(records["joint_observations"], base_indexes)
    )
    ego_state = repeat_world(
        gather_time_lanes(records["target_ego_policy_state"], base_indexes)
    )
    partner_state = repeat_world(
        gather_time_lanes(records["partner_state"], base_indexes)
    )
    ego_roles = repeat_world(gather_time_lanes(records["ego_roles"], base_indexes))
    member_indexes = jnp.tile(jnp.arange(members, dtype=jnp.int32), common_worlds)
    partner_state = _external_intervention_state(partner_state, member_indexes)
    count = common_worlds * members
    partner_codes = jnp.full(
        (count, config.partner_generator.code_dim), jnp.nan, dtype=jnp.float32
    )
    partner_sources = jnp.full((count,), 2, dtype=jnp.int32)
    ego = AnchorEgoState(
        policy_state=ego_state,
        partner_code=partner_codes,
        partner_source=partner_sources,
    )
    world = AnchorWorld(
        environment_state=environment_state,
        observations=observations,
        ego_state=ego,
        partner_state=partner_state,
        partner_episode_start=jnp.ones((count,), dtype=jnp.bool_),
        ego_roles=ego_roles,
        done=jnp.zeros((count,), dtype=jnp.bool_),
        raw_return=jnp.zeros((count,), dtype=jnp.float32),
    )
    functions = anchor_functions or make_anchor_functions(
        model=model,
        model_config=config.model,
        partner_functions=partner_functions,
        environment=environment,
    )
    runtime = AnchorRuntime(
        target_params=target_params, partner_parameters=partner_parameters
    )
    common_roots = jax.random.split(evidence_key, common_worlds)
    evidence_roots = jnp.repeat(common_roots, members, axis=0)
    world = advance_anchor_world(
        world=world,
        root_keys=evidence_roots,
        functions=functions,
        steps=int(config.anchors.probe_steps),
        gate=0.0,
        runtime=runtime,
        domain=82_001,
    )
    import numpy as np

    if bool(np.any(np.asarray(world.done))):
        raise RuntimeError("A C1 common-world evidence branch terminated early.")
    lane = jnp.arange(count, dtype=jnp.int32)
    policy_observations = world.observations[lane, world.ego_roles]
    roots = jax.random.split(continuation_key, common_worlds)
    continuation_roots = jnp.repeat(roots, members, axis=0)
    flattened_indexes = jnp.repeat(base_indexes, members)
    anchors = collect_counterfactual_anchors(
        anchor_ids=(
            jnp.asarray(anchor_domain, dtype=jnp.int64) * 1_000_000
            + 500_000
            + jnp.arange(count, dtype=jnp.int64)
        ),
        root_keys=continuation_roots,
        world=world._replace(raw_return=jnp.zeros((count,), dtype=jnp.float32)),
        rollout_flat_indexes=flattened_indexes,
        policy_states=world.ego_state.policy_state,
        observations=policy_observations,
        partner_codes=partner_codes,
        partner_sources=partner_sources,
        partner_run_ids=10_000 + member_indexes,
        functions=functions,
        action_count=6,
        fit_replicas=int(config.anchors.audit_fit_replicas),
        evaluation_replicas=int(config.anchors.audit_evaluation_replicas),
        continuation_horizon=int(config.anchors.audit_continuation_horizon),
        discount=config.ppo.gamma,
        microbatch_size=microbatch_size,
        runtime=runtime,
        chunk_kernel=chunk_kernel,
    )
    _validate_anchor_return_bounds(
        anchors,
        lower=config.anchors.return_lower_bound,
        upper=config.anchors.return_upper_bound,
    )
    return anchors, member_indexes


def collect_matched_code_audit(
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
    microbatch_size: int | None = None,
    anchor_functions: AnchorFunctions | None = None,
    chunk_kernel: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Collect only the registered post-evidence matched-code audit pairs.

    The ordinary C1 component is collected against independent external runs
    by :func:`collect_external_partner_support_audit`.  Keeping this entrypoint
    matched-only avoids silently doubling the audit budget.
    """

    import jax
    import jax.numpy as jnp

    time_count, environment_count = records["actions"].shape
    count = int(config.anchors.audit_matched_code_states)
    if count <= 0 or count > time_count * environment_count:
        raise ValueError("Registered matched-code audit count is infeasible.")
    indexes = stratified_anchor_indexes(
        jax.random.fold_in(key, 71),
        time_count=time_count,
        environment_count=environment_count,
        requested=count,
        time_bins=1,
        regret_bins=1,
        valid_mask=jnp.ones_like(records["actions"], dtype=jnp.bool_),
        regret_values=records.get(
            "decision_regret",
            jnp.zeros_like(records["actions"], dtype=jnp.float32),
        ),
        source_values=records["partner_source"],
    )
    anchors, pairs, codes, unused_indexes = _collect_anchor_batch_at_indexes(
        anchor_domain=anchor_domain,
        key=jax.random.fold_in(key, 72),
        records=records,
        environment=environment,
        model=model,
        target_params=target_params,
        config=config,
        partner_functions=partner_functions,
        partner_parameters=partner_parameters,
        enable_quotient_interventions=True,
        microbatch_size=microbatch_size,
        anchor_functions=anchor_functions,
        chunk_kernel=chunk_kernel,
        preselected_indexes=indexes,
        ordinary_count=count,
        matched_pair_count=count,
        fit_replicas=int(config.anchors.audit_fit_replicas),
        evaluation_replicas=int(config.anchors.audit_evaluation_replicas),
        continuation_horizon=int(config.anchors.audit_continuation_horizon),
        probe_steps=int(config.anchors.probe_steps),
        collect_ordinary=False,
    )
    del unused_indexes
    if pairs is None or int(anchors.anchor_ids.shape[0]) != 2 * count:
        raise RuntimeError("Matched-code audit did not produce two legal branches per pair.")
    return anchors, pairs, codes


__all__ = [
    "AnchorEgoState",
    "AnchorRuntime",
    "collect_anchor_batch",
    "collect_external_partner_support_audit",
    "collect_matched_code_audit",
    "gather_time_lanes",
    "make_quotient_pairs",
    "make_anchor_functions",
    "stratified_anchor_indexes",
]
