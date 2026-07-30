"""Selection and collection of real-return counterfactual action anchors."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

from .counterfactual_anchor import (
    AnchorFunctions,
    AnchorWorld,
    collect_counterfactual_anchors,
)
from .decision_geometry import centered_action_values, decision_distance
from .runner import observe_policy_after_transition
from .types import QuotientPairBatch


class AnchorEgoState(NamedTuple):
    """Training-only privileged continuation controller state.

    The deployable recurrent policy state is preserved verbatim.  Generator and
    snapshot partners use their continuous code through the training-only code
    teacher at every continuation step; frozen external partners use the
    full-trajectory teacher context inferred from the original completed
    rollout.  No partner identity enters the shared actor or critic.
    """

    policy_state: Any
    partner_code: Any
    partner_source: Any
    fixed_teacher_latent: Any


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



def _validate_anchor_return_bounds(anchors: Any, *, lower: float, upper: float) -> None:
    """Fail closed when preregistered bounded-return assumptions are violated."""

    import numpy as np

    fit = np.asarray(anchors.fit_returns_by_action, dtype=np.float64)
    evaluation = np.asarray(anchors.evaluation_returns_by_action, dtype=np.float64)
    if not np.isfinite(fit).all() or not np.isfinite(evaluation).all():
        raise FloatingPointError("Counterfactual anchor returns are non-finite.")
    observed_lower = float(min(fit.min(), evaluation.min()))
    observed_upper = float(max(fit.max(), evaluation.max()))
    if observed_lower < float(lower) or observed_upper > float(upper):
        raise ValueError(
            "Observed simulator continuation return violates the frozen bounds: "
            f"observed=[{observed_lower}, {observed_upper}], "
            f"registered=[{float(lower)}, {float(upper)}]."
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
    enable_quotient_interventions: bool = True,
    microbatch_size: int | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Collect split-replica all-action targets from rollout snapshots.

    Returns anchors, quotient pairs, selected generator codes, and flattened
    rollout indexes.  Ground truth is simulator raw return only.
    """

    import jax
    import jax.numpy as jnp

    time_count, environment_count = records["actions"].shape
    valid_teacher = jnp.ones(records["actions"].shape, dtype=jnp.bool_)
    selection_key, continuation_key = jax.random.split(key)
    indexes = stratified_anchor_indexes(
        selection_key,
        time_count=time_count,
        environment_count=environment_count,
        requested=config.anchors.states_per_interval,
        time_bins=config.anchors.sampling_time_bins,
        regret_bins=config.anchors.sampling_regret_bins,
        valid_mask=valid_teacher,
        regret_values=records.get(
            "decision_regret", jnp.zeros(records["actions"].shape, dtype=jnp.float32)
        ),
        source_values=records["partner_source"],
    )
    environment_state = gather_time_lanes(records["environment_state"], indexes)
    observations = gather_time_lanes(records["joint_observations"], indexes)
    ego_state = gather_time_lanes(records["ego_policy_state"], indexes)
    partner_state = gather_time_lanes(records["partner_state"], indexes)
    partner_codes = gather_time_lanes(records["partner_code"], indexes)
    partner_sources = gather_time_lanes(records["partner_source"], indexes)
    partner_run_ids = gather_time_lanes(records["partner_run_ids"], indexes)
    ego_roles = gather_time_lanes(records["ego_roles"], indexes)
    anchor_count = int(indexes.shape[0])
    # The real-return target must correspond to the frozen privileged target
    # controller specified by the method, not to the student's current belief.
    # Reconstruct its context from the completed rollout before selecting anchor
    # lanes.  Generator/snapshot lanes use the known continuous code; frozen
    # external lanes use a full-trajectory teacher inferred without identity.
    rollout_observations = jnp.concatenate(
        (
            records["observations"],
            records["response_next_observations"][-1:],
        ),
        axis=0,
    )
    full_teacher_latents = model.apply(
        {"params": target_params},
        rollout_observations,
        records["response_next_observations"],
        records["actions"],
        records["rewards"],
        records["dones"],
        method=model.full_trajectory_latents,
    )
    code_teacher_latents = model.apply(
        {"params": target_params},
        records["partner_code"],
        records["task_features"],
        1.0,
        method=model.teacher_from_code,
    ).latent
    rollout_teacher_latents = jnp.where(
        (records["partner_source"] == 2)[..., None],
        full_teacher_latents,
        code_teacher_latents,
    )
    selected_teacher_latents = gather_time_lanes(
        rollout_teacher_latents, indexes
    )
    anchor_ego_state = AnchorEgoState(
        policy_state=ego_state,
        partner_code=partner_codes,
        partner_source=partner_sources,
        fixed_teacher_latent=selected_teacher_latents,
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

    def ego_policy_step(
        state: AnchorEgoState, observation: Any, gate: Any, keys: Any
    ) -> tuple[AnchorEgoState, Any]:
        del keys, gate
        next_policy_state, online = model.apply(
            {"params": target_params},
            state.policy_state,
            observation,
            jnp.ones(state.partner_source.shape, dtype=jnp.float32),
            method=model.step,
        )
        code_latent = model.apply(
            {"params": target_params},
            state.partner_code,
            online.task_features,
            1.0,
            method=model.teacher_from_code,
        ).latent
        teacher_latent = jnp.where(
            (state.partner_source == 2)[..., None],
            state.fixed_teacher_latent,
            code_latent,
        )
        teacher = model.apply(
            {"params": target_params},
            online.task_features,
            teacher_latent,
            1.0,
            method=model.from_features_and_latent,
        )
        next_state = state._replace(
            policy_state=next_policy_state,
            fixed_teacher_latent=teacher_latent,
        )
        return next_state, jax.nn.softmax(teacher.logits, axis=-1)

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
            model_config=config.model,
        )
        return stepped._replace(policy_state=policy_state)

    def partner_policy_step(
        state: Any,
        observation: Any,
        episode_start: Any,
        keys: Any,
    ) -> tuple[Any, Any, Any]:
        action, next_state, context, unused_log_probability = partner_functions.step(
            partner_parameters, state, observation, episode_start, keys
        )
        del unused_log_probability
        return action, next_state, context

    def partner_observe(
        state: Any,
        context: Any,
        observations_at_time: Any,
        actions: Any,
        rewards: Any,
        dones: Any,
        next_observations: Any,
    ) -> Any:
        return partner_functions.observe(
            partner_parameters,
            state,
            context,
            observations_at_time,
            actions,
            rewards,
            dones,
            next_observations,
        )

    root_keys = jax.random.split(continuation_key, anchor_count)
    anchors = collect_counterfactual_anchors(
        anchor_ids=(
            jnp.asarray(anchor_domain, dtype=jnp.int64) * 1_000_000
            + jnp.arange(anchor_count, dtype=jnp.int64)
        ),
        root_keys=root_keys,
        world=world,
        rollout_flat_indexes=indexes,
        policy_states=ego_state,
        observations=gather_time_lanes(records["observations"], indexes),
        partner_codes=partner_codes,
        partner_sources=partner_sources,
        partner_run_ids=partner_run_ids,
        functions=AnchorFunctions(
            ego_policy_step=ego_policy_step,
            ego_observe=ego_observe,
            partner_policy_step=partner_policy_step,
            partner_observe=partner_observe,
            environment_step=environment.step_with_keys,
        ),
        action_count=6,
        fit_replicas=config.anchors.fit_replicas,
        evaluation_replicas=config.anchors.evaluation_replicas,
        continuation_horizon=config.anchors.continuation_horizon,
        microbatch_size=microbatch_size,
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
        int(config.partner_generator.codes_per_update),
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
        variant_initial_latent = model.apply(
            {"params": target_params},
            jnp.concatenate((code_a, code_b), axis=0),
            jnp.concatenate(
                (
                    gather_time_lanes(records["task_features"], indexes)[:pair_count],
                    gather_time_lanes(records["task_features"], indexes)[:pair_count],
                ),
                axis=0,
            ),
            1.0,
            method=model.teacher_from_code,
        ).latent
        variant_ego_state = AnchorEgoState(
            policy_state=variant_policy_state,
            partner_code=jnp.concatenate((code_a, code_b), axis=0),
            partner_source=jnp.zeros((2 * pair_count,), dtype=jnp.int32),
            fixed_teacher_latent=variant_initial_latent,
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
        base_pair_indexes = indexes[:pair_count]
        variant_indexes = jnp.concatenate(
            (base_pair_indexes, base_pair_indexes), axis=0
        )
        variant_policy_states = jax.tree_util.tree_map(
            lambda value: jnp.concatenate((value, value), axis=0),
            pair_ego_state,
        )
        selected_ego_observations = gather_time_lanes(
            records["observations"], indexes
        )[:pair_count]
        variant_observations = jnp.concatenate(
            (selected_ego_observations, selected_ego_observations), axis=0
        )
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
            functions=AnchorFunctions(
                ego_policy_step=ego_policy_step,
                ego_observe=ego_observe,
                partner_policy_step=partner_policy_step,
                partner_observe=partner_observe,
                environment_step=environment.step_with_keys,
            ),
            action_count=6,
            fit_replicas=config.anchors.fit_replicas,
            evaluation_replicas=config.anchors.evaluation_replicas,
            continuation_horizon=config.anchors.continuation_horizon,
            microbatch_size=microbatch_size,
        )
        offset = int(anchors.anchor_ids.shape[0])
        anchors = _concatenate_anchor_batches(anchors, variant)
        codes = jnp.concatenate((codes, variant_codes), axis=0)
        quotient = make_quotient_pairs(
            anchors,
            left_indexes=offset + jnp.arange(pair_count, dtype=jnp.int32),
            right_indexes=(
                offset + pair_count + jnp.arange(pair_count, dtype=jnp.int32)
            ),
        )
    _validate_anchor_return_bounds(
        anchors,
        lower=config.anchors.return_lower_bound,
        upper=config.anchors.return_upper_bound,
    )
    return anchors, quotient, codes, indexes


__all__ = [
    "AnchorEgoState",
    "collect_anchor_batch",
    "gather_time_lanes",
    "make_quotient_pairs",
    "stratified_anchor_indexes",
]
