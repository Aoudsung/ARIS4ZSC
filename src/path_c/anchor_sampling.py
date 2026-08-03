"""Fixed-budget V6 counterfactual collection (§5 legalized anchors).

Training anchors are selected only from rollout time and partner-source strata.
No learned score or post-hoc qualification decision changes the sample budget.

METHOD_SPEC §5 rebuild: every anchor snapshot comes from a real, complete
episode record (environment state, ego recurrent state, partner recurrent
state, legal ego history index, partner lineage).  Matched pairs are nearest
neighbours between genuine snapshots of *different* partner runs; mid-episode
code resets, partner-carry clearing and forced ``partner_episode_start``
splicing are abolished.  After sixteen legal probe steps the frozen §5.3
comparator three-way classification decides which pairs enter L_separation;
irreducible-ambiguity pairs are recorded and never supervised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

from .counterfactual_anchor import (
    AnchorFunctions,
    AnchorWorld,
    collect_counterfactual_anchors,
    inverse_cdf_actions,
    select_anchor_microbatch_size,
)
from .decision_geometry import centered_action_values, decision_distance
from .runner import observe_policy_after_transition
from .types import QuotientPairBatch, SeparationTerms


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


class ProbeHistory(NamedTuple):
    """Legal probe history recorded per lane (§5.2/§5.3 inputs)."""

    ego_actions: Any
    partner_actions: Any
    ego_observations: Any


def probe_matched_history(
    *,
    world: AnchorWorld,
    root_keys: Any,
    functions: AnchorFunctions,
    steps: int,
    runtime: Any,
    domain: int = 80_003,
) -> tuple[AnchorWorld, ProbeHistory]:
    """Run the sixteen-step legal probe and record the ego-visible history.

    Mirrors ``counterfactual_anchor.advance_anchor_world`` step-for-step (same
    key derivation, same active-lane masking) while additionally recording
    ego/partner actions and ego observations so the frozen comparator can be
    trained and evaluated on legal histories only.  Runs outside ``jit``.
    """

    import jax
    import jax.numpy as jnp

    if steps <= 0:
        raise ValueError("Post-evidence probe must contain at least one step.")
    roots = jnp.asarray(root_keys)
    lane_count = int(jnp.asarray(world.done).shape[0])
    if roots.shape != (lane_count, 2):
        raise ValueError("Post-evidence roots must provide one key per world.")
    lane = jnp.arange(lane_count, dtype=jnp.int32)

    current = world
    ego_action_rows: list[Any] = []
    partner_action_rows: list[Any] = []
    ego_observation_rows: list[Any] = []
    for time_index in range(int(steps)):
        active = ~jnp.asarray(current.done, dtype=jnp.bool_)
        ego_keys = jax.vmap(
            lambda key: jax.random.fold_in(jax.random.fold_in(key, domain), 10 + time_index)
        )(roots)
        partner_keys = jax.vmap(
            lambda key: jax.random.fold_in(jax.random.fold_in(key, domain), 20 + time_index)
        )(roots)
        environment_keys = jax.vmap(
            lambda key: jax.random.fold_in(jax.random.fold_in(key, domain), 30 + time_index)
        )(roots)
        uniform_keys = jax.vmap(
            lambda key: jax.random.fold_in(jax.random.fold_in(key, domain), 40 + time_index)
        )(roots)
        uniforms = jax.vmap(lambda key: jax.random.uniform(key))(uniform_keys)
        ego_observation = current.observations[lane, current.ego_roles]
        partner_observation = current.observations[lane, 1 - current.ego_roles]
        next_ego_pre, ego_probabilities = functions.ego_policy_step(
            runtime, current.ego_state, ego_observation, ego_keys
        )
        ego_actions = inverse_cdf_actions(ego_probabilities, uniforms)
        partner_actions, next_partner_pre, partner_context = functions.partner_policy_step(
            runtime,
            current.partner_state,
            partner_observation,
            current.partner_episode_start,
            partner_keys,
        )
        ego_first = jnp.stack((ego_actions, partner_actions), axis=-1)
        partner_first = jnp.stack((partner_actions, ego_actions), axis=-1)
        joint = jnp.where((current.ego_roles == 0)[:, None], ego_first, partner_first)
        next_environment, next_observations, rewards, dones, info = functions.environment_step(
            current.environment_state, joint, environment_keys
        )
        raw_by_agent = info.get("raw_rewards_by_agent")
        ego_rewards = (
            rewards
            if raw_by_agent is None
            else raw_by_agent[lane, current.ego_roles]
        )
        terminal = info["terminal_observations"]
        obs_mask = jnp.asarray(dones).reshape(
            jnp.asarray(dones).shape + (1,) * (next_observations[:, 0].ndim - 1)
        )
        ego_terminal = terminal[lane, current.ego_roles]
        partner_terminal = terminal[lane, 1 - current.ego_roles]
        ego_next = next_observations[lane, current.ego_roles]
        partner_next = next_observations[lane, 1 - current.ego_roles]
        ego_history = jnp.where(obs_mask, ego_terminal, ego_next)
        partner_history = jnp.where(obs_mask, partner_terminal, partner_next)
        next_ego = functions.ego_observe(
            next_ego_pre, ego_observation, ego_actions, ego_rewards, dones, ego_history
        )
        next_partner = functions.partner_observe(
            runtime,
            next_partner_pre,
            partner_context,
            partner_observation,
            partner_actions,
            ego_rewards,
            dones,
            partner_history,
        )
        candidate = AnchorWorld(
            environment_state=next_environment,
            observations=next_observations,
            ego_state=next_ego,
            partner_state=next_partner,
            partner_episode_start=dones,
            ego_roles=current.ego_roles,
            done=jnp.asarray(dones, dtype=jnp.bool_),
            raw_return=current.raw_return + jnp.where(active, ego_rewards, 0.0),
        )
        current = _tree_select_active(active, candidate, current)
        ego_action_rows.append(jnp.where(active, ego_actions, 0))
        partner_action_rows.append(jnp.where(active, partner_actions, 0))
        ego_observation_rows.append(ego_observation)
    history = ProbeHistory(
        ego_actions=jnp.stack(ego_action_rows, axis=1),
        partner_actions=jnp.stack(partner_action_rows, axis=1),
        ego_observations=jnp.stack(ego_observation_rows, axis=1),
    )
    return current, history


def _tree_select_active(active: Any, on_true: Any, on_false: Any) -> Any:
    import jax
    import jax.numpy as jnp

    mask = jnp.asarray(active, dtype=jnp.bool_)

    def select(true_leaf: Any, false_leaf: Any) -> Any:
        shape = (mask.shape[0],) + (1,) * (jnp.asarray(true_leaf).ndim - 1)
        return jnp.where(mask.reshape(shape), true_leaf, false_leaf)

    return jax.tree_util.tree_map(select, on_true, on_false)


def _nearest_run_partner_pairs(features: Any, run_ids: Any, pair_count: int) -> Any:
    """Greedy nearest-neighbour pairing across different partner runs (§5.2).

    Host-side numpy matching; ``collect_anchor_batch`` runs outside ``jit``.
    A candidate is consumed once, and pairs whose closest remaining partner
    belongs to the same run id are skipped (irreducible without a second
    genuine run in the batch).
    """

    import numpy as np

    feats = np.asarray(features, dtype=np.float64)
    runs = np.asarray(run_ids, dtype=np.int64).reshape((-1,))
    candidate_count = feats.shape[0]
    available = np.ones((candidate_count,), dtype=bool)
    left: list[int] = []
    right: list[int] = []
    order = np.argsort(-np.linalg.norm(feats - feats.mean(axis=0, keepdims=True), axis=1))
    for anchor_index in order:
        if len(left) >= int(pair_count):
            break
        if not available[int(anchor_index)]:
            continue
        distances = np.linalg.norm(feats - feats[int(anchor_index)], axis=1)
        distances = np.where(available, distances, np.inf)
        distances = np.where(
            runs == runs[int(anchor_index)], np.inf, distances
        )
        partner_index = int(np.argmin(distances))
        if not np.isfinite(distances[partner_index]):
            continue
        left.append(int(anchor_index))
        right.append(partner_index)
        available[int(anchor_index)] = False
        available[partner_index] = False
    return np.asarray((left, right), dtype=np.int64).T


def pair_legal_history_features(history: ProbeHistory, *, action_count: int = 6) -> Any:
    """Flattened legal ego-visible probe features (§5.3 comparator input).

    Per step: the spatially pooled ego observation planes plus the one-hot
    ego action.  Everything here is observable to the ego over its legal
    history; partner identities are never encoded directly.
    """

    import jax.numpy as jnp

    observations = jnp.asarray(history.ego_observations, dtype=jnp.float32)
    pooled = jnp.mean(
        observations.reshape(observations.shape[:2] + (-1, observations.shape[-1])),
        axis=2,
    )
    action_planes = jax_nn_one_hot(history.ego_actions, int(action_count))
    per_step = jnp.concatenate((pooled, action_planes), axis=-1)
    return per_step.reshape((per_step.shape[0], -1))


def jax_nn_one_hot(values: Any, depth: int) -> Any:
    import jax.nn
    import jax.numpy as jnp

    return jax.nn.one_hot(jnp.asarray(values, dtype=jnp.int32), int(depth))


@dataclass(frozen=True)
class FrozenPairComparator:
    """§5.3 frozen 2AFC comparator.

    The fitting procedure (features, temporal train/holdout split, logistic
    fit hyperparameters, thresholds) is registered before training runs and
    must not be tuned afterwards; threshold changes require a new ledger
    entry.
    """

    weights: Any
    bias: Any
    train_accuracy: Any
    feature_dim: int


def pair_holdout_features(features: Any, *, probe_steps: int) -> Any:
    """Causally later half of the probe features (comparator holdout)."""

    import numpy as np

    full = np.asarray(features, dtype=np.float64)
    half = int(probe_steps) // 2
    per_step = full.shape[-1] // int(probe_steps)
    return full[:, half * per_step : 2 * half * per_step]


def _two_afc_dataset(features: Any, run_ids: Any, *, maximum_pairs: int = 512):
    import numpy as np

    feats = np.asarray(features, dtype=np.float64)
    runs = np.asarray(run_ids, dtype=np.int64).reshape((-1,))
    count = feats.shape[0]
    pairs: list[tuple[int, int]] = []
    for i in range(count):
        for j in range(i + 1, count):
            if runs[i] != runs[j]:
                pairs.append((i, j))
    if len(pairs) > int(maximum_pairs):
        pairs = pairs[: int(maximum_pairs)]
    if not pairs:
        raise ValueError("2AFC comparator requires at least two partner runs.")
    matrix = np.stack(
        [np.concatenate((feats[i] - feats[j], feats[j] - feats[i])) for i, j in pairs]
    )
    labels = np.concatenate(
        [np.asarray((1.0, 0.0), dtype=np.float64) for _ in pairs]
    )
    return matrix, labels


def fit_pair_comparator(
    features: Any,
    *,
    probe_steps: int,
    partner_run_ids: Any,
    logistic_iterations: int = 100,
    learning_rate: float = 1.0,
    l2_penalty: float = 1.0e-3,
) -> FrozenPairComparator:
    """Fit the frozen comparator on the first probe half, hold out the rest.

    Training pairs use features from steps ``[0, probe_steps // 2)``;
    evaluation uses the causally later half, so reported accuracies are
    out-of-sample by construction.
    """

    import numpy as np

    full = np.asarray(features, dtype=np.float64)
    half = int(probe_steps) // 2
    if half <= 0 or 2 * half > int(probe_steps):
        raise ValueError("Comparator fitting requires at least two probe steps.")
    per_step = full.shape[-1] // int(probe_steps)
    train_features = full[:, : half * per_step]
    eval_features = pair_holdout_features(full, probe_steps=probe_steps)
    train_matrix, train_labels = _two_afc_dataset(train_features, partner_run_ids)
    dimension = train_matrix.shape[1]
    weights = np.zeros((dimension,), dtype=np.float64)
    bias = 0.0
    count = train_matrix.shape[0]
    for _ in range(int(logistic_iterations)):
        logits = train_matrix @ weights + bias
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        gradient = train_matrix.T @ (probability - train_labels) / count
        weights -= float(learning_rate) * (gradient + float(l2_penalty) * weights)
        bias -= float(learning_rate) * float(np.mean(probability - train_labels))
    eval_matrix, eval_labels = _two_afc_dataset(eval_features, partner_run_ids)
    predictions = (eval_matrix @ weights + bias) > 0.0
    accuracy = float(np.mean(predictions == (eval_labels > 0.5)))
    return FrozenPairComparator(
        weights=weights.astype(np.float32),
        bias=np.float32(bias),
        train_accuracy=np.float32(accuracy),
        feature_dim=int(dimension),
    )


def pair_comparator_accuracy(comparator: FrozenPairComparator, features: Any) -> Any:
    """Per-pair 2AFC accuracy on both presentation orders."""

    import numpy as np

    feats = np.asarray(features, dtype=np.float64)
    if feats.shape != (2, comparator.feature_dim):
        raise ValueError("Pair features must carry exactly two lanes.")
    weights = np.asarray(comparator.weights, dtype=np.float64)
    forward = float((feats[0] - feats[1]) @ weights + float(comparator.bias)) > 0.0
    backward = float((feats[1] - feats[0]) @ weights + float(comparator.bias)) > 0.0
    return np.float32((float(forward) + float(1.0 - backward)) / 2.0)


def classify_matched_pairs(
    comparator_accuracy: Any,
    signature_distance: Any,
    *,
    equivalent_accuracy_max: float,
    distinct_accuracy_min: float,
    signature_threshold: float,
) -> tuple[Any, Any, Any, Any]:
    """§5.3 three-way split: 0 observable-equivalent, 1 decision-distinct, 2
    irreducible ambiguity.  Ambiguous pairs never enter a separation or
    consistency loss; they are only recorded."""

    import jax.numpy as jnp

    accuracy = jnp.asarray(comparator_accuracy, dtype=jnp.float32)
    distance = jnp.asarray(signature_distance, dtype=jnp.float32)
    equivalent = (accuracy <= float(equivalent_accuracy_max)) & (
        distance <= float(signature_threshold)
    )
    distinct = (accuracy > float(distinct_accuracy_min)) & (
        distance > float(signature_threshold)
    )
    classes = jnp.where(equivalent, 0, jnp.where(distinct, 1, 2)).astype(jnp.int32)
    pair_count = jnp.maximum(jnp.asarray(classes.shape[0], dtype=jnp.float32), 1.0)
    fractions = jnp.stack(
        [
            jnp.sum((classes == label).astype(jnp.float32)) / pair_count
            for label in range(3)
        ]
    )
    return classes, equivalent, distinct, fractions


def separation_terms_from_matched_pairs(
    *,
    quotient: QuotientPairBatch,
    equivalent_accuracy_max: float,
    distinct_accuracy_min: float,
    signature_threshold: float,
    margin_scale: float,
) -> tuple[SeparationTerms, Mapping[str, Any]]:
    """Build the L_separation payload from the §5.3 classification (§3.2).

    Only the classification is precomputed here: equivalent/distinct masks,
    per-pair weights and ``margin = margin_scale x mean distinct signature
    distance`` stay constant between anchor triggers.  The two ego forward
    passes and the loss itself run inside ``compute_loss`` against the
    current params so the separation gradient enters the same
    ``value_and_grad`` as the other three losses (METHOD_SPEC §3.2/§3.4);
    the previous eager scalar produced outside jit was a param-independent
    constant with zero gradient.  Irreducible-ambiguity pairs keep zero
    weight (recorded only).
    """

    import jax.numpy as jnp

    if quotient.comparator_accuracy is None or quotient.ego_state_a is None:
        raise ValueError(
            "separation terms require the §5 matched-pair comparator payload."
        )
    classes, equivalent_mask, distinct_mask, fractions = classify_matched_pairs(
        quotient.comparator_accuracy,
        quotient.decision_distance,
        equivalent_accuracy_max=float(equivalent_accuracy_max),
        distinct_accuracy_min=float(distinct_accuracy_min),
        signature_threshold=float(signature_threshold),
    )
    distinct_distance = jnp.where(
        distinct_mask, quotient.decision_distance, 0.0
    )
    margin = margin_scale * jnp.sum(distinct_distance) / jnp.maximum(
        jnp.sum(distinct_mask.astype(jnp.float32)), 1.0
    )
    weights = jnp.where(
        distinct_mask,
        quotient.decision_distance,
        jnp.where(equivalent_mask, 1.0, 0.0),
    )
    terms = SeparationTerms(
        ego_state_a=quotient.ego_state_a,
        ego_state_b=quotient.ego_state_b,
        probe_observations=quotient.probe_observations,
        equivalent_mask=equivalent_mask,
        weights=weights,
        margin=margin,
    )
    readings = {
        "pair_class_fractions": fractions,
        "irreducible_ambiguity_fraction": fractions[2],
        "separation_margin": margin,
    }
    return terms, readings


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

    # §5.2: matched pairs are nearest-neighbour couplings of genuine
    # snapshots from *different* partner runs.  No generator code, no
    # mid-episode intervention, no forced episode-start splicing.
    candidate_count = 4 * pair_count
    candidate_indexes = time_source_stratified_indexes(
        matched_key,
        time_count=time_count,
        environment_count=environment_count,
        requested=candidate_count,
        source_values=records["partner_source"],
    )
    candidate_world = world_from_records(records, candidate_indexes)
    candidate_observations = gather_time_lanes(
        records["observations"], candidate_indexes
    )
    _, candidate_output = model.apply(
        {"params": target_params},
        candidate_world.ego_state,
        candidate_observations,
        jnp.zeros((candidate_count,), dtype=jnp.bool_),
        method=model.step,
    )
    candidate_run_ids = np.asarray(
        gather_time_lanes(records["partner_run_ids"], candidate_indexes)
    ).reshape((-1,))
    pair_matrix = _nearest_run_partner_pairs(
        jax.lax.stop_gradient(candidate_output.task_features),
        candidate_run_ids,
        pair_count,
    )
    paired_flat = jnp.asarray(pair_matrix.reshape((-1,), order="C"), dtype=jnp.int32)
    matched_indexes = candidate_indexes[paired_flat]
    paired_world = world_from_records(records, matched_indexes)
    pair_root_keys = jax.random.split(probe_key, pair_count)
    paired_roots = jnp.repeat(pair_root_keys, 2, axis=0)
    paired_world, probe_history = probe_matched_history(
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
    probe_features = pair_legal_history_features(probe_history)
    comparator = fit_pair_comparator(
        probe_features,
        probe_steps=int(config.anchors.probe_steps),
        partner_run_ids=gather_time_lanes(records["partner_run_ids"], matched_indexes),
    )
    eval_features = pair_holdout_features(
        probe_features, probe_steps=int(config.anchors.probe_steps)
    )
    pair_accuracies = jnp.stack(
        [
            pair_comparator_accuracy(comparator, eval_features[2 * i : 2 * i + 2])
            for i in range(pair_count)
        ]
    )
    matched = collect_counterfactual_anchors(
        anchor_ids=(
            jnp.asarray(anchor_domain, dtype=jnp.int64) * 1_000_000
            + 100_000
            + jnp.arange(2 * pair_count, dtype=jnp.int64)
        ),
        root_keys=jnp.repeat(jax.random.split(matched_roots, pair_count), 2, axis=0),
        world=paired_world,
        rollout_flat_indexes=matched_indexes,
        policy_states=paired_world.ego_state,
        observations=paired_observations,
        partner_codes=gather_time_lanes(records["partner_code"], matched_indexes),
        partner_sources=gather_time_lanes(records["partner_source"], matched_indexes),
        partner_run_ids=gather_time_lanes(records["partner_run_ids"], matched_indexes),
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
        comparator_accuracy=pair_accuracies,
        ego_state_a=paired_world.ego_state[0::2],
        ego_state_b=paired_world.ego_state[1::2],
        probe_observations=jnp.stack(
            (paired_observations[0::2], paired_observations[1::2])
        ),
    )
    _, class_readings = _pair_class_readings(
        quotient=quotient,
        equivalent_accuracy_max=float(config.anchors.observable_equivalent_accuracy_max),
        distinct_accuracy_min=float(config.anchors.decision_distinct_accuracy_min),
        signature_threshold=float(config.anchors.signature_distance_threshold),
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
        "pair_class_fractions": class_readings["pair_class_fractions"],
        "irreducible_ambiguity_fraction": class_readings[
            "irreducible_ambiguity_fraction"
        ],
        "comparator_holdout_accuracy": comparator.train_accuracy,
        "matched_pair_candidates": int(candidate_count),
        "matched_pairs_formed": int(pair_matrix.shape[0]),
    }
    return combined, quotient, budget


def collect_diversity_bank_signatures(
    *,
    anchor_domain: int,
    key: Any,
    records: Mapping[str, Any],
    codes: Any,
    functions: AnchorFunctions,
    runtime: AnchorRuntime,
    config: Any,
    microbatch_size: int | None = None,
    chunk_kernel: Any | None = None,
) -> Any:
    """§7.2 shared-state diversity signatures (review §6).

    Draw a registered bank of ``diversity_bank_size`` legal anchor snapshots
    and run real all-action CRN continuations for every diversity code on the
    *same* bank states.  Returns centered signatures shaped
    ``(code_count, bank_size, action_count)``; because the environment states
    are shared across codes, pairwise distances measure partner-behaviour
    differences only, never state-visitation differences or Q error (the
    abolished BR-diversity mixture).
    """

    import jax
    import jax.numpy as jnp

    bank_size = int(config.partner_generator.diversity_bank_size)
    code_count = int(jnp.asarray(codes).shape[0])
    bank_key, root_key = jax.random.split(key)
    time_count, environment_count = records["actions"].shape
    bank_indexes = time_source_stratified_indexes(
        bank_key,
        time_count=time_count,
        environment_count=environment_count,
        requested=bank_size,
        source_values=records["partner_source"],
    )
    world = world_from_records(records, bank_indexes)
    observations = gather_time_lanes(records["observations"], bank_indexes)
    # Tile the shared bank once per diversity code so every code is evaluated
    # on identical environment states (lane = bank_index * code_count + code).
    tiled_world = jax.tree_util.tree_map(
        lambda leaf: jnp.repeat(leaf, code_count, axis=0), world
    )
    bank = collect_counterfactual_anchors(
        anchor_ids=(
            jnp.asarray(anchor_domain, dtype=jnp.int64) * 1_000_000
            + 200_000
            + jnp.arange(bank_size * code_count, dtype=jnp.int64)
        ),
        root_keys=jax.random.split(root_key, bank_size * code_count),
        world=tiled_world,
        rollout_flat_indexes=jnp.repeat(bank_indexes, code_count, axis=0),
        policy_states=tiled_world.ego_state,
        observations=jnp.repeat(observations, code_count, axis=0),
        partner_codes=jnp.tile(jnp.asarray(codes), (bank_size, 1)),
        partner_sources=jnp.repeat(
            gather_time_lanes(records["partner_source"], bank_indexes),
            code_count,
            axis=0,
        ),
        partner_run_ids=jnp.repeat(
            gather_time_lanes(records["partner_run_ids"], bank_indexes),
            code_count,
            axis=0,
        ),
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
    signatures = centered_action_values(bank.fit_returns_by_action)
    return signatures.reshape((bank_size, code_count, -1)).transpose((1, 0, 2))


def _pair_class_readings(
    *,
    quotient: QuotientPairBatch,
    equivalent_accuracy_max: float,
    distinct_accuracy_min: float,
    signature_threshold: float,
) -> tuple[Any, Mapping[str, Any]]:
    classes, equivalent, distinct, fractions = classify_matched_pairs(
        quotient.comparator_accuracy,
        quotient.decision_distance,
        equivalent_accuracy_max=equivalent_accuracy_max,
        distinct_accuracy_min=distinct_accuracy_min,
        signature_threshold=signature_threshold,
    )
    return classes, {
        "pair_class_fractions": fractions,
        "irreducible_ambiguity_fraction": fractions[2],
    }


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
    "FrozenPairComparator",
    "ProbeHistory",
    "anchor_budget_for_updates",
    "classify_matched_pairs",
    "collect_anchor_batch",
    "collect_diversity_bank_signatures",
    "fit_pair_comparator",
    "gather_time_lanes",
    "make_anchor_functions",
    "pair_comparator_accuracy",
    "pair_holdout_features",
    "pair_legal_history_features",
    "preflight_anchor_microbatch_from_records",
    "probe_matched_history",
    "separation_terms_from_matched_pairs",
    "time_source_stratified_indexes",
    "world_from_records",
]
