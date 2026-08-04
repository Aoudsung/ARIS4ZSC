"""Fixed-budget DEPI counterfactual collection (§5 legalized anchors).

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


PAIR_EQUIVALENT = 0
PAIR_DISTINCT = 1
PAIR_AMBIGUOUS = 2
PAIR_FRACTION_ORDER = ("equivalent", "distinct", "ambiguous")
COMPARATOR_RUN_ID_CAPACITY = 64


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


def mask_supervision_partner_runs(
    action_mask: Any, partner_run_ids: Any, excluded_partner_run_ids: Any
) -> Any:
    """Disable every supervision row owned by a comparator-development run."""

    import jax.numpy as jnp

    mask = jnp.asarray(action_mask, dtype=jnp.bool_)
    run_ids = jnp.asarray(partner_run_ids, dtype=jnp.int32).reshape((-1,))
    excluded = jnp.asarray(excluded_partner_run_ids, dtype=jnp.int32).reshape((-1,))
    if mask.ndim != 2 or run_ids.shape != (mask.shape[0],):
        raise ValueError("Supervision masks and partner run IDs do not align.")
    if excluded.size == 0:
        return mask
    forbidden = jnp.any(run_ids[:, None] == excluded[None, :], axis=1)
    return mask & ~forbidden[:, None]


def time_source_stratified_indexes(
    key: Any,
    *,
    time_count: int,
    environment_count: int,
    requested: int,
    source_values: Any,
    time_bins: int = 4,
    eligible_mask: Any | None = None,
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
    eligible = (
        np.ones((total,), dtype=bool)
        if eligible_mask is None
        else np.asarray(eligible_mask, dtype=bool).reshape((-1,))
    )
    if eligible.shape != (total,):
        raise ValueError("Anchor eligibility mask does not align with rollout states.")
    if int(np.sum(eligible)) < int(requested):
        raise RuntimeError("The requested anchor stratum has too few eligible states.")
    times = np.repeat(np.arange(int(time_count)), int(environment_count))
    time_labels = np.minimum(
        (times * int(time_bins)) // max(int(time_count), 1), int(time_bins) - 1
    )
    priorities = np.asarray(
        jax.random.uniform(key, (total,), dtype=jnp.float32), dtype=np.float64
    )
    strata: dict[tuple[int, int], list[int]] = {}
    for index in range(total):
        if not eligible[index]:
            continue
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
        ranked = sorted(
            np.flatnonzero(eligible).tolist(),
            key=lambda item: priorities[item],
            reverse=True,
        )
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

    Uses the registered key derivation and active-lane masking while recording
    only ego actions and ego observations. Partner actions exist inside the
    simulator step but are never retained in the comparator artifact. Runs
    outside ``jit``.
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
        ego_observation_rows.append(ego_observation)
    history = ProbeHistory(
        ego_actions=jnp.stack(ego_action_rows, axis=1),
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


def run_disjoint_candidate_pairs(
    features: Any,
    run_ids: Any,
    *,
    pair_count: int,
    development_pair_count: int = 0,
    excluded_run_ids: Any = (),
) -> Any:
    """Form a fixed pair budget with disjoint comparator/supervision run sets."""

    import numpy as np

    feats = np.asarray(features, dtype=np.float64)
    runs = np.asarray(run_ids, dtype=np.int64).reshape((-1,))
    if feats.shape[0] != runs.size:
        raise ValueError("Candidate features and partner run IDs do not align.")
    total_pairs = int(pair_count)
    development_pairs = int(development_pair_count)
    if total_pairs <= 0 or not 0 <= development_pairs < total_pairs:
        raise ValueError("Run-disjoint pair budgets are inconsistent.")
    excluded = set(np.asarray(tuple(excluded_run_ids), dtype=np.int64).tolist())
    eligible = np.asarray(
        [index for index, run in enumerate(runs) if int(run) not in excluded],
        dtype=np.int64,
    )

    def pair_subset(indexes: np.ndarray, requested: int) -> np.ndarray | None:
        pairs = _nearest_run_partner_pairs(
            feats[indexes], runs[indexes], requested
        )
        if pairs.shape != (requested, 2):
            return None
        return indexes[pairs]

    if development_pairs == 0:
        pairs = pair_subset(eligible, total_pairs)
        if pairs is None:
            raise RuntimeError("The fixed run-disjoint matched-pair budget cannot be formed.")
        return pairs

    training_pairs = total_pairs - development_pairs
    unique_runs, counts = np.unique(runs[eligible], return_counts=True)
    if unique_runs.size < 6:
        raise RuntimeError(
            "Comparator fit/validation and DEPI supervision need at least six "
            "candidate partner runs."
        )
    count_by_run = {
        int(run): int(count) for run, count in zip(unique_runs, counts, strict=True)
    }
    by_id = [int(run) for run in sorted(unique_runs.tolist())]
    by_count_desc = sorted(by_id, key=lambda run: (-count_by_run[run], run))
    by_count_asc = sorted(by_id, key=lambda run: (count_by_run[run], run))
    interleaved: list[int] = []
    low, high = 0, len(by_count_desc) - 1
    while low <= high:
        interleaved.append(by_count_desc[low])
        low += 1
        if low <= high:
            interleaved.append(by_count_desc[high])
            high -= 1
    orders = (by_id, by_count_desc, by_count_asc, interleaved)
    best: tuple[float, np.ndarray] | None = None
    seen: set[tuple[int, ...]] = set()
    for order in orders:
        for split in range(4, len(order) - 1):
            development_runs = tuple(sorted(order[:split]))
            if development_runs in seen:
                continue
            seen.add(development_runs)
            development_set = set(development_runs)
            development_indexes = eligible[
                np.asarray([int(runs[index]) in development_set for index in eligible])
            ]
            training_indexes = eligible[
                np.asarray([int(runs[index]) not in development_set for index in eligible])
            ]
            development_matrix = pair_subset(
                development_indexes, development_pairs
            )
            training_matrix = pair_subset(training_indexes, training_pairs)
            if development_matrix is None or training_matrix is None:
                continue
            used_development_runs = set(
                runs[development_matrix.reshape((-1,))].tolist()
            )
            if len(used_development_runs) < 4:
                continue
            matrix = np.concatenate((development_matrix, training_matrix), axis=0)
            distance = float(
                np.sum(
                    np.linalg.norm(
                        feats[matrix[:, 0]] - feats[matrix[:, 1]], axis=-1
                    )
                )
            )
            if best is None or distance < best[0]:
                best = (distance, matrix)
    if best is None:
        raise RuntimeError(
            "The fixed comparator-development/training pair budget cannot be "
            "partitioned across mutually disjoint partner runs."
        )
    return best[1]


def manifest_partitioned_candidate_pairs(
    features: Any,
    run_ids: Any,
    partition_ids: Any,
    *,
    pair_count: int,
    comparator_is_frozen: bool,
) -> Any:
    """Use manifest-disjoint support/fit/validation rows for fixed pairs.

    Partition 0 is policy support and the only source of DEPI supervision;
    partitions 1 and 2 are comparator fit and validation respectively.  The
    first anchor trigger consumes half the pair budget for comparator
    development, split equally between fit and validation.  Later triggers
    consume support rows only because the comparator is frozen.
    """

    import numpy as np

    feats = np.asarray(features, dtype=np.float64)
    runs = np.asarray(run_ids, dtype=np.int64).reshape((-1,))
    partitions = np.asarray(partition_ids, dtype=np.int64).reshape((-1,))
    if feats.shape[0] != runs.size or runs.shape != partitions.shape:
        raise ValueError("Manifest-partitioned candidates do not align.")

    def form(partition: int, requested: int) -> np.ndarray:
        indexes = np.flatnonzero(partitions == int(partition))
        pairs = _nearest_run_partner_pairs(feats[indexes], runs[indexes], requested)
        if pairs.shape != (requested, 2):
            raise RuntimeError(
                f"Partner partition {partition} cannot form {requested} run-disjoint pairs."
            )
        return indexes[pairs]

    total = int(pair_count)
    if comparator_is_frozen:
        return form(0, total)
    if total < 4 or total % 4:
        raise ValueError("Initial comparator pair budget must be a positive multiple of four.")
    development = total // 2
    fit = development // 2
    validation = development - fit
    supervision = total - development
    return np.concatenate(
        (form(1, fit), form(2, validation), form(0, supervision)), axis=0
    )


def pair_legal_history_features(history: ProbeHistory, *, action_count: int = 6) -> Any:
    """Explicit semantic legal-history sequence for the frozen comparator.

    Each step contains partner visibility/relative position/direction/
    inventory/observable-change plus the ego action.  Partner action and run
    identity are deliberately excluded.
    """

    import jax.numpy as jnp
    from .response_targets import official_partner_observation_planes

    observations = jnp.asarray(history.ego_observations, dtype=jnp.float32)
    height, width, channel_count = observations.shape[-3:]
    planes = official_partner_observation_planes(channel_count)
    visibility_plane = observations[..., planes.visibility_channel] > 0
    flat_visibility = visibility_plane.reshape(
        visibility_plane.shape[:-2] + (height * width,)
    )
    visible = jnp.any(flat_visibility, axis=-1)
    position = jnp.where(
        visible,
        jnp.argmax(flat_visibility.astype(jnp.int32), axis=-1),
        height * width,
    )
    position_one_hot = jax_nn_one_hot(position, height * width + 1)
    direction_scores = jnp.sum(
        observations[..., list(planes.direction_channels)]
        * visibility_plane[..., None],
        axis=(-3, -2),
    )
    direction_one_hot = jax_nn_one_hot(jnp.argmax(direction_scores, axis=-1), 4)
    inventory = jnp.sum(
        observations[..., list(planes.inventory_channels)]
        * visibility_plane[..., None],
        axis=(-3, -2),
    )
    inventory = jnp.where(visible[..., None], inventory, 0.0)
    previous_inventory = jnp.concatenate(
        (jnp.zeros_like(inventory[:, :1]), inventory[:, :-1]), axis=1
    )
    observable_change = jnp.any(
        jnp.rint(inventory) != jnp.rint(previous_inventory), axis=-1
    ).astype(jnp.float32)
    action_planes = jax_nn_one_hot(history.ego_actions, int(action_count))
    per_step = jnp.concatenate(
        (
            visible[..., None].astype(jnp.float32),
            position_one_hot,
            direction_one_hot,
            inventory,
            observable_change[..., None],
            action_planes,
        ),
        axis=-1,
    )
    return per_step.reshape((per_step.shape[0], -1))


def jax_nn_one_hot(values: Any, depth: int) -> Any:
    import jax.nn
    import jax.numpy as jnp

    return jax.nn.one_hot(jnp.asarray(values, dtype=jnp.int32), int(depth))


class FrozenPairComparator(NamedTuple):
    """§5.3 frozen decision-regime pair comparator.

    The fitting procedure (features, temporal train/holdout split, logistic
    fit hyperparameters, thresholds) is registered before training runs and
    must not be tuned afterwards; threshold changes require a new ledger
    entry.
    """

    weights: Any
    bias: Any
    train_accuracy: Any
    validation_accuracy: Any
    validation_accuracy_interval: tuple[float, float]
    history_feature_dim: int
    pair_feature_dim: int
    development_row_count: int
    validation_row_count: int
    validation_partner_run_count: int
    development_partner_run_ids: Any
    development_partner_run_count: int


def pair_feature_rows(left_history: Any, right_history: Any) -> Any:
    """Symmetric pair representation: absolute difference and product."""

    import numpy as np

    left = np.asarray(left_history, dtype=np.float64)
    right = np.asarray(right_history, dtype=np.float64)
    if left.shape != right.shape or left.ndim not in (1, 2):
        raise ValueError("Pair histories must have equal one- or two-dimensional shapes.")
    return np.concatenate((np.abs(left - right), left * right), axis=-1)


def decision_signature_prototypes(
    signatures: Any, *, component_count: int
) -> Any:
    """Deterministic farthest-first prototypes from empirical continuations."""

    import numpy as np

    values = np.asarray(signatures, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Decision-regime development needs at least two signatures.")
    target_count = min(int(component_count), max(values.shape[0] // 2, 2))
    first = int(np.argmax(np.linalg.norm(values - values.mean(axis=0), axis=-1)))
    selected = [first]
    while len(selected) < target_count:
        distance = np.min(
            np.linalg.norm(values[:, None, :] - values[selected][None, :, :], axis=-1),
            axis=-1,
        )
        distance[selected] = -np.inf
        candidate = int(np.argmax(distance))
        if not np.isfinite(distance[candidate]) or distance[candidate] <= 1.0e-8:
            break
        selected.append(candidate)
    if len(selected) < 2:
        raise ValueError("Empirical continuation signatures contain no distinct regimes.")
    return values[np.asarray(selected, dtype=np.int64)]


def assign_decision_signature_regimes(signatures: Any, prototypes: Any) -> Any:
    """Assign signatures to the nearest frozen empirical prototype."""

    import numpy as np

    values = np.asarray(signatures, dtype=np.float64)
    centers = np.asarray(prototypes, dtype=np.float64)
    if values.ndim != 2 or centers.ndim != 2 or values.shape[1] != centers.shape[1]:
        raise ValueError("Decision signatures and prototypes do not align.")
    distance = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=-1)
    return np.argmin(distance, axis=-1).astype(np.int32)


def decision_regime_pair_dataset(
    history_features: Any,
    regime_labels: Any,
    *,
    block_ids: Any | None = None,
    maximum_pairs: int = 512,
    require_both_classes: bool = True,
) -> tuple[Any, Any, Any]:
    """Construct pair rows labelled by empirical decision-signature regime."""

    import numpy as np

    features = np.asarray(history_features, dtype=np.float64)
    regimes = np.asarray(regime_labels, dtype=np.int64).reshape((-1,))
    if features.ndim != 2 or features.shape[0] != regimes.shape[0]:
        raise ValueError("History features and regime labels must align.")
    blocks = (
        np.arange(features.shape[0], dtype=np.int64)
        if block_ids is None
        else np.asarray(block_ids).reshape((-1,))
    )
    if blocks.shape[0] != features.shape[0]:
        raise ValueError("Comparator block ids must align with histories.")
    rows: list[Any] = []
    labels: list[float] = []
    pair_blocks: list[str] = []
    for left in range(features.shape[0]):
        for right in range(left + 1, features.shape[0]):
            rows.append(pair_feature_rows(features[left], features[right]))
            labels.append(float(regimes[left] != regimes[right]))
            pair_blocks.append(
                "|".join(sorted((str(blocks[left]), str(blocks[right]))))
            )
            if len(rows) >= int(maximum_pairs):
                break
        if len(rows) >= int(maximum_pairs):
            break
    if not rows:
        raise ValueError("Comparator development requires at least two histories.")
    matrix = np.stack(rows)
    target = np.asarray(labels, dtype=np.float64)
    if require_both_classes and np.unique(target).size != 2:
        raise ValueError(
            "Comparator development split must contain same- and different-regime pairs."
        )
    return matrix, target, np.asarray(pair_blocks, dtype=object)


def _fit_logistic_pair_rows(
    pair_features: Any,
    pair_labels: Any,
    *,
    logistic_iterations: int,
    learning_rate: float,
    l2_penalty: float,
) -> tuple[Any, Any]:
    import numpy as np

    matrix = np.asarray(pair_features, dtype=np.float64)
    labels = np.asarray(pair_labels, dtype=np.float64).reshape((-1,))
    if matrix.ndim != 2 or matrix.shape[0] != labels.shape[0]:
        raise ValueError("Pair feature rows and labels must have the same sample count.")
    if matrix.shape[0] < 2 or np.unique(labels).size != 2:
        raise ValueError("Pair comparator fitting requires both binary classes.")
    weights = np.zeros((matrix.shape[1],), dtype=np.float64)
    bias = 0.0
    for _ in range(int(logistic_iterations)):
        logits = matrix @ weights + bias
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        residual = probability - labels
        gradient = matrix.T @ residual / matrix.shape[0]
        weights -= float(learning_rate) * (
            gradient + float(l2_penalty) * weights
        )
        bias -= float(learning_rate) * float(np.mean(residual))
    return weights, bias


def predict_pair_comparator(
    comparator: FrozenPairComparator, pair_features: Any
) -> Any:
    """Predict frozen pair-label probabilities from concatenated pair rows."""

    import numpy as np

    matrix = np.asarray(pair_features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None]
    if matrix.ndim != 2 or matrix.shape[1] != comparator.pair_feature_dim:
        raise ValueError("Pair rows do not match the frozen pair feature dimension.")
    logits = matrix @ np.asarray(comparator.weights, dtype=np.float64) + float(
        comparator.bias
    )
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))).astype(
        np.float32
    )


def fit_pair_comparator(
    pair_features: Any,
    pair_labels: Any,
    *,
    validation_pair_features: Any,
    validation_pair_labels: Any,
    history_feature_dim: int,
    training_block_ids: Any | None = None,
    validation_block_ids: Any | None = None,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 0,
    logistic_iterations: int = 100,
    learning_rate: float = 1.0,
    l2_penalty: float = 1.0e-3,
) -> FrozenPairComparator:
    """Fit on explicit pair rows/labels and freeze against a disjoint split.

    Labels must be derived from decision-signature regimes by the caller; run
    identity is intentionally absent from this API.
    """

    import numpy as np

    train_matrix = np.asarray(pair_features, dtype=np.float64)
    train_labels = np.asarray(pair_labels, dtype=np.float64).reshape((-1,))
    weights, bias = _fit_logistic_pair_rows(
        train_matrix,
        train_labels,
        logistic_iterations=logistic_iterations,
        learning_rate=learning_rate,
        l2_penalty=l2_penalty,
    )
    validation_matrix = np.asarray(validation_pair_features, dtype=np.float64)
    validation_labels = np.asarray(validation_pair_labels, dtype=np.float64).reshape((-1,))
    if validation_matrix.shape[0] != validation_labels.shape[0]:
        raise ValueError("Validation pair rows and labels must align.")

    def accuracy(matrix: Any, labels: Any) -> float:
        prediction = (matrix @ weights + bias) > 0.0
        return float(np.mean(prediction == (labels > 0.5)))

    pair_dim = int(train_matrix.shape[1])
    lane_dim = int(history_feature_dim)
    if pair_dim != 2 * lane_dim:
        raise ValueError("Pair feature dimension must be twice the history dimension.")
    validation_prediction = (validation_matrix @ weights + bias) > 0.0
    validation_correct = validation_prediction == (validation_labels > 0.5)
    train_blocks = (
        np.arange(train_matrix.shape[0]).astype(str)
        if training_block_ids is None
        else np.asarray(training_block_ids).astype(str).reshape((-1,))
    )
    blocks = (
        np.arange(validation_matrix.shape[0]).astype(str)
        if validation_block_ids is None
        else np.asarray(validation_block_ids).astype(str).reshape((-1,))
    )
    if train_blocks.shape[0] != train_matrix.shape[0]:
        raise ValueError("Training block ids must align with training pair rows.")
    if blocks.shape[0] != validation_matrix.shape[0]:
        raise ValueError("Validation block ids must align with validation pair rows.")

    def partner_runs(pair_blocks: Any) -> set[str]:
        return {
            run
            for block in np.asarray(pair_blocks).astype(str)
            for run in block.split("|")
        }

    training_runs = partner_runs(train_blocks)
    validation_runs = partner_runs(blocks)
    overlap = training_runs & validation_runs
    if overlap:
        raise ValueError(
            "Comparator development and validation partner runs overlap: "
            f"{sorted(overlap)}"
        )
    unique_runs = np.asarray(sorted(validation_runs), dtype=object)
    if unique_runs.size < 2:
        raise ValueError("Comparator validation needs at least two partner-run blocks.")
    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap_accuracy = np.empty((int(bootstrap_replicates),), dtype=np.float64)
    for index in range(int(bootstrap_replicates)):
        selected = rng.choice(unique_runs, size=unique_runs.size, replace=True)
        counts = {run: int(np.sum(selected == run)) for run in unique_runs}
        weights_by_pair = np.asarray(
            [
                0.5
                * sum(counts.get(run, 0) for run in str(block).split("|"))
                for block in blocks
            ],
            dtype=np.float64,
        )
        bootstrap_accuracy[index] = np.sum(
            weights_by_pair * validation_correct.astype(np.float64)
        ) / np.maximum(np.sum(weights_by_pair), 1.0)
    interval = np.quantile(bootstrap_accuracy, (0.025, 0.975))
    return FrozenPairComparator(
        weights=weights.astype(np.float32),
        bias=np.float32(bias),
        train_accuracy=np.float32(accuracy(train_matrix, train_labels)),
        validation_accuracy=np.float32(
            accuracy(validation_matrix, validation_labels)
        ),
        validation_accuracy_interval=(float(interval[0]), float(interval[1])),
        history_feature_dim=lane_dim,
        pair_feature_dim=pair_dim,
        development_row_count=int(train_matrix.shape[0]),
        validation_row_count=int(validation_matrix.shape[0]),
        validation_partner_run_count=int(unique_runs.size),
        development_partner_run_ids=np.full(
            (COMPARATOR_RUN_ID_CAPACITY,), -1, dtype=np.int32
        ),
        development_partner_run_count=0,
    )


def pair_comparator_accuracy(
    comparator: FrozenPairComparator, pair_features: Any, pair_labels: Any
) -> Any:
    """Accuracy on labelled independent pair rows."""

    import numpy as np

    probability = predict_pair_comparator(comparator, pair_features)
    labels = np.asarray(pair_labels, dtype=np.float32).reshape((-1,))
    if probability.shape != labels.shape:
        raise ValueError("Comparator predictions and labels must align.")
    return np.float32(np.mean((probability >= 0.5) == (labels >= 0.5)))


def classify_matched_pairs(
    comparator_distinct_probability: Any,
    signature_distance: Any,
    *,
    equivalent_probability_max: float,
    distinct_probability_min: float,
    signature_threshold: float,
    pair_valid: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    """§5.3 three-way split: 0 observable-equivalent, 1 decision-distinct, 2
    irreducible ambiguity.  Ambiguous pairs never enter a separation or
    consistency loss; they are only recorded."""

    import jax.numpy as jnp

    probability = jnp.asarray(
        comparator_distinct_probability, dtype=jnp.float32
    )
    distance = jnp.asarray(signature_distance, dtype=jnp.float32)
    valid = (
        jnp.ones_like(probability, dtype=jnp.bool_)
        if pair_valid is None
        else jnp.asarray(pair_valid, dtype=jnp.bool_)
    )
    equivalent = valid & (probability <= float(equivalent_probability_max)) & (
        distance <= float(signature_threshold)
    )
    distinct = valid & (probability > float(distinct_probability_min)) & (
        distance > float(signature_threshold)
    )
    classes = jnp.where(
        equivalent,
        PAIR_EQUIVALENT,
        jnp.where(distinct, PAIR_DISTINCT, PAIR_AMBIGUOUS),
    ).astype(jnp.int32)
    pair_count = jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
    fractions = jnp.stack(
        [
            jnp.sum(((classes == label) & valid).astype(jnp.float32)) / pair_count
            for label in (PAIR_EQUIVALENT, PAIR_DISTINCT, PAIR_AMBIGUOUS)
        ]
    )
    return classes, equivalent, distinct, fractions


def separation_terms_from_matched_pairs(
    *,
    quotient: QuotientPairBatch,
    equivalent_probability_max: float,
    distinct_probability_min: float,
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

    if quotient.comparator_distinct_probability is None or quotient.ego_state_a is None:
        raise ValueError(
            "separation terms require the §5 matched-pair comparator payload."
        )
    classes, equivalent_mask, distinct_mask, fractions = classify_matched_pairs(
        quotient.comparator_distinct_probability,
        quotient.decision_distance,
        equivalent_probability_max=float(equivalent_probability_max),
        distinct_probability_min=float(distinct_probability_min),
        signature_threshold=float(signature_threshold),
        pair_valid=quotient.pair_valid,
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
        pair_valid=(
            jnp.ones_like(equivalent_mask, dtype=jnp.bool_)
            if quotient.pair_valid is None
            else jnp.asarray(quotient.pair_valid, dtype=jnp.bool_)
        ),
    )
    readings = {
        "pair_class_fractions": fractions,
        "irreducible_ambiguity_fraction": fractions[2],
        "separation_margin": margin,
        "valid_pair_fraction": jnp.mean(
            (
                jnp.ones_like(equivalent_mask, dtype=jnp.float32)
                if quotient.pair_valid is None
                else jnp.asarray(quotient.pair_valid, dtype=jnp.float32)
            )
        ),
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
    frozen_comparator: FrozenPairComparator | None = None,
) -> tuple[Any, QuotientPairBatch, Mapping[str, Any], FrozenPairComparator | None]:
    """Collect exactly 32 ordinary and 16 post-evidence matched pairs."""

    import jax
    import jax.numpy as jnp
    import numpy as np

    time_count, environment_count = records["actions"].shape
    ordinary_key, matched_key, return_key, probe_key = jax.random.split(key, 4)
    ordinary_count = int(config.anchors.ordinary_states)
    pair_count = int(config.anchors.matched_history_pairs)
    partition_values = records.get("partner_partition")
    manifest_partitioned = bool(
        partition_values is not None
        and np.any(np.asarray(partition_values, dtype=np.int32) != 0)
    )
    ordinary_indexes = time_source_stratified_indexes(
        ordinary_key,
        time_count=time_count,
        environment_count=environment_count,
        requested=ordinary_count,
        source_values=records["partner_source"],
        eligible_mask=(
            np.asarray(partition_values) == 0 if partition_values is not None else None
        ),
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
            jnp.asarray(anchor_domain, dtype=jnp.int32) * 1_000_000
            + jnp.arange(ordinary_count, dtype=jnp.int32)
        ),
        root_keys=ordinary_root_keys,
        world=ordinary_world,
        rollout_flat_indexes=ordinary_indexes,
        policy_states=ordinary_world.ego_state,
        observations=gather_time_lanes(records["observations"], ordinary_indexes),
        partner_sources=gather_time_lanes(records["partner_source"], ordinary_indexes),
        partner_members=gather_time_lanes(records["partner_member"], ordinary_indexes),
        partner_family_ids=gather_time_lanes(
            records["partner_family_id"], ordinary_indexes
        ),
        partner_checkpoint_stages=gather_time_lanes(
            records["partner_checkpoint_stage"], ordinary_indexes
        ),
        partner_run_ids=gather_time_lanes(records["partner_run_ids"], ordinary_indexes),
        functions=functions,
        action_count=6,
        fit_replicas=int(config.anchors.fit_replicas),
        evaluation_replicas=int(config.anchors.evaluation_replicas),
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
    if manifest_partitioned:
        selection_keys = jax.random.split(matched_key, 3)
        comparator_frozen = frozen_comparator is not None
        if comparator_frozen:
            requested_by_partition = (candidate_count, 0, 0)
        else:
            development_pairs = pair_count // 2
            requested_by_partition = (
                4 * (pair_count - development_pairs),
                4 * (development_pairs // 2),
                4 * (development_pairs - development_pairs // 2),
            )
        selected = []
        for partition, requested in enumerate(requested_by_partition):
            if requested <= 0:
                continue
            selected.append(
                time_source_stratified_indexes(
                    selection_keys[partition],
                    time_count=time_count,
                    environment_count=environment_count,
                    requested=requested,
                    source_values=records["partner_source"],
                    eligible_mask=np.asarray(partition_values) == partition,
                )
            )
        candidate_indexes = jnp.concatenate(tuple(selected), axis=0)
    else:
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
    candidate_partition_ids = (
        None
        if partition_values is None
        else np.asarray(
            gather_time_lanes(partition_values, candidate_indexes)
        ).reshape((-1,))
    )
    frozen_development_runs = (
        ()
        if frozen_comparator is None
        else tuple(
            np.asarray(frozen_comparator.development_partner_run_ids)[
                : int(frozen_comparator.development_partner_run_count)
            ].tolist()
        )
    )
    if manifest_partitioned:
        pair_matrix = manifest_partitioned_candidate_pairs(
            jax.lax.stop_gradient(candidate_output.task_features),
            candidate_run_ids,
            candidate_partition_ids,
            pair_count=pair_count,
            comparator_is_frozen=frozen_comparator is not None,
        )
    else:
        pair_matrix = run_disjoint_candidate_pairs(
            jax.lax.stop_gradient(candidate_output.task_features),
            candidate_run_ids,
            pair_count=pair_count,
            development_pair_count=(
                pair_count // 2
                if frozen_comparator is None and pair_count >= 4
                else 0
            ),
            excluded_run_ids=frozen_development_runs,
        )
    if pair_matrix.shape != (pair_count, 2):
        raise RuntimeError(
            "The fixed matched-pair budget cannot be formed from distinct partner runs."
        )
    paired_flat = jnp.asarray(pair_matrix.reshape((-1,), order="C"), dtype=jnp.int32)
    matched_indexes = candidate_indexes[paired_flat]
    matched_partition_ids = (
        None
        if candidate_partition_ids is None
        else candidate_partition_ids[np.asarray(paired_flat)]
    )
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
    matched = collect_counterfactual_anchors(
        anchor_ids=(
            jnp.asarray(anchor_domain, dtype=jnp.int32) * 1_000_000
            + 100_000
            + jnp.arange(2 * pair_count, dtype=jnp.int32)
        ),
        root_keys=jnp.repeat(jax.random.split(matched_roots, pair_count), 2, axis=0),
        world=paired_world,
        rollout_flat_indexes=matched_indexes,
        policy_states=paired_world.ego_state,
        observations=paired_observations,
        partner_sources=gather_time_lanes(records["partner_source"], matched_indexes),
        partner_members=gather_time_lanes(records["partner_member"], matched_indexes),
        partner_family_ids=gather_time_lanes(
            records["partner_family_id"], matched_indexes
        ),
        partner_checkpoint_stages=gather_time_lanes(
            records["partner_checkpoint_stage"], matched_indexes
        ),
        partner_run_ids=gather_time_lanes(records["partner_run_ids"], matched_indexes),
        functions=functions,
        action_count=6,
        fit_replicas=int(config.anchors.fit_replicas),
        evaluation_replicas=int(config.anchors.evaluation_replicas),
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
    # Comparator development and use are disjoint at the anchor-pair level.
    # The first half supplies empirical continuation-signature regimes; only
    # the second half may enter L_separation.
    comparator = frozen_comparator
    pair_probabilities = np.full((pair_count,), 0.5, dtype=np.float32)
    comparator_validation_accuracy = float("nan")
    comparator_validation_interval = [float("nan"), float("nan")]
    training_pair_mask = np.zeros((pair_count,), dtype=bool)
    comparator_development_runs: set[int] = set()
    expected_training_pair_count = 0
    if comparator is None and pair_count >= 4:
        development_pairs = pair_count // 2
        expected_training_pair_count = pair_count - development_pairs
        development_lane_count = 2 * development_pairs
        lane_signatures = np.asarray(
            centered_action_values(matched.fit_returns_by_action)
        )
        matched_run_ids = np.asarray(
            gather_time_lanes(records["partner_run_ids"], matched_indexes)
        ).reshape((-1,))
        development_runs = np.asarray(
            sorted(set(matched_run_ids[:development_lane_count].tolist()))
        )
        comparator_development_runs = set(development_runs.tolist())
        if development_runs.size < 4:
            raise RuntimeError(
                "Frozen comparator needs four run-disjoint development blocks."
            )
        if manifest_partitioned:
            development_partitions = np.asarray(
                matched_partition_ids[:development_lane_count]
            )
            fit_lane_mask = development_partitions == 1
            validation_lane_mask = development_partitions == 2
            fit_runs = set(
                matched_run_ids[:development_lane_count][fit_lane_mask].tolist()
            )
            validation_runs = set(
                matched_run_ids[:development_lane_count][validation_lane_mask].tolist()
            )
            if fit_runs & validation_runs:
                raise RuntimeError("Manifest comparator fit/validation runs overlap.")
        else:
            split = development_runs.size // 2
            fit_runs = set(development_runs[:split].tolist())
            validation_runs = set(development_runs[split:].tolist())
            fit_lane_mask = np.asarray(
                [run in fit_runs for run in matched_run_ids[:development_lane_count]],
                dtype=bool,
            )
            validation_lane_mask = np.asarray(
                [
                    run in validation_runs
                    for run in matched_run_ids[:development_lane_count]
                ],
                dtype=bool,
            )
        if np.sum(fit_lane_mask) < 2 or np.sum(validation_lane_mask) < 2:
            raise RuntimeError(
                "Comparator run-disjoint fit/validation lanes are insufficient."
            )
        development_features = np.asarray(
            probe_features[:development_lane_count]
        )
        development_signatures = lane_signatures[:development_lane_count]
        development_run_ids = matched_run_ids[:development_lane_count]
        prototypes = decision_signature_prototypes(
            development_signatures[fit_lane_mask],
            component_count=int(config.model.protocol_components),
        )
        regime_labels = assign_decision_signature_regimes(
            development_signatures, prototypes
        )
        fit_rows, fit_labels, fit_blocks = decision_regime_pair_dataset(
            development_features[fit_lane_mask],
            regime_labels[fit_lane_mask],
            block_ids=development_run_ids[fit_lane_mask],
        )
        validation_rows, validation_labels, validation_blocks = (
            decision_regime_pair_dataset(
                development_features[validation_lane_mask],
                regime_labels[validation_lane_mask],
                block_ids=development_run_ids[validation_lane_mask],
                require_both_classes=False,
            )
        )
        comparator = fit_pair_comparator(
            fit_rows,
            fit_labels,
            validation_pair_features=validation_rows,
            validation_pair_labels=validation_labels,
            history_feature_dim=int(probe_features.shape[-1]),
            training_block_ids=fit_blocks,
            validation_block_ids=validation_blocks,
        )
        if development_runs.size > COMPARATOR_RUN_ID_CAPACITY:
            raise RuntimeError("Comparator development run capacity is insufficient.")
        stored_development_runs = np.full(
            (COMPARATOR_RUN_ID_CAPACITY,), -1, dtype=np.int32
        )
        stored_development_runs[: development_runs.size] = development_runs.astype(
            np.int32
        )
        comparator = comparator._replace(
            development_partner_run_ids=stored_development_runs,
            development_partner_run_count=int(development_runs.size),
        )
        pair_rows = pair_feature_rows(
            np.asarray(probe_features[0::2]), np.asarray(probe_features[1::2])
        )
        pair_probabilities = predict_pair_comparator(comparator, pair_rows)
        pair_runs = matched_run_ids.reshape((pair_count, 2))
        training_pair_mask = np.asarray(
            [
                pair_index >= development_pairs
                and left_run not in comparator_development_runs
                and right_run not in comparator_development_runs
                and (
                    not manifest_partitioned
                    or (
                        matched_partition_ids[2 * pair_index] == 0
                        and matched_partition_ids[2 * pair_index + 1] == 0
                    )
                )
                for pair_index, (left_run, right_run) in enumerate(pair_runs)
            ],
            dtype=bool,
        )
        if int(np.sum(training_pair_mask)) != expected_training_pair_count:
            raise RuntimeError(
                "The fixed comparator-development/training split cannot be "
                "formed from mutually run-disjoint pairs."
            )
        comparator_validation_accuracy = float(comparator.validation_accuracy)
        comparator_validation_interval = list(
            comparator.validation_accuracy_interval
        )
    elif comparator is not None:
        expected_training_pair_count = pair_count
        pair_rows = pair_feature_rows(
            np.asarray(probe_features[0::2]), np.asarray(probe_features[1::2])
        )
        pair_probabilities = predict_pair_comparator(comparator, pair_rows)
        matched_run_ids = np.asarray(
            gather_time_lanes(records["partner_run_ids"], matched_indexes)
        ).reshape((pair_count, 2))
        comparator_development_runs = set(
            np.asarray(comparator.development_partner_run_ids)[
                : int(comparator.development_partner_run_count)
            ].tolist()
        )
        training_pair_mask = np.asarray(
            [
                left_run not in comparator_development_runs
                and right_run not in comparator_development_runs
                and (
                    not manifest_partitioned
                    or (
                        matched_partition_ids[2 * pair_index] == 0
                        and matched_partition_ids[2 * pair_index + 1] == 0
                    )
                )
                for pair_index, (left_run, right_run) in enumerate(matched_run_ids)
            ],
            dtype=bool,
        )
        if int(np.sum(training_pair_mask)) != expected_training_pair_count:
            raise RuntimeError(
                "The fixed matched-pair budget is not run-disjoint from the "
                "frozen comparator development split."
            )
        comparator_validation_accuracy = float(comparator.validation_accuracy)
        comparator_validation_interval = list(
            comparator.validation_accuracy_interval
        )
    original_action_mask = np.asarray(combined.action_mask, dtype=bool)
    combined = combined._replace(
        action_mask=mask_supervision_partner_runs(
            combined.action_mask,
            combined.partner_run_ids,
            sorted(comparator_development_runs),
        )
    )
    active_anchor_mask = np.any(np.asarray(combined.action_mask, dtype=bool), axis=-1)
    if comparator_development_runs:
        active_runs = set(
            np.asarray(combined.partner_run_ids).reshape((-1,))[active_anchor_mask].tolist()
        )
        overlap = active_runs & comparator_development_runs
        if overlap:
            raise RuntimeError(
                "Comparator development partner runs leaked into DEPI supervision."
            )
        if not np.any(active_anchor_mask):
            raise RuntimeError(
                "Excluding comparator development runs leaves no DEPI supervision anchor."
            )
    pair_valid = (
        np.asarray(probe_valid[0::2])
        & np.asarray(probe_valid[1::2])
        & training_pair_mask
    )
    quotient = QuotientPairBatch(
        anchor_index_a=left,
        anchor_index_b=right,
        decision_distance=decision_distance(signatures[left], signatures[right]),
        weights=jnp.ones((pair_count,), dtype=jnp.float32),
        comparator_distinct_probability=jnp.asarray(pair_probabilities),
        ego_state_a=paired_world.ego_state[0::2],
        ego_state_b=paired_world.ego_state[1::2],
        probe_observations=jnp.stack(
            (paired_observations[0::2], paired_observations[1::2])
        ),
        pair_valid=jnp.asarray(pair_valid),
    )
    _, class_readings = _pair_class_readings(
        quotient=quotient,
        equivalent_probability_max=float(
            config.anchors.observable_equivalent_probability_max
        ),
        distinct_probability_min=float(
            config.anchors.decision_distinct_probability_min
        ),
        signature_threshold=float(config.anchors.signature_distance_threshold),
    )
    budget = {
        "counterfactual_continuation_steps": (
            (ordinary_count + 2 * pair_count)
            * 6
            * (
                int(config.anchors.fit_replicas)
                + int(config.anchors.evaluation_replicas)
            )
            * int(config.anchors.continuation_horizon)
        ),
        "matched_pair_probe_steps": pair_count * 2 * int(config.anchors.probe_steps),
        "invalid_matched_probe_states": int(
            np.sum(~np.asarray(probe_valid, dtype=bool))
        ),
        "pair_class_fractions": class_readings["pair_class_fractions"],
        "irreducible_ambiguity_fraction": class_readings[
            "irreducible_ambiguity_fraction"
        ],
        "comparator_holdout_accuracy": comparator_validation_accuracy,
        "comparator_holdout_accuracy_interval": comparator_validation_interval,
        "comparator_target": "different_empirical_decision_signature_regime",
        "comparator_development_pairs": int(pair_count // 2 if pair_count >= 4 else 0),
        "comparator_training_pairs": int(np.sum(training_pair_mask)),
        "expected_comparator_training_pairs": int(expected_training_pair_count),
        "comparator_excluded_supervision_anchors": int(
            np.sum(np.any(original_action_mask, axis=-1) & ~active_anchor_mask)
        ),
        "depi_supervision_anchor_count": int(np.sum(active_anchor_mask)),
        "matched_pair_candidates": int(candidate_count),
        "matched_pairs_formed": int(pair_matrix.shape[0]),
    }
    return combined, quotient, budget, comparator


def _pair_class_readings(
    *,
    quotient: QuotientPairBatch,
    equivalent_probability_max: float,
    distinct_probability_min: float,
    signature_threshold: float,
) -> tuple[Any, Mapping[str, Any]]:
    classes, equivalent, distinct, fractions = classify_matched_pairs(
        quotient.comparator_distinct_probability,
        quotient.decision_distance,
        equivalent_probability_max=equivalent_probability_max,
        distinct_probability_min=distinct_probability_min,
        signature_threshold=signature_threshold,
        pair_valid=quotient.pair_valid,
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
        config.anchors.matched_history_pairs
    )
    indexes = jnp.zeros((world_count,), dtype=jnp.int32)
    world = world_from_records(records, indexes)
    return select_anchor_microbatch_size(
        chunk_kernel=chunk_kernel,
        runtime=AnchorRuntime(target_params, partner_parameters),
        anchor_ids=jnp.arange(world_count, dtype=jnp.int32),
        root_keys=jax.random.split(key, world_count),
        world=world,
        rollout_flat_indexes=indexes,
        policy_states=world.ego_state,
        observations=gather_time_lanes(records["observations"], indexes),
        partner_sources=gather_time_lanes(records["partner_source"], indexes),
        partner_members=gather_time_lanes(records["partner_member"], indexes),
        partner_family_ids=gather_time_lanes(records["partner_family_id"], indexes),
        partner_checkpoint_stages=gather_time_lanes(
            records["partner_checkpoint_stage"], indexes
        ),
        partner_run_ids=gather_time_lanes(records["partner_run_ids"], indexes),
        action_count=6,
        replicas=int(config.anchors.fit_replicas)
        + int(config.anchors.evaluation_replicas),
    )


__all__ = [
    "AnchorRuntime",
    "FrozenPairComparator",
    "PAIR_AMBIGUOUS",
    "PAIR_DISTINCT",
    "PAIR_EQUIVALENT",
    "PAIR_FRACTION_ORDER",
    "ProbeHistory",
    "classify_matched_pairs",
    "collect_anchor_batch",
    "decision_regime_pair_dataset",
    "decision_signature_prototypes",
    "assign_decision_signature_regimes",
    "fit_pair_comparator",
    "manifest_partitioned_candidate_pairs",
    "gather_time_lanes",
    "make_anchor_functions",
    "mask_supervision_partner_runs",
    "pair_comparator_accuracy",
    "pair_legal_history_features",
    "pair_feature_rows",
    "predict_pair_comparator",
    "preflight_anchor_microbatch_from_records",
    "probe_matched_history",
    "run_disjoint_candidate_pairs",
    "separation_terms_from_matched_pairs",
    "time_source_stratified_indexes",
    "world_from_records",
]
