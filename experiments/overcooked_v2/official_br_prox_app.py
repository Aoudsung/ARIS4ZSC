"""Empirical local BR-Prox for every method on the common partner panel.

This audit replays the exact Official policy/environment key schedule up to a
legal-history state, clones both recurrent policies and the simulator, forces
each ego action, and evaluates the fit-selected action on independent future
replicas.  It estimates a one-action-deviation quantity under each frozen
deployed continuation policy; it is not an unrestricted best response.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import numpy as np

from experiments.overcooked_v2.common_partner_app import (
    COMMON_PARTNER_COUNT,
    VIRTUAL_PARTNER_PANEL,
    _load_common_panel_policies,
    _official_environment,
    _parse_manifests,
    _validate_common_panel,
)
from experiments.overcooked_v2.official_adapter import (
    validate_official_runtime,
)
from experiments.overcooked_v2.official_evaluation_app import FORMAL_METHODS, _load_policies
from experiments.overcooked_v2.official_policy import OfficialDEPIPolicy
from src.path_c.anchor_sampling import ProbeHistory
from src.path_c.evaluation import br_prox
from src.path_c.experiment import OFFICIAL_EVALUATION_ROOT_SEED, load_config, load_partner_manifest
from src.path_c.resources import ResourceLedger
from src.path_c.storage import (
    ensure_run_identity,
    runtime_provenance,
    sha256_path,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
    write_parquet,
)


class _TrajectoryCarry(NamedTuple):
    observations: Any
    environment_state: Any
    done: Any
    left_hstate: Any
    right_hstate: Any


class _BranchCarry(NamedTuple):
    observations: Any
    environment_state: Any
    done: Any
    left_hstate: Any
    right_hstate: Any
    raw_return: Any


def _stack_initial_hstate(policy: Any, count: int) -> Any:
    import jax
    import jax.numpy as jnp

    initial = policy.init_hstate(1)
    if initial is None:
        return None
    return jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(
            jnp.asarray(value)[None, ...], (int(count),) + jnp.asarray(value).shape
        ),
        initial,
    )


def _vmap_policy(policy: Any, observation: Any, done: Any, hstate: Any, keys: Any) -> tuple[Any, Any]:
    import jax

    if hstate is None:
        actions, unused = jax.vmap(
            lambda obs, terminal, key: policy.compute_action(obs, terminal, None, key)
        )(observation, done, keys)
        return actions, unused
    return jax.vmap(policy.compute_action)(observation, done, hstate, keys)


def _official_step_keys(episode_keys: Any, steps: int) -> tuple[Any, Any]:
    """Return exact reset and per-step keys used by Official ``get_rollout``."""

    import jax
    import jax.numpy as jnp

    split = jax.vmap(lambda key: jax.random.split(key, 2))(episode_keys)
    rollout_keys, reset_keys = split[:, 0], split[:, 1]
    per_episode = jax.vmap(lambda key: jax.random.split(key, int(steps)))(rollout_keys)
    return reset_keys, jnp.swapaxes(per_episode, 0, 1)


def _record_trajectories(
    *, left: Any, right: Any, environment: Any, episode_keys: Any
) -> Mapping[str, Any]:
    import jax
    import jax.numpy as jnp

    count = int(episode_keys.shape[0])
    reset_keys, step_keys = _official_step_keys(episode_keys, environment.max_steps)
    observations, state = jax.vmap(environment.reset)(reset_keys)
    done = {
        "agent_0": jnp.zeros((count,), dtype=jnp.bool_),
        "agent_1": jnp.zeros((count,), dtype=jnp.bool_),
        "__all__": jnp.zeros((count,), dtype=jnp.bool_),
    }
    initial = _TrajectoryCarry(
        observations=observations,
        environment_state=state,
        done=done,
        left_hstate=_stack_initial_hstate(left, count),
        right_hstate=_stack_initial_hstate(right, count),
    )

    def one(current: _TrajectoryCarry, keys: Any) -> tuple[_TrajectoryCarry, Mapping[str, Any]]:
        split = jax.vmap(lambda key: jax.random.split(key, 2))(keys)
        sample_roots, environment_keys = split[:, 0], split[:, 1]
        action_keys = jax.vmap(lambda key: jax.random.split(key, 2))(sample_roots)
        left_action, left_hstate = _vmap_policy(
            left,
            current.observations["agent_0"],
            current.done["agent_0"],
            current.left_hstate,
            action_keys[:, 0],
        )
        right_action, right_hstate = _vmap_policy(
            right,
            current.observations["agent_1"],
            current.done["agent_1"],
            current.right_hstate,
            action_keys[:, 1],
        )
        actions = {"agent_0": left_action, "agent_1": right_action}
        next_observations, next_state, unused_reward, next_done, unused_info = jax.vmap(
            environment.step
        )(environment_keys, current.environment_state, actions)
        del unused_reward, unused_info
        following = _TrajectoryCarry(
            observations=next_observations,
            environment_state=next_state,
            done=next_done,
            left_hstate=left_hstate,
            right_hstate=right_hstate,
        )
        return following, {
            "observations": current.observations,
            "environment_state": current.environment_state,
            "done": current.done,
            "left_hstate": current.left_hstate,
            "right_hstate": current.right_hstate,
            "left_action": left_action,
            "right_action": right_action,
        }

    unused_final, records = jax.lax.scan(one, initial, step_keys)
    del unused_final
    return records


def _gather_time_episode(tree: Any, time_indexes: Any, episode_indexes: Any) -> Any:
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(
        lambda value: jnp.asarray(value)[time_indexes, episode_indexes], tree
    )


def _repeat_tree(tree: Any, repeats: int) -> Any:
    import jax
    import jax.numpy as jnp

    if tree is None:
        return None
    return jax.tree_util.tree_map(
        lambda value: jnp.repeat(jnp.asarray(value), int(repeats), axis=0), tree
    )


def _select_tree(active: Any, candidate: Any, current: Any) -> Any:
    import jax
    import jax.numpy as jnp

    def choose(left: Any, right: Any) -> Any:
        mask = jnp.asarray(active, dtype=jnp.bool_)
        expanded = mask.reshape(mask.shape + (1,) * (jnp.ndim(left) - mask.ndim))
        return jnp.where(expanded, left, right)

    return jax.tree_util.tree_map(choose, candidate, current)


def _record_forced_action(policy: Any, hstate: Any, action: Any) -> Any:
    """Make the DEPI legal-history state reflect the forced diagnostic action."""

    import jax.numpy as jnp

    if not isinstance(policy, OfficialDEPIPolicy):
        return hstate
    target = jnp.asarray(hstate.previous_action)
    expanded = jnp.asarray(action, dtype=jnp.int32).reshape(
        action.shape + (1,) * (target.ndim - action.ndim)
    )
    return hstate._replace(
        previous_action=jnp.broadcast_to(expanded, target.shape),
    )


def _shared_branch_roots(anchor_roots: Any, action_count: int, replicas: int) -> Any:
    import jax
    import jax.numpy as jnp

    replica_ids = jnp.arange(int(replicas), dtype=jnp.uint32)

    def one(root: Any) -> Any:
        keys = jax.vmap(lambda replica: jax.random.fold_in(root, replica))(replica_ids)
        return jnp.broadcast_to(
            keys[None, ...], (int(action_count), int(replicas), 2)
        ).reshape((int(action_count) * int(replicas), 2))

    return jax.vmap(one)(anchor_roots).reshape((-1, 2))


def record_official_anchor_snapshots(
    *,
    left: Any,
    right: Any,
    environment: Any,
    root_key: Any,
    anchors: int,
    continuation_horizon: int,
    episodes: int = 500,
) -> Mapping[str, Any]:
    """Replay legal Official histories and return continuation-safe snapshots.

    Selection is fixed before simulation and uses the same episode-key schedule
    as the Official evaluator.  A selected time index always leaves the full
    registered continuation horizon in the episode, so no padded or already
    terminal branch can enter a mechanism control.
    """

    import jax
    import jax.numpy as jnp

    episode_count = int(episodes)
    anchor_count = int(anchors)
    horizon = int(continuation_horizon)
    available_steps = int(environment.max_steps) - horizon + 1
    if episode_count <= 0 or anchor_count <= 0 or horizon <= 0:
        raise ValueError("Snapshot episodes, anchors, and horizon must be positive.")
    if available_steps <= 0:
        raise ValueError("Continuation horizon exceeds the Official episode.")
    flat_count = available_steps * episode_count
    if anchor_count > flat_count:
        raise ValueError("Requested more mechanism anchors than legal states.")
    episode_keys = jax.random.split(root_key, episode_count)
    selection_key = jax.random.fold_in(root_key, 0x4D454348)
    flat_indexes = jnp.sort(
        jax.random.choice(
            selection_key,
            flat_count,
            shape=(anchor_count,),
            replace=False,
        )
    )
    time_indexes = flat_indexes // episode_count
    episode_indexes = flat_indexes % episode_count
    selected_episode_keys = episode_keys[episode_indexes]
    records = _record_trajectories(
        left=left,
        right=right,
        environment=environment,
        episode_keys=selected_episode_keys,
    )
    selected = _gather_time_episode(
        records,
        time_indexes,
        jnp.arange(anchor_count, dtype=jnp.int32),
    )
    anchor_episode_keys = episode_keys[episode_indexes]
    anchor_roots = jax.vmap(
        lambda key, time: jax.random.fold_in(
            jax.random.fold_in(key, 0x4D454348), time
        )
    )(anchor_episode_keys, time_indexes.astype(jnp.uint32))
    if bool(np.any(np.asarray(selected["done"]["__all__"]))):
        raise RuntimeError("A mechanism snapshot was terminal before intervention.")
    return {
        "selected": selected,
        "flat_indexes": flat_indexes,
        "time_indexes": time_indexes,
        "episode_indexes": episode_indexes,
        "anchor_roots": anchor_roots,
    }


def record_official_history_anchor_snapshots(
    *,
    left: Any,
    right: Any,
    ego_role: int,
    environment: Any,
    root_key: Any,
    anchors: int,
    history_steps: int,
    continuation_horizon: int,
    episodes: int = 500,
) -> Mapping[str, Any]:
    """Replay fixed-policy histories for the external pair comparator.

    Unlike the training-time matched probe, this collector samples only times
    with a complete preceding legal-history window and a complete future
    continuation window.  It returns ego observations/actions only; partner
    actions and partner identity never enter the comparator features.
    """

    import jax
    import jax.numpy as jnp

    role = int(ego_role)
    episode_count = int(episodes)
    anchor_count = int(anchors)
    past = int(history_steps)
    future = int(continuation_horizon)
    if role not in {0, 1}:
        raise ValueError("ego_role must be zero or one.")
    if min(episode_count, anchor_count, past, future) <= 0:
        raise ValueError(
            "Comparator episodes, anchors, history, and continuation must be positive."
        )
    eligible_time_count = int(environment.max_steps) - past - future + 1
    if eligible_time_count <= 0:
        raise ValueError("Comparator history plus continuation exceeds the episode.")
    flat_count = eligible_time_count * episode_count
    if anchor_count > flat_count:
        raise ValueError("Requested more comparator histories than legal states.")

    episode_keys = jax.random.split(root_key, episode_count)
    selection_key = jax.random.fold_in(root_key, 0x434F4D50)
    flat_indexes = jnp.sort(
        jax.random.choice(
            selection_key,
            flat_count,
            shape=(anchor_count,),
            replace=False,
        )
    )
    time_indexes = past + flat_indexes // episode_count
    episode_indexes = flat_indexes % episode_count
    selected_episode_keys = episode_keys[episode_indexes]
    records = _record_trajectories(
        left=left,
        right=right,
        environment=environment,
        episode_keys=selected_episode_keys,
    )
    lane_indexes = jnp.arange(anchor_count, dtype=jnp.int32)
    selected = _gather_time_episode(records, time_indexes, lane_indexes)
    agent = "agent_0" if role == 0 else "agent_1"
    action_field = "left_action" if role == 0 else "right_action"
    observation_rows = []
    action_rows = []
    for offset in range(past):
        history_time = time_indexes - past + offset
        observation_rows.append(
            records["observations"][agent][history_time, lane_indexes]
        )
        action_rows.append(records[action_field][history_time, lane_indexes])
    history = ProbeHistory(
        ego_actions=jnp.stack(action_rows, axis=1),
        ego_observations=jnp.stack(observation_rows, axis=1),
    )
    anchor_episode_keys = episode_keys[episode_indexes]
    anchor_roots = jax.vmap(
        lambda key, time: jax.random.fold_in(
            jax.random.fold_in(key, 0x434F4D50), time
        )
    )(anchor_episode_keys, time_indexes.astype(jnp.uint32))
    if bool(np.any(np.asarray(selected["done"]["__all__"]))):
        raise RuntimeError("A comparator snapshot was terminal before intervention.")
    return {
        "selected": selected,
        "history": history,
        "flat_indexes": flat_indexes,
        "time_indexes": time_indexes,
        "episode_indexes": episode_indexes,
        "anchor_roots": anchor_roots,
    }


def empirical_all_action_continuations_from_snapshots(
    *,
    left: Any,
    right: Any,
    ego_role: int,
    environment: Any,
    selected: Mapping[str, Any],
    anchor_roots: Any,
    fit_replicas: int,
    evaluation_replicas: int,
    continuation_horizon: int,
    ego_hstate: Any | None = None,
) -> Mapping[str, Any]:
    """Evaluate all six forced ego actions with paired real continuations.

    Fit replicas select the empirical oracle; a disjoint replica block reports
    every action value.  ``ego_hstate`` may replace only the ego recurrent
    state, which is the intervention surface used by history-shuffle,
    state-only, swap-u, and swap-c controls.  The environment, partner state,
    role, current observation, action keys, and continuation keys remain fixed.
    """

    import jax
    import jax.numpy as jnp

    role = int(ego_role)
    if role not in {0, 1}:
        raise ValueError("ego_role must be zero or one.")
    fit_count = int(fit_replicas)
    evaluation_count = int(evaluation_replicas)
    horizon = int(continuation_horizon)
    if min(fit_count, evaluation_count, horizon) <= 0:
        raise ValueError("Continuation replica counts and horizon must be positive.")
    roots_array = jnp.asarray(anchor_roots, dtype=jnp.uint32)
    anchor_count = int(roots_array.shape[0])
    if roots_array.shape != (anchor_count, 2):
        raise ValueError("Every continuation anchor needs one two-word root key.")
    left_hstate = selected["left_hstate"]
    right_hstate = selected["right_hstate"]
    if ego_hstate is not None:
        if role == 0:
            left_hstate = ego_hstate
        else:
            right_hstate = ego_hstate

    ego_name = "agent_0" if role == 0 else "agent_1"
    ego_policy = left if role == 0 else right
    selected_action_keys = jax.vmap(
        lambda key: jax.random.fold_in(key, 0x4143544E)
    )(roots_array)
    selected_actions, unused_selected_states = _vmap_policy(
        ego_policy,
        selected["observations"][ego_name],
        selected["done"][ego_name],
        left_hstate if role == 0 else right_hstate,
        selected_action_keys,
    )
    del unused_selected_states

    replica_count = fit_count + evaluation_count
    action_count = 6
    repeats = action_count * replica_count
    forced_actions = jnp.tile(
        jnp.repeat(jnp.arange(action_count, dtype=jnp.int32), replica_count),
        anchor_count,
    )
    roots = _shared_branch_roots(roots_array, action_count, replica_count)
    branch = _BranchCarry(
        observations=_repeat_tree(selected["observations"], repeats),
        environment_state=_repeat_tree(selected["environment_state"], repeats),
        done=_repeat_tree(selected["done"], repeats),
        left_hstate=_repeat_tree(left_hstate, repeats),
        right_hstate=_repeat_tree(right_hstate, repeats),
        raw_return=jnp.zeros((anchor_count * repeats,), dtype=jnp.float32),
    )

    def advance(step: int, current: _BranchCarry) -> _BranchCarry:
        active = ~jnp.asarray(current.done["__all__"], dtype=jnp.bool_)
        lane_step_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, step)
        )(roots)
        split = jax.vmap(lambda key: jax.random.split(key, 2))(lane_step_keys)
        sample_roots, environment_keys = split[:, 0], split[:, 1]
        action_keys = jax.vmap(lambda key: jax.random.split(key, 2))(sample_roots)
        left_action, next_left = _vmap_policy(
            left,
            current.observations["agent_0"],
            current.done["agent_0"],
            current.left_hstate,
            action_keys[:, 0],
        )
        right_action, next_right = _vmap_policy(
            right,
            current.observations["agent_1"],
            current.done["agent_1"],
            current.right_hstate,
            action_keys[:, 1],
        )
        first = step == 0
        if role == 0:
            left_action = jnp.where(first, forced_actions, left_action)
            forced_left = _record_forced_action(left, next_left, forced_actions)
            next_left = jax.tree_util.tree_map(
                lambda forced, normal: jnp.where(first, forced, normal),
                forced_left,
                next_left,
            )
        else:
            right_action = jnp.where(first, forced_actions, right_action)
            forced_right = _record_forced_action(right, next_right, forced_actions)
            next_right = jax.tree_util.tree_map(
                lambda forced, normal: jnp.where(first, forced, normal),
                forced_right,
                next_right,
            )
        next_observations, next_state, rewards, next_done, unused_info = jax.vmap(
            environment.step
        )(
            environment_keys,
            current.environment_state,
            {"agent_0": left_action, "agent_1": right_action},
        )
        del unused_info
        candidate = _BranchCarry(
            observations=next_observations,
            environment_state=next_state,
            done=next_done,
            left_hstate=next_left,
            right_hstate=next_right,
            raw_return=current.raw_return
            + jnp.where(active, rewards["agent_0"], 0.0),
        )
        return _select_tree(active, candidate, current)

    final = jax.lax.fori_loop(0, horizon, advance, branch)
    values = np.asarray(final.raw_return, dtype=np.float64).reshape(
        (anchor_count, action_count, replica_count)
    )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Mechanism continuation returned non-finite rewards.")
    fit = np.mean(values[..., :fit_count], axis=-1)
    evaluation = np.mean(values[..., fit_count:], axis=-1)
    selected_host = np.asarray(selected_actions, dtype=np.int64).reshape(anchor_count)
    oracle = np.argmax(fit, axis=-1)
    rows = np.arange(anchor_count)
    return {
        "fit_returns_by_action": fit,
        "fit_replica_returns_by_action": values[..., :fit_count],
        "evaluation_returns_by_action": evaluation,
        "evaluation_replica_returns_by_action": values[..., fit_count:],
        "selected_action": selected_host,
        "selected_evaluation_return": evaluation[rows, selected_host],
        "fit_oracle_action": oracle,
        "oracle_evaluation_return": evaluation[rows, oracle],
    }


def empirical_official_br_prox_pairing(
    *,
    ego_policy: Any,
    partner_policy: Any,
    ego_role: int,
    environment: Any,
    root_key: Any,
    anchors: int,
    fit_replicas: int,
    evaluation_replicas: int,
    continuation_horizon: int,
    episodes: int = 500,
) -> list[Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp

    if int(ego_role) not in {0, 1}:
        raise ValueError("ego_role must be zero or one.")
    episode_count = int(episodes)
    if episode_count <= 0:
        raise ValueError("episodes must be positive.")
    episode_keys = jax.random.split(root_key, episode_count)
    flat_count = int(environment.max_steps) * episode_count
    selection_key = jax.random.fold_in(root_key, 0x42525058)
    flat_indexes = jnp.sort(
        jax.random.choice(
            selection_key,
            flat_count,
            shape=(int(anchors),),
            replace=False,
        )
    )
    time_indexes = flat_indexes // episode_count
    episode_indexes = flat_indexes % episode_count
    left, right = (
        (ego_policy, partner_policy) if int(ego_role) == 0 else (partner_policy, ego_policy)
    )
    # Select the registered flat state indexes before simulation, then replay
    # only their episode keys.  This is exactly equivalent to recording all
    # 500x400 states because Official trajectories are independent by episode,
    # while reducing the retained state tree from 200,000 lanes to
    # anchors*400 lanes.  Duplicate selected episodes remain duplicated with
    # the same key, preserving the original trajectory and action RNG.
    selected_episode_keys = episode_keys[episode_indexes]
    records = _record_trajectories(
        left=left,
        right=right,
        environment=environment,
        episode_keys=selected_episode_keys,
    )
    selected = _gather_time_episode(
        records,
        time_indexes,
        jnp.arange(int(anchors), dtype=jnp.int32),
    )
    actual = selected["left_action"] if int(ego_role) == 0 else selected["right_action"]
    anchor_episode_keys = episode_keys[episode_indexes]
    anchor_roots = jax.vmap(
        lambda key, time: jax.random.fold_in(
            jax.random.fold_in(key, 0x42525058), time
        )
    )(anchor_episode_keys, time_indexes.astype(jnp.uint32))

    replica_count = int(fit_replicas + evaluation_replicas)
    action_count = 6
    repeats = action_count * replica_count
    forced_actions = jnp.tile(
        jnp.repeat(jnp.arange(action_count, dtype=jnp.int32), replica_count),
        int(anchors),
    )
    roots = _shared_branch_roots(anchor_roots, action_count, replica_count)
    expanded_done = _repeat_tree(selected["done"], repeats)
    branch = _BranchCarry(
        observations=_repeat_tree(selected["observations"], repeats),
        environment_state=_repeat_tree(selected["environment_state"], repeats),
        done=expanded_done,
        left_hstate=_repeat_tree(selected["left_hstate"], repeats),
        right_hstate=_repeat_tree(selected["right_hstate"], repeats),
        raw_return=jnp.zeros((int(anchors) * repeats,), dtype=jnp.float32),
    )

    def advance(step: int, current: _BranchCarry) -> _BranchCarry:
        active = ~jnp.asarray(current.done["__all__"], dtype=jnp.bool_)
        lane_step_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, step)
        )(roots)
        split = jax.vmap(lambda key: jax.random.split(key, 2))(lane_step_keys)
        sample_roots, environment_keys = split[:, 0], split[:, 1]
        action_keys = jax.vmap(lambda key: jax.random.split(key, 2))(sample_roots)
        left_action, next_left = _vmap_policy(
            left,
            current.observations["agent_0"],
            current.done["agent_0"],
            current.left_hstate,
            action_keys[:, 0],
        )
        right_action, next_right = _vmap_policy(
            right,
            current.observations["agent_1"],
            current.done["agent_1"],
            current.right_hstate,
            action_keys[:, 1],
        )
        first = step == 0
        if int(ego_role) == 0:
            left_action = jnp.where(first, forced_actions, left_action)
            forced_left = _record_forced_action(left, next_left, forced_actions)
            next_left = jax.tree_util.tree_map(
                lambda forced, normal: jnp.where(first, forced, normal),
                forced_left,
                next_left,
            )
        else:
            right_action = jnp.where(first, forced_actions, right_action)
            forced_right = _record_forced_action(right, next_right, forced_actions)
            next_right = jax.tree_util.tree_map(
                lambda forced, normal: jnp.where(first, forced, normal),
                forced_right,
                next_right,
            )
        actions = {"agent_0": left_action, "agent_1": right_action}
        next_observations, next_state, rewards, next_done, unused_info = jax.vmap(
            environment.step
        )(environment_keys, current.environment_state, actions)
        del unused_info
        candidate = _BranchCarry(
            observations=next_observations,
            environment_state=next_state,
            done=next_done,
            left_hstate=next_left,
            right_hstate=next_right,
            raw_return=current.raw_return
            + jnp.where(active, rewards["agent_0"], 0.0),
        )
        return _select_tree(active, candidate, current)

    final = jax.lax.fori_loop(0, int(continuation_horizon), advance, branch)
    values = np.asarray(final.raw_return, dtype=np.float64).reshape(
        (int(anchors), action_count, replica_count)
    )
    fit = np.mean(values[..., : int(fit_replicas)], axis=-1)
    evaluation = np.mean(values[..., int(fit_replicas) :], axis=-1)
    actual_host = np.asarray(actual, dtype=np.int64)
    oracle = np.argmax(fit, axis=-1)
    rows = []
    for index in range(int(anchors)):
        actual_return = float(evaluation[index, actual_host[index]])
        oracle_return = float(evaluation[index, oracle[index]])
        rows.append(
            {
                "anchor_index": index,
                "rollout_flat_index": int(np.asarray(flat_indexes[index])),
                "episode_index": int(np.asarray(episode_indexes[index])),
                "time_index": int(np.asarray(time_indexes[index])),
                "selected_action": int(actual_host[index]),
                "fit_oracle_action": int(oracle[index]),
                "selected_evaluation_return": actual_return,
                "oracle_evaluation_return": oracle_return,
                "raw_local_br_regret": oracle_return - actual_return,
                "br_prox": br_prox(actual_return, oracle_return),
                "oracle_action_agreement": bool(actual_host[index] == oracle[index]),
                "scope": "one_action_deviation_with_frozen_deployed_continuation",
            }
        )
    return rows


def run_common_br_prox(args: argparse.Namespace) -> None:
    import jax

    validate_formal_repository_state()
    validate_registered_python_runtime()
    runtime = validate_official_runtime()
    config = load_config(args.config, run_kind="formal")
    manifests = _parse_manifests(args.policy_manifest, layout=config.environment.layout)
    panel_path = Path(args.partner_manifest).resolve()
    panel = load_partner_manifest(
        panel_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    partners = _validate_common_panel(panel, manifests)
    output = Path(args.output).resolve()
    identity = {
        "stage": "common-partner-local-br-prox",
        "layout": config.environment.layout,
        "official_runtime": runtime,
        "repository_runtime": runtime_provenance(),
        "config_fingerprint": config.fingerprint,
        "panel_sha256": sha256_path(panel_path),
        "policy_manifests": manifests,
        "virtual_partner_panel": VIRTUAL_PARTNER_PANEL,
        "root_key": [0, OFFICIAL_EVALUATION_ROOT_SEED],
        "scope": "one_action_deviation_with_frozen_deployed_continuation",
    }
    ensure_run_identity(output, identity)
    environment = _official_environment(config)
    partners, partner_policies = _load_common_panel_policies(partners)
    root_key = jax.random.PRNGKey(OFFICIAL_EVALUATION_ROOT_SEED)
    all_rows = []
    for method in FORMAL_METHODS:
        ego_policies = _load_policies(manifests[method], config)
        for ego_index, ego in enumerate(ego_policies):
            ego_run_id = str(manifests[method]["runs"][ego_index]["run_id"])
            for partner_index, (partner_run, partner) in enumerate(
                zip(partners, partner_policies, strict=True)
            ):
                for role in (0, 1):
                    rows = empirical_official_br_prox_pairing(
                        ego_policy=ego,
                        partner_policy=partner,
                        ego_role=role,
                        environment=environment,
                        # As in the two main scoreboards, every pairing receives
                        # the identical registered episode-key vector.
                        root_key=root_key,
                        anchors=config.evaluation.br_prox_anchors_per_pairing,
                        fit_replicas=config.evaluation.br_prox_fit_replicas,
                        evaluation_replicas=config.evaluation.br_prox_evaluation_replicas,
                        continuation_horizon=config.evaluation.br_prox_continuation_horizon,
                    )
                    for row in rows:
                        all_rows.append(
                            {
                                **row,
                                "layout": config.environment.layout,
                                "method": method,
                                "ego_run_id": ego_run_id,
                                "ego_run_index": ego_index,
                                "partner_run_id": partner_run.run_id,
                                "partner_run_index": partner_index,
                                "partner_mechanism": partner_run.generation_mechanism,
                                "ego_role": role,
                            }
                        )
    write_parquet(output / "common_partner_br_prox.parquet", all_rows)
    summaries = {}
    for method in FORMAL_METHODS:
        current = [row for row in all_rows if row["method"] == method]
        summaries[method] = {
            "mean_br_prox": float(np.mean([row["br_prox"] for row in current])),
            "mean_raw_local_br_regret": float(
                np.mean([row["raw_local_br_regret"] for row in current])
            ),
            "oracle_action_agreement": float(
                np.mean([row["oracle_action_agreement"] for row in current])
            ),
            "anchor_count": len(current),
        }
    write_json(output / "common_partner_br_prox_summary.json", summaries)
    attempted = (
        len(FORMAL_METHODS)
        * 10
        * COMMON_PARTNER_COUNT
        * 2
        * config.evaluation.br_prox_anchors_per_pairing
        * 6
        * (
            config.evaluation.br_prox_fit_replicas
            + config.evaluation.br_prox_evaluation_replicas
        )
        * config.evaluation.br_prox_continuation_horizon
    )
    write_json(
        output / "budget_ledger.json",
        ResourceLedger(evaluation_steps=attempted).to_mapping(),
    )


__all__ = [
    "empirical_all_action_continuations_from_snapshots",
    "record_official_history_anchor_snapshots",
    "empirical_official_br_prox_pairing",
    "record_official_anchor_snapshots",
    "run_common_br_prox",
]
