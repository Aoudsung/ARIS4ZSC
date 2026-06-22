"""ToMZSC PECAN adapter for SERD BranchRecord probes.

This adapter runs against the project-local reproduced PECAN checkpoints.  It
loads the patched ToMZSC/JaxMARL evaluation stack used by the M3 reproduction
and emits environment-agnostic BranchRecord objects for the M4 bundle writer.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from .fixture_env import CONTROL_FAMILIES
from .serd_core import BranchRecord


@dataclass(frozen=True)
class TomzscPecanSerdConfig:
    tomzsc_root: str = (
        "results/serd_pecan_target_domain_reproduction/_patched_tomzsc"
    )
    teammate_dir: str = (
        "results/serd_pecan_target_domain_reproduction/save_teammates/"
        "overcooked_counter_circuit/"
        "teammate_pool_counter_circuit_overcooked_counter_circuit"
    )
    ego_dir: str = (
        "results/serd_pecan_target_domain_reproduction/save_pecan/"
        "overcooked_counter_circuit/pecan_1m_overcooked_counter_circuit"
    )
    cluster_labels: str = (
        "results/serd_pecan_target_domain_reproduction/clusters/pecan_clusters.json"
    )
    reproduction_manifest: str = (
        "results/serd_pecan_target_domain_reproduction/reproduction_manifest.json"
    )
    policy: str = "PECAN"
    domain: str = "ToMZSC-counter_circuit"
    layout: str = "counter_circuit"
    disruptions: tuple[str, ...] = ("missed_handoff", "route_block", "hesitation")
    probes_per_disruption: int = 4
    warmup_horizon: int = 8
    rollout_horizon: int = 20
    probe_mode: str = "policy_warmup"
    max_probe_episodes: int = 30
    seed: int = 154
    teammate_index: int = 0
    ego_index: int = 0
    teammate_agent_id: int = 0
    ego_agent_id: int = 1
    hidden_size: int = 64
    num_layers: int = 1
    norm_type: str = "layer_norm"
    norm_input: bool = False


@dataclass(frozen=True)
class TomzscPecanBundle:
    env: Any
    wrapped_env: Any
    teammate_network: Any
    ego_network: Any
    teammate_params: Any
    ego_params: Any
    teammate_paths: list[Path]
    ego_paths: list[Path]
    cluster_labels: list[int]
    config: TomzscPecanSerdConfig


ACTION_NAMES = {
    0: "up",
    1: "down",
    2: "right",
    3: "left",
    4: "stay",
    5: "interact",
}

ACTION_IDS = {name: idx for idx, name in ACTION_NAMES.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _checkpoint_paths(directory: str | Path) -> list[Path]:
    paths = sorted(_resolve(directory).glob("seed*_vmap*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no safetensors checkpoints found under {directory}")
    return paths


def _first_leaf_shape(tree: Any) -> tuple[int, ...]:
    return tuple(jax.tree_util.tree_leaves(tree)[0].shape)


def _select_index(tree: Any, index: int) -> Any:
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def load_tomzsc_pecan_bundle(config: TomzscPecanSerdConfig) -> TomzscPecanBundle:
    tomzsc_root = _resolve(config.tomzsc_root)
    if str(tomzsc_root) not in sys.path:
        sys.path.insert(0, str(tomzsc_root))

    from env.overcooked import Overcooked
    from env.overcooked_layouts import overcooked_layouts
    from models import JaxMARLLSTM, JaxMARLLSTMGCRL
    from utils.wrappers import CTRolloutManager, LogWrapper, load_params

    env = Overcooked(
        layout=overcooked_layouts[config.layout],
        pomdp=False,
        agent_view_size=5,
    )
    env = LogWrapper(env)
    wrapped_env = CTRolloutManager(env, batch_size=1, preprocess_obs=False)

    teammate_paths = _checkpoint_paths(config.teammate_dir)
    ego_paths = _checkpoint_paths(config.ego_dir)
    cluster_labels = json.loads(_resolve(config.cluster_labels).read_text(encoding="utf-8"))

    def tree_batchify(params: dict[str, Any]) -> Any:
        return jax.tree_util.tree_map(
            lambda *items: jnp.stack(items, axis=0),
            *[params[agent] for agent in ["agent_0", "agent_1"]],
        )

    def load_from_paths(paths: list[Path]) -> Any:
        params = [tree_batchify(load_params(path)) for path in paths]
        return jax.tree_util.tree_map(lambda *items: jnp.stack(items), *params)

    teammate_params_all = load_from_paths(teammate_paths)
    ego_params_all = load_from_paths(ego_paths)
    teammate_params = _select_index(teammate_params_all, config.teammate_index)
    ego_params = _select_index(ego_params_all, config.ego_index)
    if sorted((config.teammate_agent_id, config.ego_agent_id)) != [0, 1]:
        raise ValueError(
            "ToMZSC PECAN SERD expects exactly one teammate and one ego agent over ids 0/1"
        )

    if cluster_labels is None:
        raise ValueError("PECAN cluster labels are required for M4")
    if config.teammate_index >= len(teammate_paths):
        raise IndexError(
            f"teammate_index={config.teammate_index} exceeds {len(teammate_paths)} checkpoints"
        )
    if config.ego_index >= len(ego_paths):
        raise IndexError(
            f"ego_index={config.ego_index} exceeds {len(ego_paths)} checkpoints"
        )

    teammate_network = JaxMARLLSTM(
        action_dim=wrapped_env.max_action_space,
        norm_type=config.norm_type,
        norm_input=config.norm_input,
        use_lstm=False,
    )
    ego_network = JaxMARLLSTMGCRL(
        action_dim=wrapped_env.max_action_space,
        num_concepts=int(max(cluster_labels)) + 2,
        norm_type=config.norm_type,
        norm_input=config.norm_input,
        use_lstm=True,
    )
    return TomzscPecanBundle(
        env=env,
        wrapped_env=wrapped_env,
        teammate_network=teammate_network,
        ego_network=ego_network,
        teammate_params=teammate_params,
        ego_params=ego_params,
        teammate_paths=teammate_paths,
        ego_paths=ego_paths,
        cluster_labels=[int(item) for item in cluster_labels],
        config=config,
    )


def _batchify(env: Any, x: dict[str, Any]) -> Any:
    return jnp.stack([x[agent] for agent in env.agents], axis=0)


def _unbatchify(env: Any, x: Any) -> dict[str, Any]:
    return {agent: x[index] for index, agent in enumerate(env.agents)}


def _init_snapshot(bundle: TomzscPecanBundle, rng: Any) -> tuple[Any, ...]:
    from models import JaxMARLLSTM, JaxMARLLSTMGCRL

    rng, reset_key = jax.random.split(rng)
    obs, state = bundle.env.reset(reset_key)
    teammate_hs = JaxMARLLSTM.initialize_carry(
        bundle.config.hidden_size,
        len(bundle.env.agents),
        1,
    )
    ego_hs = JaxMARLLSTMGCRL.initialize_carry(
        bundle.config.hidden_size,
        len(bundle.env.agents),
        1,
    )
    done = jnp.bool_(False)
    return rng, state, obs, teammate_hs, ego_hs, done


def _make_transition_fn(bundle: TomzscPecanBundle) -> Any:
    @jax.jit
    def transition_fn(
        rng: Any,
        state: Any,
        obs: Any,
        teammate_hs: Any,
        ego_hs: Any,
        done: Any,
        override_actor: Any,
        override_action: Any,
        override_enabled: Any,
    ) -> tuple[Any, ...]:
        valid_actions = bundle.wrapped_env.get_valid_actions(state.env_state)
        obs_batch = _batchify(bundle.env, obs)[:, None, None]
        done_batch = done[None, None]

        next_teammate_hs, teammate_q = jax.vmap(
            bundle.teammate_network.apply,
            in_axes=(0, 0, 0, None, None),
        )(
            bundle.teammate_params,
            teammate_hs,
            obs_batch,
            done_batch,
            False,
        )
        teammate_q = teammate_q.squeeze((1, 2))
        valid_batch = _batchify(bundle.env, valid_actions).squeeze(1)
        teammate_actions = jnp.argmax(teammate_q - (1 - valid_batch) * 1000, axis=-1)
        actions = _unbatchify(bundle.env, teammate_actions)

        next_ego_hs, ego_output = jax.vmap(
            bundle.ego_network.apply,
            in_axes=(0, 0, 0, None),
        )(
            bundle.ego_params,
            ego_hs,
            obs_batch,
            done_batch,
        )
        ego_q = ego_output[0].squeeze((1, 2))
        ego_actions = jnp.argmax(ego_q - (1 - valid_batch) * 1000, axis=-1)
        ego_agent = f"agent_{bundle.config.ego_agent_id}"
        actions[ego_agent] = _unbatchify(bundle.env, ego_actions)[ego_agent]

        enabled = override_enabled.astype(jnp.bool_)
        for agent_index, agent_name in enumerate(bundle.env.agents):
            actions[agent_name] = jnp.where(
                enabled & (override_actor == agent_index),
                override_action,
                actions[agent_name],
            ).astype(jnp.int32)

        rng, step_key = jax.random.split(rng)
        next_obs, next_state, reward, next_done, info = bundle.env.step(
            step_key,
            state,
            actions,
        )
        return (
            rng,
            next_state,
            next_obs,
            next_teammate_hs,
            next_ego_hs,
            next_done["__all__"],
            reward,
            info,
            actions,
        )

    return transition_fn


def _transition(
    bundle: TomzscPecanBundle,
    transition_fn: Any,
    rng: Any,
    state: Any,
    obs: Any,
    teammate_hs: Any,
    ego_hs: Any,
    done: Any,
    override: tuple[int, str] | None,
) -> tuple[Any, ...]:
    if override is None:
        actor_index = 0
        action_id = 0
        enabled = 0
    else:
        actor_index, action_name = override
        action_id = ACTION_IDS[action_name]
        enabled = 1
    return transition_fn(
        rng,
        state,
        obs,
        teammate_hs,
        ego_hs,
        done,
        jnp.asarray(actor_index, dtype=jnp.int32),
        jnp.asarray(action_id, dtype=jnp.int32),
        jnp.asarray(enabled, dtype=jnp.int32),
    )


def _warmup(
    bundle: TomzscPecanBundle,
    transition_fn: Any,
    rng: Any,
    horizon: int,
) -> tuple[Any, ...]:
    rng, state, obs, teammate_hs, ego_hs, done = _init_snapshot(bundle, rng)
    for _ in range(horizon):
        (
            rng,
            state,
            obs,
            teammate_hs,
            ego_hs,
            done,
            _reward,
            _info,
            _actions,
        ) = _transition(
            bundle,
            transition_fn,
            rng,
            state,
            obs,
            teammate_hs,
            ego_hs,
            done,
            None,
        )
    return rng, state, obs, teammate_hs, ego_hs, done


def _reward_event_snapshots(
    bundle: TomzscPecanBundle,
    transition_fn: Any,
    rng: Any,
    needed: int,
) -> tuple[Any, list[tuple[Any, ...]], dict[str, Any]]:
    snapshots: list[tuple[Any, ...]] = []
    event_meta: list[dict[str, int | float]] = []
    episode_index = 0
    max_steps = int(getattr(bundle.env, "max_steps", 400))
    while len(snapshots) < needed and episode_index < bundle.config.max_probe_episodes:
        rng, episode_rng = jax.random.split(rng)
        snapshot = _init_snapshot(bundle, episode_rng)
        for step_index in range(max_steps):
            before = snapshot
            (
                next_rng,
                next_state,
                next_obs,
                next_teammate_hs,
                next_ego_hs,
                next_done,
                reward,
                _info,
                _actions,
            ) = _transition(
                bundle,
                transition_fn,
                *snapshot,
                None,
            )
            reward_value = _reward_scalar(bundle, reward)
            if reward_value > 0.0:
                snapshots.append(before)
                event_meta.append(
                    {
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "reward": reward_value,
                    }
                )
                if len(snapshots) >= needed:
                    break
            snapshot = (
                next_rng,
                next_state,
                next_obs,
                next_teammate_hs,
                next_ego_hs,
                next_done,
            )
            if bool(jax.device_get(jnp.asarray(next_done))):
                break
        episode_index += 1
    if len(snapshots) < needed:
        raise RuntimeError(
            "reward_event probe construction found "
            f"{len(snapshots)} reward-relevant snapshots, needed {needed}; "
            f"increase max_probe_episodes or repair policy/domain role assignment"
        )
    return rng, snapshots, {
        "probe_mode": bundle.config.probe_mode,
        "reward_event_snapshots_found": len(snapshots),
        "reward_event_episodes_scanned": episode_index,
        "reward_event_meta": event_meta,
    }


def _reward_scalar(bundle: TomzscPecanBundle, reward: Any) -> float:
    return float(jax.device_get(jnp.asarray(reward[f"agent_{bundle.config.ego_agent_id}"])))


def _rollout_return(
    bundle: TomzscPecanBundle,
    transition_fn: Any,
    snapshot: tuple[Any, ...],
    horizon: int,
    first_override: tuple[int, str] | None,
) -> tuple[float, float]:
    rng, state, obs, teammate_hs, ego_hs, done = snapshot
    total = 0.0
    for step_idx in range(horizon):
        override = first_override if step_idx == 0 else None
        (
            rng,
            state,
            obs,
            teammate_hs,
            ego_hs,
            done,
            reward,
            _info,
            _actions,
        ) = _transition(
            bundle,
            transition_fn,
            rng,
            state,
            obs,
            teammate_hs,
            ego_hs,
            done,
            override,
        )
        total += _reward_scalar(bundle, reward)
    return total, 1.0 if first_override is not None else 0.0


def _state_value(raw: Any, attr: str, default: float = 0.0) -> Any:
    return getattr(raw, attr, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(jax.device_get(jnp.asarray(value)))
    except Exception:
        return default


def _phi_pre_h(snapshot: tuple[Any, ...], probe_index: int) -> dict[str, float]:
    _rng, state, _obs, _teammate_hs, _ego_hs, _done = snapshot
    raw = jax.device_get(state.env_state)
    agent_pos = _state_value(raw, "agent_pos", None)
    agent_dir = _state_value(raw, "agent_dir_idx", None)
    agent_inv = _state_value(raw, "agent_inv", None)
    time_value = _state_value(raw, "time", probe_index)
    phi = {
        "probe_index": float(probe_index),
        "time": _safe_float(time_value),
    }
    if agent_pos is not None:
        phi.update(
            {
                "p0_x": _safe_float(agent_pos[0, 0]),
                "p0_y": _safe_float(agent_pos[0, 1]),
                "p1_x": _safe_float(agent_pos[1, 0]),
                "p1_y": _safe_float(agent_pos[1, 1]),
            }
        )
    if agent_dir is not None:
        phi.update(
            {
                "p0_dir": _safe_float(agent_dir[0]),
                "p1_dir": _safe_float(agent_dir[1]),
            }
        )
    if agent_inv is not None:
        phi.update(
            {
                "p0_holding": float(_safe_float(agent_inv[0], 1.0) != 1.0),
                "p1_holding": float(_safe_float(agent_inv[1], 1.0) != 1.0),
            }
        )
    return phi


def _semantic_override(disruption: str, probe_index: int) -> tuple[int, str]:
    if disruption == "missed_handoff":
        return (0, "stay")
    if disruption == "route_block":
        return (1, "stay")
    if disruption == "hesitation":
        return (probe_index % 2, "stay")
    raise ValueError(f"unknown disruption: {disruption}")


def _control_override(disruption: str, family: str, probe_index: int) -> tuple[int, str]:
    actor = 0 if disruption == "missed_handoff" else 1
    if disruption == "hesitation":
        actor = probe_index % 2
    action_by_family = {
        "random_lag": "interact",
        "state_block": "up",
        "reward_shaping": "down",
        "naive_replanning": "left",
    }
    if family not in action_by_family:
        raise ValueError(f"unknown control family: {family}")
    return actor, action_by_family[family]


def make_tomzsc_pecan_branch_records(
    config: TomzscPecanSerdConfig,
) -> tuple[list[BranchRecord], list[BranchRecord], dict[str, Any]]:
    bundle = load_tomzsc_pecan_bundle(config)
    transition_fn = _make_transition_fn(bundle)
    semantic_records: list[BranchRecord] = []
    control_records: list[BranchRecord] = []
    rng = jax.random.PRNGKey(config.seed)
    needed_snapshots = len(config.disruptions) * config.probes_per_disruption
    reward_event_meta: dict[str, Any] = {}
    reward_event_snapshots: list[tuple[Any, ...]] = []
    if config.probe_mode == "reward_event":
        rng, reward_event_snapshots, reward_event_meta = _reward_event_snapshots(
            bundle,
            transition_fn,
            rng,
            needed_snapshots,
        )
    elif config.probe_mode != "policy_warmup":
        raise ValueError(f"unknown probe_mode: {config.probe_mode}")

    snapshot_index = 0
    for disruption in config.disruptions:
        for probe_index in range(config.probes_per_disruption):
            if config.probe_mode == "reward_event":
                snapshot = reward_event_snapshots[snapshot_index]
                snapshot_index += 1
            else:
                rng, probe_rng = jax.random.split(rng)
                snapshot = _warmup(
                    bundle,
                    transition_fn,
                    probe_rng,
                    config.warmup_horizon + probe_index,
                )
            phi = _phi_pre_h(snapshot, probe_index)
            no_shock_return, _zero = _rollout_return(
                bundle,
                transition_fn,
                snapshot,
                horizon=config.rollout_horizon,
                first_override=None,
            )
            sem_return, sem_mag = _rollout_return(
                bundle,
                transition_fn,
                snapshot,
                horizon=config.rollout_horizon,
                first_override=_semantic_override(disruption, probe_index),
            )
            probe_id = f"{config.policy}:{config.domain}:{disruption}:{probe_index:04d}"
            semantic_records.append(
                BranchRecord(
                    probe_id=probe_id,
                    policy=config.policy,
                    domain=config.domain,
                    disruption=disruption,
                    family="semantic",
                    no_shock_return=no_shock_return,
                    branch_return=sem_return,
                    shock_magnitude=sem_mag,
                    phi_pre_h=phi,
                )
            )
            for family in CONTROL_FAMILIES:
                control_return, control_mag = _rollout_return(
                    bundle,
                    transition_fn,
                    snapshot,
                    horizon=config.rollout_horizon,
                    first_override=_control_override(disruption, family, probe_index),
                )
                control_records.append(
                    BranchRecord(
                        probe_id=probe_id,
                        policy=config.policy,
                        domain=config.domain,
                        disruption=disruption,
                        family=family,
                        no_shock_return=no_shock_return,
                        branch_return=control_return,
                        shock_magnitude=control_mag,
                        phi_pre_h=phi,
                    )
                )

    manifest_path = _resolve(config.reproduction_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_paths = {
        "teammates": [str(path) for path in bundle.teammate_paths],
        "ego": [str(path) for path in bundle.ego_paths],
        "cluster_labels": str(_resolve(config.cluster_labels)),
        "reproduction_manifest": str(manifest_path),
    }
    provenance = {
        "policy_family": "PECAN",
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_hashes": {
            "teammates": {str(path): sha256_file(path) for path in bundle.teammate_paths},
            "ego": {str(path): sha256_file(path) for path in bundle.ego_paths},
            "cluster_labels": sha256_file(_resolve(config.cluster_labels)),
            "reproduction_manifest": sha256_file(manifest_path),
        },
        "source_repository_path": str(_resolve(config.tomzsc_root)),
        "adapter_route": "TOMZSC_PECAN_NATIVE_JAXMARL_SERD_SELECTED",
        "environment_name": f"overcooked_{config.layout}",
        "target_domain": config.domain,
        "random_seeds": [config.seed],
        "m3_acceptance_artifact": "refine-logs/M3_PECAN_TARGET_DOMAIN_ACCEPTANCE.md",
        "m3_reproduction_status": manifest.get("status"),
        "policy_param_shapes": {
            "teammate_first_leaf": _first_leaf_shape(bundle.teammate_params),
            "ego_first_leaf": _first_leaf_shape(bundle.ego_params),
        },
        "cluster_labels": bundle.cluster_labels,
        "teammate_index": config.teammate_index,
        "ego_index": config.ego_index,
        "teammate_agent_id": config.teammate_agent_id,
        "ego_agent_id": config.ego_agent_id,
        "probe_mode": config.probe_mode,
        "max_probe_episodes": config.max_probe_episodes,
        **reward_event_meta,
    }
    return semantic_records, control_records, provenance
