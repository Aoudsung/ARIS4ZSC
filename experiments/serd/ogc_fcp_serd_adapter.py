"""Native OGC FCP adapter for FCP-only SERD diagnostic branch records.

This module is designed to run on the remote OGC host. It uses OGC's own
Overcooked environment, FCP population checkpoint, and heterogenous MAPPO policy
call path to emit SERD-compatible BranchRecord rows. It is diagnostic-only until
target-domain PECAN evidence exists.
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
class OgcFcpSerdConfig:
    ogc_src: str = "/apps/users/cxw/ZSC_coordinator/external/OGC/src"
    population_json: str = "populations/fcp/Overcooked-CounterCircuit6_9/population.json"
    log_dir: str = "/apps/users/cxw/logs/minimax"
    ego_xpid: str = ""
    checkpoint_name: str = "checkpoint"
    agent_id: int = 1
    band: str = "low"
    agent_idx: int = 0
    env_name: str = "Overcooked-CounterCircuit6_9"
    policy: str = "FCP"
    domain: str = "OGC-CounterCircuit6_9"
    disruptions: tuple[str, ...] = ("missed_handoff", "route_block", "hesitation")
    probes_per_disruption: int = 4
    warmup_horizon: int = 8
    rollout_horizon: int = 20
    seed: int = 1


@dataclass(frozen=True)
class OgcPolicyBundle:
    runner: Any
    benv: Any
    pop: Any
    params_0: Any
    params_1: Any
    ego_checkpoint: Path
    ego_meta: Path
    partner_checkpoint: Path
    partner_meta: Path
    population_path: Path
    population_index: int
    source_commit_hash: str | None


ACTION_NAMES = {
    0: "right",
    1: "down",
    2: "left",
    3: "up",
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


def resolve_under(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def checkpoint_entry(
    population: dict[str, Any],
    agent_id: int,
    band: str,
) -> tuple[int, str, str]:
    offset = {"low": 1, "mid": 2, "high": 3}[band]
    population_index = (agent_id - 1) * 3 + offset
    key = str(population_index)
    meta_key = f"{population_index}_meta"
    if key not in population:
        raise KeyError(f"population entry missing: {key}")
    if meta_key not in population:
        raise KeyError(f"population meta entry missing: {meta_key}")
    return population_index, str(population[key]), str(population[meta_key])


def _first_leaf_shape(tree: Any) -> tuple[int, ...]:
    return tuple(jax.tree_util.tree_leaves(tree)[0].shape)


def _select_agent_params(params: Any, agent_idx: int) -> Any:
    return jax.tree_util.tree_map(
        lambda value: jnp.take(value, indices=jnp.array([agent_idx]), axis=0),
        params,
    )


def _checkpoint_params(runner_state: Any) -> Any:
    state = runner_state[1]
    if "params" in state:
        return state["params"]
    if "actor_params" in state:
        return state["actor_params"]
    raise ValueError("No params or actor_params found in checkpoint state")


def _source_commit_hash(ogc_src: Path) -> str | None:
    git_dir = ogc_src.parent / ".git"
    head = git_dir / "HEAD"
    if not head.exists():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = git_dir / text.split(" ", 1)[1]
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    return text


def load_ogc_policy_bundle(config: OgcFcpSerdConfig) -> OgcPolicyBundle:
    ogc_src = Path(config.ogc_src).expanduser().resolve()
    if str(ogc_src) not in sys.path:
        sys.path.insert(0, str(ogc_src))

    from minimax.util.checkpoint import load_config, load_pkl_object
    from minimax.util.rl import AgentPopHeterogenous
    import minimax.agents as agents
    import minimax.models as models
    from minimax.runners_ma import EvalRunnerHeterogenous

    population_path = resolve_under(ogc_src, config.population_json)
    population = json.loads(population_path.read_text(encoding="utf-8"))
    population_index, partner_rel, partner_meta_rel = checkpoint_entry(
        population,
        agent_id=config.agent_id,
        band=config.band,
    )
    partner_checkpoint = resolve_under(ogc_src, partner_rel)
    partner_meta = resolve_under(ogc_src, partner_meta_rel)

    if not config.ego_xpid:
        raise ValueError("ego_xpid is required for OGC FCP SERD diagnostic")
    ego_dir = Path(config.log_dir).expanduser() / config.ego_xpid
    ego_checkpoint = ego_dir / f"{config.checkpoint_name}.pkl"
    ego_meta = ego_dir / "meta.json"

    for path in (population_path, partner_checkpoint, partner_meta, ego_checkpoint, ego_meta):
        if not path.exists():
            raise FileNotFoundError(str(path))

    ego_cfg = load_config(str(ego_meta))
    partner_cfg = load_config(str(partner_meta))

    ego_state = load_pkl_object(str(ego_checkpoint))
    partner_state = load_pkl_object(str(partner_checkpoint))
    params_0 = _select_agent_params(_checkpoint_params(ego_state), config.agent_idx)
    params_1 = _select_agent_params(_checkpoint_params(partner_state), config.agent_idx)

    student_model = models.make(
        env_name=ego_cfg.env_name,
        model_name=ego_cfg.student_model_name,
        **ego_cfg.student_model_args,
    )
    partner_model = models.make(
        env_name=partner_cfg.env_name,
        model_name=partner_cfg.student_model_name,
        **partner_cfg.student_model_args,
    )
    pop = AgentPopHeterogenous(
        agent_0=agents.MAPPOAgent(actor=student_model, critic=None),
        agent_1=agents.MAPPOAgent(actor=partner_model, critic=None),
        n_agents=1,
    )
    runner = EvalRunnerHeterogenous(
        pop=pop,
        env_names=config.env_name,
        env_kwargs=ego_cfg.eval_env_args,
        n_episodes=1,
        render_mode=None,
        agent_idxs="*",
    )
    if len(runner.benvs) != 1:
        raise ValueError(f"expected one OGC env, got {len(runner.benvs)}")

    return OgcPolicyBundle(
        runner=runner,
        benv=runner.benvs[0],
        pop=pop,
        params_0=params_0,
        params_1=params_1,
        ego_checkpoint=ego_checkpoint,
        ego_meta=ego_meta,
        partner_checkpoint=partner_checkpoint,
        partner_meta=partner_meta,
        population_path=population_path,
        population_index=population_index,
        source_commit_hash=_source_commit_hash(ogc_src),
    )


def _init_snapshot(bundle: OgcPolicyBundle, rng: Any) -> tuple[Any, ...]:
    runner = bundle.runner
    pop = bundle.pop
    rng, *reset_rngs = jax.random.split(rng, pop.n_agents + 1)
    obs, state, extra = bundle.benv.reset(jnp.array(reset_rngs))

    if pop.agent_0.is_recurrent:
        rng, subrng = jax.random.split(rng)
        carry_0, _ = pop.init_carry_agent_0(subrng, obs["agent_0"])
    else:
        carry_0 = None
    if pop.agent_1.is_recurrent:
        rng, subrng = jax.random.split(rng)
        carry_1, _ = pop.init_carry_agent_1(subrng, obs["agent_1"])
    else:
        carry_1 = None
    done = jnp.zeros((pop.n_agents, 1), dtype=jnp.bool_)
    return rng, state, obs, carry_0, carry_1, extra, done


def _sample_policy_action(
    bundle: OgcPolicyBundle,
    rng: Any,
    obs: Any,
    carry_0: Any,
    carry_1: Any,
    done: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    pop = bundle.pop
    _, _, pi_0_params, pi_1_params, next_carry_0, next_carry_1 = pop.act(
        (bundle.params_0, bundle.params_1),
        obs,
        (carry_0, carry_1),
        done,
    )
    pi_0 = pop.get_action_0_dist(pi_0_params, dtype=bundle.runner.action_dtype)
    pi_1 = pop.get_action_1_dist(pi_1_params, dtype=bundle.runner.action_dtype)
    rng, subrng = jax.random.split(rng)
    action_0 = pi_0.sample(seed=subrng)
    rng, subrng = jax.random.split(rng)
    action_1 = pi_1.sample(seed=subrng)
    return rng, action_0, action_1, next_carry_0, next_carry_1


def _force_action(action: Any, actor_index: int, action_name: str) -> Any:
    forced = jnp.array([[ACTION_IDS[action_name]]], dtype=action.dtype)
    return forced


def _replace_actions(
    action_0: Any,
    action_1: Any,
    actor_index: int,
    action_name: str,
) -> tuple[Any, Any]:
    if actor_index == 0:
        return _force_action(action_0, 0, action_name), action_1
    return action_0, _force_action(action_1, 1, action_name)


def _transition(
    bundle: OgcPolicyBundle,
    rng: Any,
    state: Any,
    obs: Any,
    carry_0: Any,
    carry_1: Any,
    extra: Any,
    done: Any,
    override: tuple[int, str] | None,
) -> tuple[Any, ...]:
    rng, action_0, action_1, next_carry_0, next_carry_1 = _sample_policy_action(
        bundle,
        rng,
        obs,
        carry_0,
        carry_1,
        done,
    )
    if override is not None:
        action_0, action_1 = _replace_actions(
            action_0,
            action_1,
            actor_index=override[0],
            action_name=override[1],
        )
    env_action = {"agent_0": action_0, "agent_1": action_1}
    rng, *vrngs = jax.random.split(rng, bundle.pop.n_agents + 1)
    next_obs, next_state, reward, next_done, info, next_extra = bundle.benv.step(
        jnp.array(vrngs),
        state,
        env_action,
        extra,
    )
    return (
        rng,
        next_state,
        next_obs,
        next_carry_0,
        next_carry_1,
        next_extra,
        next_done,
        reward,
        info,
        action_0,
        action_1,
    )


def _warmup(bundle: OgcPolicyBundle, rng: Any, horizon: int) -> tuple[Any, ...]:
    rng, state, obs, carry_0, carry_1, extra, done = _init_snapshot(bundle, rng)
    for _ in range(horizon):
        (
            rng,
            state,
            obs,
            carry_0,
            carry_1,
            extra,
            done,
            _reward,
            _info,
            _action_0,
            _action_1,
        ) = _transition(bundle, rng, state, obs, carry_0, carry_1, extra, done, None)
    return rng, state, obs, carry_0, carry_1, extra, done


def _return_scalar(info: Any) -> float:
    sparse = info["sparse_reward"]
    return float(jax.device_get(jnp.asarray(sparse)[0, 0, 0]))


def _rollout_return(
    bundle: OgcPolicyBundle,
    snapshot: tuple[Any, ...],
    horizon: int,
    first_override: tuple[int, str] | None,
) -> tuple[float, float, tuple[str, str], tuple[str, str]]:
    rng, state, obs, carry_0, carry_1, extra, done = snapshot
    total = 0.0
    first_base: tuple[str, str] | None = None
    first_branch: tuple[str, str] | None = None
    for step_idx in range(horizon):
        override = first_override if step_idx == 0 else None
        (
            rng,
            state,
            obs,
            carry_0,
            carry_1,
            extra,
            done,
            _reward,
            info,
            action_0,
            action_1,
        ) = _transition(bundle, rng, state, obs, carry_0, carry_1, extra, done, override)
        if step_idx == 0:
            a0 = int(jax.device_get(jnp.asarray(action_0)[0, 0]))
            a1 = int(jax.device_get(jnp.asarray(action_1)[0, 0]))
            first_branch = (ACTION_NAMES[a0], ACTION_NAMES[a1])
            first_base = first_branch
        total += _return_scalar(info)
    if first_base is None or first_branch is None:
        raise ValueError("rollout_horizon must be positive")
    shock_magnitude = 1.0 if first_override is not None else 0.0
    return total, shock_magnitude, first_base, first_branch


def _phi_pre_h(bundle: OgcPolicyBundle, snapshot: tuple[Any, ...], probe_index: int) -> dict[str, float]:
    _rng, state, _obs, _carry_0, _carry_1, _extra, _done = snapshot
    raw = jax.device_get(state)
    agent_pos = raw.agent_pos[0, 0]
    agent_dir = raw.agent_dir_idx[0, 0]
    agent_inv = raw.agent_inv[0, 0]
    pot_status = raw.maze_map[0, 0, :, :, 2]
    return {
        "probe_index": float(probe_index),
        "time": float(raw.time[0, 0]),
        "p0_x": float(agent_pos[0, 0]),
        "p0_y": float(agent_pos[0, 1]),
        "p1_x": float(agent_pos[1, 0]),
        "p1_y": float(agent_pos[1, 1]),
        "p0_dir": float(agent_dir[0]),
        "p1_dir": float(agent_dir[1]),
        "p0_holding": float(agent_inv[0] != 1),
        "p1_holding": float(agent_inv[1] != 1),
        "pot_status_sum": float(jnp.asarray(pot_status).sum()),
        "env_time_remaining": float(bundle.benv.env.max_episode_steps() - raw.time[0, 0]),
    }


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


def make_ogc_fcp_branch_records(
    config: OgcFcpSerdConfig,
) -> tuple[list[BranchRecord], list[BranchRecord], dict[str, Any]]:
    bundle = load_ogc_policy_bundle(config)
    semantic_records: list[BranchRecord] = []
    control_records: list[BranchRecord] = []
    rng = jax.random.PRNGKey(config.seed)

    for disruption in config.disruptions:
        for probe_index in range(config.probes_per_disruption):
            rng, probe_rng = jax.random.split(rng)
            snapshot = _warmup(
                bundle,
                probe_rng,
                horizon=config.warmup_horizon + probe_index,
            )
            phi = _phi_pre_h(bundle, snapshot, probe_index)
            no_shock_return, _zero, _base_action, _no_shock_action = _rollout_return(
                bundle,
                snapshot,
                horizon=config.rollout_horizon,
                first_override=None,
            )
            sem_return, sem_mag, _sem_base, _sem_action = _rollout_return(
                bundle,
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
                control_return, control_mag, _control_base, _control_action = _rollout_return(
                    bundle,
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

    provenance = {
        "policy_family": "FCP",
        "checkpoint_paths": {
            "ego": str(bundle.ego_checkpoint),
            "partner": str(bundle.partner_checkpoint),
            "population_json": str(bundle.population_path),
        },
        "checkpoint_hashes": {
            "ego": sha256_file(bundle.ego_checkpoint),
            "partner": sha256_file(bundle.partner_checkpoint),
            "population_json": sha256_file(bundle.population_path),
        },
        "metadata_paths": {
            "ego": str(bundle.ego_meta),
            "partner": str(bundle.partner_meta),
        },
        "source_repository_path": str(Path(config.ogc_src).expanduser().resolve()),
        "source_commit_hash": bundle.source_commit_hash,
        "adapter_route": "FCP_ROUTE_NATIVE_OGC_SERD_SELECTED",
        "environment_name": config.env_name,
        "target_domain": config.domain,
        "population_index": bundle.population_index,
        "random_seeds": [config.seed],
        "m3_acceptance_artifact": "refine-logs/M3_REMOTE_FCP_SMOKE_ACCEPTANCE.md",
        "policy_param_shapes": {
            "ego_first_leaf": _first_leaf_shape(bundle.params_0),
            "partner_first_leaf": _first_leaf_shape(bundle.params_1),
        },
    }
    return semantic_records, control_records, provenance


def all_records(
    semantic_records: list[BranchRecord],
    control_records: list[BranchRecord],
) -> list[BranchRecord]:
    return list(semantic_records) + list(control_records)
