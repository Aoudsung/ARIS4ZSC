"""Raw-backed evaluation for unified DELTA-ZSC."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from experiments.overcooked_v2.official_adapter import (
    official_pairing_rollouts,
    official_policy,
    restore_official_checkpoint,
    validate_official_runtime,
)
from src.delta_zsc.config import METHOD_VERSION
from src.delta_zsc.deployment import Deployment, load_deployment, reset_state
from src.delta_zsc.model import FULL_VARIANT, JOINT_VARIANT, METHOD_VARIANTS
from src.delta_zsc.runner import deployment_action
from src.path_c.experiment import load_partner_manifest
from src.path_c.resources import ResourceLedger, gpu_hours_for_wall_seconds
from src.path_c.storage import runtime_provenance, sha256_path, write_parquet


class UnifiedPolicy:
    """Official evaluator adapter exposing only legal local history."""

    def __init__(self, deployment: Deployment) -> None:
        self.deployment = deployment

    def init_hstate(self, batch_size: int) -> Any:
        return reset_state(self.deployment, batch_size=int(batch_size))

    def compute_action(
        self,
        observation: Any,
        done: Any,
        hstate: Any,
        key: Any,
    ) -> tuple[Any, Any]:
        import jax
        import jax.numpy as jnp

        obs = jnp.asarray(observation, dtype=jnp.float32)
        batched = obs.ndim == 4
        if not batched:
            obs = obs[None]
        done_array = jnp.asarray(done, dtype=jnp.bool_)
        if done_array.ndim == 0:
            done_array = done_array[None]
        fresh = reset_state(self.deployment, batch_size=int(obs.shape[0]))
        state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(
                done_array.reshape(done_array.shape + (1,) * (jnp.ndim(new) - 1)),
                new,
                old,
            ),
            fresh,
            hstate,
        )
        keys = jnp.asarray(key)
        if keys.ndim == 1:
            keys = keys[None]
        stepped, action, _, _ = deployment_action(
            agent=self.deployment.agent,
            base_params=self.deployment.base_params,
            latent_params=self.deployment.latent_params,
            state=state,
            observation=obs,
            keys=keys,
            variant=self.deployment.variant,
        )
        next_state = stepped._replace(
            previous_action=jnp.asarray(action, dtype=jnp.int32),
            episode_start=jnp.zeros_like(done_array),
        )
        return (
            action if batched else action[0],
            next_state,
        )


def _official_environment(config: Any) -> Any:
    import jaxmarl
    from jaxmarl.environments.overcooked_v2.overcooked import ObservationType

    return jaxmarl.make(
        "overcooked_v2",
        layout=config.environment.layout,
        max_steps=config.environment.episode_steps,
        observation_type=ObservationType.DEFAULT,
        agent_view_size=config.environment.agent_view_size,
        negative_rewards=config.environment.negative_rewards,
        random_agent_positions=config.environment.random_agent_positions,
        sample_recipe_on_delivery=config.environment.sample_recipe_on_delivery,
        indicate_successful_delivery=(
            config.environment.indicate_successful_delivery
        ),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def run_evaluation(args: Any) -> None:
    import jax

    started = time.perf_counter()
    runtime = validate_official_runtime()
    deployment_path = Path(args.deployment).resolve()
    deployment = load_deployment(deployment_path)
    override = getattr(args, "variant_override", None)
    if override is not None:
        requested = str(override).lower()
        if requested not in METHOD_VARIANTS:
            raise ValueError("Evaluation variant override is not registered.")
        # Full and joint are two analytic deployment policies over the same
        # jointly trained emissions.  No parameter or checkpoint is changed.
        if {deployment.variant, requested} - {JOINT_VARIANT, FULL_VARIANT}:
            raise ValueError("Only joint/full may share one trained parameter bundle.")
        deployment = replace(deployment, variant=requested)
    config = deployment.config
    manifest_path = Path(args.partner_manifest).resolve()
    manifest = load_partner_manifest(
        manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    role = str(getattr(args, "partner_role", "confirmatory"))
    partners = tuple(manifest.by_role(role))
    if not partners:
        raise ValueError(f"Evaluation partner role {role!r} is empty.")
    episodes = int(
        config.evaluation.episodes_per_pairing
        if getattr(args, "episodes", None) is None
        else args.episodes
    )
    if episodes <= 0:
        raise ValueError("Evaluation episode count must be positive.")
    ego = UnifiedPolicy(deployment)
    environment = _official_environment(config)
    root = jax.random.PRNGKey(int(getattr(args, "evaluation_seed", 0)))
    rows = []
    for partner_index, partner_run in enumerate(partners):
        partner_config, partner_params = restore_official_checkpoint(
            partner_run.checkpoint
        )
        partner = official_policy(partner_params, partner_config)
        roles = (0, 1) if config.evaluation.evaluate_both_roles else (0,)
        for ego_role in roles:
            left, right = (ego, partner) if ego_role == 0 else (partner, ego)
            pairing_key = jax.random.fold_in(
                jax.random.fold_in(root, partner_index), ego_role
            )
            rollouts, _ = official_pairing_rollouts(
                left_policy=left,
                right_policy=right,
                environment=environment,
                root_key=pairing_key,
                episodes=episodes,
            )
            returns = np.asarray(rollouts.total_reward, dtype=np.float64)
            for episode_index, raw_return in enumerate(returns):
                rows.append(
                    {
                        "method": METHOD_VERSION,
                        "variant": deployment.variant,
                        "layout": config.environment.layout,
                        "ego_run_id": deployment_path.parent.name,
                        "partner_run_id": str(partner_run.run_id),
                        "partner_parent_run_id": str(
                            partner_run.parent_training_run_id
                        ),
                        "partner_mechanism": str(
                            partner_run.generation_mechanism
                        ),
                        "ego_role": int(ego_role),
                        "episode_index": int(episode_index),
                        "raw_return": float(raw_return),
                    }
                )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "episode_returns.parquet"
    write_parquet(raw_path, rows)
    values = np.asarray([row["raw_return"] for row in rows], dtype=np.float64)
    elapsed = time.perf_counter() - started
    ledger = ResourceLedger(
        evaluation_steps=len(rows) * config.environment.episode_steps,
        gpu_hours=gpu_hours_for_wall_seconds(elapsed),
        wall_clock_hours=elapsed / 3600.0,
    )
    identity = {
        "stage": "evaluate-unified-delta-zsc",
        "method": METHOD_VERSION,
        "variant": deployment.variant,
        "layout": config.environment.layout,
        "runtime": runtime,
        "repository_runtime": runtime_provenance(),
        "deployment": {
            "path": str(deployment_path),
            "sha256": sha256_path(deployment_path),
        },
        "partner_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_path(manifest_path),
            "role": role,
        },
        "episodes_per_pairing": episodes,
        "evaluation_seed": int(getattr(args, "evaluation_seed", 0)),
    }
    _write_json(output / "run_identity.json", identity)
    _write_json(
        output / "evaluation_summary.json",
        {
            "version": 1,
            "artifact_type": "unified_delta_raw_evaluation",
            "method": METHOD_VERSION,
            "variant": deployment.variant,
            "layout": config.environment.layout,
            "episode_count": len(rows),
            "partner_run_count": len(partners),
            "raw_return_mean": float(np.mean(values)),
            "raw_return_standard_error": float(
                np.std(values, ddof=1) / np.sqrt(max(values.size, 1))
            ),
            "raw": {"path": str(raw_path), "sha256": sha256_path(raw_path)},
            "resource_ledger": ledger.to_mapping(),
        },
    )
    _write_json(output / "resource_ledger.json", ledger.to_mapping())


__all__ = ["UnifiedPolicy", "run_evaluation"]
