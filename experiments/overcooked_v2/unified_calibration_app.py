"""Held-out posterior-predictive diagnostics for unified DELTA-ZSC.

Calibration is a diagnostic, not an independent paper claim or training gate.
The only nested comparison is the trained component-residual response model
against its jointly trained base response distribution and a per-step uniform
mixture of the same components.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from experiments.overcooked_v2.official_adapter import FrozenPartnerPool, VectorEnvironment
from src.delta_zsc.deployment import load_deployment
from src.delta_zsc.filtering import sequence_log_likelihood
from src.delta_zsc.response_model import response_log_probability
from src.delta_zsc.runner import collect_rollout, initialize_runner
from src.delta_zsc.training import replay_behavior_features
from src.path_c.experiment import load_partner_manifest
from src.path_c.external_partner import make_external_partner_functions
from src.path_c.resources import ResourceLedger, gpu_hours_for_wall_seconds
from src.path_c.storage import runtime_provenance, sha256_path


def _environment(config: Any, num_envs: int) -> VectorEnvironment:
    shim = SimpleNamespace(
        environment=SimpleNamespace(
            layout=config.environment.layout,
            episode_steps=config.environment.episode_steps,
            agent_view_size=config.environment.agent_view_size,
            indicate_successful_delivery=config.environment.indicate_successful_delivery,
            num_envs=int(num_envs),
        )
    )
    return VectorEnvironment.create(shim)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _score_block(deployment: Any, batch: Any) -> Mapping[str, float]:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    statistics = replay_behavior_features(batch)
    full_logits = deployment.latent_model.apply(
        {"params": deployment.latent_params},
        batch.observations[:-1],
        statistics,
        batch.actions,
        include_component_residual=True,
        method=deployment.latent_model.response_logits,
    )
    base_logits = deployment.latent_model.apply(
        {"params": deployment.latent_params},
        batch.observations[:-1],
        statistics,
        batch.actions,
        include_component_residual=False,
        method=deployment.latent_model.response_logits,
    )
    from src.path_c.response_targets import (
        extract_partner_response_targets,
        official_partner_observation_planes,
    )

    planes = official_partner_observation_planes(batch.observations.shape[-1])
    targets = extract_partner_response_targets(
        batch.observations[:-1], batch.response_next_observations, planes=planes
    )
    full_logp = response_log_probability(full_logits, targets)
    base_logp = response_log_probability(base_logits, targets)
    transition = deployment.latent_model.apply(
        {"params": deployment.latent_params},
        method=deployment.latent_model.transition,
    )
    full = sequence_log_likelihood(
        transition=transition,
        response_log_likelihood=full_logp,
        episode_starts=batch.episode_starts[:-1],
        valid_mask=batch.ppo_mask,
        initial_belief=None,
    )
    base = sequence_log_likelihood(
        transition=transition,
        response_log_likelihood=base_logp,
        episode_starts=batch.episode_starts[:-1],
        valid_mask=batch.ppo_mask,
        initial_belief=None,
    )
    uniform_step_logp = logsumexp(
        full_logp - jnp.log(float(full_logp.shape[-1])), axis=-1
    )
    mask = jnp.asarray(batch.ppo_mask, dtype=jnp.float32)
    uniform_nll = -jnp.sum(mask * uniform_step_logp) / jnp.maximum(
        jnp.sum(mask), 1.0
    )
    return {
        "posterior_predictive_nll": float(full.response_nll),
        "base_response_nll": float(base.response_nll),
        "uniform_mixture_nll": float(uniform_nll),
        "component_gain_vs_base": float(base.response_nll - full.response_nll),
        "filter_gain_vs_uniform": float(uniform_nll - full.response_nll),
        "mean_posterior_entropy": float(
            jnp.mean(
                -jnp.sum(
                    full.posterior_sequence
                    * jnp.log(jnp.maximum(full.posterior_sequence, 1.0e-12)),
                    axis=-1,
                )
            )
        ),
    }


def _bootstrap(rows: list[Mapping[str, float]], *, replicates: int, seed: int) -> Mapping[str, Any]:
    names = tuple(rows[0])
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(0, matrix.shape[0], size=(int(replicates), matrix.shape[0]))
    draws = np.mean(matrix[indexes], axis=1)
    low, high = np.quantile(draws, (0.025, 0.975), axis=0)
    mean = np.mean(matrix, axis=0)
    return {
        name: {
            "mean": float(value),
            "interval_95": [float(lo), float(hi)],
        }
        for name, value, lo, hi in zip(names, mean, low, high, strict=True)
    }


def run_calibration(args: Any) -> None:
    import jax
    import jax.numpy as jnp

    started = time.perf_counter()
    deployment = load_deployment(args.deployment)
    config = deployment.config
    manifest_path = Path(args.partner_manifest).resolve()
    manifest = load_partner_manifest(
        manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    runs = tuple(manifest.by_role("calibration"))
    if len(runs) < 4:
        raise ValueError("Calibration requires at least four independent partner runs.")
    episodes_per_run = int(getattr(args, "episodes_per_run", 32))
    if episodes_per_run <= 0:
        raise ValueError("episodes_per_run must be positive.")
    root = jax.random.PRNGKey(int(getattr(args, "seed", 0)))
    rows = []
    total_steps = 0
    for index, run in enumerate(runs):
        environment = _environment(config, episodes_per_run)
        pool = FrozenPartnerPool.from_checkpoints(
            (run.checkpoint,),
            parent_training_run_ids=(run.parent_training_run_id,),
        )
        partner_functions = make_external_partner_functions(
            pool=pool,
            member_indexes=jnp.asarray(0, dtype=jnp.int32),
            run_ids=jnp.asarray(index, dtype=jnp.int32),
        )
        runner = initialize_runner(
            environment=environment,
            task_hidden_dim=config.architecture.task_hidden_dim,
            component_count=config.method.latent_components,
            partner_functions=partner_functions,
            random_key=jax.random.fold_in(root, index),
        )
        _, batch, _ = collect_rollout(
            state=runner,
            length=config.environment.episode_steps,
            environment=environment,
            agent=deployment.agent,
            base_params=deployment.base_params,
            latent_params=deployment.latent_params,
            partner_functions=partner_functions,
            partner_parameters=None,
            variant=deployment.variant,
            task_hidden_dim=config.architecture.task_hidden_dim,
            component_count=config.method.latent_components,
            shaping_factor=0.0,
            record_anchor_state=False,
        )
        score = _score_block(deployment, batch)
        rows.append(
            {
                "partner_run_id": str(run.run_id),
                "partner_mechanism": str(run.generation_mechanism),
                **score,
            }
        )
        total_steps += episodes_per_run * config.environment.episode_steps
    numeric = [
        {key: float(value) for key, value in row.items() if key not in {"partner_run_id", "partner_mechanism"}}
        for row in rows
    ]
    summary = _bootstrap(
        numeric,
        replicates=config.evaluation.bootstrap_replicates,
        seed=int(getattr(args, "seed", 0)),
    )
    elapsed = time.perf_counter() - started
    ledger = ResourceLedger(
        calibration_steps=total_steps,
        gpu_hours=gpu_hours_for_wall_seconds(elapsed),
        wall_clock_hours=elapsed / 3600.0,
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "calibration.json",
        {
            "version": 1,
            "artifact_type": "unified_delta_posterior_predictive_calibration",
            "diagnostic_only": True,
            "method": deployment.config.fingerprint,
            "variant": deployment.variant,
            "layout": config.environment.layout,
            "run_count": len(rows),
            "per_run": rows,
            "summary": summary,
            "sources": {
                "deployment": {
                    "path": str(Path(args.deployment).resolve()),
                    "sha256": sha256_path(args.deployment),
                },
                "partner_manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_path(manifest_path),
                },
            },
            "resource_ledger": ledger.to_mapping(),
            "repository_runtime": runtime_provenance(),
        },
    )
    _write_json(output / "resource_ledger.json", ledger.to_mapping())


__all__ = ["run_calibration"]
