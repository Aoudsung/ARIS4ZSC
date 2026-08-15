"""Crossed causal decision-value intervention for DELTA belief use."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.delta_zsc.anchors import collect_anchor_batch
from src.delta_zsc.config import METHOD_VERSION, OFFICIAL_ACTION_COUNT, load_config
from src.delta_zsc.manifest import load_partner_manifest
from src.delta_zsc.mirror_policy import mirror_policy_logits
from src.delta_zsc.partners import make_static_partner_functions
from src.delta_zsc.resources import ResourceLedger
from src.delta_zsc.runner import collect_rollout, initialize_runner
from src.delta_zsc.storage import ensure_run_identity, write_json

from .deployment import load_deployment


def _crossed_bootstrap(matrix: np.ndarray, *, replicates: int = 9_999) -> np.ndarray:
    """Resample ego and partner nodes independently."""

    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError(
            "Belief intervention needs at least two ego and two partner runs."
        )
    rng = np.random.default_rng(0)
    draws = np.empty((int(replicates),), dtype=np.float64)
    for index in range(int(replicates)):
        ego = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        partner = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        draws[index] = float(np.mean(matrix[np.ix_(ego, partner)]))
    return draws


def run_belief_value_intervention(args: argparse.Namespace) -> None:
    """Estimate correct-belief minus shuffled-belief value in the same world.

    Belief vectors are exchanged across independent ego-history lanes, but the
    task state, current partner, all-action continuation vector, role, and CRN
    keys remain fixed.  Formal inference resamples both ego and partner runs.
    """

    started = time.perf_counter()
    import jax
    import jax.numpy as jnp

    from .official_adapter import FrozenPartnerPool, VectorEnvironment
    from .training_app import _anchor_functions

    config = load_config(args.config, run_kind=args.run_kind)
    deployment_paths = tuple(Path(value).resolve() for value in args.deployment)
    deployments = tuple(load_deployment(path) for path in deployment_paths)
    for deployment in deployments:
        if deployment.config.environment.layout != config.environment.layout:
            raise ValueError("Belief-intervention deployment layout differs.")
        if deployment.config.method_variant not in {"delta_passive", "delta_active"}:
            raise ValueError("Belief intervention requires a DELTA deployment.")
    if config.run_kind == "formal" and len(deployments) < config.evaluation.minimum_ego_runs:
        raise ValueError("Formal H3 requires the registered number of ego runs.")

    manifest_path = Path(args.partner_manifest).resolve()
    manifest = load_partner_manifest(
        manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_file_check),
    )
    runs = (
        manifest.by_role("confirmatory")
        if config.run_kind == "formal"
        else (manifest.by_role("development_coverage") or manifest.by_role("confirmatory"))
    )
    if len(runs) < 2:
        raise ValueError("Belief intervention needs at least two held-out partner runs.")

    effects = np.empty((len(deployments), len(runs)), dtype=np.float64)
    node_rows = []
    total_steps = 0
    for ego_index, deployment in enumerate(deployments):
        for partner_index, run in enumerate(runs):
            pool = FrozenPartnerPool.from_checkpoints(
                (run.checkpoint,),
                parent_training_run_ids=(run.parent_training_run_id,),
            )
            partner_functions = make_static_partner_functions(
                pool=pool,
                probabilities=jnp.asarray([1.0]),
                run_ids=jnp.asarray([0]),
            )
            environment = VectorEnvironment.create(config)
            root = jax.random.fold_in(
                jax.random.fold_in(jax.random.PRNGKey(30_000), ego_index),
                partner_index,
            )
            runner = initialize_runner(
                environment=environment,
                model=deployment.model,
                partner_functions=partner_functions,
                random_key=root,
            )
            _, batch, records = collect_rollout(
                state=runner,
                length=config.training.rollout_length,
                environment=environment,
                model=deployment.model,
                base_params=deployment.base_params,
                latent_params=deployment.latent_params,
                partner_functions=partner_functions,
                partner_parameters=None,
                official_shaping_factor=0.0,
                record_anchors=True,
                use_deployment_policy=True,
            )
            functions = _anchor_functions(
                model=deployment.model,
                partner_functions=partner_functions,
                partner_parameters=None,
                environment=environment,
            )
            anchors = collect_anchor_batch(
                key=jax.random.fold_in(root, 31_000),
                records=records,
                functions=functions,
                base_params=deployment.base_params,
                latent_params=deployment.latent_params,
                states_per_trigger=config.anchors.states_per_trigger,
                action_count=OFFICIAL_ACTION_COUNT,
                fit_replicas=config.anchors.fit_replicas,
                evaluation_replicas=config.anchors.evaluation_replicas,
                horizon=config.method.continuation_horizon,
                gamma=config.ppo.gamma,
                collect_successor=False,
            )
            _, output = deployment.model.sequence(
                deployment.base_params,
                deployment.latent_params,
                batch.initial_policy_state,
                batch.observations,
                batch.previous_actions,
                batch.episode_starts,
            )
            time, lane = anchors.time_indexes, anchors.lane_indexes
            belief = output.belief[time, lane]
            if belief.shape[0] < 2:
                raise ValueError("Belief intervention needs at least two anchor histories.")
            shuffled = jnp.roll(belief, shift=1, axis=0)
            means = output.component_decision_means[time, lane]
            correct_q = jnp.sum(belief[..., :, None] * means, axis=-2)
            shuffled_q = jnp.sum(shuffled[..., :, None] * means, axis=-2)
            base_logits = output.base_policy_logits[time, lane]
            correct_logits, _, _ = mirror_policy_logits(
                base_logits,
                correct_q,
                kl_budget=config.method.adaptation_kl_budget,
            )
            shuffled_logits, _, _ = mirror_policy_logits(
                base_logits,
                shuffled_q,
                kl_budget=config.method.adaptation_kl_budget,
            )
            correct_policy = jax.nn.softmax(correct_logits, axis=-1)
            shuffled_policy = jax.nn.softmax(shuffled_logits, axis=-1)
            returns = anchors.evaluation_returns_by_action
            current_effects = np.asarray(
                jnp.sum((correct_policy - shuffled_policy) * returns, axis=-1),
                dtype=np.float64,
            )
            value = float(np.mean(current_effects))
            effects[ego_index, partner_index] = value
            node_rows.append(
                {
                    "ego_run_index": ego_index,
                    "ego_run_id": deployment.ego_run_id,
                    "partner_run_index": partner_index,
                    "partner_run_id": run.run_id,
                    "partner_mechanism": run.generation_mechanism,
                    "anchor_count": int(current_effects.size),
                    "mean_effect": value,
                }
            )
            total_steps += (
                config.environment.num_envs * config.training.rollout_length
                + config.anchors.states_per_trigger
                * 6
                * (config.anchors.fit_replicas + config.anchors.evaluation_replicas)
                * config.method.continuation_horizon
            )

    boot = _crossed_bootstrap(effects)
    output_dir = Path(args.output).resolve()
    deployment_sources = [{"path": str(path)} for path in deployment_paths]
    ensure_run_identity(
        output_dir,
        {
            "stage": "belief-value-intervention",
            "method": METHOD_VERSION,
            "layout": config.environment.layout,
            "deployments": deployment_sources,
            "partner_manifest": {"path": str(manifest_path)},
        },
    )
    elapsed = time.perf_counter() - started
    ledger = ResourceLedger(
        intervention_steps=total_steps,
        measurement_wall_clock_hours=elapsed / 3600.0,
        measurement_gpu_hours=(
            elapsed / 3600.0
            if any(device.platform == "gpu" for device in jax.devices())
            else 0.0
        ),
    )
    write_json(
        output_dir / "belief_value_intervention.json",
        {
            "version": 2,
            "artifact_type": "delta_belief_value_intervention",
            "method": METHOD_VERSION,
            "execution_mode": "active",
            "layout": config.environment.layout,
            "estimand": (
                "same-source-world expected return of correct legal-history "
                "belief policy minus shuffled-belief policy"
            ),
            "estimate": float(np.mean(effects)),
            "interval_95": [float(v) for v in np.quantile(boot, (0.025, 0.975))],
            "one_sided_lcb": float(np.quantile(boot, 0.05)),
            "ego_run_count": int(effects.shape[0]),
            "partner_run_count": int(effects.shape[1]),
            "node_effects": node_rows,
            "resource_ledger": ledger.to_mapping(),
        },
    )
    write_json(output_dir / "resource_ledger.json", ledger.to_mapping())


__all__ = ["run_belief_value_intervention"]
