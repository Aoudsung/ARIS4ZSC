"""Held-out posterior-predictive diagnostics for the unified latent model."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.delta_zsc.behavior_statistics import behavior_features
from src.delta_zsc.config import load_config
from src.delta_zsc.manifest import load_partner_manifest
from src.delta_zsc.observation import extract_response_target
from src.delta_zsc.partners import make_static_partner_functions
from src.delta_zsc.response_model import response_joint_log_probability, response_predict
from src.delta_zsc.transition import predict_belief
from src.delta_zsc.resources import ResourceLedger
from src.delta_zsc.runner import collect_rollout, initialize_runner
from src.delta_zsc.storage import ensure_run_identity, sha256_path, write_json

from .deployment import load_deployment


def run_posterior_predictive_diagnostics(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp

    from .official_adapter import FrozenPartnerPool, VectorEnvironment

    config = load_config(args.config, run_kind=args.run_kind)
    deployment = load_deployment(args.deployment)
    if (
        deployment.config.environment.layout != config.environment.layout
        or deployment.config.method_variant != config.method_variant
        or deployment.config.method != config.method
    ):
        raise ValueError("Posterior-diagnostic deployment/config identity differs.")
    manifest_path = Path(args.partner_manifest).resolve()
    manifest = load_partner_manifest(
        manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    runs = manifest.by_role("calibration")
    if not runs:
        raise ValueError("Calibration panel is empty.")
    per_run = []
    steps = 0
    for index, run in enumerate(runs):
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
        runner = initialize_runner(
            environment=environment,
            model=deployment.model,
            partner_functions=partner_functions,
            random_key=jax.random.fold_in(jax.random.PRNGKey(20_000), index),
        )
        _, batch, _ = collect_rollout(
            state=runner,
            length=config.environment.episode_steps,
            environment=environment,
            model=deployment.model,
            base_params=deployment.base_params,
            latent_params=deployment.latent_params,
            partner_functions=partner_functions,
            partner_parameters=None,
            official_shaping_factor=0.0,
            record_anchors=False,
            use_deployment_policy=True,
        )
        _, output = deployment.model.sequence(
            deployment.base_params,
            deployment.latent_params,
            batch.initial_policy_state,
            batch.observations,
            batch.previous_actions,
            batch.episode_starts,
        )
        valid_transition = ~np.asarray(batch.episode_starts[1:], dtype=bool)
        model_nll_all = np.asarray(output.response_negative_log_likelihood[1:])
        model_nll = model_nll_all[valid_transition]
        uniform_rows = []
        no_history_rows = []
        event_probabilities = []
        event_labels = []
        event_masks = []
        component_count = config.method.latent_components
        uniform = jnp.full((config.environment.num_envs, component_count), 1.0 / component_count)
        for time in range(1, batch.observations.shape[0]):
            target = extract_response_target(
                batch.observations[time - 1], batch.observations[time]
            )
            prediction = output.response_prediction[time]
            component_logp = response_joint_log_probability(prediction, target)
            uniform_logp = jsp.special.logsumexp(
                jnp.log(uniform) + component_logp, axis=-1
            )
            transition_valid = ~np.asarray(batch.episode_starts[time], dtype=bool)
            uniform_rows.append(
                np.asarray(-uniform_logp)[transition_valid]
            )
            zero_behavior = jnp.zeros_like(output.behavior_features[time - 1])
            no_history_prediction = response_predict(
                deployment.latent_params["response"],
                deployment.latent_params["component_embeddings"],
                batch.observations[time - 1],
                zero_behavior,
                batch.previous_actions[time],
            )
            no_history_component = response_joint_log_probability(
                no_history_prediction, target
            )
            no_history_rows.append(
                np.asarray(
                    -jsp.special.logsumexp(
                        jnp.log(uniform) + no_history_component, axis=-1
                    )
                )[transition_valid]
            )
            event_component = jax.nn.sigmoid(prediction.inventory_change_logit)
            predictive_belief = predict_belief(
                output.belief[time - 1],
                deployment.latent_params["transition_logits"],
            )
            predictive_belief = jnp.where(
                batch.episode_starts[time][..., None],
                uniform,
                predictive_belief,
            )
            event_probabilities.append(
                np.asarray(jnp.sum(predictive_belief * event_component, axis=-1))[
                    transition_valid
                ]
            )
            event_labels.append(
                np.asarray(target.inventory_change)[transition_valid]
            )
            event_masks.append(np.asarray(target.event_mask)[transition_valid])
        uniform_nll = np.concatenate(uniform_rows, axis=0)
        no_history_nll = np.concatenate(no_history_rows, axis=0)
        event_p = np.concatenate(event_probabilities, axis=0)
        event_y = np.concatenate(event_labels, axis=0)
        event_m = np.concatenate(event_masks, axis=0) > 0.5
        brier = float(np.mean((event_p[event_m] - event_y[event_m]) ** 2)) if np.any(event_m) else None
        voi = np.asarray(output.active_voi[:-1], dtype=np.float64)
        voi_raw = np.asarray(output.active_voi_raw[:-1], dtype=np.float64)
        information_gain = np.asarray(
            output.active_information_gain[:-1], dtype=np.float64
        )
        quadrature_error = np.asarray(
            output.active_voi_quadrature_error[:-1], dtype=np.float64
        )
        adaptation_kl = np.asarray(output.adaptation_kl[:-1], dtype=np.float64)
        base_action = np.argmax(np.asarray(output.base_policy_logits[:-1]), axis=-1)
        deployed_action = np.argmax(np.asarray(output.policy_logits[:-1]), axis=-1)
        per_run.append(
            {
                "partner_run_id": run.run_id,
                "partner_mechanism": run.generation_mechanism,
                "model_nll": float(np.mean(model_nll)),
                "uniform_nll": float(np.mean(uniform_nll)),
                "no_history_nll": float(np.mean(no_history_nll)),
                "event_brier": brier,
                "event_prevalence": (
                    None if not np.any(event_m) else float(np.mean(event_y[event_m]))
                ),
                "valid_transition_count": int(model_nll.size),
                "mean_belief_entropy": float(
                    np.mean(
                        -np.sum(
                            np.asarray(output.belief)
                            * np.log(np.maximum(np.asarray(output.belief), 1.0e-12)),
                            axis=-1,
                        )
                    )
                ),
                "mean_action_voi": float(np.mean(voi)),
                "mean_max_action_voi": float(np.mean(np.max(voi, axis=-1))),
                "raw_voi_negative_fraction": float(np.mean(voi_raw < -1.0e-7)),
                "mean_information_gain": float(np.mean(information_gain)),
                "mean_voi_quadrature_error": float(np.mean(quadrature_error)),
                "p95_voi_quadrature_error": float(np.quantile(quadrature_error, 0.95)),
                "maximum_voi_quadrature_error": float(np.max(quadrature_error)),
                "mean_adaptation_kl": float(np.mean(adaptation_kl)),
                "greedy_action_disagreement_rate": float(
                    np.mean(base_action != deployed_action)
                ),
            }
        )
        steps += config.environment.num_envs * config.environment.episode_steps
    output_dir = Path(args.output).resolve()
    ensure_run_identity(
        output_dir,
        {
            "stage": "posterior-predictive-diagnostics",
            "deployment": {
                "path": str(Path(args.deployment).resolve()),
                "sha256": sha256_path(args.deployment),
            },
            "partner_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_path(manifest_path),
            },
            "config_fingerprint": config.fingerprint,
        },
    )
    model_values = np.asarray([row["model_nll"] for row in per_run], dtype=np.float64)
    uniform_values = np.asarray([row["uniform_nll"] for row in per_run], dtype=np.float64)
    no_history_values = np.asarray(
        [row["no_history_nll"] for row in per_run], dtype=np.float64
    )
    improvement_uniform = uniform_values - model_values
    improvement_history = no_history_values - model_values
    rng = np.random.default_rng(0)

    def interval(values: np.ndarray) -> list[float]:
        if values.size < 2:
            return [float("nan"), float("nan")]
        draws = np.mean(
            values[rng.integers(0, values.size, size=(9_999, values.size))],
            axis=1,
        )
        return [float(value) for value in np.quantile(draws, (0.025, 0.975))]

    brier_values = np.asarray(
        [row["event_brier"] for row in per_run if row["event_brier"] is not None],
        dtype=np.float64,
    )
    voi_values = np.asarray([row["mean_action_voi"] for row in per_run], dtype=np.float64)
    max_voi_values = np.asarray(
        [row["mean_max_action_voi"] for row in per_run], dtype=np.float64
    )
    information_values = np.asarray(
        [row["mean_information_gain"] for row in per_run], dtype=np.float64
    )
    quadrature_values = np.asarray(
        [row["mean_voi_quadrature_error"] for row in per_run], dtype=np.float64
    )
    disagreement_values = np.asarray(
        [row["greedy_action_disagreement_rate"] for row in per_run],
        dtype=np.float64,
    )
    elapsed = time.perf_counter() - started
    import jax

    ledger = ResourceLedger(
        calibration_steps=steps,
        measurement_wall_clock_hours=elapsed / 3600.0,
        measurement_gpu_hours=(
            elapsed / 3600.0
            if any(device.platform == "gpu" for device in jax.devices())
            else 0.0
        ),
    )
    write_json(
        output_dir / "posterior_predictive_diagnostics.json",
        {
            "version": 2,
            "artifact_type": "delta_posterior_predictive_diagnostics",
            "layout": config.environment.layout,
            "claim_role": "diagnostic_only",
            "primary_unit": "partner_run",
            "per_partner_run": per_run,
            "mean_model_nll": float(np.mean(model_values)),
            "mean_improvement_vs_uniform": float(np.mean(improvement_uniform)),
            "improvement_vs_uniform_interval_95": interval(improvement_uniform),
            "mean_improvement_vs_no_history": float(np.mean(improvement_history)),
            "improvement_vs_no_history_interval_95": interval(improvement_history),
            "mean_event_brier": (
                None if brier_values.size == 0 else float(np.mean(brier_values))
            ),
            "mean_action_voi": float(np.mean(voi_values)),
            "mean_max_action_voi": float(np.mean(max_voi_values)),
            "mean_information_gain": float(np.mean(information_values)),
            "mean_voi_quadrature_error": float(np.mean(quadrature_values)),
            "mean_greedy_action_disagreement_rate": float(
                np.mean(disagreement_values)
            ),
            "event_brier_interval_95": (
                None if brier_values.size < 2 else interval(brier_values)
            ),
            "resource_ledger": ledger.to_mapping(),
        },
    )
    write_json(output_dir / "resource_ledger.json", ledger.to_mapping())



__all__ = ["run_posterior_predictive_diagnostics"]
