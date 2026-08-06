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
from src.delta_zsc.response_model import (
    response_factor_log_probabilities,
    response_joint_log_probability,
    response_predict,
)
from src.delta_zsc.transition import predict_belief
from src.delta_zsc.resources import ResourceLedger
from src.delta_zsc.runner import collect_rollout, initialize_runner
from src.delta_zsc.storage import ensure_run_identity, write_json

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
        verify_files=not bool(args.skip_manifest_file_check),
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
        model_rows = []
        uniform_rows = []
        no_history_rows = []
        event_probabilities = []
        event_labels = []
        event_masks = []
        event_conditional_nll = []
        event_other = []
        event_classes = []
        component_js = []
        factor_rows: dict[str, list[np.ndarray]] = {}
        component_count = config.method.latent_components
        uniform = jnp.full((config.environment.num_envs, component_count), 1.0 / component_count)
        for time in range(batch.actions.shape[0]):
            target = extract_response_target(
                batch.observations[time],
                batch.response_next_observations[time],
                batch.actions[time],
                jnp.zeros_like(batch.dones[time], dtype=jnp.bool_),
            )
            prediction = response_predict(
                deployment.latent_params["response"],
                deployment.latent_params["component_embeddings"],
                batch.observations[time],
                output.behavior_features[time],
                batch.actions[time],
            )
            component_logp = response_joint_log_probability(prediction, target)
            predictive_belief = predict_belief(
                output.belief[time], deployment.latent_params["transition_logits"]
            )
            model_logp = jsp.special.logsumexp(
                jnp.log(jnp.maximum(predictive_belief, 1.0e-30)) + component_logp,
                axis=-1,
            )
            model_rows.append(np.asarray(-model_logp))
            uniform_logp = jsp.special.logsumexp(
                jnp.log(uniform) + component_logp, axis=-1
            )
            uniform_rows.append(np.asarray(-uniform_logp))
            zero_behavior = jnp.zeros_like(output.behavior_features[time])
            no_history_prediction = response_predict(
                deployment.latent_params["response"],
                deployment.latent_params["component_embeddings"],
                batch.observations[time],
                zero_behavior,
                batch.actions[time],
            )
            no_history_component = response_joint_log_probability(
                no_history_prediction, target
            )
            no_history_rows.append(np.asarray(-jsp.special.logsumexp(
                jnp.log(uniform) + no_history_component, axis=-1
            )))
            event_component = jax.nn.sigmoid(prediction.interface_change_logit)
            event_probabilities.append(
                np.asarray(jnp.sum(predictive_belief * event_component, axis=-1))
            )
            event_labels.append(np.asarray(target.interface_changed))
            event_masks.append(np.asarray(target.interface_available))
            event_logp = jax.nn.log_softmax(prediction.interface_event_logits, axis=-1)
            # This is p(E | C=1,H), so the component mixture must first be
            # updated by the observed change event.  Using the predictive
            # belief directly would report a different, unregistered score.
            changed_component_logp = (
                jnp.log(jnp.maximum(predictive_belief, 1.0e-30))
                + jax.nn.log_sigmoid(prediction.interface_change_logit)
            )
            mixture_event = jsp.special.logsumexp(
                changed_component_logp[..., :, None] + event_logp,
                axis=-2,
            ) - jsp.special.logsumexp(changed_component_logp, axis=-1)[..., None]
            selected_event_logp = jnp.take_along_axis(
                mixture_event, target.interface_event[..., None], axis=-1
            )[..., 0]
            changed_mask = target.interface_available * target.interface_changed
            event_conditional_nll.append(np.asarray(-selected_event_logp)[np.asarray(changed_mask) > 0.5])
            event_other.append(np.asarray(target.interface_event)[np.asarray(changed_mask) > 0.5] == 30)
            event_classes.append(
                np.asarray(target.interface_event)[np.asarray(changed_mask) > 0.5]
            )
            component_probability = np.asarray(jax.nn.softmax(prediction.interface_event_logits, axis=-1))
            mean_probability = np.mean(component_probability, axis=-2, keepdims=True)
            component_js.append(np.mean(np.sum(
                component_probability * (
                    np.log(np.maximum(component_probability, 1.0e-12))
                    - np.log(np.maximum(mean_probability, 1.0e-12))
                ), axis=-1
            ), axis=-1))
            for name, component_factor in response_factor_log_probabilities(prediction, target).items():
                factor_mixture = jsp.special.logsumexp(
                    jnp.log(jnp.maximum(predictive_belief, 1.0e-30)) + component_factor,
                    axis=-1,
                )
                direct = target.direct
                factor_mask = {
                    "visibility": jnp.ones_like(target.interface_available),
                    "position": direct.visible_mask,
                    "direction": direct.visible_mask,
                    "inventory": direct.visible_mask,
                    "inventory_change": direct.event_mask,
                    "interface_availability": jnp.ones_like(target.interface_available),
                    "interface_change": target.interface_available,
                    "interface_event": changed_mask,
                    "recipe_change": target.recipe_mask,
                }[name]
                selected_factor = np.asarray(-factor_mixture)[
                    np.asarray(factor_mask) > 0.5
                ]
                factor_rows.setdefault(name, []).append(selected_factor)
        model_nll = np.concatenate(model_rows, axis=0)
        uniform_nll = np.concatenate(uniform_rows, axis=0)
        no_history_nll = np.concatenate(no_history_rows, axis=0)
        event_p = np.concatenate(event_probabilities, axis=0)
        event_y = np.concatenate(event_labels, axis=0)
        event_m = np.concatenate(event_masks, axis=0) > 0.5
        observed_event_classes = (
            np.concatenate(event_classes)
            if any(row.size for row in event_classes)
            else np.asarray([], dtype=np.int32)
        )
        event_histogram = np.bincount(observed_event_classes, minlength=31).astype(np.float64)
        event_distribution = event_histogram / max(float(np.sum(event_histogram)), 1.0)
        brier = float(np.mean((event_p[event_m] - event_y[event_m]) ** 2)) if np.any(event_m) else None
        voi = np.asarray(output.active_voi[:-1], dtype=np.float64)
        information_gain = np.asarray(
            output.active_information_gain[:-1], dtype=np.float64
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
                "interface_coverage_rate": float(np.mean(event_m)),
                "event_conditional_nll": (
                    None if not any(row.size for row in event_conditional_nll)
                    else float(np.mean(np.concatenate(event_conditional_nll)))
                ),
                "event_other_multi_rate": (
                    None if not any(row.size for row in event_other)
                    else float(np.mean(np.concatenate(event_other)))
                ),
                "interface_event_count": int(observed_event_classes.size),
                "interface_event_distribution": event_distribution.tolist(),
                "mean_component_event_js": float(np.mean(np.concatenate(component_js))),
                "factor_nll": {
                    name: (
                        None if not any(row.size for row in rows)
                        else float(np.mean(np.concatenate(rows)))
                    )
                    for name, rows in factor_rows.items()
                },
                "factor_count": {
                    name: int(sum(row.size for row in rows))
                    for name, rows in factor_rows.items()
                },
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
                "minimum_exact_voi": float(np.min(voi)),
                "exact_voi_negative_fraction": float(np.mean(voi < 0.0)),
                "mean_information_gain": float(np.mean(information_gain)),
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
            "deployment": {"path": str(Path(args.deployment).resolve())},
            "partner_manifest": {"path": str(manifest_path)},
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
    disagreement_values = np.asarray(
        [row["greedy_action_disagreement_rate"] for row in per_run],
        dtype=np.float64,
    )
    target_distribution_by_mechanism = {}
    for mechanism in sorted({row["partner_mechanism"] for row in per_run}):
        mechanism_rows = [
            row for row in per_run if row["partner_mechanism"] == mechanism
        ]
        weights = np.asarray(
            [row["interface_event_count"] for row in mechanism_rows], dtype=np.float64
        )
        distributions = np.asarray(
            [row["interface_event_distribution"] for row in mechanism_rows],
            dtype=np.float64,
        )
        target_distribution_by_mechanism[mechanism] = (
            np.sum(distributions * weights[:, None], axis=0)
            / max(float(np.sum(weights)), 1.0)
        ).tolist()
    sp_op_total_variation = None
    if "sp" in target_distribution_by_mechanism and "op" in target_distribution_by_mechanism:
        sp_op_total_variation = float(
            0.5
            * np.sum(
                np.abs(
                    np.asarray(target_distribution_by_mechanism["sp"])
                    - np.asarray(target_distribution_by_mechanism["op"])
                )
            )
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
            "version": 3,
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
            "interface_event_distribution_by_mechanism": target_distribution_by_mechanism,
            "sp_op_interface_event_total_variation": sp_op_total_variation,
            "sp_op_measurement_has_no_performance_gate": True,
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
