"""Calibration and posterior-predictive diagnostics for DELTA-ZSC v4."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.delta_zsc.config import METHOD_VERSION, OFFICIAL_ACTION_COUNT, load_config
from src.delta_zsc.manifest import load_partner_manifest
from src.delta_zsc.observation import (
    INTERFACE_EVENT_CLASSES,
    extract_probe_response_target,
    extract_response_target,
)
from src.delta_zsc.partners import make_static_partner_functions
from src.delta_zsc.resources import ResourceLedger
from src.delta_zsc.response_model import (
    probe_response_predict,
    probe_response_semantic_component_log_probability,
    probe_response_shared_log_probability,
    response_factor_log_probabilities,
    response_predict,
    response_semantic_component_log_probability,
    response_shared_log_probability,
)
from src.delta_zsc.runner import collect_rollout, initialize_runner
from src.delta_zsc.semantic_initializer import (
    fit_spectral_simplex_initializer,
    save_semantic_initializer,
)
from src.delta_zsc.storage import ensure_run_identity, write_json

from .deployment import load_deployment


# The pooled conditional model is an initializer instrument, not a method
# mechanism. A fixed label-free Rademacher projection preserves information from
# the complete 5x5 frame while keeping cross-fitting and parent-disjoint oracle
# diagnostics feasible on the formal panel.
EVENT_FRAME_PROJECTION_DIM = 48
EVENT_MODEL_OPTIMIZATION_STEPS = 100
EVENT_MODEL_PROJECTION_SEED = 41_903


def _project_event_frames(frames: np.ndarray) -> np.ndarray:
    value = np.asarray(frames, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError("Event frame matrix must have shape [event,frame_feature].")
    rng = np.random.default_rng(EVENT_MODEL_PROJECTION_SEED)
    projection = rng.integers(0, 2, size=(value.shape[1], EVENT_FRAME_PROJECTION_DIM))
    projection = (2.0 * projection.astype(np.float32) - 1.0) / np.sqrt(
        float(EVENT_FRAME_PROJECTION_DIM)
    )
    return value @ projection


def _mixture_logp(belief: Any, component_logp: Any) -> Any:
    import jax.numpy as jnp
    import jax.scipy as jsp

    return jsp.special.logsumexp(
        jnp.log(jnp.maximum(belief, 1.0e-30)) + component_logp, axis=-1
    )


def _softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.maximum(np.sum(exponential, axis=-1, keepdims=True), 1.0e-30)



def _sigmoid(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    positive = array >= 0.0
    result = np.empty_like(array, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _masked_mean(value: Any, mask: Any) -> float | None:
    selected = np.asarray(value, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    return None if selected.size == 0 else float(np.mean(selected))


def _bernoulli_brier(logit: Any, target: Any, mask: Any) -> float | None:
    return _masked_mean(
        np.square(_sigmoid(logit) - np.asarray(target, dtype=np.float64)), mask
    )


def _categorical_brier(
    probability: Any, target: Any, mask: Any
) -> float | None:
    probs = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(target, dtype=np.int64)
    one_hot = np.eye(probs.shape[-1], dtype=np.float64)[labels]
    return _masked_mean(np.sum(np.square(probs - one_hot), axis=-1), mask)


def _interface_outcome_diagnostics(
    *,
    available: Any,
    changed: Any,
    event: Any,
    change_probability: Any,
    event_probability: Any,
    valid: Any | None = None,
) -> tuple[np.ndarray, float | None, int]:
    """Return the available-step 32-outcome distribution and Brier score.

    Outcome zero is ``NO_CHANGE`` and outcomes 1..31 are the structured
    interface events. Availability is scored separately as a shared Bernoulli
    factor, so unavailable rows are not silently recoded as no-change.
    """

    availability = np.asarray(available, dtype=np.float64) > 0.5
    mask = availability
    if valid is not None:
        mask = mask & (np.asarray(valid, dtype=np.float64) > 0.5)
    change = np.asarray(changed, dtype=np.float64) > 0.5
    event_label = np.asarray(event, dtype=np.int64)
    target = np.where(change, event_label + 1, 0)
    histogram = np.bincount(target[mask], minlength=32).astype(np.float64)
    distribution = histogram / max(float(np.sum(histogram)), 1.0)
    change_p = np.asarray(change_probability, dtype=np.float64)
    event_p = np.asarray(event_probability, dtype=np.float64)
    probability = np.concatenate(
        ((1.0 - change_p)[..., None], change_p[..., None] * event_p), axis=-1
    )
    return distribution, _categorical_brier(probability, target, mask), int(np.sum(mask))


def _fit_pooled_linear_event_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    predict_x: np.ndarray,
    *,
    class_count: int,
) -> np.ndarray:
    """Deterministic pooled q0(E|X) used only to form initializer residuals."""

    x = np.asarray(train_x, dtype=np.float32)
    y = np.asarray(train_y, dtype=np.int64)
    query = np.asarray(predict_x, dtype=np.float32)
    mean = np.mean(x, axis=0, keepdims=True)
    scale = np.std(x, axis=0, keepdims=True)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    x = (x - mean) / scale
    query = (query - mean) / scale
    count = int(class_count)
    weights = np.zeros((x.shape[1], count), dtype=np.float32)
    histogram = np.bincount(y, minlength=count).astype(np.float32) + 1.0
    bias = np.log(histogram / np.sum(histogram))
    first_w = np.zeros_like(weights)
    second_w = np.zeros_like(weights)
    first_b = np.zeros_like(bias)
    second_b = np.zeros_like(bias)
    one_hot = np.eye(count, dtype=np.float32)[y]
    for step in range(1, EVENT_MODEL_OPTIMIZATION_STEPS + 1):
        probability = _softmax(x @ weights + bias)
        error = (probability - one_hot) / float(max(x.shape[0], 1))
        grad_w = x.T @ error + 1.0e-4 * weights
        grad_b = np.sum(error, axis=0)
        first_w = 0.9 * first_w + 0.1 * grad_w
        second_w = 0.999 * second_w + 0.001 * np.square(grad_w)
        first_b = 0.9 * first_b + 0.1 * grad_b
        second_b = 0.999 * second_b + 0.001 * np.square(grad_b)
        correction1 = 1.0 - 0.9**step
        correction2 = 1.0 - 0.999**step
        rate = 0.01
        weights -= rate * (first_w / correction1) / (
            np.sqrt(second_w / correction2) + 1.0e-8
        )
        bias -= rate * (first_b / correction1) / (
            np.sqrt(second_b / correction2) + 1.0e-8
        )
    return _softmax(query @ weights + bias)


def _cross_fitted_pooled_event_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
    *,
    class_count: int,
) -> np.ndarray:
    unique = np.unique(episode_ids)
    folds = min(5, int(unique.size))
    if folds < 2:
        histogram = np.bincount(labels, minlength=class_count).astype(np.float64) + 1.0
        return np.broadcast_to(
            (histogram / np.sum(histogram))[None],
            (labels.size, class_count),
        ).copy()
    assignment = {int(episode): index % folds for index, episode in enumerate(unique)}
    result = np.empty((labels.size, class_count), dtype=np.float64)
    for fold in range(folds):
        held = np.asarray([assignment[int(value)] == fold for value in episode_ids])
        train = ~held
        result[held] = _fit_pooled_linear_event_model(
            features[train], labels[train], features[held], class_count=class_count
        )
    return result


def _parent_disjoint_conditional_oracle(
    features: np.ndarray,
    labels: np.ndarray,
    mechanisms: np.ndarray,
    parent_ids: np.ndarray,
    *,
    class_count: int,
) -> dict[str, Any]:
    """Measure residual mechanism information without training DELTA on labels."""

    unique_parents = np.unique(parent_ids)
    unique_mechanisms = sorted(str(value) for value in np.unique(mechanisms))
    if unique_parents.size < 2 or len(unique_mechanisms) < 2:
        return {
            "status": "insufficient_parent_disjoint_support",
            "parent_count": int(unique_parents.size),
            "mechanisms": unique_mechanisms,
            "pooled_nll": None,
            "oracle_nll": None,
            "conditional_oracle_gain": None,
        }
    mechanism_index = {value: index for index, value in enumerate(unique_mechanisms)}
    one_hot = np.eye(len(unique_mechanisms), dtype=np.float32)[
        np.asarray([mechanism_index[str(value)] for value in mechanisms], dtype=np.int64)
    ]
    pooled_losses: list[np.ndarray] = []
    oracle_losses: list[np.ndarray] = []
    evaluated_parents: list[str] = []
    skipped_parents: list[dict[str, Any]] = []
    evaluated_event_count = 0
    for held_parent in unique_parents:
        held = parent_ids == held_parent
        train = ~held
        if not np.any(held) or np.sum(train) < max(int(class_count), 2):
            skipped_parents.append(
                {
                    "parent": str(held_parent),
                    "reason": "insufficient_training_events",
                }
            )
            continue
        held_mechanisms = {str(value) for value in np.unique(mechanisms[held])}
        train_mechanisms = {str(value) for value in np.unique(mechanisms[train])}
        missing = sorted(held_mechanisms - train_mechanisms)
        if missing:
            # A mechanism indicator cannot be evaluated parent-disjointly if
            # the held parent is the only training-support source of that
            # mechanism.  Skipping avoids a privileged-label extrapolation
            # artifact masquerading as residual information.
            skipped_parents.append(
                {
                    "parent": str(held_parent),
                    "reason": "held_mechanism_absent_from_training_fold",
                    "missing_mechanisms": missing,
                }
            )
            continue
        pooled_probability = _fit_pooled_linear_event_model(
            features[train], labels[train], features[held], class_count=class_count
        )
        oracle_probability = _fit_pooled_linear_event_model(
            np.concatenate((features[train], one_hot[train]), axis=-1),
            labels[train],
            np.concatenate((features[held], one_hot[held]), axis=-1),
            class_count=class_count,
        )
        held_labels = labels[held]
        rows = np.arange(held_labels.size)
        pooled_losses.append(
            -np.log(np.maximum(pooled_probability[rows, held_labels], 1.0e-12))
        )
        oracle_losses.append(
            -np.log(np.maximum(oracle_probability[rows, held_labels], 1.0e-12))
        )
        evaluated_parents.append(str(held_parent))
        evaluated_event_count += int(held_labels.size)
    if not pooled_losses:
        return {
            "status": "no_evaluable_parent_fold",
            "parent_count": int(unique_parents.size),
            "mechanisms": unique_mechanisms,
            "pooled_nll": None,
            "oracle_nll": None,
            "conditional_oracle_gain": None,
            "skipped_parents": skipped_parents,
            "evaluated_event_count": 0,
        }
    pooled_nll = float(np.mean(np.concatenate(pooled_losses)))
    oracle_nll = float(np.mean(np.concatenate(oracle_losses)))
    return {
        "status": "measured",
        "parent_count": int(unique_parents.size),
        "evaluated_parents": evaluated_parents,
        "skipped_parents": skipped_parents,
        "evaluated_event_count": int(evaluated_event_count),
        "mechanisms": unique_mechanisms,
        "pooled_nll": pooled_nll,
        "oracle_nll": oracle_nll,
        "conditional_oracle_gain": pooled_nll - oracle_nll,
        "uses_privileged_labels_for_diagnostic_only": True,
    }


def build_semantic_initializer(args: argparse.Namespace) -> None:
    """Build the unlabeled spectral-simplex event initializer."""

    import jax
    import jax.numpy as jnp

    from .official_adapter import FrozenPartnerPool, VectorEnvironment

    started = time.perf_counter()
    config = load_config(args.config, run_kind=args.run_kind)
    deployment = load_deployment(args.deployment)
    if deployment.config.environment.layout != config.environment.layout:
        raise ValueError("Initializer deployment/config layout differs.")
    if deployment.config.official_protocol != config.official_protocol:
        raise ValueError("Initializer deployment/config Official protocol differs.")
    manifest_path = Path(args.partner_manifest).resolve()
    manifest = load_partner_manifest(
        manifest_path,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_file_check),
    )
    role = str(getattr(args, "partner_role", "calibration"))
    if role != "calibration":
        raise ValueError("Semantic initializer must use the calibration panel.")
    runs = manifest.by_role(role)
    if not runs:
        raise ValueError(f"Semantic initializer panel is empty for role {role!r}.")

    event_features: list[np.ndarray] = []
    event_labels: list[np.ndarray] = []
    event_episode_ids: list[np.ndarray] = []
    event_mechanisms: list[np.ndarray] = []
    event_parent_ids: list[np.ndarray] = []
    next_episode_id = 0
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
            random_key=jax.random.fold_in(jax.random.PRNGKey(30_000), index),
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
            use_deployment_policy=False,
        )
        target = extract_response_target(
            batch.observations[:-1],
            batch.response_next_observations,
            batch.actions,
            jnp.zeros_like(batch.dones, dtype=jnp.bool_),
        )
        changed = np.asarray(
            target.interface_available * target.interface_changed
        ) > 0.5
        frames = np.asarray(batch.observations[:-1], dtype=np.float32)
        # Match the legal conditioning variables of the response predictor: full
        # current frame, analytic history statistics, and ego action.  The
        # pooled model is cross-fitted, so the residual cannot merely memorize
        # an episode.
        _, history_output = deployment.model.sequence(
            deployment.base_params,
            deployment.latent_params,
            batch.initial_policy_state,
            batch.observations,
            batch.previous_actions,
            batch.episode_starts,
            compute_latent=True,
            compute_decision=False,
            execute_adaptation=False,
        )
        flat_frame = frames.reshape(frames.shape[:2] + (-1,))
        history = np.asarray(history_output.behavior_features[:-1], dtype=np.float32)
        actions = np.eye(OFFICIAL_ACTION_COUNT, dtype=np.float32)[np.asarray(batch.actions)]
        event_frame = _project_event_frames(flat_frame[changed])
        features = np.concatenate(
            (event_frame, history[changed], actions[changed]), axis=-1
        )
        dones = np.asarray(batch.dones, dtype=bool)
        local_episode = np.zeros_like(dones, dtype=np.int64)
        for lane in range(dones.shape[1]):
            lane_episode = np.cumsum(
                np.concatenate(([0], dones[:-1, lane].astype(np.int64)))
            )
            local_episode[:, lane] = lane_episode + next_episode_id
            next_episode_id += int(np.max(lane_episode)) + 1
        event_features.append(features)
        event_labels.append(np.asarray(target.interface_event)[changed])
        event_episode_ids.append(local_episode[changed])
        event_mechanisms.append(
            np.full(int(np.sum(changed)), str(run.generation_mechanism), dtype=object)
        )
        event_parent_ids.append(
            np.full(
                int(np.sum(changed)),
                str(run.parent_training_run_id),
                dtype=object,
            )
        )
        steps += config.environment.num_envs * config.environment.episode_steps

    if not event_features or not any(value.shape[0] for value in event_features):
        raise ValueError("Calibration panel contains no structured interface events.")
    features = np.concatenate(event_features, axis=0)
    labels = np.concatenate(event_labels, axis=0).astype(np.int64)
    episode_ids = np.concatenate(event_episode_ids, axis=0).astype(np.int64)
    mechanisms = np.concatenate(event_mechanisms, axis=0)
    parent_ids = np.concatenate(event_parent_ids, axis=0)
    requested_components = tuple(
        sorted(
            set(
                int(value)
                for value in (
                    getattr(args, "component_count", None)
                    or (config.method.latent_components,)
                )
            )
        )
    )
    if labels.size < max(requested_components):
        raise ValueError(
            "Calibration panel contains fewer structured events than the "
            "largest requested component count."
        )
    pooled = _cross_fitted_pooled_event_probabilities(
        features, labels, episode_ids, class_count=INTERFACE_EVENT_CLASSES
    )
    conditional_oracle = _parent_disjoint_conditional_oracle(
        features,
        labels,
        mechanisms,
        parent_ids,
        class_count=INTERFACE_EVENT_CLASSES,
    )
    rows = []
    used_episode_ids = []
    for episode in np.unique(episode_ids):
        mask = episode_ids == episode
        count = int(np.sum(mask))
        if count == 0:
            continue
        residual = np.sum(np.eye(INTERFACE_EVENT_CLASSES)[labels[mask]] - pooled[mask], axis=0) / np.sqrt(
            float(count)
        )
        rows.append(residual)
        used_episode_ids.append(int(episode))
    if len(rows) < max(requested_components):
        raise ValueError(
            "Calibration panel contains fewer event-bearing episodes than the "
            "largest requested component count."
        )
    source_payload = {
        "method": METHOD_VERSION,
        "layout": config.environment.layout,
        "official_protocol_version": config.official_protocol.protocol_version,
        "official_source_commit": config.official_protocol.source_commit,
        "partner_role": role,
        "partner_run_ids": [run.run_id for run in runs],
        "parent_training_run_ids": sorted(
            {str(run.parent_training_run_id) for run in runs}
        ),
        "event_count": int(labels.size),
        "episode_ids": used_episode_ids,
        "pooled_predictor": (
            "five_fold_linear_softmax_fixed_rademacher_frame_projection_"
            "plus_history_action"
        ),
        "frame_projection_dim": EVENT_FRAME_PROJECTION_DIM,
        "frame_projection_seed": EVENT_MODEL_PROJECTION_SEED,
        "optimization_steps": EVENT_MODEL_OPTIMIZATION_STEPS,
        "uses_partner_labels": False,
        "conditional_oracle": conditional_oracle,
    }
    output = Path(args.output).resolve()
    ensure_run_identity(
        output,
        {
            "stage": "semantic-initializer",
            "method": METHOD_VERSION,
            "layout": config.environment.layout,
            "deployment": {"path": str(Path(args.deployment).resolve())},
            "partner_manifest": {"path": str(manifest_path)},
            "partner_role": role,
            "component_counts": list(requested_components),
        },
    )
    artifacts = []
    residual_matrix = np.asarray(rows, dtype=np.float32)
    for component_count in requested_components:
        initializer = fit_spectral_simplex_initializer(
            residual_matrix,
            component_count=component_count,
            source={**source_payload, "component_count": int(component_count)},
        )
        artifact_root = (
            output
            if len(requested_components) == 1
            else output / f"k-{component_count}"
        )
        initializer_npz, initializer_json = save_semantic_initializer(
            artifact_root, initializer
        )
        artifacts.append(
            {
                "component_count": int(component_count),
                "npz": str(initializer_npz),
                "json": str(initializer_json),
                "artifact": initializer.to_mapping(include_bias=False),
            }
        )
    elapsed = time.perf_counter() - started
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
        output / "semantic_initializer_summary.json",
        {
            "version": 1,
            "artifact_type": "delta_v4_semantic_initializer_summary",
            "initializers": artifacts,
            "event_count": int(labels.size),
            "episode_count": int(len(rows)),
            "uses_partner_labels": False,
            "conditional_oracle": conditional_oracle,
            "resource_ledger": ledger.to_mapping(),
        },
    )
    write_json(output / "resource_ledger.json", ledger.to_mapping())


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
        target = extract_response_target(
            batch.observations[:-1],
            batch.response_next_observations,
            batch.actions,
            jnp.zeros_like(batch.dones, dtype=jnp.bool_),
        )
        prediction = response_predict(
            deployment.latent_params["response"],
            deployment.latent_params["component_embeddings"],
            batch.observations[:-1],
            output.behavior_features[:-1],
            batch.actions,
        )
        shared_logp = response_shared_log_probability(prediction, target)
        component_semantic = response_semantic_component_log_probability(prediction, target)
        model_semantic = _mixture_logp(output.belief[:-1], component_semantic)
        uniform = jnp.full_like(output.belief[:-1], 1.0 / config.method.latent_components)
        uniform_semantic = _mixture_logp(uniform, component_semantic)
        model_nll = np.asarray(-(shared_logp + model_semantic))
        uniform_nll = np.asarray(-(shared_logp + uniform_semantic))

        event_mask = np.asarray(
            target.interface_available * target.interface_changed
        ) > 0.5
        event_logp = jax.nn.log_softmax(prediction.interface_event_logits, axis=-1)
        mixture_event = jsp.special.logsumexp(
            jnp.log(jnp.maximum(output.belief[:-1], 1.0e-30))[..., :, None]
            + event_logp,
            axis=-2,
        )
        selected_event = jnp.take_along_axis(
            mixture_event, target.interface_event[..., None], axis=-1
        )[..., 0]
        event_classes = np.asarray(target.interface_event)[event_mask]
        histogram = np.bincount(event_classes, minlength=INTERFACE_EVENT_CLASSES).astype(np.float64)
        distribution = histogram / max(float(np.sum(histogram)), 1.0)
        probabilities = np.asarray(jax.nn.softmax(prediction.interface_event_logits, axis=-1))
        mean_probability = np.mean(probabilities, axis=-2, keepdims=True)
        component_js = np.mean(
            np.sum(
                probabilities
                * (
                    np.log(np.maximum(probabilities, 1.0e-12))
                    - np.log(np.maximum(mean_probability, 1.0e-12))
                ),
                axis=-1,
            ),
            axis=-1,
        )
        belief_for_response = np.asarray(output.belief[:-1], dtype=np.float64)
        mixture_event_probability = np.exp(np.asarray(mixture_event, dtype=np.float64))
        interface_outcome_distribution, interface_outcome_brier, interface_available_count = (
            _interface_outcome_diagnostics(
                available=target.interface_available,
                changed=target.interface_changed,
                event=target.interface_event,
                change_probability=_sigmoid(prediction.interface_change_logit),
                event_probability=mixture_event_probability,
            )
        )
        factor_nll = {}
        factor_count = {}
        factor_masks = {
            "visibility": np.ones_like(np.asarray(target.interface_available), dtype=bool),
            "inventory_change": np.asarray(target.direct.event_mask) > 0.5,
            "interface_availability": np.ones_like(
                np.asarray(target.interface_available), dtype=bool
            ),
            "interface_change": np.asarray(target.interface_available) > 0.5,
            "recipe_change": np.asarray(target.recipe_mask) > 0.5,
            "position": np.asarray(target.direct.visible_mask) > 0.5,
            "direction": np.asarray(target.direct.visible_mask) > 0.5,
            "inventory": np.asarray(target.direct.visible_mask) > 0.5,
            "interface_event": event_mask,
        }
        for name, factor in response_factor_log_probabilities(prediction, target).items():
            if factor.ndim == output.belief[:-1].ndim:
                scored = _mixture_logp(output.belief[:-1], factor)
            else:
                scored = factor
            mask = factor_masks[name]
            values = np.asarray(-scored)[mask]
            factor_nll[name] = None if values.size == 0 else float(np.mean(values))
            factor_count[name] = int(values.size)

        factor_brier = {
            "visibility": _bernoulli_brier(
                prediction.direct.visibility_logit,
                target.direct.visibility,
                factor_masks["visibility"],
            ),
            "inventory_change": _bernoulli_brier(
                prediction.direct.inventory_change_logit,
                target.direct.inventory_change,
                factor_masks["inventory_change"],
            ),
            "interface_availability": _bernoulli_brier(
                prediction.interface_availability_logit,
                target.interface_available,
                factor_masks["interface_availability"],
            ),
            "interface_change": _bernoulli_brier(
                prediction.interface_change_logit,
                target.interface_changed,
                factor_masks["interface_change"],
            ),
            "recipe_change": _bernoulli_brier(
                prediction.recipe_change_logit,
                target.recipe_changed,
                factor_masks["recipe_change"],
            ),
        }
        position_probability = np.sum(
            belief_for_response[..., :, None]
            * _softmax(np.asarray(prediction.direct.relative_position_logits)),
            axis=-2,
        )
        direction_probability = np.sum(
            belief_for_response[..., :, None]
            * _softmax(np.asarray(prediction.direct.direction_logits)),
            axis=-2,
        )
        inventory_probability = np.sum(
            belief_for_response[..., :, None, None]
            * _softmax(np.asarray(prediction.direct.inventory_logits)),
            axis=-3,
        )
        factor_brier["position"] = _categorical_brier(
            position_probability, target.direct.relative_position, factor_masks["position"]
        )
        factor_brier["direction"] = _categorical_brier(
            direction_probability, target.direct.direction, factor_masks["direction"]
        )
        inventory_one_hot = np.eye(
            inventory_probability.shape[-1], dtype=np.float64
        )[np.asarray(target.direct.inventory, dtype=np.int64)]
        inventory_brier_by_row = np.mean(
            np.sum(np.square(inventory_probability - inventory_one_hot), axis=-1),
            axis=-1,
        )
        factor_brier["inventory"] = _masked_mean(
            inventory_brier_by_row, factor_masks["inventory"]
        )
        factor_brier["interface_event"] = _categorical_brier(
            mixture_event_probability, target.interface_event, event_mask
        )

        delayed = {
            "valid_count": 0,
            "event_count": 0,
            "shared_nll": None,
            "semantic_nll": None,
            "shared_factor_brier": {},
            "event_conditional_brier": None,
            "interface_outcome_brier": None,
            "interface_outcome_distribution": [0.0] * 32,
            "interface_available_count": 0,
            "mean_component_event_js": 0.0,
        }
        if batch.actions.shape[0] >= 2:
            invalid_window = jnp.asarray(batch.dones[:-1], dtype=jnp.bool_) | jnp.asarray(
                batch.dones[1:], dtype=jnp.bool_
            )
            probe_target = extract_probe_response_target(
                batch.response_next_observations[:-1],
                batch.response_next_observations[1:],
                batch.actions[1:],
                invalid_window,
            )
            probe_prediction = probe_response_predict(
                deployment.latent_params["probe_response"],
                deployment.latent_params["component_embeddings"],
                batch.observations[:-2],
                output.behavior_features[:-2],
                batch.actions[:-1],
            )
            probe_valid = np.asarray(probe_target.valid_mask) > 0.5
            probe_event_mask = (
                probe_valid
                & (np.asarray(probe_target.interface_available) > 0.5)
                & (np.asarray(probe_target.interface_changed) > 0.5)
            )
            probe_shared = np.asarray(
                -probe_response_shared_log_probability(
                    probe_prediction, probe_target
                )
            )
            probe_component = probe_response_semantic_component_log_probability(
                probe_prediction, probe_target
            )
            probe_semantic = np.asarray(
                -_mixture_logp(output.belief[:-2], probe_component)
            )
            probe_probability = np.asarray(
                jax.nn.softmax(probe_prediction.interface_event_logits, axis=-1)
            )
            probe_mean = np.mean(probe_probability, axis=-2, keepdims=True)
            probe_js = np.mean(
                np.sum(
                    probe_probability
                    * (
                        np.log(np.maximum(probe_probability, 1.0e-12))
                        - np.log(np.maximum(probe_mean, 1.0e-12))
                    ),
                    axis=-1,
                ),
                axis=-1,
            )
            probe_belief = np.asarray(output.belief[:-2], dtype=np.float64)
            probe_mixture_event_probability = np.sum(
                probe_belief[..., :, None] * probe_probability, axis=-2
            )
            (
                probe_outcome_distribution,
                probe_outcome_brier,
                probe_available_count,
            ) = _interface_outcome_diagnostics(
                available=probe_target.interface_available,
                changed=probe_target.interface_changed,
                event=probe_target.interface_event,
                change_probability=_sigmoid(probe_prediction.interface_change_logit),
                event_probability=probe_mixture_event_probability,
                valid=probe_target.valid_mask,
            )
            probe_shared_brier = {
                "visibility": _bernoulli_brier(
                    probe_prediction.visibility_logit,
                    probe_target.visibility,
                    probe_valid,
                ),
                "interface_availability": _bernoulli_brier(
                    probe_prediction.interface_availability_logit,
                    probe_target.interface_available,
                    probe_valid,
                ),
                "interface_change": _bernoulli_brier(
                    probe_prediction.interface_change_logit,
                    probe_target.interface_changed,
                    probe_valid
                    & (np.asarray(probe_target.interface_available) > 0.5),
                ),
            }
            delayed = {
                "valid_count": int(np.sum(probe_valid)),
                "event_count": int(np.sum(probe_event_mask)),
                "shared_nll": (
                    None
                    if not np.any(probe_valid)
                    else float(np.mean(probe_shared[probe_valid]))
                ),
                "semantic_nll": (
                    None
                    if not np.any(probe_event_mask)
                    else float(np.mean(probe_semantic[probe_event_mask]))
                ),
                "shared_factor_brier": probe_shared_brier,
                "event_conditional_brier": _categorical_brier(
                    probe_mixture_event_probability,
                    probe_target.interface_event,
                    probe_event_mask,
                ),
                "interface_outcome_brier": probe_outcome_brier,
                "interface_outcome_distribution": probe_outcome_distribution.tolist(),
                "interface_available_count": int(probe_available_count),
                "mean_component_event_js": (
                    0.0
                    if not np.any(probe_event_mask)
                    else float(np.mean(probe_js[probe_event_mask]))
                ),
            }

        voi = np.asarray(output.active_voi[:-1], dtype=np.float64)
        information = np.asarray(output.active_information_gain[:-1], dtype=np.float64)
        active_eligible = np.asarray(
            output.active_probe_eligible[:-1], dtype=bool
        )
        voi_spread = np.max(voi, axis=-1) - np.min(voi, axis=-1)
        information_spread = np.max(information, axis=-1) - np.min(information, axis=-1)

        def eligible_mean(value: np.ndarray) -> float:
            selected = np.asarray(value)[active_eligible]
            return 0.0 if selected.size == 0 else float(np.mean(selected))
        base_action = np.argmax(np.asarray(output.base_policy_logits[:-1]), axis=-1)
        deployed_action = np.argmax(np.asarray(output.policy_logits[:-1]), axis=-1)
        belief = np.asarray(output.belief)
        predictive = np.asarray(output.predictive_belief)
        filter_kl = np.sum(
            belief
            * (
                np.log(np.maximum(belief, 1.0e-12))
                - np.log(np.maximum(predictive, 1.0e-12))
            ),
            axis=-1,
        )
        per_run.append(
            {
                "partner_run_id": run.run_id,
                "partner_mechanism": run.generation_mechanism,
                "model_nll": float(np.mean(model_nll)),
                "uniform_nll": float(np.mean(uniform_nll)),
                "factor_nll": factor_nll,
                "factor_count": factor_count,
                "factor_brier": factor_brier,
                "interface_available_count": int(interface_available_count),
                "interface_outcome_brier": interface_outcome_brier,
                "interface_outcome_distribution": interface_outcome_distribution.tolist(),
                "interface_event_count": int(event_classes.size),
                "interface_event_distribution": distribution.tolist(),
                "event_conditional_nll": (
                    None
                    if not np.any(event_mask)
                    else float(np.mean(np.asarray(-selected_event)[event_mask]))
                ),
                "event_conditional_brier": factor_brier["interface_event"],
                "event_other_multi_rate": (
                    None
                    if event_classes.size == 0
                    else float(np.mean(event_classes == 30))
                ),
                "mean_component_event_js": float(np.mean(component_js[event_mask]))
                if np.any(event_mask)
                else 0.0,
                "mean_belief_entropy": float(
                    np.mean(-np.sum(belief * np.log(np.maximum(belief, 1.0e-12)), axis=-1))
                ),
                "mean_filter_kl": float(np.mean(filter_kl)),
                "delayed_probe_response": delayed,
                "active_probe_eligible_count": int(np.sum(active_eligible)),
                "mean_action_voi": eligible_mean(voi),
                "mean_max_action_voi": eligible_mean(np.max(voi, axis=-1)),
                "mean_action_voi_spread": eligible_mean(voi_spread),
                "mean_information_gain": eligible_mean(information),
                "mean_information_gain_spread": eligible_mean(information_spread),
                "mean_adaptation_kl": eligible_mean(
                    np.asarray(output.adaptation_kl[:-1])
                ),
                "greedy_action_disagreement_rate": eligible_mean(
                    (base_action != deployed_action).astype(np.float64)
                ),
            }
        )
        steps += config.environment.num_envs * config.environment.episode_steps

    output_dir = Path(args.output).resolve()
    ensure_run_identity(
        output_dir,
        {
            "stage": "posterior-predictive-diagnostics",
            "method": METHOD_VERSION,
            "deployment": {"path": str(Path(args.deployment).resolve())},
            "partner_manifest": {"path": str(manifest_path)},
        },
    )
    by_mechanism = {}
    outcome_by_mechanism = {}
    for mechanism in sorted({row["partner_mechanism"] for row in per_run}):
        rows = [row for row in per_run if row["partner_mechanism"] == mechanism]
        weights = np.asarray([row["interface_event_count"] for row in rows], dtype=np.float64)
        distributions = np.asarray(
            [row["interface_event_distribution"] for row in rows], dtype=np.float64
        )
        by_mechanism[mechanism] = (
            np.sum(distributions * weights[:, None], axis=0)
            / max(float(np.sum(weights)), 1.0)
        ).tolist()
        outcome_weights = np.asarray(
            [row["interface_available_count"] for row in rows], dtype=np.float64
        )
        outcome_distributions = np.asarray(
            [row["interface_outcome_distribution"] for row in rows], dtype=np.float64
        )
        outcome_by_mechanism[mechanism] = (
            np.sum(outcome_distributions * outcome_weights[:, None], axis=0)
            / max(float(np.sum(outcome_weights)), 1.0)
        ).tolist()
    sp_op_tv = None
    sp_op_joint_tv = None
    if "sp" in by_mechanism and "op" in by_mechanism:
        sp_op_tv = float(
            0.5
            * np.sum(
                np.abs(np.asarray(by_mechanism["sp"]) - np.asarray(by_mechanism["op"]))
            )
        )
        sp_op_joint_tv = float(
            0.5
            * np.sum(
                np.abs(
                    np.asarray(outcome_by_mechanism["sp"])
                    - np.asarray(outcome_by_mechanism["op"])
                )
            )
        )
    elapsed = time.perf_counter() - started
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
            "version": 4,
            "artifact_type": "delta_v4_posterior_predictive_diagnostics",
            "method": METHOD_VERSION,
            "layout": config.environment.layout,
            "claim_role": "diagnostic_only",
            "per_partner_run": per_run,
            "mean_model_nll": float(np.mean([row["model_nll"] for row in per_run])),
            "mean_improvement_vs_uniform": float(
                np.mean([row["uniform_nll"] - row["model_nll"] for row in per_run])
            ),
            "interface_event_distribution_by_mechanism": by_mechanism,
            "sp_op_interface_event_total_variation": sp_op_tv,
            "interface_outcome_distribution_by_mechanism": outcome_by_mechanism,
            "sp_op_interface_outcome_total_variation": sp_op_joint_tv,
            "mean_interface_outcome_brier": float(
                np.mean(
                    [
                        row["interface_outcome_brier"]
                        for row in per_run
                        if row["interface_outcome_brier"] is not None
                    ]
                )
            )
            if any(row["interface_outcome_brier"] is not None for row in per_run)
            else None,
            "mean_action_voi_spread": float(
                np.mean([row["mean_action_voi_spread"] for row in per_run])
            ),
            "mean_information_gain_spread": float(
                np.mean([row["mean_information_gain_spread"] for row in per_run])
            ),
            "resource_ledger": ledger.to_mapping(),
        },
    )
    write_json(output_dir / "resource_ledger.json", ledger.to_mapping())


__all__ = ["build_semantic_initializer", "run_posterior_predictive_diagnostics"]
