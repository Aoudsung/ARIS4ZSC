"""Held-out posterior calibration for DEPI (METHOD_SPEC §2.4).

The command evaluates the categorical protocol posterior with one shared
joint component likelihood.  Components are exchangeable; no algorithm family
is assigned to a latent index.

* primary   -- posterior-predictive log score (per-step NLL, nats);
* secondary -- Brier score of the interaction_change event;
* coverage  -- posterior-predictive 90% sets for position and direction;
* gate      -- ``calibration_pass_decision`` (failure triggers SCIENTIFIC_SPEC
  Φ5 downgrade of the Bayes claim).

The no-history baseline retains the current physical frame but resets the
posterior to uniform, capability to zero, and the protocol summary to its
uniform-prior value.  Partner run is the primary aggregation unit; pooled
scores are descriptive only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from experiments.overcooked_v2.deployment import (
    PRIMARY_ARTIFACT_NAME,
    deployable_parameters,
    load_training_model,
)
from experiments.overcooked_v2.official_adapter import (
    FrozenPartnerPool,
    VectorEnvironment,
    validate_official_runtime,
)
from src.path_c.protocol_mixture import mixture_summary
from src.path_c.calibration import (
    event_brier_score,
    highest_probability_set_coverage,
    per_step_log_score,
    posterior_predictive_probabilities,
    uniform_prior_log_score,
)
from src.path_c.experiment import (
    METHOD_VERSION,
    OFFICIAL_PROTOCOL_VERSION,
    OFFICIAL_SOURCE_COMMIT,
    load_config,
    load_partner_manifest,
    official_training_domain_keys,
)
from src.path_c.external_partner import make_external_partner_functions
from src.path_c.protocol_encoder import posterior_entropy
from src.path_c.resources import ResourceLedger, gpu_hours_for_wall_seconds
from src.path_c.runner import collect_rollout, initialize_runner
from src.path_c.storage import (
    calibration_identity,
    ensure_run_identity,
    pytree_fingerprint,
    sha256_path,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
    write_parquet,
)
from src.path_c.types import ContextOutput


POSTERIOR_CALIBRATION_ARTIFACT_NAME = "DEPI-Posterior-Calibration"
POSTERIOR_CALIBRATION_SCHEMA_VERSION = 3
_MECHANISM_ALIASES = {
    "rnn-sp": "sp",
    "sp": "sp",
    "rnn-op": "op",
    "op": "op",
    "state-augmented": "sa",
    "sa": "sa",
    "fcp": "fcp",
}
def _owned_calibration_runs(manifest: Any, seed_index: int) -> tuple[Any, ...]:
    del seed_index
    return tuple(
        run
        for run in manifest.by_role("calibration")
        if run.owner_seed_index is None
    )


def _validate_calibration_runs(runs: tuple[Any, ...], *, formal: bool) -> None:
    parent_ids = [str(run.parent_training_run_id) for run in runs]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("Posterior calibration requires independent partner-run blocks.")
    if formal:
        counts = Counter(_MECHANISM_ALIASES.get(run.generation_mechanism) for run in runs)
        if counts != Counter({"sp": 5, "op": 5, "sa": 5, "fcp": 5}):
            raise ValueError(
                "Formal posterior calibration requires five independent runs "
                "from each of SP, OP, SA, and FCP."
            )


def _collect_calibration_records(
    *, config: Any, deployment: Any, run: Any, key: Any, run_numeric_id: int
) -> tuple[Any, int]:
    """Roll the frozen held-out partner and record legal-history episodes."""

    import jax
    import jax.numpy as jnp

    calibration_config = replace(
        config,
        environment=replace(
            config.environment,
            num_envs=int(config.posterior_calibration.episodes_per_run),
        ),
    )
    environment = VectorEnvironment.create(calibration_config)
    pool = FrozenPartnerPool.from_checkpoints(
        (run.checkpoint,),
        parent_training_run_ids=(run.parent_training_run_id,),
    )
    partner_functions = make_external_partner_functions(
        pool=pool,
        member_indexes=jnp.asarray(0, dtype=jnp.int32),
        run_ids=jnp.asarray(run_numeric_id, dtype=jnp.int32),
    )
    runner_key, rollout_key = jax.random.split(key, 2)
    runner = initialize_runner(
        environment=environment,
        model_config=config.model,
        partner_functions=partner_functions,
        random_key=runner_key,
    )._replace(random_key=rollout_key)
    # Held-out scoring uses the full posterior: context dropout is disabled so
    # the recursive Bayes filter reads the complete legal history (METHOD_SPEC
    # §2.4: this set never receives a training gradient).
    _, batch, _ = collect_rollout(
        state=runner,
        length=config.environment.episode_steps,
        environment=environment,
        model=deployment.model,
        params=deployment.params,
        target_params=deployment.params,
        model_config=config.model,
        partner_functions=partner_functions,
        partner_parameters=None,
        context_dropout_probability_value=0.0,
        context_dropout_root=jax.random.fold_in(rollout_key, 9_001),
        official_shaping_factor=0.0,
        record_mode="anchor_full",
    )
    steps = int(config.posterior_calibration.episodes_per_run) * int(
        config.environment.episode_steps
    )
    return batch, steps


def _score_calibration_block(*, deployment: Any, batch: Any) -> Mapping[str, Any]:
    """Score one held-out partner block with the §2.4 proper scoring rules.

    Returns the per-step tensors needed for pooled aggregation plus the
    discrete-posterior context readouts (posterior entropy and the mixture
    summary ``c = Σ_k π_k m_k`` of the mean posterior).
    """

    import jax
    import jax.numpy as jnp

    from src.path_c.response_targets import (
        component_joint_log_probability,
        extract_partner_response_targets,
        official_partner_observation_planes,
    )
    from jax.scipy.special import logsumexp

    model = deployment.model
    params = deployment.params
    _, prediction = model.apply(
        {"params": params},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        batch.actions,
        method=model.response_sequence,
    )
    planes = official_partner_observation_planes(batch.observations.shape[-1])
    targets = extract_partner_response_targets(
        batch.observations[:-1], batch.response_next_observations, planes=planes
    )
    component_log_probabilities = component_joint_log_probability(prediction, targets)
    log_pi = prediction.posterior_log_probabilities
    posterior = jnp.exp(log_pi)
    model_log_probabilities = logsumexp(
        log_pi + component_log_probabilities, axis=-1
    )
    # Build a real no-history counterfactual: physical task/frame features are
    # retained, while every partner-history carrier is reset to its prior.
    _, context = model.apply(
        {"params": params},
        batch.initial_policy_state,
        batch.observations,
        batch.previous_actions,
        batch.episode_starts,
        method=model.context_sequence,
    )
    sliced = ContextOutput(
        context.task_features[:-1],
        context.instant_partner[:-1],
        context.capability[:-1],
        context.protocol_probabilities[:-1],
        context.protocol_embedding[:-1],
    )
    component_matrix = params["protocol_component_embeddings"]["embedding"]
    uniform = jnp.full_like(
        sliced.protocol_probabilities,
        1.0 / float(sliced.protocol_probabilities.shape[-1]),
    )
    no_history_context = ContextOutput(
        task_features=sliced.task_features,
        instant_partner=sliced.instant_partner,
        capability=jnp.zeros_like(sliced.capability),
        protocol_probabilities=uniform,
        protocol_embedding=mixture_summary(uniform, component_matrix),
    )
    no_history = model.apply(
        {"params": params},
        no_history_context,
        batch.observations[:-1],
        batch.actions,
        method=model.response_from_context_and_action,
    )
    no_history_component_logp = component_joint_log_probability(no_history, targets)
    no_history_log_probabilities = logsumexp(
        no_history.posterior_log_probabilities + no_history_component_logp,
        axis=-1,
    )
    event_sigmoid = jax.nn.sigmoid(prediction.interaction_change_logit)
    prior_event_sigmoid = jax.nn.sigmoid(no_history.interaction_change_logit)
    mean_posterior = jnp.mean(posterior, axis=(0, 1))
    return {
        "model_log_probabilities": np.asarray(model_log_probabilities),
        "component_log_probabilities": np.asarray(component_log_probabilities),
        "no_history_log_probabilities": np.asarray(no_history_log_probabilities),
        "event_probabilities": np.asarray(jnp.sum(posterior * event_sigmoid, axis=-1)),
        "prior_event_probabilities": np.asarray(
            jnp.sum(uniform * prior_event_sigmoid, axis=-1)
        ),
        "event_labels": np.asarray(targets.interaction_change),
        "position_probabilities": np.asarray(
            posterior_predictive_probabilities(
                posterior, prediction.relative_position_logits
            )
        ),
        "position_labels": np.asarray(targets.relative_position),
        "direction_probabilities": np.asarray(
            posterior_predictive_probabilities(
                posterior, prediction.direction_logits
            )
        ),
        "direction_labels": np.asarray(targets.direction),
        "visible_mask": np.asarray(targets.visible_mask),
        "event_mask": np.asarray(targets.event_mask),
        "posterior_probabilities": np.asarray(posterior),
        "mean_posterior_entropy": float(jnp.mean(posterior_entropy(posterior))),
        "mean_posterior_mixture_summary": np.asarray(
            mixture_summary(mean_posterior, component_matrix)
        ),
    }


_METRIC_NAMES = (
    "log_score",
    "uniform_baseline_log_score",
    "no_history_baseline_log_score",
    "delta_nll_no_c",
    "event_brier",
    "prior_baseline_brier",
    "coverage_position",
    "coverage_direction",
)


def _metric_values(
    payload: Mapping[str, Any], index: Any, *, credibility: float
) -> Mapping[str, float]:
    """Score one episode slice with the shared calibration primitives."""

    visible = payload["visible_mask"][index]
    event_valid = payload["event_mask"][index]
    model_score = float(per_step_log_score(payload["model_log_probabilities"][index]))
    no_history_score = float(
        per_step_log_score(payload["no_history_log_probabilities"][index])
    )
    return {
        "log_score": model_score,
        "uniform_baseline_log_score": float(
            uniform_prior_log_score(payload["component_log_probabilities"][index])
        ),
        "no_history_baseline_log_score": no_history_score,
        "delta_nll_no_c": no_history_score - model_score,
        "event_brier": float(
            event_brier_score(
                payload["event_probabilities"][index],
                payload["event_labels"][index],
                mask=event_valid,
            )
        ),
        "prior_baseline_brier": float(
            event_brier_score(
                payload["prior_event_probabilities"][index],
                payload["event_labels"][index],
                mask=event_valid,
            )
        ),
        "coverage_position": float(
            highest_probability_set_coverage(
                payload["position_probabilities"][index],
                payload["position_labels"][index],
                credibility=float(credibility),
                mask=visible,
            )
        ),
        "coverage_direction": float(
            highest_probability_set_coverage(
                payload["direction_probabilities"][index],
                payload["direction_labels"][index],
                credibility=float(credibility),
                mask=visible,
            )
        ),
    }


def _episode_metric_matrix(
    payload: Mapping[str, Any], *, credibility: float = 0.90
) -> np.ndarray:
    """Return ``(episode, metric)`` scores; no long episode gets extra weight."""

    shape = np.asarray(payload["model_log_probabilities"]).shape
    if len(shape) != 2:
        raise ValueError("Calibration log probabilities must have (time, episode) axes.")
    rows = [
        _metric_values(
            payload, (slice(None), episode), credibility=float(credibility)
        )
        for episode in range(shape[1])
    ]
    return np.asarray(
        [[row[name] for name in _METRIC_NAMES] for row in rows], dtype=np.float64
    )


def _block_metrics(
    payload: Mapping[str, Any], *, credibility: float = 0.90
) -> Mapping[str, float]:
    """Partner-run mean with episodes as equal secondary units."""

    mean = np.mean(
        _episode_metric_matrix(payload, credibility=float(credibility)), axis=0
    )
    return {
        name: float(value) for name, value in zip(_METRIC_NAMES, mean, strict=True)
    }


def _aggregate_metrics(
    payloads: list[Mapping[str, Any]], *, credibility: float = 0.90
) -> Mapping[str, float]:
    """Pooled descriptive readouts; partner runs remain the primary unit."""

    def stacked(name: str) -> np.ndarray:
        return np.concatenate([payload[name] for payload in payloads], axis=0)

    visible = stacked("visible_mask")
    event_valid = stacked("event_mask")
    model_score = float(per_step_log_score(stacked("model_log_probabilities")))
    no_history_score = float(
        per_step_log_score(stacked("no_history_log_probabilities"))
    )
    return {
        "log_score": model_score,
        "uniform_baseline_log_score": float(
            uniform_prior_log_score(stacked("component_log_probabilities"))
        ),
        "no_history_baseline_log_score": no_history_score,
        "delta_nll_no_c": no_history_score - model_score,
        "event_brier": float(
            event_brier_score(
                stacked("event_probabilities"), stacked("event_labels"), mask=event_valid
            )
        ),
        "prior_baseline_brier": float(
            event_brier_score(
                stacked("prior_event_probabilities"),
                stacked("event_labels"),
                mask=event_valid,
            )
        ),
        "coverage_position": float(
            highest_probability_set_coverage(
                stacked("position_probabilities"),
                stacked("position_labels"),
                credibility=float(credibility),
                mask=visible,
            )
        ),
        "coverage_direction": float(
            highest_probability_set_coverage(
                stacked("direction_probabilities"),
                stacked("direction_labels"),
                credibility=float(credibility),
                mask=visible,
            )
        ),
        "posterior_entropy": float(
            np.mean([payload["mean_posterior_entropy"] for payload in payloads])
        ),
    }


def _event_calibration_diagnostics(
    payloads: list[Mapping[str, Any]], *, bin_count: int = 10
) -> Mapping[str, Any]:
    probabilities = np.concatenate(
        [np.asarray(payload["event_probabilities"]).reshape((-1,)) for payload in payloads]
    )
    labels = np.concatenate(
        [np.asarray(payload["event_labels"]).reshape((-1,)) for payload in payloads]
    )
    mask = np.concatenate(
        [np.asarray(payload["event_mask"]).reshape((-1,)) for payload in payloads]
    ) > 0.5
    probabilities = probabilities[mask]
    labels = labels[mask]
    if probabilities.size == 0:
        edges = np.linspace(0.0, 1.0, int(bin_count) + 1)
        return {
            "positive_brier": None,
            "negative_brier": None,
            "event_prevalence": None,
            "reliability_curve": [
                {
                    "bin_low": float(edges[index]),
                    "bin_high": float(edges[index + 1]),
                    "count": 0,
                    "mean_probability": None,
                    "event_frequency": None,
                }
                for index in range(int(bin_count))
            ],
        }
    positive = labels > 0.5
    negative = ~positive
    edges = np.linspace(0.0, 1.0, int(bin_count) + 1)
    indexes = np.minimum(
        np.searchsorted(edges, probabilities, side="right") - 1,
        int(bin_count) - 1,
    )
    reliability = []
    for index in range(int(bin_count)):
        selected = indexes == index
        reliability.append(
            {
                "bin_low": float(edges[index]),
                "bin_high": float(edges[index + 1]),
                "count": int(np.sum(selected)),
                "mean_probability": (
                    None if not np.any(selected) else float(np.mean(probabilities[selected]))
                ),
                "event_frequency": (
                    None if not np.any(selected) else float(np.mean(labels[selected]))
                ),
            }
        )
    return {
        "positive_brier": (
            None
            if not np.any(positive)
            else float(np.mean(np.square(1.0 - probabilities[positive])))
        ),
        "negative_brier": (
            None
            if not np.any(negative)
            else float(np.mean(np.square(probabilities[negative])))
        ),
        "event_prevalence": float(np.mean(labels)),
        "reliability_curve": reliability,
    }


def _run_level_summary(
    episode_matrices: list[np.ndarray], *, bootstrap_replicates: int, seed: int
) -> tuple[Mapping[str, float], Mapping[str, list[float]]]:
    """Two-level bootstrap: partner run first, episode second."""

    matrix = np.stack([values.mean(axis=0) for values in episode_matrices])
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("Run-level calibration needs at least two partner runs.")
    mean = matrix.mean(axis=0)
    rng = np.random.default_rng(int(seed))
    boot = np.empty((int(bootstrap_replicates), matrix.shape[1]), dtype=np.float64)
    for replicate in range(int(bootstrap_replicates)):
        selected_runs = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        sampled_run_means = []
        for run_index in selected_runs:
            episodes = episode_matrices[int(run_index)]
            selected_episodes = rng.integers(
                0, episodes.shape[0], size=episodes.shape[0]
            )
            sampled_run_means.append(episodes[selected_episodes].mean(axis=0))
        boot[replicate] = np.mean(sampled_run_means, axis=0)
    low, high = np.quantile(boot, (0.025, 0.975), axis=0)
    return (
        {name: float(value) for name, value in zip(_METRIC_NAMES, mean, strict=True)},
        {
            name: [float(lo), float(hi)]
            for name, lo, hi in zip(_METRIC_NAMES, low, high, strict=True)
        },
    )


def _run_level_contrast_intervals(
    episode_matrices: list[np.ndarray],
    *,
    bootstrap_replicates: int,
    seed: int,
    brier_ratio: float,
) -> Mapping[str, list[float]]:
    """Paired run/episode bootstrap intervals used by the registered gate."""

    indexes = {name: index for index, name in enumerate(_METRIC_NAMES)}
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(bootstrap_replicates), 3), dtype=np.float64)
    for replicate in range(int(bootstrap_replicates)):
        selected_runs = rng.integers(
            0, len(episode_matrices), size=len(episode_matrices)
        )
        sampled = []
        for run_index in selected_runs:
            episodes = episode_matrices[int(run_index)]
            selected_episodes = rng.integers(
                0, episodes.shape[0], size=episodes.shape[0]
            )
            sampled.append(episodes[selected_episodes].mean(axis=0))
        mean = np.mean(sampled, axis=0)
        draws[replicate] = (
            mean[indexes["log_score"]]
            - mean[indexes["uniform_baseline_log_score"]],
            mean[indexes["log_score"]]
            - mean[indexes["no_history_baseline_log_score"]],
            mean[indexes["event_brier"]]
            - float(brier_ratio) * mean[indexes["prior_baseline_brier"]],
        )
    low, high = np.quantile(draws, (0.025, 0.975), axis=0)
    names = (
        "log_score_minus_uniform",
        "log_score_minus_no_history",
        "event_brier_minus_registered_baseline",
    )
    return {
        name: [float(lo), float(hi)]
        for name, lo, hi in zip(names, low, high, strict=True)
    }


def run_posterior_calibration(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp

    started = time.perf_counter()
    config = load_config(args.config, run_kind=args.run_kind)
    if config.run_kind == "formal":
        validate_formal_repository_state()
        validate_registered_python_runtime()
        validate_official_runtime()
    manifest = load_partner_manifest(
        args.partner_manifest,
        expected_layout=config.environment.layout,
        verify_files=not bool(args.skip_manifest_hash_check),
    )
    runs = _owned_calibration_runs(manifest, int(args.seed_index))
    maximum_runs = getattr(args, "maximum_runs", None)
    if maximum_runs is not None:
        if config.run_kind != "mechanical":
            raise ValueError("Calibration run limiting is mechanical-only.")
        runs = runs[: int(maximum_runs)]
    _validate_calibration_runs(runs, formal=config.run_kind == "formal")
    calibration = config.posterior_calibration
    if len(runs) < calibration.minimum_run_count:
        raise RuntimeError(
            "Held-out calibration has too few independent partner-run blocks; "
            "the §2.4 calibration readout is not exported."
        )
    output = Path(args.output).resolve()
    ensure_run_identity(
        output,
        calibration_identity(
            config=config,
            seed_index=int(args.seed_index),
            training_run=args.training_run,
            manifest=manifest,
        ),
    )
    deployment = load_training_model(args.training_run, config)
    # Registered independent calibration key domain (root-seed separation from
    # training/evaluation streams; METHOD_SPEC §2.4).
    root = jnp.asarray(
        official_training_domain_keys(int(args.seed_index))["posterior_calibration"],
        dtype=jnp.uint32,
    )
    root = jax.random.fold_in(root, int(calibration.root_seed_offset))
    payloads: list[Mapping[str, Any]] = []
    table: list[Mapping[str, Any]] = []
    attempted = 0
    for index, run in enumerate(runs):
        batch, block_steps = _collect_calibration_records(
            config=config,
            deployment=deployment,
            run=run,
            key=jax.random.fold_in(root, index),
            run_numeric_id=index,
        )
        attempted += int(block_steps)
        payload = _score_calibration_block(deployment=deployment, batch=batch)
        payloads.append(payload)
        metrics = _block_metrics(
            payload, credibility=float(calibration.credibility)
        )
        row: dict[str, Any] = {
            "partner_run_id": run.run_id,
            "partner_mechanism": run.generation_mechanism,
            "episodes": int(calibration.episodes_per_run),
            "steps": int(block_steps),
            "posterior_entropy": float(payload["mean_posterior_entropy"]),
            "protocol_embedding_norm": float(
                np.linalg.norm(payload["mean_posterior_mixture_summary"])
            ),
            "positive_event_count": int(
                np.sum(
                    np.asarray(payload["event_labels"])
                    * np.asarray(payload["event_mask"])
                )
            ),
            "valid_event_count": int(np.sum(np.asarray(payload["event_mask"]))),
        }
        row.update({key: float(value) for key, value in metrics.items()})
        table.append(row)
    episode_matrices = [
            _episode_metric_matrix(
                payload, credibility=float(calibration.credibility)
            )
            for payload in payloads
        ]
    aggregate, run_bootstrap_ci = _run_level_summary(
        episode_matrices,
        bootstrap_replicates=int(calibration.bootstrap_replicates),
        seed=int(calibration.bootstrap_seed),
    )
    contrast_ci = _run_level_contrast_intervals(
        episode_matrices,
        bootstrap_replicates=int(calibration.bootstrap_replicates),
        seed=int(calibration.bootstrap_seed) + 1,
        brier_ratio=float(calibration.brier_ratio),
    )
    pooled_descriptive = _aggregate_metrics(
        payloads, credibility=float(calibration.credibility)
    )
    positive_event_count = int(sum(row["positive_event_count"] for row in table))
    positive_by_family = {
        family: int(
            sum(
                row["positive_event_count"]
                for row in table
                if row["partner_mechanism"] == family
            )
        )
        for family in sorted({str(row["partner_mechanism"]) for row in table})
    }
    event_estimable = bool(
        positive_event_count >= int(calibration.minimum_positive_event_count)
        and positive_by_family
        and min(positive_by_family.values())
        >= int(calibration.minimum_positive_event_count_per_family)
    )
    event_diagnostics = _event_calibration_diagnostics(payloads)
    run_prevalence = np.asarray(
        [
            row["positive_event_count"] / max(row["valid_event_count"], 1)
            for row in table
        ],
        dtype=np.float64,
    )
    prevalence_rng = np.random.default_rng(int(calibration.bootstrap_seed) + 2)
    prevalence_draws = np.mean(
        run_prevalence[
            prevalence_rng.integers(
                0,
                run_prevalence.size,
                size=(int(calibration.bootstrap_replicates), run_prevalence.size),
            )
        ],
        axis=1,
    )
    event_prevalence_interval = [
        float(value) for value in np.quantile(prevalence_draws, (0.025, 0.975))
    ]
    pass_flags = {
        "log_score_vs_uniform": bool(
            contrast_ci["log_score_minus_uniform"][1]
            <= -float(calibration.log_score_margin)
        ),
        "log_score_vs_no_history": bool(
            contrast_ci["log_score_minus_no_history"][1]
            <= -float(calibration.log_score_margin)
        ),
        "position_coverage_in_band": bool(
            run_bootstrap_ci["coverage_position"][0]
            >= float(calibration.coverage_low)
            and run_bootstrap_ci["coverage_position"][1]
            <= float(calibration.coverage_high)
        ),
        "direction_coverage_in_band": bool(
            run_bootstrap_ci["coverage_direction"][0]
            >= float(calibration.coverage_low)
            and run_bootstrap_ci["coverage_direction"][1]
            <= float(calibration.coverage_high)
        ),
        "event_brier": bool(event_estimable and (
            contrast_ci["event_brier_minus_registered_baseline"][1] <= 0.0
        )),
    }
    overall_pass = bool(all(pass_flags.values()))
    elapsed_wall_seconds = time.perf_counter() - started
    ledger = ResourceLedger(
        calibration_steps=attempted,
        gpu_hours=gpu_hours_for_wall_seconds(elapsed_wall_seconds),
        wall_clock_hours=elapsed_wall_seconds / 3_600.0,
    )
    config_path = Path(args.config).resolve()
    manifest_path = Path(args.partner_manifest).resolve()
    training_path = Path(args.training_run).resolve()
    deployment_path = training_path / "final_deployment"
    score_path = output / "calibration_scores.parquet"
    write_parquet(score_path, table)
    write_json(
        output / "posterior_calibration.json",
        {
            "version": POSTERIOR_CALIBRATION_SCHEMA_VERSION,
            "artifact_type": "depi_posterior_calibration",
            "artifact_name": POSTERIOR_CALIBRATION_ARTIFACT_NAME,
            "method": METHOD_VERSION,
            "method_variant": config.method_variant,
            "layout": config.environment.layout,
            "scientific_readout_allowed": config.run_kind == "formal",
            "official_protocol_version": OFFICIAL_PROTOCOL_VERSION,
            "official_source_commit": OFFICIAL_SOURCE_COMMIT,
            "config_fingerprint": config.fingerprint,
            "sources": {
                "config": {"path": str(config_path), "sha256": sha256_path(config_path)},
                "training_run": {
                    "path": str(training_path),
                    "sha256": sha256_path(training_path),
                },
                "primary_deployment": {
                    "path": str(deployment_path),
                    "sha256": sha256_path(deployment_path),
                },
                "partner_manifest": {
                    "path": str(manifest_path),
                    "sha256": sha256_path(manifest_path),
                },
                "calibration_scores": {
                    "path": str(score_path),
                    "sha256": sha256_path(score_path),
                },
            },
            "resource_ledger": ledger.to_mapping(),
            "protocol": (
                "METHOD_SPEC §2.4 held-out calibration of the exchangeable protocol "
                "posterior (log score primary, interaction-change Brier "
                "secondary, 90% highest-probability-set coverage)"
            ),
            "primary_deployment": str(deployment_path),
            "primary_artifact_name": PRIMARY_ARTIFACT_NAME,
            "model_fingerprint": pytree_fingerprint(
                deployable_parameters(deployment.params)
            ),
            "registered": {
                "log_score_margin": float(calibration.log_score_margin),
                "coverage_low": float(calibration.coverage_low),
                "coverage_high": float(calibration.coverage_high),
                "brier_ratio": float(calibration.brier_ratio),
                "credibility": float(calibration.credibility),
                "coverage_targets": ["relative_position", "direction"],
                "no_history_frame_policy": "retain_current_physical_frame",
                "primary_aggregation_unit": calibration.primary_unit,
                "secondary_aggregation_unit": calibration.secondary_unit,
                "root_seed_offset": int(calibration.root_seed_offset),
                "protocol_component_count": int(config.model.protocol_components),
                "minimum_positive_event_count": int(
                    calibration.minimum_positive_event_count
                ),
                "minimum_positive_event_count_per_family": int(
                    calibration.minimum_positive_event_count_per_family
                ),
            },
            "aggregate": aggregate,
            "run_bootstrap_95_ci": run_bootstrap_ci,
            "run_bootstrap_contrast_95_ci": contrast_ci,
            "pooled_descriptive": pooled_descriptive,
            "event_calibration": {
                "status": "estimable" if event_estimable else "not_estimable",
                "positive_event_count": positive_event_count,
                "positive_event_count_by_partner_family": positive_by_family,
                **event_diagnostics,
                "partner_run_bootstrap_prevalence_95_ci": (
                    event_prevalence_interval
                ),
            },
            "pass": {**pass_flags, "overall": bool(overall_pass)},
            "runs": table,
        },
    )
    write_json(output / "resource_ledger.json", ledger.to_mapping())
    write_json(
        output / "run_metadata.json",
        {
            "artifact_name": POSTERIOR_CALIBRATION_ARTIFACT_NAME,
            "partner_run_blocks": len(runs),
            "calibration_rows": len(table),
            "wall_seconds": elapsed_wall_seconds,
            "scientific_readout": config.run_kind == "formal",
            "scientific_readout_allowed": config.run_kind == "formal",
            "calibration_pass": bool(overall_pass),
            "note": (
                "METHOD_SPEC §2.4 held-out calibration readout; failure "
                "triggers the SCIENTIFIC_SPEC Φ5 downgrade of the Bayes claim."
            ),
        },
    )
    verdict = "PASS" if overall_pass else "FAIL (SCIENTIFIC_SPEC Φ5 downgrade)"
    print(
        f"Complete held-out calibration ({verdict}): "
        f"log_score={aggregate['log_score']:.4f} "
        f"coverage={aggregate['coverage_position']:.3f} "
        f"event_brier={aggregate['event_brier']:.4f} -> {output}"
    )


__all__ = [
    "POSTERIOR_CALIBRATION_ARTIFACT_NAME",
    "POSTERIOR_CALIBRATION_SCHEMA_VERSION",
    "run_posterior_calibration",
]
