"""Held-out protocol calibration evaluation for DEPI (METHOD_SPEC §2.4).

Migrated from the legacy V6 safety-wrapper app: the continuous Gaussian
belief semantics (``prior_gaussian_summary``, ``belief_summary``/``belief_mean``
latents, Mahalanobis support scoring, predicted/empirical policy gain) are
replaced by the registered K=4 categorical protocol-posterior scoring rules:

* primary   -- posterior-predictive log score (per-step NLL, nats);
* secondary -- Brier score of the interaction_change event;
* coverage  -- empirical coverage of the 90% highest-probability categorical
  prediction sets, registered band [0.85, 0.95];
* gate      -- ``calibration_pass_decision`` (failure triggers SCIENTIFIC_SPEC
  Φ5 downgrade of the Bayes claim).

The posterior context is the protocol mixture summary ``c_t = Σ_k π_{t,k} m_k``
(``mixture_summary``); the prior context is the uniform-posterior mixture of
§1.2/§2.1.  CLI wiring is unchanged: ``calibrate-safety`` in
``experiments/overcooked_v2/path_c.py`` dispatches ``run_safety_calibration``.
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
from src.path_c.belief_set_encoder import mixture_summary
from src.path_c.calibration import (
    calibration_pass_decision,
    event_brier_score,
    highest_probability_set_coverage,
    per_step_log_score,
    uniform_prior_log_score,
)
from src.path_c.experiment import (
    ENGINEERING_SEED_INDEX,
    METHOD_VERSION,
    load_config,
    load_partner_manifest,
    official_training_domain_keys,
)
from src.path_c.external_partner import make_external_partner_functions
from src.path_c.protocol_encoder import posterior_entropy
from src.path_c.resources import ResourceLedger
from src.path_c.runner import collect_rollout, initialize_runner
from src.path_c.storage import (
    calibration_identity,
    ensure_run_identity,
    pytree_fingerprint,
    validate_formal_repository_state,
    validate_registered_python_runtime,
    write_json,
    write_parquet,
)
from src.path_c.types import ContextOutput


SAFETY_ARTIFACT_NAME = "DELTA-ZSC-E2E+Safety"
_MECHANISM_ALIASES = {
    "rnn-sp": "sp",
    "sp": "sp",
    "rnn-op": "op",
    "op": "op",
    "state-augmented": "sa",
    "sa": "sa",
    "fcp": "fcp",
}
# METHOD_SPEC §2.1 registered class anchors SP→1, OP→2, SA→3, FCP→4
# (converted to zero-based posterior indexes).
_REGIME_INDEX_BY_MECHANISM = {"sp": 0, "op": 1, "sa": 2, "fcp": 3}
_REGISTERED_LOG_SCORE_MARGIN = 0.02
_REGISTERED_COVERAGE_LOW = 0.85
_REGISTERED_COVERAGE_HIGH = 0.95
_REGISTERED_BRIER_RATIO = 0.90
_REGISTERED_CREDIBILITY = 0.90


def _owned_calibration_runs(manifest: Any, seed_index: int) -> tuple[Any, ...]:
    owner = None if int(seed_index) == ENGINEERING_SEED_INDEX else int(seed_index)
    return tuple(
        run
        for run in manifest.by_role("calibration")
        if run.owner_seed_index == owner
    )


def _validate_calibration_runs(runs: tuple[Any, ...], *, formal: bool) -> None:
    parent_ids = [str(run.parent_training_run_id) for run in runs]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("Safety calibration requires independent partner-run blocks.")
    if formal:
        counts = Counter(_MECHANISM_ALIASES.get(run.generation_mechanism) for run in runs)
        if counts != Counter({"sp": 5, "op": 5, "sa": 5, "fcp": 5}):
            raise ValueError(
                "Formal V6 safety calibration requires five independent runs "
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
            config.environment, num_envs=int(config.calibration.episodes_per_run)
        ),
    )
    environment = VectorEnvironment.create(calibration_config)
    pool = FrozenPartnerPool.from_checkpoints(
        (run.checkpoint,),
        parent_training_run_ids=(run.parent_training_run_id,),
    )
    # DEPI replaces the deleted V6 Gaussian ``latent_dim`` with the context
    # dimension concat(u, c_t) (METHOD_SPEC §1.1); the external-partner runtime
    # only needs a size for its metadata slot.
    context_dim = int(config.model.capability_dim) + int(
        config.model.component_embedding_dim
    )
    partner_functions = make_external_partner_functions(
        pool=pool,
        member_indexes=jnp.asarray(0, dtype=jnp.int32),
        latent_dim=context_dim,
        code_dim=config.partner_generator.code_dim,
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
    steps = int(config.calibration.episodes_per_run) * int(
        config.environment.episode_steps
    )
    return batch, steps


def _component_group_log_probabilities(prediction: Any, targets: Any) -> Any:
    """Per-component (..., K) grouped log p_k(y_{t+1}).

    Uses the same per-head log-probability primitives and the same
    visibility masking as ``mixture_response_loss`` (METHOD_SPEC §2.3), with
    the per-component grouping
    ``log p(vis) + log p(pos) + visible * (log p(dir) + log p(inv) + log p(event))``.

    Note on comparability: this grouping shares the *grouping structure* of
    the trained NLL, but NOT its marginalization order.  Training
    marginalizes each head separately (Σ_k logsumexp per head, then sums the
    heads); the held-out score here first sums the heads inside each
    component and marginalizes once.  The two are generally not numerically
    identical (log of a sum ≠ sum of logs across mixture components), so the
    held-out log score must be read as a same-family calibration quantity,
    not as the exact trained NLL evaluated out-of-sample.
    """

    import jax.numpy as jnp

    # Shared single-source-of-truth primitives of mixture_response_loss.
    from src.path_c.response_targets import (
        _bernoulli_log_probability,
        _categorical_log_probability,
    )

    visibility = _bernoulli_log_probability(
        prediction.visibility_logit, targets.visibility
    )
    position = _categorical_log_probability(
        prediction.relative_position_logits, targets.relative_position
    )
    direction = _categorical_log_probability(
        prediction.direction_logits, targets.direction
    )
    inventory = jnp.sum(
        _categorical_log_probability(prediction.inventory_logits, targets.inventory),
        axis=-1,
    )
    event = _bernoulli_log_probability(
        prediction.interaction_change_logit, targets.interaction_change
    )
    visible = jnp.asarray(targets.visible_mask, dtype=jnp.float32)[..., None]
    return visibility + position + visible * (direction + inventory + event)


def _mixture_log_probability(
    posterior_log_probabilities: Any, component_log_probabilities: Any
) -> Any:
    from jax.scipy.special import logsumexp

    import jax.numpy as jnp

    return logsumexp(
        jnp.asarray(posterior_log_probabilities, dtype=jnp.float32)
        + jnp.asarray(component_log_probabilities, dtype=jnp.float32),
        axis=-1,
    )


def _score_calibration_block(*, deployment: Any, batch: Any) -> Mapping[str, Any]:
    """Score one held-out partner block with the §2.4 proper scoring rules.

    Returns the per-step tensors needed for pooled aggregation plus the
    discrete-posterior context readouts (posterior entropy and the mixture
    summary ``c = Σ_k π_k m_k`` of the mean posterior).
    """

    import jax
    import jax.numpy as jnp

    from src.path_c.response_targets import (
        extract_partner_response_targets,
        official_partner_observation_planes,
    )

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
    component_log_probabilities = _component_group_log_probabilities(
        prediction, targets
    )
    log_pi = prediction.posterior_log_probabilities
    posterior = jnp.exp(log_pi)
    model_log_probabilities = _mixture_log_probability(log_pi, component_log_probabilities)
    # Registered baseline 2 (§2.4): no-frame -- kinematic heads receive zero
    # frame features; the protocol event head never reads the frame (§2.2).
    # Side effect to keep in mind when reading this baseline: the posterior
    # carried into ``sliced`` below was still produced by recursive Bayes
    # filtering over the *real* observation sequence, i.e. the no-frame
    # baseline removes frame evidence from the response heads only, while the
    # protocol posterior keeps accumulating observation-driven updates.
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
        context.capability[:-1],
        context.protocol_probabilities[:-1],
        context.protocol_embedding[:-1],
    )
    no_frame = model.apply(
        {"params": params},
        sliced,
        jnp.zeros_like(batch.observations[:-1]),
        batch.actions,
        method=model.response_from_context_and_action,
    )
    no_frame_log_probabilities = _mixture_log_probability(
        no_frame.posterior_log_probabilities,
        _component_group_log_probabilities(no_frame, targets),
    )
    event_sigmoid = jax.nn.sigmoid(prediction.interaction_change_logit)
    component_matrix = params["protocol_component_embeddings"]["embedding"]
    mean_posterior = jnp.mean(posterior, axis=(0, 1))
    return {
        "model_log_probabilities": np.asarray(model_log_probabilities),
        "component_log_probabilities": np.asarray(component_log_probabilities),
        "no_frame_log_probabilities": np.asarray(no_frame_log_probabilities),
        "event_probabilities": np.asarray(jnp.sum(posterior * event_sigmoid, axis=-1)),
        "prior_event_probabilities": np.asarray(jnp.mean(event_sigmoid, axis=-1)),
        "event_labels": np.asarray(targets.interaction_change),
        "position_probabilities": np.asarray(
            jax.nn.softmax(prediction.relative_position_logits, axis=-1)
        ),
        "position_labels": np.asarray(targets.relative_position),
        "direction_probabilities": np.asarray(
            jax.nn.softmax(prediction.direction_logits, axis=-1)
        ),
        "direction_labels": np.asarray(targets.direction),
        "visible_mask": np.asarray(targets.visible_mask),
        "posterior_probabilities": np.asarray(posterior),
        "mean_posterior_entropy": float(jnp.mean(posterior_entropy(posterior))),
        "mean_posterior_mixture_summary": np.asarray(
            mixture_summary(mean_posterior, component_matrix)
        ),
    }


def _block_metrics(payload: Mapping[str, Any]) -> Mapping[str, float]:
    """Per-block §2.4 readouts computed with the shared calibration primitives."""

    visible = payload["visible_mask"]
    return {
        "log_score": float(per_step_log_score(payload["model_log_probabilities"])),
        "uniform_log_score": float(
            uniform_prior_log_score(payload["component_log_probabilities"])
        ),
        "no_frame_log_score": float(
            per_step_log_score(payload["no_frame_log_probabilities"])
        ),
        "event_brier": float(
            event_brier_score(
                payload["event_probabilities"], payload["event_labels"], mask=visible
            )
        ),
        "prior_event_brier": float(
            event_brier_score(
                payload["prior_event_probabilities"],
                payload["event_labels"],
                mask=visible,
            )
        ),
        "coverage_position": float(
            highest_probability_set_coverage(
                payload["position_probabilities"],
                payload["position_labels"],
                credibility=_REGISTERED_CREDIBILITY,
                mask=visible,
            )
        ),
        "coverage_direction": float(
            highest_probability_set_coverage(
                payload["direction_probabilities"],
                payload["direction_labels"],
                credibility=_REGISTERED_CREDIBILITY,
                mask=visible,
            )
        ),
    }


def _aggregate_metrics(payloads: list[Mapping[str, Any]]) -> Mapping[str, float]:
    """Pooled §2.4 readouts over all run-disjoint held-out blocks."""

    def stacked(name: str) -> np.ndarray:
        return np.concatenate([payload[name] for payload in payloads], axis=0)

    visible = stacked("visible_mask")
    return {
        "log_score": float(per_step_log_score(stacked("model_log_probabilities"))),
        "uniform_baseline_log_score": float(
            uniform_prior_log_score(stacked("component_log_probabilities"))
        ),
        "no_frame_baseline_log_score": float(
            per_step_log_score(stacked("no_frame_log_probabilities"))
        ),
        "event_brier": float(
            event_brier_score(
                stacked("event_probabilities"), stacked("event_labels"), mask=visible
            )
        ),
        "prior_baseline_brier": float(
            event_brier_score(
                stacked("prior_event_probabilities"),
                stacked("event_labels"),
                mask=visible,
            )
        ),
        "coverage_position": float(
            highest_probability_set_coverage(
                stacked("position_probabilities"),
                stacked("position_labels"),
                credibility=_REGISTERED_CREDIBILITY,
                mask=visible,
            )
        ),
        "coverage_direction": float(
            highest_probability_set_coverage(
                stacked("direction_probabilities"),
                stacked("direction_labels"),
                credibility=_REGISTERED_CREDIBILITY,
                mask=visible,
            )
        ),
        "posterior_entropy": float(
            np.mean([payload["mean_posterior_entropy"] for payload in payloads])
        ),
    }


def run_safety_calibration(args: argparse.Namespace) -> None:
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
    _validate_calibration_runs(runs, formal=config.run_kind == "formal")
    if len(runs) < config.calibration.minimum_run_count:
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
        official_training_domain_keys(int(args.seed_index))["calibration"],
        dtype=jnp.uint32,
    )
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
        metrics = _block_metrics(payload)
        posterior = payload["posterior_probabilities"]
        regime_index = _REGIME_INDEX_BY_MECHANISM.get(
            _MECHANISM_ALIASES.get(run.generation_mechanism, run.generation_mechanism)
        )
        row: dict[str, Any] = {
            "partner_run_id": run.run_id,
            "partner_mechanism": run.generation_mechanism,
            "episodes": int(config.calibration.episodes_per_run),
            "steps": int(block_steps),
            "posterior_entropy": float(payload["mean_posterior_entropy"]),
            "protocol_embedding_norm": float(
                np.linalg.norm(payload["mean_posterior_mixture_summary"])
            ),
        }
        row.update({key: float(value) for key, value in metrics.items()})
        if regime_index is not None:
            row["regime_posterior_mass"] = float(posterior[..., regime_index].mean())
            row["regime_argmax_accuracy"] = float(
                (posterior.argmax(axis=-1) == regime_index).mean()
            )
        table.append(row)
    aggregate = _aggregate_metrics(payloads)
    pass_flags = {
        "log_score_vs_uniform": bool(
            aggregate["log_score"]
            <= aggregate["uniform_baseline_log_score"] - _REGISTERED_LOG_SCORE_MARGIN
        ),
        "log_score_vs_no_frame": bool(
            aggregate["log_score"]
            <= aggregate["no_frame_baseline_log_score"] - _REGISTERED_LOG_SCORE_MARGIN
        ),
        "coverage_in_band": bool(
            _REGISTERED_COVERAGE_LOW
            <= aggregate["coverage_position"]
            <= _REGISTERED_COVERAGE_HIGH
        ),
        "event_brier": bool(
            aggregate["event_brier"]
            <= _REGISTERED_BRIER_RATIO * aggregate["prior_baseline_brier"]
        ),
    }
    overall_pass = calibration_pass_decision(
        model_log_score=aggregate["log_score"],
        uniform_baseline_log_score=aggregate["uniform_baseline_log_score"],
        no_frame_baseline_log_score=aggregate["no_frame_baseline_log_score"],
        coverage=aggregate["coverage_position"],
        event_brier=aggregate["event_brier"],
        prior_baseline_brier=aggregate["prior_baseline_brier"],
        log_score_margin=_REGISTERED_LOG_SCORE_MARGIN,
        coverage_low=_REGISTERED_COVERAGE_LOW,
        coverage_high=_REGISTERED_COVERAGE_HIGH,
        brier_ratio=_REGISTERED_BRIER_RATIO,
    )
    write_json(
        output / "held_out_calibration.json",
        {
            "artifact_name": SAFETY_ARTIFACT_NAME,
            "method": METHOD_VERSION,
            "protocol": (
                "METHOD_SPEC §2.4 held-out calibration of the K=4 protocol "
                "posterior (log score primary, interaction-change Brier "
                "secondary, 90% highest-probability-set coverage)"
            ),
            "primary_deployment": str(
                Path(args.training_run).resolve() / "final_deployment"
            ),
            "primary_artifact_name": PRIMARY_ARTIFACT_NAME,
            "model_fingerprint": pytree_fingerprint(
                deployable_parameters(deployment.params)
            ),
            "registered": {
                "log_score_margin": _REGISTERED_LOG_SCORE_MARGIN,
                "coverage_low": _REGISTERED_COVERAGE_LOW,
                "coverage_high": _REGISTERED_COVERAGE_HIGH,
                "brier_ratio": _REGISTERED_BRIER_RATIO,
                "credibility": _REGISTERED_CREDIBILITY,
                "coverage_target": "relative_position",
            },
            "aggregate": aggregate,
            "pass": {**pass_flags, "overall": bool(overall_pass)},
            "runs": table,
        },
    )
    write_parquet(output / "calibration_scores.parquet", table)
    ledger = ResourceLedger(calibration_steps=attempted)
    write_json(output / "resource_ledger.json", ledger.to_mapping())
    write_json(
        output / "run_metadata.json",
        {
            "artifact_name": SAFETY_ARTIFACT_NAME,
            "partner_run_blocks": len(runs),
            "calibration_rows": len(table),
            "wall_seconds": time.perf_counter() - started,
            "scientific_readout": True,
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


__all__ = ["SAFETY_ARTIFACT_NAME", "run_safety_calibration"]
