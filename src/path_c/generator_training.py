"""Initialization and continuous PPO updates for the one V6 partner generator.

METHOD_SPEC §7.1/§7.2 geometry: the code space is the 3-simplex spanned by a
regular tetrahedron of affinely independent source anchors, sampling is
restricted to the fitted Dirichlet support, diversity is measured by mean
pairwise signature distance on a shared environment-state bank, and after the
imitation phase (progress < mixture_ramp_fraction) the generator backbone is
frozen while adversarial updates act only on the code distribution.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, NamedTuple

import numpy as np

from .decision_geometry import (
    mean_pairwise_signature_contributions,
    mean_pairwise_signature_distance,
)
from .partner_episode import (
    complete_episode_raw_returns,
    generator_ppo_objective,
    generator_weight_schedule,
    lower_tail_cvar_jax,
    source_logit_distillation_loss,
)
from .partner_generator import initial_generator_carry


def source_code_anchors(count: int, dimension: int) -> Any:
    """Regular tetrahedron vertices of the §7.1 code simplex.

    v0=(1,1,1)/√3 (SP), v1=(1,-1,-1)/√3 (OP), v2=(-1,1,-1)/√3 (SA),
    v3=(-1,-1,1)/√3 (FCP).  Pairwise inner products equal -1/3, so the four
    vertices are affinely independent (full rank 3), unlike the legacy
    rank-2 construction v2=-v0, v3=-v1.
    """

    import jax.numpy as jnp

    if int(count) != 4:
        raise ValueError("V6 initialization requires one SP/OP/SA/FCP source.")
    if int(dimension) != 3:
        raise ValueError(
            "METHOD_SPEC §7.1 fixes the generator code space to the "
            "3-dimensional tetrahedral simplex; code_dim must be 3."
        )
    scale = 1.0 / math.sqrt(3.0)
    anchors = scale * np.asarray(
        (
            (1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        ),
        dtype=np.float32,
    )
    return jnp.asarray(anchors)


def _default_simplex_anchors() -> Any:
    import jax.numpy as jnp

    scale = 1.0 / math.sqrt(3.0)
    return jnp.asarray(
        scale
        * np.asarray(
            (
                (1.0, 1.0, 1.0),
                (1.0, -1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
            ),
            dtype=np.float32,
        )
    )


def barycentric_coordinates(codes: Any, anchors: Any | None = None) -> Any:
    """Barycentric weights w (sum to 1) with code = V^T w (§7.1 map)."""

    import jax.numpy as jnp

    vertices = (
        jnp.asarray(anchors, dtype=jnp.float32)
        if anchors is not None
        else _default_simplex_anchors()
    )
    code = jnp.asarray(codes, dtype=jnp.float32)
    flat = code.reshape((-1, 3))
    # Solve [V^T; 1^T] w = [code; 1]; the tetrahedron is affinely independent
    # so the 4x4 augmented matrix is always invertible.
    matrix = jnp.concatenate((vertices.T, jnp.ones((1, 4), dtype=jnp.float32)), axis=0)
    rhs = jnp.concatenate(
        (flat.T, jnp.ones((1, flat.shape[0]), dtype=jnp.float32)), axis=0
    )
    weights = jnp.linalg.solve(matrix, rhs).T
    weights = jnp.maximum(weights, 0.0)
    weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1.0e-12)
    return weights.reshape(code.shape[:-1] + (4,))


def codes_from_barycentric(weights: Any, anchors: Any | None = None) -> Any:
    """Inverse §7.1 map: code = V^T w."""

    import jax.numpy as jnp

    vertices = (
        jnp.asarray(anchors, dtype=jnp.float32)
        if anchors is not None
        else _default_simplex_anchors()
    )
    weights = jnp.asarray(weights, dtype=jnp.float32)
    return weights @ vertices


def fit_dirichlet_alpha(codes: Any, anchors: Any | None = None) -> Any:
    """Method-of-moments Dirichlet concentration fit over simplex codes."""

    import jax.numpy as jnp

    weights = barycentric_coordinates(codes, anchors)
    flat = weights.reshape((-1, 4))
    mean = jnp.mean(flat, axis=0)
    variance = jnp.maximum(jnp.var(flat, axis=0), 1.0e-8)
    concentration = mean * (1.0 - mean) / variance - 1.0
    alpha_0 = jnp.maximum(jnp.mean(concentration), 1.0e-3)
    return jnp.maximum(mean * alpha_0, 1.0e-3)


def simplex_neighbor_codes(
    codes: Any,
    *,
    step: float = 0.05,
    anchors: Any | None = None,
) -> Any:
    """Neighbouring codes strictly inside the fitted simplex support.

    Barycentric weights are moved towards the simplex centroid and renormalised,
    which keeps every perturbed code a valid convex combination of the four
    anchors (the §7.1 replacement for the legacy cube clipping).
    """

    import jax.numpy as jnp

    weights = barycentric_coordinates(codes, anchors)
    centroid = jnp.full(weights.shape[-1:], 0.25, dtype=jnp.float32)
    perturbed = (1.0 - float(step)) * weights + float(step) * centroid
    perturbed = perturbed / jnp.maximum(
        jnp.sum(perturbed, axis=-1, keepdims=True), 1.0e-12
    )
    return codes_from_barycentric(perturbed, anchors)


class GeneratorCodeArchive(NamedTuple):
    """§7.2 retained code archive; capacity fixed at 256 entries."""

    codes: Any
    contributions: Any
    count: Any


def empty_code_archive(capacity: int = 256) -> GeneratorCodeArchive:
    import jax.numpy as jnp

    return GeneratorCodeArchive(
        codes=jnp.zeros((int(capacity), 3), dtype=jnp.float32),
        contributions=jnp.full((int(capacity),), -jnp.inf, dtype=jnp.float32),
        count=jnp.asarray(0, dtype=jnp.int32),
    )


def archive_insert(
    archive: GeneratorCodeArchive, codes: Any, contributions: Any
) -> GeneratorCodeArchive:
    """Merge new codes and keep the top ``capacity`` diversity contributions."""

    import jax.numpy as jnp

    capacity = int(archive.codes.shape[0])
    merged_codes = jnp.concatenate((archive.codes, jnp.asarray(codes)), axis=0)
    merged_contributions = jnp.concatenate(
        (archive.contributions, jnp.asarray(contributions)), axis=0
    )
    order = jnp.argsort(merged_contributions, descending=True)
    return GeneratorCodeArchive(
        codes=merged_codes[order][:capacity],
        contributions=merged_contributions[order][:capacity],
        count=jnp.minimum(
            archive.count + jnp.asarray(codes).shape[0], capacity
        ),
    )


def distill_generator_sources(
    *,
    generator: Any,
    params: Any,
    batch: Any,
    code_anchors: Any,
    optimizer: Any,
    optimizer_state: Any,
    schedule: Any,
    hidden_dim: int,
) -> tuple[Any, Any, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp
    import optax

    indexes = jnp.asarray(schedule, dtype=jnp.int32).reshape(
        (-1, int(schedule.shape[-1]))
    )

    def one(carry: Any, lane_indexes: Any):
        current, current_optimizer_state = carry
        members = batch.source_members[:, lane_indexes]
        codes = code_anchors[members]
        observations = batch.observations[:-1, lane_indexes]
        starts = batch.episode_starts[:-1, lane_indexes]
        keys = jnp.zeros(observations.shape[:2] + (2,), dtype=jnp.uint32)
        initial = initial_generator_carry(int(lane_indexes.shape[0]), hidden_dim)

        def objective(candidate: Any):
            _, output = generator.apply(
                {"params": candidate},
                initial,
                observations,
                codes,
                starts,
                keys,
                method=generator.sequence,
            )
            return source_logit_distillation_loss(
                output.logits,
                batch.owner_logits[:, lane_indexes],
                batch.valid_mask[:, lane_indexes],
            )

        loss, gradients = jax.value_and_grad(objective)(current)
        updates, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current
        )
        return (
            optax.apply_updates(current, updates),
            next_optimizer_state,
        ), loss

    (next_params, next_state), losses = jax.lax.scan(
        one, (params, optimizer_state), indexes
    )
    return next_params, next_state, {
        "generator_source_distillation_kl": losses[-1],
        "generator_source_distillation_updates": jnp.asarray(indexes.shape[0]),
    }


def episode_decision_signatures(batch: Any) -> Any:
    import jax.numpy as jnp

    mask = jnp.asarray(batch.valid_mask, dtype=jnp.float32)
    signatures = jnp.asarray(batch.ego_action_signatures, dtype=jnp.float32)
    return jnp.sum(mask[..., None] * signatures, axis=0) / jnp.maximum(
        jnp.sum(mask, axis=0)[:, None], 1.0
    )


def bank_pairwise_signature_diversity(signatures_by_code: Any) -> Any:
    """§7.2 diversity: mean pairwise code distance on the shared bank.

    ``signatures_by_code`` has shape ``(code_count, bank_size, action_count)``
    and holds real all-action continuation signatures produced on identical
    environment states; the mean over bank states removes any state-visit
    mixture from the diversity measure (review §6).
    """

    import jax.numpy as jnp

    signatures = jnp.asarray(signatures_by_code, dtype=jnp.float32)
    per_state = mean_pairwise_signature_distance(
        signatures.transpose((1, 0, 2))
    )
    return jnp.mean(per_state)


def generator_episode_bonus(
    *,
    batch: Any,
    progress: Any,
    competence_multiplier: Any,
    reference_cvar: Any,
    cvar_level: float,
    maximum_probability: float,
    ramp_fraction: float,
    kernel_bandwidth: float,
    kernel_jitter: float,
    bank_signatures: Any = None,
) -> tuple[Any, Mapping[str, Any]]:
    import jax
    import jax.numpy as jnp

    _, unused_imitation, diversity_weight = generator_weight_schedule(
        progress,
        maximum_probability=maximum_probability,
        ramp_fraction=ramp_fraction,
    )
    del unused_imitation, kernel_bandwidth, kernel_jitter
    raw_returns = jax.lax.stop_gradient(
        jnp.sum(
            jnp.asarray(batch.raw_rewards, dtype=jnp.float32)
            * jnp.asarray(batch.valid_mask, dtype=jnp.float32),
            axis=0,
        )
    )
    # §7.2: BR-diversity abolished.  The only legal diversity source is the
    # mean pairwise signature distance of real all-action continuations on
    # the shared environment-state bank; without a bank payload the diversity
    # bonus contributes zero (never a per-episode mixed-state substitute).
    bank_available = bank_signatures is not None
    diversity_scalar = (
        bank_pairwise_signature_diversity(bank_signatures)
        if bank_available
        else jnp.asarray(0.0, dtype=jnp.float32)
    )
    diversity = jnp.full(raw_returns.shape, jax.lax.stop_gradient(diversity_scalar))
    sorted_returns = jnp.sort(raw_returns)
    tail_count = max(int(math.ceil(float(cvar_level) * raw_returns.shape[0])), 1)
    threshold = sorted_returns[tail_count - 1]
    tail = (raw_returns <= threshold).astype(jnp.float32)
    competence = (
        jnp.asarray(competence_multiplier, dtype=jnp.float32)
        * tail
        * (raw_returns - jnp.asarray(reference_cvar, dtype=jnp.float32))
        / jnp.maximum(jnp.mean(tail), 1.0e-8)
    )
    bonus = (
        diversity_weight * diversity
        + competence
    )
    return jax.lax.stop_gradient(bonus), {
        "generator_diversity_weight": diversity_weight,
        "generator_diversity_bonus_mean": jnp.mean(diversity),
        "generator_diversity_bank_available": jnp.asarray(
            float(bank_available), dtype=jnp.float32
        ),
        "generator_competence_bonus_mean": jnp.mean(competence),
        "generator_episode_raw_cvar": lower_tail_cvar_jax(
            raw_returns, level=cvar_level
        ),
    }


def _smoothness_loss(
    *, generator: Any, params: Any, batch: Any, lane_indexes: Any, hidden_dim: int
) -> Any:
    import jax
    import jax.numpy as jnp

    observations = batch.observations[:, lane_indexes]
    codes = batch.codes[:, lane_indexes]
    starts = jnp.zeros(batch.dones[:, lane_indexes].shape, dtype=jnp.bool_).at[0].set(True)
    keys = jnp.zeros(observations.shape[:2] + (2,), dtype=jnp.uint32)
    initial = jax.tree_util.tree_map(lambda value: value[lane_indexes], batch.initial_carries)
    # §7.1: smoothness probes move inside the simplex support instead of the
    # legacy cube-clipped direction perturbation.
    neighbor = simplex_neighbor_codes(codes, step=0.05)
    _, original = generator.apply(
        {"params": params}, initial, observations, codes, starts, keys,
        method=generator.sequence,
    )
    _, shifted = generator.apply(
        {"params": params}, initial, observations, neighbor, starts, keys,
        method=generator.sequence,
    )
    original_log = jax.nn.log_softmax(original.logits, axis=-1)
    shifted_log = jax.nn.log_softmax(shifted.logits, axis=-1)
    probability = jnp.exp(original_log)
    pointwise = jnp.sum(probability * (original_log - shifted_log), axis=-1)
    mask = jnp.asarray(batch.valid_mask[:, lane_indexes], dtype=jnp.float32)
    return jnp.sum(mask * pointwise) / jnp.maximum(jnp.sum(mask), 1.0)


def update_generator(
    *,
    generator: Any,
    params: Any,
    optimizer_state: Any,
    optimizer: Any,
    batch: Any,
    source_batch: Any,
    code_anchors: Any,
    schedule: Any,
    config: Any,
    progress: Any,
    competence_multiplier: Any,
    reference_cvar: Any,
    bank_signatures: Any = None,
) -> tuple[Any, Any, Mapping[str, Any]]:
    """Run the fixed four-epoch/eight-minibatch generator update.

    ``bank_signatures`` carries the §7.2 shared-state diversity payload
    ``(code_count, bank_size, action_count)`` produced by real all-action
    continuations on the registered anchor bank.
    """

    import jax
    import jax.numpy as jnp
    import optax

    indexes = jnp.asarray(schedule, dtype=jnp.int32).reshape(
        (-1, int(schedule.shape[-1]))
    )
    _, imitation_weight, _ = generator_weight_schedule(
        progress,
        maximum_probability=config.partner_generator.maximum_generator_probability,
        ramp_fraction=config.partner_generator.mixture_ramp_fraction,
    )
    episode_bonus, bonus_metrics = generator_episode_bonus(
        batch=batch,
        progress=progress,
        competence_multiplier=competence_multiplier,
        reference_cvar=reference_cvar,
        cvar_level=config.partner_generator.cvar_level,
        maximum_probability=config.partner_generator.maximum_generator_probability,
        ramp_fraction=config.partner_generator.mixture_ramp_fraction,
        kernel_bandwidth=config.partner_generator.kernel_bandwidth,
        kernel_jitter=config.partner_generator.kernel_jitter,
        bank_signatures=bank_signatures,
    )

    def one(carry: Any, lane_indexes: Any):
        current, current_optimizer_state = carry
        observations = batch.observations[:, lane_indexes]
        codes = batch.codes[:, lane_indexes]
        starts = jnp.zeros(batch.dones[:, lane_indexes].shape, dtype=jnp.bool_).at[0].set(True)
        keys = jnp.zeros(observations.shape[:2] + (2,), dtype=jnp.uint32)
        initial = jax.tree_util.tree_map(
            lambda value: value[lane_indexes], batch.initial_carries
        )
        source_members = source_batch.source_members[:, lane_indexes]
        source_codes = code_anchors[source_members]
        source_observations = source_batch.observations[:-1, lane_indexes]
        source_starts = source_batch.episode_starts[:-1, lane_indexes]
        source_keys = jnp.zeros(source_observations.shape[:2] + (2,), dtype=jnp.uint32)
        source_initial = initial_generator_carry(
            int(lane_indexes.shape[0]), config.partner_generator.hidden_dim
        )

        def objective(candidate: Any):
            _, output = generator.apply(
                {"params": candidate}, initial, observations, codes, starts, keys,
                method=generator.sequence,
            )
            _, source_output = generator.apply(
                {"params": candidate}, source_initial, source_observations,
                source_codes, source_starts, source_keys, method=generator.sequence,
            )
            imitation = source_logit_distillation_loss(
                source_output.logits,
                source_batch.owner_logits[:, lane_indexes],
                source_batch.valid_mask[:, lane_indexes],
            )
            smoothness = _smoothness_loss(
                generator=generator,
                params=candidate,
                batch=batch,
                lane_indexes=lane_indexes,
                hidden_dim=config.partner_generator.hidden_dim,
            )
            return generator_ppo_objective(
                new_logits=output.logits,
                new_values=output.value,
                actions=batch.actions[:, lane_indexes],
                behavior_log_probabilities=batch.behavior_log_probabilities[:, lane_indexes],
                behavior_values=batch.behavior_values[:, lane_indexes],
                official_shaped_rewards=batch.official_shaped_rewards[:, lane_indexes],
                dones=batch.dones[:, lane_indexes],
                valid_mask=batch.valid_mask[:, lane_indexes],
                gamma=config.ppo.gamma,
                gae_lambda=config.ppo.gae_lambda,
                clip_epsilon=config.ppo.clip_epsilon,
                value_clip_epsilon=config.ppo.value_clip_epsilon,
                value_weight=config.ppo.value_weight,
                entropy_weight=config.ppo.entropy_weight,
                normalize_advantages=config.ppo.normalize_advantages,
                diversity_episode_bonus=episode_bonus[lane_indexes],
                imitation_loss=imitation,
                imitation_weight=imitation_weight,
                smoothness_loss=smoothness,
                smoothness_weight=config.partner_generator.smoothness_weight,
            )

        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(current)
        updates, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current
        )
        next_params = optax.apply_updates(current, updates)
        return (next_params, next_optimizer_state), {**metrics, "generator_loss": loss}

    (scan_params, scan_state), scan_metrics = jax.lax.scan(
        one, (params, optimizer_state), indexes
    )
    # §7.2: once imitation progress completes (progress >= ramp fraction) the
    # generator backbone is frozen; adversarial updates act only on the code
    # distribution, reported here as the fitted Dirichlet concentration.
    frozen = jnp.asarray(progress, dtype=jnp.float32) >= jnp.asarray(
        config.partner_generator.mixture_ramp_fraction, dtype=jnp.float32
    )
    next_params, next_state = jax.lax.cond(
        frozen,
        lambda _: (params, optimizer_state),
        lambda _: (scan_params, scan_state),
        operand=None,
    )
    frozen_metrics = jax.tree_util.tree_map(
        lambda value: value[-1], scan_metrics
    )
    adversarial_alpha = jax.lax.cond(
        frozen,
        lambda _: fit_dirichlet_alpha(batch.codes),
        lambda _: jnp.zeros((4,), dtype=jnp.float32),
        operand=None,
    )
    final_metrics = jax.lax.cond(
        frozen,
        lambda _: jax.tree_util.tree_map(jnp.zeros_like, frozen_metrics),
        lambda _: frozen_metrics,
        operand=None,
    )
    return next_params, next_state, {
        **final_metrics,
        **bonus_metrics,
        "generator_imitation_schedule_weight": imitation_weight,
        "generator_frozen": frozen.astype(jnp.float32),
        "generator_code_alpha": adversarial_alpha,
    }


__all__ = [
    "GeneratorCodeArchive",
    "archive_insert",
    "bank_pairwise_signature_diversity",
    "barycentric_coordinates",
    "codes_from_barycentric",
    "distill_generator_sources",
    "empty_code_archive",
    "episode_decision_signatures",
    "fit_dirichlet_alpha",
    "generator_episode_bonus",
    "simplex_neighbor_codes",
    "source_code_anchors",
    "update_generator",
]
