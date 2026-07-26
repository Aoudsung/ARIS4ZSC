"""Bellman-mixture, outcome, and value-preserving response objectives."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


class FrozenAssignments(NamedTuple):
    responsibilities: Any
    responsibility_energies: Any
    bootstrap_mask: Any
    bellman_targets: Any
    response_signature_targets: Any
    response_code_targets: Any


class LossBundle(NamedTuple):
    total: Any
    metrics: Mapping[str, Any]


def gather_actions(values: Any, actions: Any, *, action_axis: int = -1) -> Any:
    import jax.numpy as jnp

    source = jnp.asarray(values)
    indexes = jnp.asarray(actions, dtype=jnp.int32)
    axis = action_axis if action_axis >= 0 else source.ndim + action_axis
    shape = list(indexes.shape)
    while len(shape) < source.ndim:
        shape.insert(axis, 1)
    expanded = indexes.reshape(tuple(shape))
    gathered = jnp.take_along_axis(source, expanded, axis=axis)
    return jnp.squeeze(gathered, axis=axis)


def huber(error: Any, delta: float = 1.0) -> Any:
    import jax.numpy as jnp

    absolute = jnp.abs(jnp.asarray(error))
    quadratic = jnp.minimum(absolute, delta)
    return 0.5 * jnp.square(quadratic) + delta * (absolute - quadratic)


def bellman_targets(
    *,
    rewards: Any,
    dones: Any,
    next_execution_probabilities: Any,
    target_next_q_values: Any,
    gamma: float,
) -> Any:
    """Build per-slot targets using target-policy expectation and clipped Q."""

    import jax.numpy as jnp

    target_q = jnp.min(jnp.asarray(target_next_q_values), axis=-3)
    expected = jnp.einsum(
        "...a,...ma->...m", jnp.asarray(next_execution_probabilities), target_q
    )
    return jnp.asarray(rewards)[..., None] + float(gamma) * (
        1.0 - jnp.asarray(dones, dtype=jnp.float32)
    )[..., None] * expected


def gaussian_negative_log_likelihood(
    value: Any, mean: Any, log_standard_deviation: Any
) -> Any:
    import jax.numpy as jnp

    log_std = jnp.asarray(log_standard_deviation)
    normalized = (jnp.asarray(value) - jnp.asarray(mean)) * jnp.exp(-log_std)
    return 0.5 * jnp.square(normalized) + log_std


def _td_evidence_items(*, q_values: Any, actions: Any, targets: Any) -> Any:
    """Return per-transition, per-slot twin-averaged Bellman evidence."""

    import jax.numpy as jnp

    selected = gather_actions(q_values, actions, action_axis=-1)
    errors = huber(selected - jnp.asarray(targets)[..., None, :])
    return jnp.mean(errors, axis=-2)


def slot_outcome_nll_items(
    *,
    response_logits: Any,
    reward_mean: Any,
    reward_log_standard_deviation: Any,
    next_q_mean: Any,
    next_q_log_standard_deviation: Any,
    response_codes: Any,
    rewards: Any,
    target_next_q_values: Any,
    dones: Any,
    actions: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """Return per-transition, per-slot control-outcome negative evidence."""

    import jax
    import jax.numpy as jnp

    action_response_logits = gather_actions(
        response_logits, actions, action_axis=-2
    )
    response_target = jnp.asarray(response_codes, dtype=jnp.int32)
    response_nll = -jnp.take_along_axis(
        jax.nn.log_softmax(action_response_logits, axis=-1),
        response_target[..., None, None],
        axis=-1,
    )[..., 0]
    selected_reward_mean = gather_actions(reward_mean, actions, action_axis=-1)
    selected_reward_log_std = gather_actions(
        reward_log_standard_deviation, actions, action_axis=-1
    )
    reward_nll = gaussian_negative_log_likelihood(
        jnp.asarray(rewards)[..., None],
        selected_reward_mean,
        selected_reward_log_std,
    )
    selected_next_mean = gather_actions(next_q_mean, actions, action_axis=-3)
    selected_next_log_std = gather_actions(
        next_q_log_standard_deviation, actions, action_axis=-3
    )
    selected_response_mean = gather_actions(
        selected_next_mean, response_codes, action_axis=-2
    )
    selected_response_log_std = gather_actions(
        selected_next_log_std, response_codes, action_axis=-2
    )
    continuation_target = jnp.asarray(target_next_q_values)
    continuation_target = jnp.where(
        jnp.asarray(dones, dtype=jnp.bool_)[..., None, None, None],
        jnp.zeros_like(continuation_target),
        continuation_target,
    )
    continuation_nll_full = gaussian_negative_log_likelihood(
        continuation_target,
        selected_response_mean,
        selected_response_log_std,
    )
    continuation_nll = jnp.mean(continuation_nll_full, axis=(-3, -1))
    total = response_nll + reward_nll + continuation_nll
    return total, {
        "response_nll_items": response_nll,
        "reward_nll_items": reward_nll,
        "next_q_nll_items": continuation_nll,
    }


def episode_responsibility_evidence(
    *,
    q_values: Any,
    actions: Any,
    targets: Any,
    temperature: float,
    response_logits: Any | None = None,
    reward_mean: Any | None = None,
    reward_log_standard_deviation: Any | None = None,
    next_q_mean: Any | None = None,
    next_q_log_standard_deviation: Any | None = None,
    response_codes: Any | None = None,
    rewards: Any | None = None,
    target_next_q_values: Any | None = None,
    dones: Any | None = None,
    availability_mask: Any | None = None,
) -> tuple[Any, Any]:
    """Infer one stopped latent-expert posterior per complete episode lane.

    Evidence is accumulated over the complete episode instead of averaged over
    400 steps. Optional outcome tensors add response, reward, and next-control
    likelihoods. No partner identity or provenance enters this calculation.
    """

    import jax
    import jax.numpy as jnp

    if not 0.0 < float(temperature) < float("inf"):
        raise ValueError("Responsibility temperature must be finite and positive.")
    evidence_items = _td_evidence_items(
        q_values=q_values, actions=actions, targets=targets
    )
    optional = (
        response_logits,
        reward_mean,
        reward_log_standard_deviation,
        next_q_mean,
        next_q_log_standard_deviation,
        response_codes,
        rewards,
        target_next_q_values,
        dones,
    )
    if any(value is not None for value in optional):
        if any(value is None for value in optional):
            raise ValueError(
                "Joint responsibility evidence requires every outcome tensor."
            )
        outcome_items, unused_parts = slot_outcome_nll_items(
            response_logits=response_logits,
            reward_mean=reward_mean,
            reward_log_standard_deviation=reward_log_standard_deviation,
            next_q_mean=next_q_mean,
            next_q_log_standard_deviation=next_q_log_standard_deviation,
            response_codes=response_codes,
            rewards=rewards,
            target_next_q_values=target_next_q_values,
            dones=dones,
            actions=actions,
        )
        del unused_parts
        evidence_items = evidence_items + outcome_items
    energies = jnp.sum(evidence_items, axis=0)
    if availability_mask is None:
        available = jnp.ones_like(energies, dtype=jnp.bool_)
    else:
        available = jnp.asarray(availability_mask, dtype=jnp.bool_)
        if available.shape != energies.shape:
            raise ValueError(
                "Responsibility availability must match [environment, slot]."
            )
        empty = ~jnp.any(available, axis=-1)
        available = available.at[..., 0].set(available[..., 0] | empty)
    minimum = jnp.min(
        jnp.where(available, energies, jnp.inf), axis=-1, keepdims=True
    )
    logits = jnp.where(
        available,
        -(energies - minimum) / float(temperature),
        -jnp.inf,
    )
    responsibilities = jax.nn.softmax(logits, axis=-1)
    return (
        jax.lax.stop_gradient(responsibilities),
        jax.lax.stop_gradient(energies),
    )


def episode_responsibilities(**kwargs: Any) -> Any:
    """Compatibility wrapper returning only stopped responsibilities."""

    responsibilities, unused_energies = episode_responsibility_evidence(**kwargs)
    del unused_energies
    return responsibilities


def sample_bootstrap_mask(
    key: Any,
    *,
    environment_count: int,
    slot_count: int,
    probability: float,
) -> Any:
    """Sample episode-level expert support and retain one slot if empty."""

    import jax
    import jax.numpy as jnp

    mask_key, fallback_key = jax.random.split(key)
    mask = jax.random.bernoulli(
        mask_key, probability, (environment_count, slot_count)
    )
    fallback = jax.random.randint(
        fallback_key, (environment_count,), 0, slot_count
    )
    fallback_mask = jax.nn.one_hot(fallback, slot_count, dtype=jnp.bool_)
    return jnp.where(jnp.any(mask, axis=-1, keepdims=True), mask, fallback_mask)


def bellman_loss(
    *,
    q_values: Any,
    actions: Any,
    targets: Any,
    stopped_responsibilities: Any,
    bootstrap_mask: Any,
) -> LossBundle:
    import jax
    import jax.numpy as jnp

    selected = gather_actions(q_values, actions, action_axis=-1)
    per_item = huber(selected - jnp.asarray(targets)[..., None, :])
    weights = jax.lax.stop_gradient(
        jnp.asarray(stopped_responsibilities)
        * jnp.asarray(bootstrap_mask, dtype=jnp.float32)
    )
    time_weights = weights[None, :, None, :]
    denominator = (
        jnp.asarray(q_values).shape[0] * 2.0 * jnp.sum(weights) + 1.0e-8
    )
    loss = jnp.sum(per_item * time_weights) / denominator
    return LossBundle(
        total=loss,
        metrics={
            "bellman": loss,
            "mean_absolute_td_error": jnp.sum(
                jnp.abs(selected - jnp.asarray(targets)[..., None, :])
                * time_weights
            )
            / denominator,
        },
    )


def outcome_loss(
    *,
    response_logits: Any,
    reward_mean: Any,
    reward_log_standard_deviation: Any,
    next_q_mean: Any,
    next_q_log_standard_deviation: Any,
    response_codes: Any,
    rewards: Any,
    target_next_q_values: Any,
    dones: Any,
    actions: Any,
    stopped_responsibilities: Any,
    bootstrap_mask: Any | None = None,
    terminal_response: int = 15,
) -> LossBundle:
    """Fit slot outcomes using the stopped latent responsibility."""

    import jax
    import jax.numpy as jnp

    slot_terms, parts = slot_outcome_nll_items(
        response_logits=response_logits,
        reward_mean=reward_mean,
        reward_log_standard_deviation=reward_log_standard_deviation,
        next_q_mean=next_q_mean,
        next_q_log_standard_deviation=next_q_log_standard_deviation,
        response_codes=response_codes,
        rewards=rewards,
        target_next_q_values=target_next_q_values,
        dones=dones,
        actions=actions,
    )
    weights = jnp.asarray(stopped_responsibilities)
    if bootstrap_mask is not None:
        weights = weights * jnp.asarray(bootstrap_mask, dtype=jnp.float32)
    weights = jax.lax.stop_gradient(weights)[None, ...]
    denominator = (
        jnp.asarray(response_logits).shape[0] * jnp.sum(weights) + 1.0e-8
    )
    loss = jnp.sum(slot_terms * weights) / denominator
    terminal_mean = next_q_mean[..., terminal_response, :]
    terminal_penalty = jnp.mean(jnp.square(terminal_mean))
    total = loss + terminal_penalty
    return LossBundle(
        total=total,
        metrics={
            "slot_outcome": loss,
            "response_nll": jnp.sum(parts["response_nll_items"] * weights)
            / denominator,
            "reward_nll": jnp.sum(parts["reward_nll_items"] * weights)
            / denominator,
            "next_q_nll": jnp.sum(parts["next_q_nll_items"] * weights)
            / denominator,
            "terminal_continuation_penalty": terminal_penalty,
        },
    )


def response_encoder_loss(
    *,
    predicted_signatures: Any,
    target_signatures: Any,
    target_codes: Any,
    codebook_embeddings: Any,
) -> LossBundle:
    import jax
    import jax.numpy as jnp

    predicted = jnp.asarray(predicted_signatures)
    target = jax.lax.stop_gradient(jnp.asarray(target_signatures))
    codebook = jax.lax.stop_gradient(jnp.asarray(codebook_embeddings))
    logits = -jnp.sum(jnp.square(predicted[..., None, :] - codebook), axis=-1)
    targets = jnp.asarray(target_codes, dtype=jnp.int32)
    valid = targets < codebook.shape[0]
    safe_targets = jnp.where(valid, targets, 0)
    classification_items = -jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=-1),
        safe_targets[..., None],
        axis=-1,
    )[..., 0]
    denominator = jnp.sum(valid.astype(jnp.float32)) + 1.0e-8
    classification = jnp.sum(
        classification_items * valid.astype(jnp.float32)
    ) / denominator
    commitment = jnp.sum(
        jnp.mean(jnp.square(predicted - target), axis=-1)
        * valid.astype(jnp.float32)
    ) / denominator
    total = classification + commitment
    return LossBundle(
        total=total,
        metrics={
            "response_encoder_classification": classification,
            "response_encoder_commitment": commitment,
        },
    )


def assert_training_batch_has_no_audit_labels(batch_mapping: Mapping[str, Any]) -> None:
    forbidden = {
        "family",
        "family_id",
        "seed",
        "training_seed",
        "checkpoint_index",
        "training_run_id",
        "partner_index",
        "partner_member_index",
    }
    overlap = forbidden & set(batch_mapping)
    if overlap:
        raise ValueError(
            "Differentiable training batches cannot contain audit labels: "
            + ", ".join(sorted(overlap))
        )


__all__ = [
    "FrozenAssignments",
    "LossBundle",
    "assert_training_batch_has_no_audit_labels",
    "bellman_loss",
    "bellman_targets",
    "episode_responsibilities",
    "episode_responsibility_evidence",
    "gaussian_negative_log_likelihood",
    "gather_actions",
    "huber",
    "outcome_loss",
    "response_encoder_loss",
    "sample_bootstrap_mask",
    "slot_outcome_nll_items",
]
