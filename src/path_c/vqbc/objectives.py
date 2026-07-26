"""Behavior-consistent Bellman-mixture and outcome objectives for VQBC V4.2."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple


class FrozenAssignments(NamedTuple):
    responsibilities: Any
    responsibility_energies: Any
    bootstrap_mask: Any
    bellman_targets: Any
    stale_bellman_targets: Any
    stale_current_beliefs: Any
    response_signature_targets: Any
    response_code_targets: Any
    continuation_use_targets: Any
    continuation_mask_targets: Any


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
    return jnp.squeeze(
        jnp.take_along_axis(source, indexes.reshape(tuple(shape)), axis=axis),
        axis=axis,
    )


def huber(error: Any, delta: float = 1.0) -> Any:
    import jax.numpy as jnp

    absolute = jnp.abs(jnp.asarray(error))
    quadratic = jnp.minimum(absolute, float(delta))
    return 0.5 * jnp.square(quadratic) + float(delta) * (absolute - quadratic)


def gaussian_negative_log_likelihood(
    value: Any, mean: Any, log_standard_deviation: Any
) -> Any:
    import jax.numpy as jnp

    log_std = jnp.asarray(log_standard_deviation)
    normalized = (jnp.asarray(value) - jnp.asarray(mean)) * jnp.exp(-log_std)
    return 0.5 * jnp.square(normalized) + log_std


def raw_policy_continuation_targets(
    *, execution_probabilities: Any, target_q_values: Any, dones: Any
) -> Any:
    """Raw expected Q under the exact target execution distribution."""

    import jax.numpy as jnp

    probabilities = jnp.asarray(execution_probabilities, dtype=jnp.float32)
    q_values = jnp.asarray(target_q_values, dtype=jnp.float32)
    if q_values.shape[-3] != 2:
        raise ValueError("Continuation targets require two Q estimators.")
    if q_values.shape[:-3] != probabilities.shape[:-1]:
        raise ValueError("Policy and target Q prefix axes differ.")
    if q_values.shape[-1] != probabilities.shape[-1]:
        raise ValueError("Policy and target Q action axes differ.")
    expected = jnp.einsum("...a,...ema->...em", probabilities, q_values)
    return jnp.where(
        jnp.asarray(dones, dtype=jnp.bool_)[..., None, None],
        jnp.zeros_like(expected),
        expected,
    )


def bellman_targets(
    *,
    rewards: Any,
    dones: Any,
    next_execution_probabilities: Any,
    target_next_q_values: Any,
    gamma: float,
) -> Any:
    import jax.numpy as jnp

    target_q = jnp.min(jnp.asarray(target_next_q_values), axis=-3)
    expected = jnp.einsum(
        "...a,...ma->...m", jnp.asarray(next_execution_probabilities), target_q
    )
    return jnp.asarray(rewards)[..., None] + float(gamma) * (
        1.0 - jnp.asarray(dones, dtype=jnp.float32)
    )[..., None] * expected


def _td_evidence_items(*, q_values: Any, actions: Any, targets: Any) -> Any:
    import jax.numpy as jnp

    selected = gather_actions(q_values, actions, action_axis=-1)
    return jnp.mean(
        huber(selected - jnp.asarray(targets)[..., None, :]), axis=-2
    )


def _selected_outcome_items(
    *,
    response_logits: Any,
    reward_mean: Any,
    reward_log_standard_deviation: Any,
    continuation_use_mean: Any,
    continuation_use_log_standard_deviation: Any,
    continuation_mask_mean: Any,
    continuation_mask_log_standard_deviation: Any,
    response_codes: Any,
    rewards: Any,
    continuation_use_targets: Any,
    continuation_mask_targets: Any,
    actions: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """Joint per-transition control evidence with shape [T,B,slot]."""

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

    def continuation_items(mean: Any, log_std: Any, target: Any) -> Any:
        selected_action_mean = gather_actions(mean, actions, action_axis=-2)
        selected_action_std = gather_actions(log_std, actions, action_axis=-2)
        selected_mean = gather_actions(
            selected_action_mean, response_codes, action_axis=-1
        )
        selected_std = gather_actions(
            selected_action_std, response_codes, action_axis=-1
        )
        return jnp.mean(
            gaussian_negative_log_likelihood(target, selected_mean, selected_std),
            axis=-2,
        )

    use_nll = continuation_items(
        continuation_use_mean,
        continuation_use_log_standard_deviation,
        continuation_use_targets,
    )
    mask_nll = continuation_items(
        continuation_mask_mean,
        continuation_mask_log_standard_deviation,
        continuation_mask_targets,
    )
    return response_nll + reward_nll + use_nll + mask_nll, {
        "response_nll_items": response_nll,
        "reward_nll_items": reward_nll,
        "continuation_use_nll_items": use_nll,
        "continuation_mask_nll_items": mask_nll,
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
    continuation_use_mean: Any | None = None,
    continuation_use_log_standard_deviation: Any | None = None,
    continuation_mask_mean: Any | None = None,
    continuation_mask_log_standard_deviation: Any | None = None,
    response_codes: Any | None = None,
    rewards: Any | None = None,
    continuation_use_targets: Any | None = None,
    continuation_mask_targets: Any | None = None,
    availability_mask: Any | None = None,
) -> tuple[Any, Any]:
    """Infer stopped slot responsibilities from full-episode evidence."""

    import jax
    import jax.numpy as jnp

    evidence_items = _td_evidence_items(
        q_values=q_values, actions=actions, targets=targets
    )
    optional = (
        response_logits,
        reward_mean,
        reward_log_standard_deviation,
        continuation_use_mean,
        continuation_use_log_standard_deviation,
        continuation_mask_mean,
        continuation_mask_log_standard_deviation,
        response_codes,
        rewards,
        continuation_use_targets,
        continuation_mask_targets,
    )
    if any(value is not None for value in optional):
        if any(value is None for value in optional):
            raise ValueError(
                "Joint responsibility evidence requires every outcome tensor."
            )
        outcome_items, unused = _selected_outcome_items(
            response_logits=response_logits,
            reward_mean=reward_mean,
            reward_log_standard_deviation=reward_log_standard_deviation,
            continuation_use_mean=continuation_use_mean,
            continuation_use_log_standard_deviation=(
                continuation_use_log_standard_deviation
            ),
            continuation_mask_mean=continuation_mask_mean,
            continuation_mask_log_standard_deviation=(
                continuation_mask_log_standard_deviation
            ),
            response_codes=response_codes,
            rewards=rewards,
            continuation_use_targets=continuation_use_targets,
            continuation_mask_targets=continuation_mask_targets,
            actions=actions,
        )
        del unused
        evidence_items = evidence_items + outcome_items
    energies = jnp.sum(evidence_items, axis=0)
    available = (
        jnp.ones_like(energies, dtype=jnp.bool_)
        if availability_mask is None
        else jnp.asarray(availability_mask, dtype=jnp.bool_)
    )
    if available.shape != energies.shape:
        raise ValueError("Responsibility availability must be [environment, slot].")
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
    return (
        jax.lax.stop_gradient(jax.nn.softmax(logits, axis=-1)),
        jax.lax.stop_gradient(energies),
    )


def episode_responsibilities(**kwargs: Any) -> Any:
    values, unused = episode_responsibility_evidence(**kwargs)
    del unused
    return values


def sample_bootstrap_mask(
    key: Any,
    *,
    environment_count: int,
    slot_count: int,
    probability: float,
) -> Any:
    import jax
    import jax.numpy as jnp

    mask_key, fallback_key = jax.random.split(key)
    mask = jax.random.bernoulli(
        mask_key, float(probability), (environment_count, slot_count)
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
    metric_prefix: str = "bellman",
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
    denominator = q_values.shape[0] * 2.0 * jnp.sum(weights) + 1.0e-8
    loss = jnp.sum(per_item * time_weights) / denominator
    return LossBundle(
        total=loss,
        metrics={
            metric_prefix: loss,
            f"{metric_prefix}_mean_absolute_td_error": jnp.sum(
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
    continuation_use_mean: Any,
    continuation_use_log_standard_deviation: Any,
    continuation_mask_mean: Any,
    continuation_mask_log_standard_deviation: Any,
    response_codes: Any,
    rewards: Any,
    continuation_use_targets: Any,
    continuation_mask_targets: Any,
    actions: Any,
    stopped_responsibilities: Any,
    bootstrap_mask: Any | None = None,
    terminal_response: int = 15,
) -> LossBundle:
    import jax
    import jax.numpy as jnp

    terms, parts = _selected_outcome_items(
        response_logits=response_logits,
        reward_mean=reward_mean,
        reward_log_standard_deviation=reward_log_standard_deviation,
        continuation_use_mean=continuation_use_mean,
        continuation_use_log_standard_deviation=(
            continuation_use_log_standard_deviation
        ),
        continuation_mask_mean=continuation_mask_mean,
        continuation_mask_log_standard_deviation=(
            continuation_mask_log_standard_deviation
        ),
        response_codes=response_codes,
        rewards=rewards,
        continuation_use_targets=continuation_use_targets,
        continuation_mask_targets=continuation_mask_targets,
        actions=actions,
    )
    weights = jnp.asarray(stopped_responsibilities)
    if bootstrap_mask is not None:
        weights = weights * jnp.asarray(bootstrap_mask, dtype=jnp.float32)
    weights = jax.lax.stop_gradient(weights)[None, ...]
    denominator = response_logits.shape[0] * jnp.sum(weights) + 1.0e-8
    fitted = jnp.sum(terms * weights) / denominator
    terminal_use = continuation_use_mean[..., int(terminal_response)]
    terminal_mask = continuation_mask_mean[..., int(terminal_response)]
    terminal_penalty = 0.5 * (
        jnp.mean(jnp.square(terminal_use))
        + jnp.mean(jnp.square(terminal_mask))
    )
    return LossBundle(
        total=fitted + terminal_penalty,
        metrics={
            "slot_outcome": fitted,
            "response_nll": jnp.sum(parts["response_nll_items"] * weights)
            / denominator,
            "reward_nll": jnp.sum(parts["reward_nll_items"] * weights)
            / denominator,
            "continuation_use_nll": jnp.sum(
                parts["continuation_use_nll_items"] * weights
            )
            / denominator,
            "continuation_mask_nll": jnp.sum(
                parts["continuation_mask_nll_items"] * weights
            )
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
    items = -jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=-1), safe_targets[..., None], axis=-1
    )[..., 0]
    denominator = jnp.sum(valid.astype(jnp.float32)) + 1.0e-8
    classification = jnp.sum(items * valid.astype(jnp.float32)) / denominator
    commitment = jnp.sum(
        jnp.mean(jnp.square(predicted - target), axis=-1)
        * valid.astype(jnp.float32)
    ) / denominator
    return LossBundle(
        total=classification + commitment,
        metrics={
            "response_encoder_classification": classification,
            "response_encoder_commitment": commitment,
        },
    )


def assert_training_batch_has_no_audit_labels(
    batch_mapping: Mapping[str, Any]
) -> None:
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
    "raw_policy_continuation_targets",
    "response_encoder_loss",
    "sample_bootstrap_mask",
]
