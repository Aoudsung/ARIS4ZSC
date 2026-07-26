"""Deterministic complete-link value quotients and quotient-level belief updates."""

from __future__ import annotations

from typing import Any, NamedTuple


class QuotientAggregation(NamedTuple):
    class_ids: Any
    class_mask: Any
    class_counts: Any
    class_belief: Any
    response_probabilities: Any
    reward_mean: Any
    next_q_mean: Any


def centered_advantages(q_values: Any) -> Any:
    import jax.numpy as jnp

    values = jnp.asarray(q_values)
    return values - jnp.mean(values, axis=-1, keepdims=True)


def value_signatures(q_values: Any) -> tuple[Any, Any]:
    """Return the mean centered advantage and the twin-disagreement radius."""

    import jax.numpy as jnp

    values = jnp.asarray(q_values)
    if values.shape[-3] != 2:
        raise ValueError("Value signatures require exactly two Q estimators.")
    centered = centered_advantages(values)
    signature = 0.5 * (centered[..., 0, :, :] + centered[..., 1, :, :])
    radius = jnp.max(
        jnp.abs(centered[..., 0, :, :] - centered[..., 1, :, :]), axis=-1
    )
    return signature, radius


def complete_link_class_ids(
    signatures: Any, radii: Any, *, tolerance: float = 1.0e-6
) -> Any:
    """Group slots in index order, requiring compatibility with every member."""

    import jax
    import jax.numpy as jnp

    signature_values = jnp.asarray(signatures)
    radius_values = jnp.asarray(radii)
    if (
        signature_values.ndim < 2
        or radius_values.shape != signature_values.shape[:-1]
    ):
        raise ValueError("Signatures and radii have incompatible shapes.")
    slot_count = int(signature_values.shape[-2])
    flat_signatures = signature_values.reshape((-1, slot_count, signature_values.shape[-1]))
    flat_radii = radius_values.reshape((-1, slot_count))

    def one(signatures_at_state: Any, radii_at_state: Any) -> Any:
        distances = jnp.max(
            jnp.abs(
                signatures_at_state[:, None, :]
                - signatures_at_state[None, :, :]
            ),
            axis=-1,
        )
        compatible = distances <= (
            radii_at_state[:, None] + radii_at_state[None, :] + tolerance
        )
        initial = (
            jnp.full((slot_count,), -1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

        def assign(index: int, carry: tuple[Any, Any]) -> tuple[Any, Any]:
            class_ids, class_count = carry
            class_indexes = jnp.arange(slot_count, dtype=jnp.int32)
            member_matrix = class_ids[None, :] == class_indexes[:, None]
            earlier = jnp.arange(slot_count) < index
            members = member_matrix & earlier[None, :]
            nonempty = jnp.any(members, axis=-1)
            compatible_with_all = jnp.all(
                jnp.where(members, compatible[index][None, :], True), axis=-1
            )
            eligible = nonempty & compatible_with_all & (class_indexes < class_count)
            first = jnp.min(
                jnp.where(eligible, class_indexes, jnp.asarray(slot_count, jnp.int32))
            )
            selected = jnp.where(first < slot_count, first, class_count)
            next_count = jnp.maximum(class_count, selected + 1)
            return class_ids.at[index].set(selected), next_count

        return jax.lax.fori_loop(0, slot_count, assign, initial)[0]

    flat_ids = jax.vmap(one)(flat_signatures, flat_radii)
    return flat_ids.reshape(signature_values.shape[:-2] + (slot_count,))


def class_membership(class_ids: Any, slot_count: int) -> tuple[Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    ids = jnp.asarray(class_ids, dtype=jnp.int32)
    membership = jax.nn.one_hot(ids, slot_count, dtype=jnp.float32)
    counts = jnp.sum(membership, axis=-2)
    mask = counts > 0.0
    return membership, counts, mask


def class_sum(values: Any, membership: Any) -> Any:
    import jax.numpy as jnp

    source = jnp.asarray(values)
    members = jnp.asarray(membership)
    return jnp.einsum("...mc,...m->...c", members, source)


def class_mean(values: Any, membership: Any, counts: Any) -> Any:
    """Uniformly average a slot-axis tensor into its current classes."""

    import jax.numpy as jnp

    source = jnp.asarray(values)
    members = jnp.asarray(membership)
    count_values = jnp.asarray(counts)
    prefix_rank = members.ndim - 2
    slot_axis = prefix_rank
    flattened = source.reshape(
        source.shape[:prefix_rank] + (source.shape[slot_axis], -1)
    )
    reduced = jnp.einsum("...mc,...mZ->...cZ", members, flattened)
    safe_counts = jnp.maximum(count_values, 1.0)[..., :, None]
    result = reduced / safe_counts
    suffix = source.shape[slot_axis + 1 :]
    return result.reshape(result.shape[:-1] + suffix)


def aggregate_slots(
    *,
    class_ids: Any,
    slot_log_belief: Any,
    response_probabilities: Any,
    reward_mean: Any,
    next_q_mean: Any,
) -> QuotientAggregation:
    """Build all class-level kernels using uniform within-class averaging."""

    import jax.numpy as jnp

    response = jnp.asarray(response_probabilities)
    slot_count = int(response.shape[-3])
    membership, counts, mask = class_membership(class_ids, slot_count)
    slot_probabilities = jnp.exp(
        jnp.asarray(slot_log_belief)
        - jnp.max(jnp.asarray(slot_log_belief), axis=-1, keepdims=True)
    )
    slot_probabilities = slot_probabilities / jnp.sum(
        slot_probabilities, axis=-1, keepdims=True
    )
    class_belief = class_sum(slot_probabilities, membership)
    class_belief = jnp.where(mask, class_belief, 0.0)
    class_belief = class_belief / jnp.sum(class_belief, axis=-1, keepdims=True)
    # Continuation values carry an estimator axis before the slot axis.  Move
    # it behind the slot while applying the same uniform class mean, then
    # restore the public [..., estimator, class, action, response, next-action]
    # layout.
    continuation_by_slot = jnp.moveaxis(jnp.asarray(next_q_mean), -5, -4)
    continuation_by_class = class_mean(
        continuation_by_slot, membership, counts
    )
    continuation_by_class = jnp.moveaxis(continuation_by_class, -4, -5)
    return QuotientAggregation(
        class_ids=class_ids,
        class_mask=mask,
        class_counts=counts,
        class_belief=class_belief,
        response_probabilities=class_mean(response, membership, counts),
        reward_mean=class_mean(reward_mean, membership, counts),
        next_q_mean=continuation_by_class,
    )


def quotient_bayes_update(
    *,
    class_ids: Any,
    class_belief: Any,
    class_response_probabilities: Any,
    action: Any,
    response_code: Any,
    class_counts: Any,
    class_mask: Any,
    probability_floor: float = 1.0e-8,
) -> Any:
    """Update class belief, then write each class uniformly to its slots."""

    import jax
    import jax.numpy as jnp

    probabilities = jnp.asarray(class_response_probabilities)
    action_index = jnp.asarray(action, dtype=jnp.int32)
    response_index = jnp.asarray(response_code, dtype=jnp.int32)
    selected_action = jnp.take_along_axis(
        probabilities,
        action_index[..., None, None, None],
        axis=-2,
    )[..., 0, :]
    likelihood = jnp.take_along_axis(
        selected_action, response_index[..., None, None], axis=-1
    )[..., 0]
    likelihood = jnp.maximum(likelihood, probability_floor)
    posterior = jnp.asarray(class_belief) * likelihood
    posterior = jnp.where(class_mask, posterior, 0.0)
    posterior = posterior / jnp.sum(posterior, axis=-1, keepdims=True)
    membership = jax.nn.one_hot(
        jnp.asarray(class_ids, dtype=jnp.int32),
        int(probabilities.shape[-3]),
        dtype=posterior.dtype,
    )
    per_class_slot_probability = posterior / jnp.maximum(class_counts, 1.0)
    slot_probability = jnp.einsum(
        "...mc,...c->...m", membership, per_class_slot_probability
    )
    return jnp.log(jnp.maximum(slot_probability, probability_floor))


__all__ = [
    "QuotientAggregation",
    "aggregate_slots",
    "centered_advantages",
    "class_mean",
    "class_membership",
    "class_sum",
    "complete_link_class_ids",
    "quotient_bayes_update",
    "value_signatures",
]
